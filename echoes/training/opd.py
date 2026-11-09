from __future__ import annotations

import math
import random

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

from echoes.training.checkpoint import append_metrics, prepare_output, restore, save
from echoes.training.config import load_config
from echoes.training.distributed import average_metrics, initialize, summed, timed
from echoes.training.engine import create_engine
from echoes.training.objectives import full_vocab_forward_kl, mixture_weight, token_log_probs
from echoes.training.records import assert_disjoint, fingerprint, fixed_validation, load_trajectories, targets
from echoes.training.rollout import collect, make_batch, to_device
from echoes.training.validation import paired_scores


def update(student, teacher, samples, mixture: float, config: dict) -> dict[str, float]:
    settings = config["training"]
    batch = make_batch(samples, student.tokenizer.pad_token_id)
    global_tokens = summed(float(batch.batch["distillation_mask"].sum()), student.device)
    world = dist.get_world_size()
    actor_config = OmegaConf.create({"clip_ratio": settings["clip_ratio"], "clip_ratio_low": None,
                                    "clip_ratio_high": None, "clip_ratio_c": 1e9, "global_batch_info": {}})
    student.module.train()
    teacher.module.eval()
    student.optimizer_zero_grad()
    totals = {"opd_loss": 0.0, "grpo_loss": 0.0, "loss": 0.0}
    for key in samples[0].metrics:
        totals[key] = sum(sample.metrics[key] for sample in samples) / len(samples)
    for row, sample in enumerate(samples):
        response = sample.response_ids.to(student.device)
        size = response.numel()
        teacher_logits = None
        if mixture > 0:
            with torch.no_grad():
                teacher_logits = teacher.response_logits(to_device(sample.teacher_inputs, student.device), response)
        student_logits = student.response_logits(to_device(sample.student_inputs, student.device), response)
        log_probs = token_log_probs(student_logits, response, settings["logit_chunk_size"])
        # Each rollout contributes its token-mean policy loss; OPD uses one global token mean.
        grpo_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=batch.batch["old_log_probs"][row:row + 1, :size].to(student.device),
            log_prob=log_probs,
            advantages=batch.batch["advantages"][row:row + 1, :size].to(student.device),
            response_mask=batch.batch["response_mask"][row:row + 1, :size].to(student.device),
            loss_agg_mode="seq-mean-token-mean", config=actor_config,
        )
        grpo_loss = grpo_loss / len(samples)
        if teacher_logits is not None:
            token_kl = full_vocab_forward_kl(student_logits, teacher_logits, settings["logit_chunk_size"])
            opd_mask = sample.distillation_mask.to(student.device).unsqueeze(0)
            opd_loss = (token_kl * opd_mask).sum() * world / max(1.0, global_tokens)
        else:
            opd_loss = log_probs.sum() * 0
        loss = mixture * opd_loss + (1 - mixture) * grpo_loss
        loss.backward()
        totals["opd_loss"] += float(opd_loss.detach())
        totals["grpo_loss"] += float(grpo_loss.detach())
        totals["loss"] += float(loss.detach())
        del teacher_logits, student_logits, log_probs, loss, opd_loss, grpo_loss
    norm = float(student.optimizer_step())
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm; the update was rejected")
    student.lr_scheduler_step()
    totals["gradient_norm"] = norm
    totals["distilled_tokens"] = global_tokens
    return average_metrics(totals, student.device)


def main():
    config, _ = load_config("opd")
    rank, world, _ = initialize(config["runtime"]["seed"])
    output = prepare_output(config)
    train = load_trajectories(config["data"]["train"], config["data"]["image_root"], "train")
    validation = load_trajectories(config["data"]["validation"], config["data"]["image_root"], "validation")
    assert_disjoint(train, validation, config["data"]["image_root"])
    validation = fixed_validation(validation, config["data"]["validation_trajectories"], config["runtime"]["seed"])
    identities = {"stage": "opd", "train_fingerprint": fingerprint(train, config["data"]["image_root"]),
                  "validation_fingerprint": fingerprint(validation, config["data"]["image_root"])}
    settings = config["training"]
    total_steps = settings["rounds"] * settings["steps_per_round"]
    student = create_engine(config, total_steps)
    teacher = create_engine(config, total_steps, teacher=True)
    state = restore(student, config, identities)
    if state is None:
        initial_scores, validation_seconds = timed(paired_scores, student, validation, config, device=student.device)
        state = {**identities, "global_step": 0, "completed_rounds": 0,
                 "initial_gap": initial_scores["gap"], "last_gap": initial_scores["gap"]}
        append_metrics(output, {"event": "initial_validation", "global_step": 0,
                                "validation_seconds": validation_seconds, **initial_scores})
    if state["global_step"] != state["completed_rounds"] * settings["steps_per_round"]:
        raise ValueError("OPD checkpoints must be saved at teacher-refresh boundaries")
    selected_targets = targets(train)
    for round_index in range(state["completed_rounds"], settings["rounds"]):
        _, refresh_seconds = timed(teacher.refresh_from, student, device=student.device)
        append_metrics(output, {"event": "teacher_refresh", "round": round_index + 1,
                                "global_step": state["global_step"], "refresh_seconds": refresh_seconds})
        mixture = mixture_weight(config["schedule"], state["last_gap"], state["initial_gap"], round_index, settings["rounds"])
        for _ in range(settings["steps_per_round"]):
            generator = random.Random(config["runtime"]["seed"] + state["global_step"] * world + rank)
            selected = generator.choices(selected_targets, k=settings["prompts_per_rank"])
            samples, rollout_seconds = timed(collect, student, train, selected, config, device=student.device)
            metrics, update_seconds = timed(update, student, teacher, samples, mixture, config, device=student.device)
            state["global_step"] += 1
            append_metrics(output, {"event": "update", "round": round_index + 1,
                                    "global_step": state["global_step"], "lambda_opd": mixture,
                                    "rollout_seconds": rollout_seconds, "update_seconds": update_seconds, **metrics})
        scores, validation_seconds = timed(paired_scores, student, validation, config, device=student.device)
        state.update(completed_rounds=round_index + 1, last_gap=scores["gap"])
        next_weight = mixture_weight(config["schedule"], scores["gap"], state["initial_gap"],
                                     round_index + 1, settings["rounds"])
        append_metrics(output, {"event": "round_validation", "round": round_index + 1,
                                "global_step": state["global_step"], "lambda_next": next_weight,
                                "validation_seconds": validation_seconds, **scores})
        if state["global_step"] % settings["save_every"] == 0 or round_index + 1 == settings["rounds"]:
            save(student, output, config, state)
    student.export_hf(str(output / "final"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
