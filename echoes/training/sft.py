from __future__ import annotations

import math
import random

import torch
import torch.distributed as dist

from echoes.cta import format_cta
from echoes.prompts import student_messages
from echoes.training.checkpoint import append_metrics, prepare_output, restore, save
from echoes.training.config import load_config
from echoes.training.distributed import average_metrics, initialize, summed
from echoes.training.engine import create_engine
from echoes.training.objectives import token_log_probs
from echoes.training.records import fingerprint, image_path, load_trajectories, targets
from echoes.training.rollout import encode


def target_example(engine, trajectory: dict, index: int, config: dict):
    step = trajectory["steps"][index]
    previous = trajectory["steps"][index - 1]["compression"] if index else ""
    for text in (previous, step["compression"]):
        if len(engine.tokenizer.encode(text, add_special_tokens=False)) > config["training"]["memory_budget"]:
            raise ValueError("SFT compression exceeds the configured token budget")
    inputs = encode(engine, student_messages(trajectory["instruction"], image_path(config, step), previous), config)
    output = format_cta(step["compression"], step["think"], step["action"])
    response = engine.tokenizer.encode(output, add_special_tokens=False) + [engine.tokenizer.eos_token_id]
    if len(response) > config["training"]["max_new_tokens"]:
        raise ValueError("SFT target exceeds max_new_tokens; increase the limit to retain the complete CTA target")
    return inputs, torch.tensor(response, device=engine.device)


def main():
    config, _ = load_config("sft")
    rank, world, _ = initialize(config["runtime"]["seed"])
    output = prepare_output(config)
    trajectories = load_trajectories(config["data"]["train"], config["data"]["image_root"], "train")
    selected = targets(trajectories)
    settings = config["training"]
    global_batch = world * settings["prompts_per_rank"]
    steps_per_epoch = math.ceil(len(selected) / global_batch)
    total_steps = steps_per_epoch * settings["epochs"]
    engine = create_engine(config, total_steps)
    for trajectory in trajectories:
        for step in trajectory["steps"]:
            if not all(isinstance(step.get(key), str) for key in ("compression", "think")):
                raise ValueError("Every retained SFT prefix requires compression and think annotations")
            if len(engine.tokenizer.encode(step["compression"], add_special_tokens=False)) > settings["memory_budget"]:
                raise ValueError("SFT compression exceeds the 128-token budget")
            format_cta(step["compression"], step["think"], step["action"])
    identities = {"stage": "sft", "train_fingerprint": fingerprint(trajectories, config["data"]["image_root"])}
    state = restore(engine, config, identities) or {**identities, "global_step": 0}
    for epoch in range(settings["epochs"]):
        order = list(selected)
        random.Random(config["runtime"]["seed"] + epoch).shuffle(order)
        for batch_index in range(steps_per_epoch):
            expected_step = epoch * steps_per_epoch + batch_index
            if expected_step < state["global_step"]:
                continue
            examples = []
            for offset in range(settings["prompts_per_rank"]):
                global_index = batch_index * global_batch + rank * settings["prompts_per_rank"] + offset
                active = global_index < len(order)
                trajectory_index, step_index = order[global_index if active else 0]
                inputs, response = target_example(engine, trajectories[trajectory_index], step_index, config)
                examples.append((inputs, response, active))
            global_tokens = summed(sum(response.numel() for _, response, active in examples if active), engine.device)
            engine.module.train()
            engine.optimizer_zero_grad()
            local_loss = 0.0
            for inputs, response, active in examples:
                logits = engine.response_logits(inputs, response)
                loss = -token_log_probs(logits, response, settings["logit_chunk_size"]).sum()
                loss = loss * (world / global_tokens if active else 0.0)
                loss.backward()
                local_loss += float(loss.detach())
                del logits, loss
            norm = float(engine.optimizer_step())
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm; the update was rejected")
            engine.lr_scheduler_step()
            state["global_step"] += 1
            metrics = average_metrics({"sft_loss": local_loss, "gradient_norm": norm}, engine.device)
            append_metrics(output, {"event": "update", "epoch": epoch + 1, "global_step": state["global_step"], **metrics})
            if state["global_step"] % settings["save_every"] == 0 or state["global_step"] == total_steps:
                save(engine, output, config, state)
    engine.export_hf(str(output / "final"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
