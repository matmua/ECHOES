from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from echoes.cta import decode_response, distillation_token_mask, parse_cta, truncate_memory
from echoes.inference import encode_messages
from echoes.prompts import privileged_messages, student_messages
from echoes.training.distributed import maximum
from echoes.training.objectives import reward, token_log_probs
from echoes.training.records import image_path


@dataclass
class Sample:
    student_inputs: dict
    teacher_inputs: dict
    response_ids: torch.Tensor
    old_log_probs: torch.Tensor
    distillation_mask: torch.Tensor
    group: int
    metrics: dict[str, float]


def to_device(inputs: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def next_memory(text: str, previous: str, tokenizer, budget: int) -> str:
    try:
        return truncate_memory(parse_cta(text).compression, tokenizer, budget)
    except (ValueError, TypeError):
        return previous


def encode(engine, messages, config: dict):
    return encode_messages(engine.processor, messages, engine.device,
                           config["model"]["max_prompt_tokens"])


@torch.no_grad()
def rollout_prefix(engine, trajectory: dict, target_step: int, config: dict) -> str:
    memory = ""
    steps = trajectory["steps"]
    for index in range(maximum(target_step, engine.device)):
        active = index < target_step
        step = steps[index] if active else steps[0]
        messages = student_messages(trajectory["instruction"], image_path(config, step), memory if active else "")
        inputs = encode(engine, messages, config)
        ids = engine.generate_ids(inputs, config["training"]["max_new_tokens"], config["training"]["temperature"])
        if active:
            memory = next_memory(decode_response(ids, engine.tokenizer), memory,
                                 engine.tokenizer, config["training"]["memory_budget"])
    return memory


@torch.no_grad()
def collect(engine, trajectories: list[dict], selected: list[tuple[int, int]], config: dict) -> list[Sample]:
    engine.module.eval()
    samples = []
    settings = config["training"]
    reward_config = {**config["reward"], "memory_budget": settings["memory_budget"]}
    for group, (trajectory_index, step_index) in enumerate(selected):
        trajectory = trajectories[trajectory_index]
        step = trajectory["steps"][step_index]
        memory = rollout_prefix(engine, trajectory, step_index, config)
        student_inputs = encode(engine, student_messages(trajectory["instruction"], image_path(config, step), memory), config)
        preceding = trajectory["steps"][max(0, step_index - settings["history_frames"]):step_index + 1]
        teacher_inputs = encode(engine, privileged_messages(trajectory["instruction"],
                                                             [image_path(config, item) for item in preceding]), config)
        for _ in range(settings["group_size"]):
            response = engine.generate_ids(student_inputs, settings["max_new_tokens"], settings["temperature"])
            logits = engine.response_logits(student_inputs, response)
            old_log_probs = token_log_probs(logits, response, settings["logit_chunk_size"]).squeeze(0).cpu()
            del logits
            text = decode_response(response, engine.tokenizer)
            distillation_mask = torch.tensor(distillation_token_mask(response.cpu(), engine.tokenizer), dtype=torch.float32)
            scores = reward(text, step["action"], engine.tokenizer, reward_config, bbox=step.get("bbox"))
            samples.append(Sample(to_device(student_inputs, "cpu"), to_device(teacher_inputs, "cpu"),
                                  response.cpu(), old_log_probs, distillation_mask, group, scores))
    return samples


def make_batch(samples: list[Sample], pad_token_id: int) -> DataProto:
    length = max(sample.response_ids.numel() for sample in samples)
    responses = torch.full((len(samples), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(responses, dtype=torch.float32)
    distillation_mask = torch.zeros_like(mask)
    old_log_probs = torch.zeros_like(mask)
    rewards = torch.zeros_like(mask)
    for row, sample in enumerate(samples):
        size = sample.response_ids.numel()
        responses[row, :size] = sample.response_ids
        mask[row, :size] = 1
        distillation_mask[row, :size] = sample.distillation_mask
        old_log_probs[row, :size] = sample.old_log_probs
        rewards[row, size - 1] = sample.metrics["reward"]
    groups = np.array([sample.group for sample in samples])
    advantages, returns = compute_grpo_outcome_advantage(rewards, mask, groups, norm_adv_by_std_in_grpo=True)
    return DataProto.from_dict(tensors={"responses": responses, "response_mask": mask,
                                       "distillation_mask": distillation_mask,
                                       "old_log_probs": old_log_probs, "token_level_rewards": rewards,
                                       "advantages": advantages, "returns": returns},
                               non_tensors={"group": groups})
