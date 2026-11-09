from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from echoes.actions import action_match
from echoes.cta import compression_text, parse_cta


def _forward_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    log_student = F.log_softmax(student.float(), dim=-1)
    log_teacher = F.log_softmax(teacher.detach().float(), dim=-1)
    return (log_teacher.exp() * (log_teacher - log_student)).sum(-1)


def full_vocab_forward_kl(student: torch.Tensor, teacher: torch.Tensor,
                         chunk_size: int = 32) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Teacher and student must score the same response positions and vocabulary")
    parts = []
    for start in range(0, student.shape[1], chunk_size):
        s, t = student[:, start:start + chunk_size], teacher[:, start:start + chunk_size]
        parts.append(checkpoint(_forward_kl, s, t, use_reentrant=False) if s.requires_grad else _forward_kl(s, t))
    return torch.cat(parts, dim=1)


def _token_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="none").reshape(ids.shape)


def token_log_probs(logits: torch.Tensor, ids: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    parts = []
    for start in range(0, ids.shape[1], chunk_size):
        chunk, target = logits[:, start:start + chunk_size], ids[:, start:start + chunk_size]
        parts.append(checkpoint(_token_log_probs, chunk, target, use_reentrant=False)
                     if chunk.requires_grad else _token_log_probs(chunk, target))
    return torch.cat(parts, dim=1)


def reward(text: str, target: dict, tokenizer, config: dict, bbox=None) -> dict[str, float]:
    length = len(tokenizer.encode(compression_text(text), add_special_tokens=False))
    length_cost = config["length_weight"] * max(0, length - config["memory_budget"]) / config["memory_budget"]
    try:
        cta = parse_cta(text)
        correct = float(action_match(cta.action, target, bbox=bbox, point_tolerance=config["point_tolerance"]))
        valid = 1.0
    except (ValueError, TypeError):
        correct, valid = 0.0, 0.0
    return {"reward": config["action_weight"] * correct + config["format_weight"] * valid - length_cost,
            "action_accuracy": correct, "format_validity": valid,
            "compression_tokens": float(length), "length_penalty": length_cost}


def pmf_weight(gap: float, initial_gap: float, lower: float = 0.2, upper: float = 0.8,
               epsilon: float = 1e-6) -> float:
    return min(upper, max(lower, lower + (upper - lower) * max(0.0, gap) / (initial_gap + epsilon)))


def mixture_weight(config: dict, gap: float, initial_gap: float, round_index: int, rounds: int) -> float:
    mode = config["mode"]
    if mode == "rl_only":
        return 0.0
    if mode == "opd_only":
        return 1.0
    if mode == "fixed":
        return float(config["fixed_weight"])
    if mode == "linear":
        progress = min(1.0, max(0.0, round_index / max(1, rounds - 1)))
        return config["lambda_max"] + progress * (config["lambda_min"] - config["lambda_max"])
    return pmf_weight(gap, initial_gap, config["lambda_min"], config["lambda_max"], config["epsilon"])
