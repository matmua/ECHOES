from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean

from echoes.actions import action_match, normalize_text


def _anls(first, second):
    first, second = normalize_text(first), normalize_text(second)
    if not first and not second:
        return 1.0
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return 1 - previous[-1] / max(len(first), len(second))


def score_action(prediction, target, bbox=None, point_tolerance=0.14,
                 text_match="exact", anls_threshold=0.5):
    if text_match not in {"exact", "anls"} or not 0 <= anls_threshold <= 1:
        raise ValueError("Invalid text matching protocol")
    valid = prediction is not None
    type_match = valid and prediction.get("type") == target["type"]
    exact = valid and action_match(prediction, target, bbox=bbox,
                                   point_tolerance=point_tolerance)
    if type_match and target["type"] == "TYPE" and text_match == "anls":
        exact = _anls(prediction["text"], target["text"]) >= anls_threshold
    grounding = None
    if target["type"] in {"CLICK", "LONG_PRESS"}:
        if valid and prediction.get("type") in {"CLICK", "LONG_PRESS"}:
            location = dict(prediction, type=target["type"])
            grounding = bool(action_match(location, target, bbox=bbox,
                                          point_tolerance=point_tolerance))
        else:
            grounding = False
    return {"type_match": bool(type_match), "exact_match": bool(exact),
            "grounding_match": grounding}


def aggregate_steps(rows):
    if not rows:
        return {"steps": 0}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["trajectory_id"])].append(row)
    ground = [row["grounding_match"] for row in rows if row["grounding_match"] is not None]
    return {
        "steps": len(rows), "trajectories": len(groups),
        "TM": mean(float(row["type_match"]) for row in rows),
        "EM": mean(float(row["exact_match"]) for row in rows),
        "action_type_accuracy": mean(float(row["type_match"]) for row in rows),
        "step_accuracy": mean(float(row["exact_match"]) for row in rows),
        "grounding_accuracy": mean(ground) if ground else None,
        "grounding_steps": len(ground),
        "trajectory_exact_match": mean(all(r["exact_match"] for r in group)
                                       for group in groups.values()),
        "parse_failure_rate": mean(bool(row.get("parse_error")) for row in rows),
        "mean_input_tokens": mean(row["input_tokens"] for row in rows),
        "mean_output_tokens": mean(row["output_tokens"] for row in rows),
        "mean_latency_seconds": mean(row["latency_seconds"] for row in rows),
        "total_latency_seconds": sum(row["latency_seconds"] for row in rows),
    }


def gui_report(rows):
    datasets = sorted({row["dataset"] for row in rows})
    report = {"overall": aggregate_steps(rows), "datasets": {
        name: aggregate_steps([r for r in rows if r["dataset"] == name]) for name in datasets
    }}
    report["trajectory_length"] = {
        "at_most_12": aggregate_steps([r for r in rows if r["trajectory_length"] <= 12]),
        "over_12": aggregate_steps([r for r in rows if r["trajectory_length"] > 12]),
    }
    report["memory_dependency"] = {
        "dependent": aggregate_steps([r for r in rows if r.get("memory_dependent") is True]),
        "independent": aggregate_steps([r for r in rows if r.get("memory_dependent") is False]),
        "unannotated_steps": sum(r.get("memory_dependent") is None for r in rows),
    }
    return report


def _quantile(ordered, q):
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap(first, second, resamples=10_000, seed=42, confidence=0.95):
    """Resample paired trajectories and recompute the pooled step-level EM."""
    if resamples < 2 or not 0 < confidence < 1:
        raise ValueError("resamples must be at least 2 and confidence must lie in (0, 1)")
    def index(rows):
        result = {}
        for row in rows:
            key = (row["dataset"], row["trajectory_id"], row["step_id"])
            if key in result:
                raise ValueError(f"Duplicate evaluation key: {key}")
            result[key] = row
        return result
    a, b = index(first), index(second)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired bootstrap requires identical, nonempty evaluated step sets")
    grouped = defaultdict(lambda: [0.0, 0.0, 0])
    for key in a:
        if a[key].get("reference") != b[key].get("reference"):
            raise ValueError(f"Paired evaluation references differ at {key}")
        item = grouped[key[:2]]
        item[0] += float(a[key]["exact_match"])
        item[1] += float(b[key]["exact_match"])
        item[2] += 1
    trajectories = list(grouped.values())
    if len(trajectories) < 2:
        raise ValueError("Trajectory bootstrap requires at least two distinct trajectories")
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        selected = rng.choices(trajectories, k=len(trajectories))
        draws.append(sum(x[0] - x[1] for x in selected) / sum(x[2] for x in selected))
    draws.sort()
    alpha = (1 - confidence) / 2
    nonpositive = (sum(x <= 0 for x in draws) + 1) / (resamples + 1)
    nonnegative = (sum(x >= 0 for x in draws) + 1) / (resamples + 1)
    delta = sum(x[0] - x[1] for x in trajectories) / sum(x[2] for x in trajectories)
    centered_tail = (sum(abs(x - delta) >= abs(delta) for x in draws) + 1) / (resamples + 1)
    return {
        "delta_EM": delta,
        "confidence_interval": [_quantile(draws, alpha), _quantile(draws, 1 - alpha)],
        "two_sided_centered_bootstrap_p": centered_tail,
        "two_sided_bootstrap_sign_tail_probability": min(1.0, 2 * min(nonpositive, nonnegative)),
        "confidence": confidence, "resamples": resamples, "seed": seed,
        "paired_trajectories": len(trajectories), "paired_steps": len(a),
    }
