from __future__ import annotations

import torch
import torch.distributed as dist

from echoes.actions import action_match
from echoes.cta import decode_response, parse_cta
from echoes.prompts import privileged_messages, student_messages
from echoes.training.distributed import maximum, summed
from echoes.training.records import image_path
from echoes.training.rollout import encode, next_memory


@torch.no_grad()
def paired_scores(engine, trajectories: list[dict], config: dict) -> dict[str, float]:
    if not trajectories or any(item.get("split") != "validation" for item in trajectories):
        raise ValueError("PMF scheduling requires a nonempty held-out validation set")
    engine.module.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    local = trajectories[rank::world]
    scores = {"deployable": 0.0, "privileged": 0.0}
    count = 0
    for slot in range(maximum(len(local), engine.device)):
        real_trajectory = slot < len(local)
        trajectory = local[slot] if real_trajectory else trajectories[0]
        steps = trajectory["steps"]
        memory = ""
        for index in range(maximum(len(steps) if real_trajectory else 0, engine.device)):
            active = real_trajectory and index < len(steps)
            step = steps[index] if active else steps[0]
            deployable = student_messages(trajectory["instruction"], image_path(config, step), memory if active else "")
            history = steps[max(0, index - config["training"]["history_frames"]):index + 1] if active else [step]
            privileged = privileged_messages(trajectory["instruction"], [image_path(config, item) for item in history])
            for view, messages in (("deployable", deployable), ("privileged", privileged)):
                ids = engine.generate_ids(encode(engine, messages, config), config["training"]["max_new_tokens"], 0.0)
                text = decode_response(ids, engine.tokenizer)
                if active:
                    try:
                        cta = parse_cta(text)
                        scores[view] += float(action_match(cta.action, step["action"], bbox=step.get("bbox"),
                                                           point_tolerance=config["reward"]["point_tolerance"]))
                    except (ValueError, TypeError):
                        pass
                    if view == "deployable":
                        memory = next_memory(text, memory, engine.tokenizer, config["training"]["memory_budget"])
            count += int(active)
    total = summed(count, engine.device)
    if total == 0:
        raise ValueError("No validation steps")
    scores = {view: summed(value, engine.device) / total for view, value in scores.items()}
    return {**scores, "gap": max(0.0, scores["privileged"] - scores["deployable"]), "steps": int(total)}
