from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from echoes.cta import parse_cta, truncate_memory
from echoes.io import load_image, read_jsonl, write_jsonl
from echoes.prompts import privileged_messages, student_messages
from .metrics import gui_report, paired_bootstrap, score_action


def evaluate_trajectories(policy, trajectories, image_root, *, view="deployable",
                          memory_mode="recursive", memory_budget=128, stale_lag=2,
                          point_tolerance=0.14, seed=42, text_match="exact", anls_threshold=0.5):
    if view not in {"deployable", "privileged"}:
        raise ValueError("view must be deployable or privileged")
    if memory_mode not in {"recursive", "none", "shuffle", "stale", "naive"}:
        raise ValueError("Unknown memory intervention")
    if view == "privileged" and memory_mode != "recursive":
        raise ValueError("Memory interventions apply only to the deployable view")
    if memory_budget < 1 or stale_lag < 2:
        raise ValueError("memory_budget must be positive and stale_lag must be at least 2")
    trajectories = list(trajectories)
    splits = {item.get("split") for item in trajectories}
    if len(splits) != 1 or not splits <= {"validation", "test"}:
        raise ValueError("Evaluation requires a single held-out validation or test split")
    keys = [(t["dataset"], t["trajectory_id"]) for t in trajectories]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate trajectory identifiers")
    donor_memory = {}
    donor_map = {}
    if memory_mode == "shuffle":
        if len(trajectories) < 2:
            raise ValueError("Shuffled memory requires at least two trajectories")
        baseline, _ = evaluate_trajectories(policy, trajectories, image_root,
                                          memory_budget=memory_budget,
                                          point_tolerance=point_tolerance, seed=seed,
                                          text_match=text_match, anls_threshold=anls_threshold)
        for row in baseline:
            donor_memory.setdefault((row["dataset"], row["trajectory_id"]), []).append(row["compression"])
        order = list(keys)
        random.Random(seed).shuffle(order)
        donor_map = {key: order[(i + 1) % len(order)] for i, key in enumerate(order)}
    root = Path(image_root)
    rows = []
    for trajectory in trajectories:
        memory, states = "", []
        steps = trajectory["steps"]
        if not steps:
            raise ValueError("Evaluation trajectories must contain observations")
        for index, step in enumerate(steps):
            image = load_image(root / step["image"])
            memory_input = memory
            if memory_mode == "none":
                memory_input = ""
            elif memory_mode == "stale":
                memory_input = states[index - stale_lag] if index >= stale_lag else ""
            elif memory_mode == "shuffle":
                donor = donor_memory[donor_map[(trajectory["dataset"], trajectory["trajectory_id"])]]
                memory_input = donor[min(index - 1, len(donor) - 1)] if index else ""
            if view == "privileged":
                images = [load_image(root / item["image"])
                          for item in steps[max(0, index - 4):index + 1]]
                messages = privileged_messages(trajectory["instruction"], images)
            else:
                messages = student_messages(trajectory["instruction"], image, memory_input)
            generation = policy.generate(messages)
            prediction, error = None, None
            try:
                output = parse_cta(generation.text)
                prediction = output.action
                memory = truncate_memory(output.compression, policy.tokenizer, memory_budget)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            extra_tokens, extra_output, extra_latency = 0, 0, 0.0
            if memory_mode == "naive" and view == "deployable":
                summary = policy.generate([{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Update a concise factual summary of the interface and "
                     "the observed interaction history. Return only the updated summary. "
                     f"Previous summary:\n{memory_input}"},
                ]}])
                memory = truncate_memory(summary.text, policy.tokenizer, memory_budget)
                extra_tokens, extra_output = summary.input_tokens, summary.output_tokens
                extra_latency = summary.latency_seconds
            states.append(memory)
            row = {
                "dataset": trajectory["dataset"], "trajectory_id": trajectory["trajectory_id"],
                "step_id": step["step_id"], "trajectory_length": len(steps),
                "memory_dependent": step.get("memory_dependent"), "view": view,
                "memory_mode": memory_mode, "memory_input": memory_input if view == "deployable" else None,
                "compression": memory, "prediction": prediction, "reference": step["action"],
                "response": generation.text, "parse_error": error,
                "input_tokens": generation.input_tokens + extra_tokens,
                "output_tokens": generation.output_tokens + extra_output,
                "latency_seconds": generation.latency_seconds + extra_latency,
            }
            row.update(score_action(prediction, step["action"], step.get("bbox"), point_tolerance,
                                    text_match=text_match, anls_threshold=anls_threshold))
            rows.append(row)
    return rows, gui_report(rows)


def validation_gap(policy, trajectories, image_root, **kwargs):
    if any(t.get("split") != "validation" for t in trajectories):
        raise ValueError("Training-time PMF scheduling requires the held-out validation split")
    _, deployable = evaluate_trajectories(policy, trajectories, image_root, view="deployable", **kwargs)
    _, privileged = evaluate_trajectories(policy, trajectories, image_root, view="privileged", **kwargs)
    d, p = deployable["overall"]["EM"], privileged["overall"]["EM"]
    return {"deployable": d, "privileged": p, "gap": max(0.0, p - d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=["deployable", "privileged", "paired"], default="deployable")
    parser.add_argument("--memory-mode", choices=["recursive", "none", "shuffle", "stale", "naive"], default="recursive")
    parser.add_argument("--memory-budget", type=int, default=128)
    parser.add_argument("--stale-lag", type=int, default=2)
    parser.add_argument("--point-tolerance", type=float, default=0.14)
    parser.add_argument("--text-match", choices=["exact", "anls"], default="exact")
    parser.add_argument("--anls-threshold", type=float, default=0.5)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    from echoes.inference import Policy
    trajectories = read_jsonl(args.data)
    if not trajectories or any(t.get("split") != args.split for t in trajectories):
        raise ValueError("All evaluation trajectories must belong to the requested split")
    if args.view != "deployable" and args.memory_mode != "recursive":
        raise ValueError("Memory interventions apply to deployable evaluation")
    policy = Policy(args.model, device=args.device, max_new_tokens=args.max_new_tokens, temperature=0.0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for view in (["deployable", "privileged"] if args.view == "paired" else [args.view]):
        rows, report = evaluate_trajectories(
            policy, trajectories, args.image_root, view=view, memory_mode=args.memory_mode,
            memory_budget=args.memory_budget, stale_lag=args.stale_lag,
            point_tolerance=args.point_tolerance, seed=args.seed,
            text_match=args.text_match, anls_threshold=args.anls_threshold,
        )
        write_jsonl(output / f"{view}_steps.jsonl", rows)
        reports[view] = report
    if args.view == "paired":
        p = reports["privileged"]["overall"]["EM"]
        d = reports["deployable"]["overall"]["EM"]
        reports["privileged_deployable_gap"] = max(0.0, p - d)
    reports["protocol"] = {"name": "canonical_nine_action", "split": args.split,
                           "memory_mode": args.memory_mode, "text_match": args.text_match,
                           "anls_threshold": args.anls_threshold,
                           "point_tolerance": args.point_tolerance, "memory_budget": args.memory_budget,
                           "point_rule": "reference_bbox_hit_else_normalized_euclidean_distance",
                           "scroll_rule": "exact_direction", "observation_protocol": "recorded_trajectory_replay"}
    (output / "metrics.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = paired_bootstrap(read_jsonl(args.first), read_jsonl(args.second), args.resamples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
