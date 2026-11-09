from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from statistics import mean

from PIL import Image

from echoes.cta import truncate_memory
from echoes.io import write_jsonl


_OUTPUT = re.compile(
    r"\A\s*<compression>(.*?)</compression>\s*<think>(.*?)</think>\s*"
    r"<action>\s*(Up|Down|Left|Right)\s*</action>\s*\Z", re.DOTALL,
)
_FORMAT = (
    "Choose exactly one native Sokoban move. Return "
    "<compression>updated compact task state</compression>"
    "<think>reasoning about the current board</think>"
    "<action>Up or Down or Left or Right</action>. "
    "Compression should retain progress, useful spatial facts, and observed outcomes; "
    "keep it within {memory_budget} tokens."
)


def _observation_content(observation, resolution):
    content = []
    for item in observation.info:
        if item["type"] == "text":
            content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            image = item["image_url"].convert("RGB")
            if max(image.size) != resolution:
                scale = resolution / max(image.size)
                image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                                     resample=Image.Resampling.LANCZOS)
            content.append({"type": "image", "image": image})
        else:
            raise ValueError(f"Unexpected native observation modality: {item['type']}")
    if sum(item["type"] == "image" for item in content) != 1:
        raise ValueError("Sokoban observations must contain exactly one current screenshot")
    return content


def evaluate_sokoban(policy, magebench_root, scenarios, output, *, max_steps=100,
                      resolution=256, memory_budget=128, memory_mode="recursive"):
    if max_steps < 1 or resolution < 1 or memory_budget < 1:
        raise ValueError("Step, image-resolution, and memory budgets must be positive")
    if memory_mode not in {"recursive", "none"}:
        raise ValueError("Sokoban memory mode must be recursive or none")
    root, output = Path(magebench_root).resolve(), Path(output)
    if not (root / "env" / "sokoban_env.py").is_file():
        raise ValueError("magebench_root must point to the official MageBench checkout")
    sys.path.insert(0, str(root))
    native_env = importlib.import_module("env")
    if Path(native_env.__file__).resolve().parent != root / "env":
        raise RuntimeError("A different env package is already imported; start a fresh process")
    env_class, action_class = native_env.get_env("Sokoban")
    rows, episodes = [], []
    for number, scenario in enumerate(scenarios):
        scenario_path = Path(scenario)
        if not scenario_path.is_absolute():
            scenario_path = root / "resources" / scenario_path
        episode_dir = output / f"episode_{number:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        environment = env_class(scenario_path.resolve().as_posix(), str(episode_dir), resolution)
        try:
            observation = environment.get_init_observation()
            memory, parsing_errors = "", 0
            done = False
            for step in range(max_steps):
                content = _observation_content(observation, resolution)
                if memory_mode == "recursive":
                    content.append({"type": "text", "text": "Previous task state:\n" + memory})
                content.append({"type": "text", "text": "Act according to the current board.\n"
                                + _FORMAT.format(memory_budget=memory_budget)})
                generation = policy.generate([
                    {"role": "system", "content": environment.system_prompt},
                    {"role": "user", "content": content},
                ])
                match = _OUTPUT.fullmatch(generation.text)
                action = None
                if match is None:
                    parsing_errors += 1
                else:
                    memory = truncate_memory(match[1].strip(), policy.tokenizer, memory_budget)
                    action = match[3]
                    observation, done = environment.take_action(action_class(action))
                rows.append({"episode": number, "scenario": scenario_path.name, "step": step,
                             "response": generation.text, "action": action, "compression": memory,
                             "parse_error": match is None, "input_tokens": generation.input_tokens,
                             "output_tokens": generation.output_tokens,
                             "latency_seconds": generation.latency_seconds})
                if done:
                    break
            native_score, details = environment.Evaluator()
            base_environment = getattr(environment.env, "unwrapped", environment.env)
            solved = bool(base_environment._check_if_all_boxes_on_target())
            episodes.append({"episode": number, "scenario": scenario_path.name,
                             "native_score": float(native_score),
                             "normalized_score": float(native_score) / 100.0,
                             "fully_solved": solved, "environment_done": bool(done),
                             "steps": step + 1, "parsing_errors": parsing_errors,
                             "native_details": details})
        finally:
            environment.env.close()
    if not episodes:
        raise ValueError("At least one Sokoban scenario is required")
    write_jsonl(output / "steps.jsonl", rows)
    write_jsonl(output / "episodes.jsonl", episodes)
    report = {"episodes": len(episodes), "normalized_task_score": mean(r["normalized_score"] for r in episodes),
              "native_task_score": mean(r["native_score"] for r in episodes),
              "fully_solved_tasks": sum(r["fully_solved"] for r in episodes),
              "mean_input_tokens": mean(r["input_tokens"] for r in rows),
              "mean_latency_seconds": mean(r["latency_seconds"] for r in rows),
              "memory_mode": memory_mode, "max_steps": max_steps, "resolution": resolution}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magebench-root", required=True)
    parser.add_argument("--scenarios", required=True, help="JSON array of native scenario paths relative to resources")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--memory-budget", type=int, default=128)
    parser.add_argument("--memory-mode", choices=["recursive", "none"], default="recursive")
    args = parser.parse_args()
    from echoes.inference import Policy
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not all(isinstance(path, str) for path in scenarios):
        raise ValueError("Scenarios must be a JSON array of native scenario paths")
    policy = Policy(args.model, device=args.device, max_new_tokens=args.max_new_tokens, temperature=0.0)
    evaluate_sokoban(policy, args.magebench_root, scenarios, args.output,
                     max_steps=args.max_steps, resolution=args.resolution,
                     memory_budget=args.memory_budget, memory_mode=args.memory_mode)


if __name__ == "__main__":
    main()
