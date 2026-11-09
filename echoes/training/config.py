from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import yaml


def load_config(stage: str) -> tuple[dict, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"configs/{stage}.yaml")
    for name in ("model", "train", "validation", "image-root", "output", "resume"):
        parser.add_argument("--" + name)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    mapping = {"model": ("model", "path"), "train": ("data", "train"),
               "validation": ("data", "validation"), "image_root": ("data", "image_root"),
               "output": ("training", "output"), "resume": ("training", "resume")}
    for key, (section, field) in mapping.items():
        if getattr(args, key) is not None:
            config[section][field] = getattr(args, key)
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            parser.error("--set requires KEY=VALUE")
        path = key.split(".")
        node = config
        for part in path[:-1]:
            if part not in node or not isinstance(node[part], dict):
                parser.error(f"Unknown configuration key: {key}")
            node = node[part]
        if path[-1] not in node:
            parser.error(f"Unknown configuration key: {key}")
        node[path[-1]] = yaml.safe_load(value)
    validate_config(config, stage)
    return config, args


def validate_config(config: dict, stage: str) -> None:
    for section in ("model", "data", "optimizer", "runtime", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for key in ("max_new_tokens", "memory_budget", "prompts_per_rank", "save_every", "logit_chunk_size"):
        if int(config["training"][key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if config["training"]["memory_budget"] != 128:
        raise ValueError("The CTA prompt and paper configuration use a 128-token memory budget")
    if not math.isfinite(config["optimizer"]["lr"]) or config["optimizer"]["lr"] <= 0:
        raise ValueError("optimizer.lr must be positive and finite")
    if stage == "opd":
        train = config["training"]
        if train["group_size"] < 2 or train["steps_per_round"] < 1 or train["rounds"] < 1:
            raise ValueError("Invalid group size or round schedule")
        if train["temperature"] != 1.0:
            raise ValueError("OPD--GRPO uses temperature 1.0 for matching rollout and scoring distributions")
        schedule = config["schedule"]
        if not 0 <= schedule["lambda_min"] <= schedule["lambda_max"] <= 1 or schedule["epsilon"] <= 0:
            raise ValueError("Invalid PMF schedule bounds")
        if schedule["mode"] not in {"pmf", "linear", "fixed", "rl_only", "opd_only"}:
            raise ValueError("Invalid mixture schedule")
        if not 0 <= schedule["fixed_weight"] <= 1:
            raise ValueError("fixed_weight must lie in [0, 1]")
        if not 0 <= train["history_frames"] <= 4:
            raise ValueError("Privileged history must contain at most four preceding frames")
        if config["data"]["validation_trajectories"] < 1:
            raise ValueError("A fixed nonempty validation set is required")
    elif config["training"]["epochs"] < 1:
        raise ValueError("training.epochs must be positive")


def resume_signature(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["training"].pop("resume", None)
    config["training"].pop("output", None)
    return config
