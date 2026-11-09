from __future__ import annotations

import hashlib
import json
from pathlib import Path

from echoes.actions import validate_action
from echoes.io import load_image, read_jsonl, resolve_image


def load_trajectories(path: str, image_root: str, required_split: str) -> list[dict]:
    records = read_jsonl(path)
    if not records:
        raise ValueError("A trajectory file cannot be empty")
    seen = set()
    for trajectory in records:
        identity = trajectory["trajectory_id"]
        if identity in seen:
            raise ValueError("Duplicate trajectory ID")
        seen.add(identity)
        if trajectory.get("split") != required_split:
            raise ValueError(f"Expected split={required_split}")
        if not isinstance(trajectory.get("instruction"), str) or not trajectory["instruction"].strip():
            raise ValueError("Every trajectory requires an instruction")
        steps = trajectory["steps"]
        if not steps:
            raise ValueError("A trajectory cannot be empty")
        for index, step in enumerate(steps):
            if type(step["step_id"]) is not int or step["step_id"] != index:
                raise ValueError("Steps must be contiguous, starting at zero")
            if not isinstance(step.get("trainable", True), bool):
                raise ValueError("trainable must be a boolean")
            validate_action(step["action"])
            resolve_image(image_root, step["image"])
    return records


def targets(trajectories: list[dict]) -> list[tuple[int, int]]:
    selected = [(i, j) for i, trajectory in enumerate(trajectories)
                for j, step in enumerate(trajectory["steps"]) if step.get("trainable", True)]
    if not selected:
        raise ValueError("No trainable steps were selected")
    return selected


def _visual_hashes(trajectories: list[dict], image_root: str) -> set[str]:
    hashes = set()
    for name in {step["image"] for trajectory in trajectories for step in trajectory["steps"]}:
        image = load_image(resolve_image(image_root, name))
        digest = hashlib.sha256()
        digest.update(str(image.size).encode())
        digest.update(image.tobytes())
        hashes.add(digest.hexdigest())
    return hashes


def assert_disjoint(train: list[dict], validation: list[dict], image_root: str) -> None:
    if {t["trajectory_id"] for t in train} & {t["trajectory_id"] for t in validation}:
        raise ValueError("Training and validation trajectories overlap")
    if _visual_hashes(train, image_root) & _visual_hashes(validation, image_root):
        raise ValueError("Training and validation contain identical screenshot pixels")


def fingerprint(trajectories: list[dict], image_root: str) -> str:
    digest = hashlib.sha256(json.dumps(trajectories, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for value in sorted(_visual_hashes(trajectories, image_root)):
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def fixed_validation(trajectories: list[dict], count: int, seed: int) -> list[dict]:
    def key(record):
        return hashlib.sha256(f"{seed}:{record['trajectory_id']}".encode()).hexdigest()
    return sorted(trajectories, key=key)[:count]


def image_path(config: dict, step: dict) -> Path:
    return resolve_image(config["data"]["image_root"], step["image"])
