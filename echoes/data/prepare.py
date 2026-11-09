from __future__ import annotations

import argparse
import copy
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from echoes.actions import normalize_action
from echoes.cta import format_cta
from echoes.data.common import content_digest, copy_image, normalized_bbox, read_records, write_json, write_records


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def normalize_trajectory(
    record: dict[str, Any], image_root: Path, destination: Path, coordinates: str, dataset: str | None,
) -> dict[str, Any]:
    source_dataset = dataset or record.get("dataset")
    if not isinstance(source_dataset, str) or not source_dataset.strip():
        raise ValueError("Each trajectory needs a dataset name")
    if "trajectory_id" not in record or not str(record["trajectory_id"]).strip():
        raise ValueError("Each trajectory needs a nonempty trajectory_id")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Each trajectory needs a nonempty instruction")
    raw_steps = record.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Each trajectory needs a nonempty steps list")
    split = record.get("split")
    split = {"val": "validation", "dev": "validation"}.get(split, split)
    if split not in {None, "train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    trajectory = {
        "trajectory_id": hashlib.sha256(
            f"{source_dataset}\0{record['trajectory_id']}".encode("utf-8")
        ).hexdigest()[:24],
        "dataset": source_dataset.strip(),
        "instruction": instruction.strip(),
        "split": split,
        "steps": [],
    }
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict) or not isinstance(raw_step.get("image"), str):
            raise ValueError("Each step needs a relative image path")
        if "step_id" in raw_step and (type(raw_step["step_id"]) is not int or raw_step["step_id"] != index):
            raise ValueError("Supplied step IDs must follow chronological list order, starting at zero")
        image, width, height, checksum = copy_image(image_root, raw_step["image"], destination)
        if "width" in raw_step and int(raw_step["width"]) != width:
            raise ValueError("Annotated image width does not match the screenshot")
        if "height" in raw_step and int(raw_step["height"]) != height:
            raise ValueError("Annotated image height does not match the screenshot")
        if not isinstance(raw_step.get("action"), dict):
            raise ValueError("Each step needs an action object")
        action = copy.deepcopy(raw_step["action"])
        if "x" in action or "y" in action:
            if "x" not in action or "y" not in action:
                raise ValueError("Both action coordinates are required")
            if isinstance(action["x"], bool) or isinstance(action["y"], bool):
                raise ValueError("Action coordinates cannot be booleans")
            x, y = float(action["x"]), float(action["y"])
            if coordinates == "pixels":
                x, y = x / width, y / height
            elif coordinates == "1000":
                x, y = x / 1000, y / 1000
            action["x"], action["y"] = x, y
        action = normalize_action(action)
        format_cta("", "", action)
        step: dict[str, Any] = {
            "step_id": index,
            "image": image,
            "image_sha256": checksum,
            "width": width,
            "height": height,
            "action": action,
            "trainable": True,
        }
        if "bbox" in raw_step:
            step["bbox"] = normalized_bbox(raw_step["bbox"], width, height, coordinates)
        if "memory_dependent" in raw_step:
            if not isinstance(raw_step["memory_dependent"], bool):
                raise ValueError("memory_dependent must be a boolean")
            step["memory_dependent"] = raw_step["memory_dependent"]
        if "compression" in raw_step or "think" in raw_step:
            if not all(isinstance(raw_step.get(key), str) for key in ("compression", "think")):
                raise ValueError("CTA supervision requires both compression and think strings")
            step["compression"] = raw_step["compression"].strip()
            step["think"] = raw_step["think"].strip()
            format_cta(step["compression"], step["think"], action)
        trajectory["steps"].append(step)
    return trajectory


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    contents: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["trajectory_id"]
        if key in unique:
            previous = unique[key]
            if content_digest(previous) != content_digest(record):
                raise ValueError(f"Conflicting trajectory records or split labels: {key}")
        else:
            unique[key] = record
    for key in sorted(unique):
        record = unique[key]
        digest = content_digest({key: value for key, value in record.items() if key not in {"trajectory_id", "split"}})
        if digest in contents:
            previous = contents[digest]
            labels = {row["split"] for row in (previous, record) if row["split"] is not None}
            if len(labels) > 1:
                raise ValueError("Duplicate trajectory content appears in different explicit splits")
            if labels:
                previous["split"] = labels.pop()
        else:
            contents[digest] = record
    return sorted(contents.values(), key=lambda row: row["trajectory_id"])


def assign_splits(
    trajectories: list[dict[str, Any]], seed: int, validation_fraction: float, test_fraction: float,
) -> list[dict[str, Any]]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must lie in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")
    groups = DisjointSet(len(trajectories))
    image_owners: dict[str, int] = {}
    for index, trajectory in enumerate(trajectories):
        for step in trajectory["steps"]:
            checksum = step["image_sha256"]
            if checksum in image_owners:
                groups.union(index, image_owners[checksum])
            image_owners[checksum] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(trajectories)):
        components[groups.find(index)].append(index)
    pending: list[list[int]] = []
    counts: Counter[str] = Counter()
    total_steps = sum(len(item["steps"]) for item in trajectories)
    fractions = {
        "train": 1 - validation_fraction - test_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    for indices in components.values():
        explicit = {trajectories[i]["split"] for i in indices if trajectories[i]["split"] is not None}
        if len(explicit) > 1:
            raise ValueError("A screenshot is shared across explicitly assigned data splits")
        if explicit:
            split = explicit.pop()
            for i in indices:
                trajectories[i]["split"] = split
                counts[split] += len(trajectories[i]["steps"])
        else:
            pending.append(indices)
    random.Random(seed).shuffle(pending)
    pending.sort(key=lambda group: sum(len(trajectories[i]["steps"]) for i in group), reverse=True)
    for indices in pending:
        allowed = [name for name, fraction in fractions.items() if fraction > 0]
        split = max(allowed, key=lambda name: total_steps * fractions[name] - counts[name])
        for i in indices:
            trajectories[i]["split"] = split
            counts[split] += len(trajectories[i]["steps"])
    assert_disjoint(trajectories)
    return trajectories


def assert_disjoint(trajectories: list[dict[str, Any]]) -> None:
    trajectory_splits: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    for trajectory in trajectories:
        split = trajectory["split"]
        identifier = trajectory["trajectory_id"]
        if identifier in trajectory_splits and trajectory_splits[identifier] != split:
            raise ValueError("A trajectory is shared across data splits")
        trajectory_splits[identifier] = split
        for step in trajectory["steps"]:
            checksum = step["image_sha256"]
            if checksum in image_splits and image_splits[checksum] != split:
                raise ValueError("A screenshot is shared across data splits")
            image_splits[checksum] = split


def select_steps(trajectories: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Training step counts must be positive")
    candidates = [(i, j) for i, row in enumerate(trajectories) for j in range(len(row["steps"]))]
    if len(candidates) < count:
        raise ValueError(f"Requested {count} training steps but only {len(candidates)} are available")
    selected = set(random.Random(seed).sample(candidates, count))
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for i, j in selected:
        by_trajectory[i].append(j)
    output = []
    for i in sorted(by_trajectory):
        trajectory = copy.deepcopy(trajectories[i])
        trajectory["steps"] = trajectory["steps"][: max(by_trajectory[i]) + 1]
        for j, step in enumerate(trajectory["steps"]):
            step["trainable"] = (i, j) in selected
        output.append(trajectory)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize trajectories and prepare leakage-free training splits")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--coordinates", choices=("normalized", "pixels", "1000"), default="normalized")
    parser.add_argument("--sft-steps", type=int, default=10000)
    parser.add_argument("--rl-steps", type=int, default=9500)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [row for path in args.input for row in read_records(path)]
    if not records:
        raise ValueError("The input contains no trajectories")
    normalized = [
        normalize_trajectory(row, args.image_root, args.output, args.coordinates, args.dataset)
        for row in records
    ]
    trajectories = assign_splits(
        deduplicate(normalized), args.seed, args.validation_fraction, args.test_fraction,
    )
    splits = {name: [row for row in trajectories if row["split"] == name] for name in ("train", "validation", "test")}
    for name, fraction in (("validation", args.validation_fraction), ("test", args.test_fraction)):
        if fraction > 0 and not splits[name]:
            raise ValueError(f"No {name} trajectories remain after grouping; provide a disjoint explicit {name} split")
    sft = select_steps(splits["train"], args.sft_steps, args.seed + 1)
    rl = select_steps(splits["train"], args.rl_steps, args.seed + 2)
    for name, rows in {**splits, "sft": sft, "rl": rl}.items():
        write_records(args.output / f"{name}.jsonl", rows)
    manifest = {
        "seed": args.seed,
        "coordinate_space": "normalized",
        "splits": {
            name: {
                "trajectories": len(rows),
                "steps": sum(len(row["steps"]) for row in rows),
                "trainable_steps": sum(step["trainable"] for row in rows for step in row["steps"]),
                "sha256": content_digest(rows),
            }
            for name, rows in {**splits, "sft": sft, "rl": rl}.items()
        },
    }
    write_json(args.output / "manifest.json", manifest)
    print("Prepared " + ", ".join(f"{name}: {info['steps']} steps" for name, info in manifest["splits"].items()))


if __name__ == "__main__":
    main()
