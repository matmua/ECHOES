from __future__ import annotations

import argparse
import copy
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from echoes.data.common import content_digest, copy_image, image_digest, normalized_bbox, read_records, write_json, write_records


TASK_LEVELS = {
    "bbox_to_function": "L1",
    "semantic_to_bbox": "L1",
    "active_tab": "L2",
    "foreground_layer": "L2",
    "blocking_state": "L2",
    "screen_selection": "L2",
    "screen_state_diff": "L3",
}


def reviewed_queries(screen: dict[str, Any], coordinates: str = "normalized") -> list[dict[str, Any]]:
    image = screen["image"]
    common = {"image": image}
    for key in ("width", "height"):
        if key in screen:
            common[key] = screen[key]
    result: list[dict[str, Any]] = []

    def append(task: str, annotation: dict[str, Any], question: str, answer: Any, **extra: Any) -> None:
        if annotation.get("human_verified") is not True:
            return
        query = {
            **common, "task": task, "question": annotation.get("question", question),
            "answer": answer, "human_verified": True, **extra,
        }
        if "choices" in annotation:
            if annotation.get("choices_human_verified") is not True:
                raise ValueError("MCQ choices require choices_human_verified=true")
            query["choices"] = annotation["choices"]
            query["answer"] = annotation["correct_choice"]
        result.append(query)

    for element in screen.get("elements", []):
        if element.get("human_verified") is not True:
            continue
        function = element.get("function")
        if not isinstance(function, str) or not function.strip():
            raise ValueError("Reviewed elements require a nonempty function description")
        box = element["bbox"]
        append(
            "bbox_to_function", element,
            f"What is the function of the interface element inside bounding box {box} in {coordinates} coordinates?",
            function, bbox=box,
        )
        if "request" in element:
            localization = {key: value for key, value in element.items() if key not in {"question", "choices"}}
            append("semantic_to_bbox", localization, element["request"], box, bbox=box)
    if screen.get("active_tab", {}).get("human_verified") is True:
        annotation = screen["active_tab"]
        append("active_tab", annotation, "Which navigation tab is currently active?", annotation["answer"])
    if screen.get("foreground_layer", {}).get("human_verified") is True:
        annotation = screen["foreground_layer"]
        append(
            "foreground_layer", annotation,
            "Is a foreground interface layer present? Describe it if present.",
            {"present": annotation["present"], "description": annotation.get("description", "")},
        )
    for annotation in screen.get("blocking_states", []):
        if annotation.get("human_verified") is True:
            append("blocking_state", annotation, annotation["question"], {"blocked": annotation["blocked"]})
    if screen.get("screen_selection", {}).get("human_verified") is True:
        annotation = screen["screen_selection"]
        append(
            "screen_selection", annotation, "Which items or values are currently selected?",
            annotation["selected"],
        )
    for transition in screen.get("transitions", []):
        if transition.get("human_verified") is not True:
            continue
        append(
            "screen_state_diff", transition,
            "Describe the observed interface-state changes from the first image to the second image.",
            transition["description"], images=[image, transition["next_image"]],
        )
    return result


def validate_answer(task: str, answer: Any, choices: dict[str, str] | None) -> None:
    if choices is not None:
        if task not in {"bbox_to_function", "screen_state_diff"}:
            raise ValueError("MCQ alternatives are supported for element functions and state differences")
        if not isinstance(choices, dict) or len(choices) < 2:
            raise ValueError("MCQ choices must be an object with at least two alternatives")
        if any(not isinstance(key, str) or len(key) != 1 or not key.isupper() for key in choices):
            raise ValueError("MCQ choices must use single uppercase letter keys")
        if any(not isinstance(value, str) or not value.strip() for value in choices.values()):
            raise ValueError("MCQ alternatives must be nonempty strings")
        normalized = [value.casefold().strip() for value in choices.values()]
        if len(set(normalized)) != len(normalized) or answer not in choices:
            raise ValueError("MCQ alternatives must be distinct and include the reference answer")
        return
    if task in {"bbox_to_function", "active_tab", "screen_state_diff"}:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{task} requires a nonempty reference string")
    elif task == "foreground_layer":
        if not isinstance(answer, dict) or not isinstance(answer.get("present"), bool):
            raise ValueError("Foreground references require a boolean present field")
        if not isinstance(answer.get("description", ""), str):
            raise ValueError("Foreground descriptions must be strings")
    elif task == "blocking_state":
        if not isinstance(answer, dict) or not isinstance(answer.get("blocked"), bool):
            raise ValueError("Blocking references require a boolean blocked field")
    elif task == "screen_selection":
        if not isinstance(answer, (list, dict)):
            raise ValueError("Screen-selection references must be a list or an item-to-value object")
        entries = answer if isinstance(answer, list) else list(answer) + list(answer.values())
        if any(not isinstance(value, (str, int, float, bool)) for value in entries):
            raise ValueError("Selection items and values must be scalar strings, numbers, or booleans")
        if any(isinstance(value, float) and not math.isfinite(value) for value in entries):
            raise ValueError("Selection values must be finite")


def normalize_query(
    row: dict[str, Any], image_root: Path, output: Path, coordinates: str, forbidden_images: set[str],
) -> dict[str, Any]:
    if row.get("human_verified") is not True:
        raise ValueError("Only human-verified queries can enter the diagnostic release")
    task = row.get("task")
    if task not in TASK_LEVELS:
        raise ValueError("Unknown diagnostic task")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each diagnostic instance needs a nonempty question")
    names = row.get("images", [row.get("image")])
    if not isinstance(names, list) or len(names) != (2 if task == "screen_state_diff" else 1):
        raise ValueError("Single-screen queries require one image and state differences require two")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("Diagnostic image paths must be strings")
    copied = [copy_image(image_root, name, output) for name in names]
    if any(checksum in forbidden_images for _, _, _, checksum in copied):
        raise ValueError("A diagnostic screenshot overlaps a supplied training or validation split")
    image, width, height, checksum = copied[0]
    if "width" in row and int(row["width"]) != width:
        raise ValueError("Diagnostic width does not match the screenshot")
    if "height" in row and int(row["height"]) != height:
        raise ValueError("Diagnostic height does not match the screenshot")
    answer = copy.deepcopy(row.get("answer"))
    choices = copy.deepcopy(row.get("choices"))
    validate_answer(task, answer, choices)
    result: dict[str, Any] = {
        "task": task,
        "level": TASK_LEVELS[task],
        "screen_id": checksum,
        "width": width,
        "height": height,
        "question": question.strip(),
        "answer": answer,
        "human_verified": True,
    }
    if task == "screen_state_diff":
        result["images"] = [item[0] for item in copied]
        result["next_width"], result["next_height"] = copied[1][1:3]
    else:
        result["image"] = image
    if "bbox" in row:
        result["bbox"] = normalized_bbox(row["bbox"], width, height, coordinates)
    if task == "bbox_to_function" and "bbox" not in result:
        raise ValueError("BBox-to-Function queries require the localized reference bounding box")
    if task == "semantic_to_bbox":
        if "bbox" not in result:
            raise ValueError("Semantic-to-BBox queries require a reference bounding box")
        box = result["bbox"]
        result["answer"] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    if choices is not None:
        result["choices"] = choices
    result["instance_id"] = content_digest(result)[:24]
    return result


def build_reviewed_mcq(row: dict[str, Any], distractors: list[str], seed: int) -> dict[str, Any]:
    if row.get("human_verified") is not True or row.get("distractors_human_verified") is not True:
        raise ValueError("MCQ construction requires verified references and verified distractors")
    if row.get("task") not in {"bbox_to_function", "screen_state_diff"}:
        raise ValueError("MCQ construction requires an element-function or state-difference query")
    answer = row["answer"]
    if not isinstance(answer, str):
        raise ValueError("MCQ reference must be a string")
    alternatives = [answer]
    normalized = {answer.strip().casefold()}
    for value in distractors:
        key = value.strip().casefold()
        if key and key not in normalized:
            alternatives.append(value)
            normalized.add(key)
        if len(alternatives) == 4:
            break
    if len(alternatives) < 2:
        raise ValueError("At least one distinct verified distractor is required")
    random.Random(seed).shuffle(alternatives)
    result = copy.deepcopy(row)
    result["choices"] = {chr(65 + i): value for i, value in enumerate(alternatives)}
    result["answer"] = next(key for key, value in result["choices"].items() if value == answer)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the seven-task diagnostic benchmark from reviewed annotations")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("queries", "screens"), default="queries")
    parser.add_argument("--coordinates", choices=("normalized", "pixels", "1000"), default="normalized")
    parser.add_argument("--exclude-trajectories", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude-image-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    forbidden: set[str] = set()
    digest_cache: dict[tuple[str, str], str] = {}
    for path in args.exclude_trajectories:
        excluded_root = (args.exclude_image_root or path.parent).resolve()
        for trajectory in read_records(path):
            for step in trajectory["steps"]:
                key = (str(excluded_root), step["image"])
                if key not in digest_cache:
                    digest_cache[key] = image_digest(excluded_root, step["image"])
                checksum = digest_cache[key]
                if "image_sha256" in step and step["image_sha256"] != checksum:
                    raise ValueError("An excluded trajectory's image hash does not match its screenshot")
                forbidden.add(checksum)
    source = [row for path in args.input for row in read_records(path)]
    queries = [query for screen in source for query in reviewed_queries(screen, args.coordinates)] if args.mode == "screens" else source
    retained: dict[str, dict[str, Any]] = {}
    rejected = 0
    for index, row in enumerate(queries):
        if row.get("human_verified") is not True:
            rejected += 1
            continue
        if "distractors" in row:
            row = build_reviewed_mcq(row, row["distractors"], args.seed + index)
        query = normalize_query(row, args.image_root, args.output, args.coordinates, forbidden)
        retained[query["instance_id"]] = query
    if not retained:
        raise ValueError("No human-verified diagnostic instances were supplied")
    records = [retained[key] for key in sorted(retained)]
    write_records(args.output / "benchmark.jsonl", records)
    task_counts = Counter(row["task"] for row in records)
    level_counts = Counter(row["level"] for row in records)
    images = {name for row in records for name in row.get("images", [row.get("image")])}
    manifest = {
        "instances": len(records),
        "unique_images": len(images),
        "tasks": {task: task_counts[task] for task in TASK_LEVELS},
        "levels": {level: level_counts[level] for level in ("L1", "L2", "L3")},
        "human_verified_instances": len(records),
        "human_review_coverage": 1.0,
        "sha256": content_digest(records),
    }
    if args.mode == "queries":
        manifest["excluded_unverified_queries"] = rejected
    write_json(args.output / "manifest.json", manifest)
    print(f"Built {len(records)} reviewed diagnostic instances from {len(images)} unique images")


if __name__ == "__main__":
    main()
