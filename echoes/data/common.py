from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        text = stream.read()
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Expected JSON objects in {path.name}")
    return records


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def content_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_path(root: str | Path, name: str) -> Path:
    root = Path(root).expanduser().resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Image paths must stay inside the image root")
    if not candidate.is_file():
        raise FileNotFoundError(f"Image does not exist: {name}")
    return candidate


def rgb_digest(image: Any) -> str:
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(f"{width}:{height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def image_digest(root: str | Path, name: str) -> str:
    from PIL import Image, ImageOps

    with Image.open(image_path(root, name)) as opened:
        return rgb_digest(ImageOps.exif_transpose(opened).convert("RGB"))


def copy_image(root: Path, name: str, destination: Path) -> tuple[str, int, int, str]:
    from PIL import Image, ImageOps

    source = image_path(root, name)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        checksum = rgb_digest(image)
        relative = f"images/{checksum}.png"
        target = destination / relative
        if target.exists():
            if image_digest(destination, relative) != checksum:
                raise ValueError("An existing image cache entry does not match its pixel hash")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            clean = Image.frombytes("RGB", image.size, image.tobytes())
            clean.save(target, format="PNG")
    return relative, width, height, checksum


def normalized_bbox(value: Any, width: int, height: int, coordinates: str) -> list[float]:
    import math

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    if any(isinstance(v, bool) for v in value):
        raise ValueError("Bounding-box coordinates cannot be booleans")
    box = [float(v) for v in value]
    if coordinates == "pixels":
        box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    elif coordinates == "1000":
        box = [v / 1000 for v in box]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
        raise ValueError("Bounding boxes must be finite and inside the image")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Bounding boxes must have positive width and height")
    return box
