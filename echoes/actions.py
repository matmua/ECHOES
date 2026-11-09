from __future__ import annotations

import math
import unicodedata
from typing import Any


ACTION_TYPES = frozenset({
    "CLICK", "LONG_PRESS", "TYPE", "SCROLL", "PRESS_BACK", "PRESS_HOME",
    "PRESS_RECENT", "COMPLETE", "IMPOSSIBLE",
})
POINT_ACTIONS = frozenset({"CLICK", "LONG_PRESS"})
TERMINAL_ACTIONS = frozenset({"COMPLETE", "IMPOSSIBLE"})


def _coordinate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Coordinates must be real numbers")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Coordinates must lie in [0, 1]")
    return value


def validate_action(action: dict) -> dict:
    if not isinstance(action, dict) or action.get("type") not in ACTION_TYPES:
        raise ValueError("Action must have one of the nine canonical types")
    kind = action["type"]
    result = {"type": kind}
    fields = {"type"}
    if kind in POINT_ACTIONS:
        fields |= {"x", "y"}
        result.update(x=_coordinate(action.get("x")), y=_coordinate(action.get("y")))
    elif kind == "TYPE":
        fields.add("text")
        if not isinstance(action.get("text"), str):
            raise ValueError("TYPE requires a text string")
        result["text"] = action["text"]
    elif kind == "SCROLL":
        fields.add("direction")
        if action.get("direction") not in {"up", "down", "left", "right"}:
            raise ValueError("SCROLL requires up, down, left, or right")
        result["direction"] = action["direction"]
    if set(action) != fields:
        raise ValueError(f"{kind} requires exactly these fields: {sorted(fields)}")
    return result


def normalize_action(action: dict, width: int | None = None, height: int | None = None) -> dict:
    if not isinstance(action, dict):
        raise ValueError("An action must be a dictionary")
    aliases = {"TAP": "CLICK", "LONG_CLICK": "LONG_PRESS", "INPUT_TEXT": "TYPE",
               "BACK": "PRESS_BACK", "HOME": "PRESS_HOME", "RECENT": "PRESS_RECENT",
               "SUCCESS": "COMPLETE", "FAIL": "IMPOSSIBLE"}
    kind = str(action.get("type", "")).strip().upper().replace("-", "_")
    kind = aliases.get(kind, kind)
    result = {"type": kind}
    if kind in POINT_ACTIONS:
        x, y = action.get("x"), action.get("y")
        if width is not None or height is not None:
            if not width or not height or width <= 0 or height <= 0:
                raise ValueError("Pixel coordinates require positive width and height")
            x, y = float(x) / width, float(y) / height
        result.update(x=x, y=y)
    elif kind == "TYPE":
        result["text"] = action.get("text")
    elif kind == "SCROLL":
        result["direction"] = str(action.get("direction", "")).lower()
    return validate_action(result)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def action_match(pred: dict, target: dict, bbox: list[float] | None = None,
                 point_tolerance: float = 0.14) -> bool:
    if not 0 <= point_tolerance <= math.sqrt(2):
        raise ValueError("Invalid point tolerance")
    try:
        pred, target = validate_action(pred), validate_action(target)
    except (ValueError, TypeError):
        return False
    if pred["type"] != target["type"]:
        return False
    kind = target["type"]
    if kind in POINT_ACTIONS:
        if bbox is not None:
            if len(bbox) != 4 or any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in bbox):
                raise ValueError("A target box must be normalized xyxy")
            x0, y0, x1, y1 = bbox
            if x0 > x1 or y0 > y1:
                raise ValueError("A target box must be ordered xyxy")
            return x0 <= pred["x"] <= x1 and y0 <= pred["y"] <= y1
        return math.hypot(pred["x"] - target["x"], pred["y"] - target["y"]) <= point_tolerance
    if kind == "TYPE":
        return normalize_text(pred["text"]) == normalize_text(target["text"])
    if kind == "SCROLL":
        return pred["direction"] == target["direction"]
    return True
