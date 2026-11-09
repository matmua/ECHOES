from __future__ import annotations

from echoes.io import load_image


CTA_SYSTEM = """You are a visual GUI agent. Produce exactly this ordered structure:
<compression>updated task state</compression>
<think>reasoning for the next action</think>
<action>{JSON action}</action>
Compression must retain task progress, critical facts, completed operations, and observed outcomes needed for later decisions. Rewrite it recursively in at most 128 tokens. Do not claim that the next action has already succeeded. Think must use the available observations and task state. Output one action with normalized coordinates in [0,1]. The nine action schemas are:
{"type":"CLICK","x":0.5,"y":0.5}
{"type":"LONG_PRESS","x":0.5,"y":0.5}
{"type":"TYPE","text":"text to enter"}
{"type":"SCROLL","direction":"up|down|left|right"}
{"type":"PRESS_BACK"}
{"type":"PRESS_HOME"}
{"type":"PRESS_RECENT"}
{"type":"COMPLETE"}
{"type":"IMPOSSIBLE"}
Do not output text outside the three tags."""


def _image(image) -> dict:
    return {"type": "image", "image": load_image(image)}


def student_messages(instruction: str, image, compression: str) -> list[dict]:
    return [
        {"role": "system", "content": CTA_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Task: {instruction}\nPrevious Compression:\n{compression or '(empty)'}\nCurrent screenshot:"},
            _image(image),
        ]},
    ]


def privileged_messages(instruction: str, images: list) -> list[dict]:
    if not images or len(images) > 5:
        raise ValueError("The privileged view requires one current and at most four historical screenshots")
    content = [{"type": "text", "text": f"Task: {instruction}\nScreenshots are ordered from oldest to current."}]
    for index, image in enumerate(images):
        label = "Current screenshot:" if index == len(images) - 1 else f"Historical screenshot {index + 1}:"
        content.extend([{"type": "text", "text": label}, _image(image)])
    return [{"role": "system", "content": CTA_SYSTEM}, {"role": "user", "content": content}]
