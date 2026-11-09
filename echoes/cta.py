from __future__ import annotations

import json
import re
from dataclasses import dataclass

from echoes.actions import validate_action


_CTA = re.compile(
    r"\A\s*<compression>(?P<compression>.*?)</compression>\s*"
    r"<think>(?P<think>.*?)</think>\s*<action>(?P<action>.*?)</action>\s*\Z", re.DOTALL,
)
_RESERVED = re.compile(r"</?(?:compression|think|action)>")
_MODEL_CONTROL = re.compile(r"<\|[^<>]*\|>")


@dataclass(frozen=True)
class CTA:
    compression: str
    think: str
    action: dict


def parse_cta(text: str) -> CTA:
    if _MODEL_CONTROL.search(text):
        raise ValueError("Model control markers cannot occur inside CTA output")
    match = _CTA.fullmatch(text)
    if match is None:
        raise ValueError("Expected exactly one ordered Compression, Think, Action sequence")
    compression, think = match["compression"].strip(), match["think"].strip()
    if _RESERVED.search(compression) or _RESERVED.search(think):
        raise ValueError("Nested CTA tags are invalid")
    action = validate_action(json.loads(match["action"]))
    return CTA(compression, think, action)


def format_cta(compression: str, think: str, action: dict) -> str:
    if not isinstance(compression, str) or not isinstance(think, str):
        raise ValueError("Compression and Think must be strings")
    if _MODEL_CONTROL.search(compression + think + json.dumps(action, ensure_ascii=False)):
        raise ValueError("CTA content cannot contain model control markers")
    if _RESERVED.search(compression) or _RESERVED.search(think):
        raise ValueError("CTA content cannot contain structural tags")
    action = validate_action(action)
    return (f"<compression>{compression.strip()}</compression>\n"
            f"<think>{think.strip()}</think>\n"
            f"<action>{json.dumps(action, ensure_ascii=False, separators=(',', ':'))}</action>")


def compression_text(text: str) -> str:
    match = re.search(r"<compression>(.*?)(?:</compression>|\Z)", text, re.DOTALL)
    return match[1].strip() if match else ""


def decode_response(token_ids, tokenizer, eos_token_ids=None) -> str:
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    endings = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if eos_token_ids is not None:
        endings.update([eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids)
    while ids and ids[-1] in endings:
        ids.pop()
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def distillation_token_mask(token_ids, tokenizer) -> list[float]:
    """Select the ordered CTA sections on the original sampled tokenization."""
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    spans, cursor = [], 0
    for name in ("compression", "think", "action"):
        match = re.search(rf"<{name}>.*?</{name}>", text[cursor:], re.DOTALL)
        if match is None:
            break
        start, end = cursor + match.start(), cursor + match.end()
        spans.append((start, end))
        cursor = end
    if not spans:
        return [0.0] * len(ids)
    # Decode prefixes to preserve sampled BPE boundaries, even when re-encoding merges tokens.
    boundaries = [0] + [len(tokenizer.decode(ids[:i], skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)) for i in range(1, len(ids) + 1)]
    specials = set(tokenizer.all_special_ids)
    mask = []
    for i, token in enumerate(ids):
        low, high = sorted((boundaries[i], boundaries[i + 1]))
        overlap = any(max(left, low) < min(right, high) if low != high else left <= low < right
                      for left, right in spans)
        mask.append(float(token not in specials and overlap))
    return mask


def truncate_memory(text: str, tokenizer, budget: int = 128) -> str:
    if budget < 1:
        raise ValueError("Memory budget must be positive")
    text = _MODEL_CONTROL.sub("", text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    end = budget
    result = tokenizer.decode(ids[:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    while end and (not text.startswith(result) or len(tokenizer.encode(result, add_special_tokens=False)) > budget):
        end -= 1
        result = tokenizer.decode(ids[:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result
