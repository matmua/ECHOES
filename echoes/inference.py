from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from echoes.cta import decode_response, parse_cta, truncate_memory
from echoes.prompts import student_messages


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def encode_messages(processor, messages: list[dict], device=None, max_prompt_tokens: int = 16384):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, list):
            images.extend(part["image"] for part in content if part.get("type") == "image")
    kwargs = {"text": [text], "return_tensors": "pt", "padding": False}
    if images:
        kwargs["images"] = images
    encoded = processor(**kwargs)
    # Qwen multimodal position IDs are computed by the native model from image_grid_thw.
    encoded.pop("token_type_ids", None)
    if encoded["input_ids"].shape[-1] > max_prompt_tokens:
        raise ValueError("Prompt exceeds max_prompt_tokens; reduce the image resolution or memory budget")
    return encoded.to(device) if device is not None else encoded


class Policy:
    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 768,
                 temperature: float = 0.0, min_pixels: int = 65536, max_pixels: int = 1048576,
                 max_prompt_tokens: int = 16384, attention_backend: str = "sdpa", **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected policy arguments: {sorted(kwargs)}")
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError("Invalid generation settings")
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=attention_backend,
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = temperature

    @torch.inference_mode()
    def generate(self, messages: list[dict]) -> Generation:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        inputs = encode_messages(self.processor, messages, self.device, self.max_prompt_tokens)
        options = dict(max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                       pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
                       top_p=1.0, top_k=0, repetition_penalty=1.0)
        if self.temperature > 0:
            options["temperature"] = self.temperature
        output = self.model.generate(**inputs, **options)
        response = output[0, inputs["input_ids"].shape[-1]:]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return Generation(decode_response(response, self.tokenizer, self.model.generation_config.eos_token_id),
                          inputs["input_ids"].shape[-1], response.numel(), time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--compression", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    policy = Policy(args.model, device=args.device)
    previous = truncate_memory(args.compression, policy.tokenizer)
    result = policy.generate(student_messages(args.instruction, args.image, previous))
    cta = parse_cta(result.text)
    print(json.dumps({"compression": truncate_memory(cta.compression, policy.tokenizer),
                      "think": cta.think, "action": cta.action}, ensure_ascii=False))


if __name__ == "__main__":
    main()
