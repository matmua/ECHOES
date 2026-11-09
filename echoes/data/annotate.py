from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from echoes.actions import validate_action
from echoes.cta import format_cta, parse_cta
from echoes.data.common import content_digest, image_digest, image_path, read_records, write_records


def source_digest(trajectory: dict[str, Any]) -> str:
    clean = copy.deepcopy(trajectory)
    clean.pop("annotation", None)
    for step in clean["steps"]:
        step.pop("compression", None)
        step.pop("think", None)
    return content_digest(clean)


def annotation_messages(
    trajectory: dict[str, Any], index: int, image_root: Path, previous_compression: str, budget: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Task: {trajectory['instruction']}\n"
            f"Previous textual state: {previous_compression or '(empty)'}\n"
            "The following observations are in chronological order. An action after an observation "
            "was executed only when explicitly marked as executed."
        ),
    }]
    for position, step in enumerate(trajectory["steps"][: index + 1]):
        content.append({"type": "text", "text": f"Observation {position}:"})
        path = image_path(image_root, step["image"])
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        if position < index:
            content.append({
                "type": "text",
                "text": "Executed action: " + json.dumps(step["action"], ensure_ascii=False),
            })
    content.append({
        "type": "text",
        "text": (
            "At the current observation the demonstrated next action is "
            + json.dumps(trajectory["steps"][index]["action"], ensure_ascii=False)
            + ". It has not been executed. Write the updated compression and the reasoning for this next action. "
            + "Do not claim that this action has succeeded, and do not infer later observations."
        ),
    })
    return [
        {
            "role": "system",
            "content": (
                "Annotate a causal GUI trajectory prefix for Compression--Think--Action training. "
                "Return exactly a JSON object with string fields compression and think. "
                f"The compression must fit within {budget} model tokens, retain task progress, "
                "decision-relevant facts, critical state changes and observed execution outcomes, "
                "and replace the previous state rather than append to it. Preserve exact names, "
                "numbers and values that future decisions may require. Include only information "
                "supported by the supplied prefix. The think field explains the next demonstrated "
                "action from the current observation and retained state. Do not include XML tags."
            ),
        },
        {"role": "user", "content": content},
    ]


class AnnotationClient:
    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float, retries: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def complete(self, messages: list[dict[str, Any]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for attempt in range(self.retries):
            request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                message = result["choices"][0]["message"]["content"]
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("The annotation service returned an empty response")
                return message
            except urllib.error.HTTPError as error:
                if error.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt + 1 == self.retries:
                    raise RuntimeError(f"Annotation request failed with HTTP {error.code}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt + 1 == self.retries:
                    raise RuntimeError("Annotation request failed after retrying the connection") from None
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Annotation request did not return a response")


def decode_annotation(text: str, tokenizer: Any, budget: int, action: dict[str, Any]) -> tuple[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    answer = json.loads(text)
    if not isinstance(answer, dict) or not all(isinstance(answer.get(k), str) for k in ("compression", "think")):
        raise ValueError("Both compression and think must be strings")
    compression, think = answer["compression"].strip(), answer["think"].strip()
    if not think:
        raise ValueError("The think field must be nonempty")
    if re.search(r"</?(?:compression|think|action)>", compression + think, flags=re.IGNORECASE):
        raise ValueError("The annotation must not contain CTA delimiters")
    length = len(tokenizer.encode(compression, add_special_tokens=False))
    if length > budget:
        raise ValueError(f"Compression contains {length} tokens; the maximum is {budget}")
    parse_cta(format_cta(compression, think, action))
    return compression, think


def annotate_trajectory(
    trajectory: dict[str, Any], client: AnnotationClient, tokenizer: Any, tokenizer_name: str,
    image_root: Path, budget: int, content_attempts: int, reuse_existing: bool,
) -> dict[str, Any]:
    result = copy.deepcopy(trajectory)
    previous_compression = ""
    for index, step in enumerate(result["steps"]):
        if reuse_existing and "compression" in step and "think" in step:
            compression, think = decode_annotation(
                json.dumps({"compression": step["compression"], "think": step["think"]}),
                tokenizer, budget, step["action"],
            )
        else:
            messages = annotation_messages(result, index, image_root, previous_compression, budget)
            last_error = ""
            for attempt in range(content_attempts):
                response = client.complete(messages)
                try:
                    compression, think = decode_annotation(response, tokenizer, budget, step["action"])
                    break
                except (ValueError, TypeError, KeyError) as error:
                    last_error = str(error)
                    messages.extend([
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": f"Revise the JSON annotation: {last_error}. Keep it causal and concise."},
                    ])
            else:
                raise ValueError(f"Invalid CTA annotation at step {index}: {last_error}")
        step["compression"], step["think"] = compression, think
        previous_compression = compression
    result["annotation"] = {
        "model": client.model,
        "tokenizer": tokenizer_name,
        "compression_budget": budget,
        "source_sha256": source_digest(trajectory),
        "causal_prefix": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal CTA targets from trajectory prefixes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--compression-budget", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--content-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if min(args.compression_budget, args.workers, args.retries, args.content_attempts) < 1 or args.timeout <= 0:
        raise ValueError("Budgets, workers, timeouts and retry counts must be positive")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and annotation output must be different files")
    trajectories = read_records(args.input)
    digest_cache: dict[str, str] = {}
    for trajectory in trajectories:
        if not isinstance(trajectory.get("instruction"), str) or not trajectory["instruction"].strip():
            raise ValueError("Each annotation trajectory requires a nonempty instruction")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Each annotation trajectory requires a nonempty steps list")
        for index, step in enumerate(steps):
            if type(step.get("step_id")) is not int or step["step_id"] != index:
                raise ValueError("Annotation step IDs must be contiguous, starting at zero")
            if not isinstance(step.get("trainable", True), bool):
                raise ValueError("trainable must be a boolean")
            validate_action(step["action"])
            format_cta("", "", step["action"])
            name = step["image"]
            if name not in digest_cache:
                digest_cache[name] = image_digest(args.image_root, name)
            if "image_sha256" in step and step["image_sha256"] != digest_cache[name]:
                raise ValueError("An annotation screenshot does not match its recorded pixel hash")
            step["image_sha256"] = digest_cache[name]
    ids = [row["trajectory_id"] for row in trajectories]
    if not trajectories or len(set(ids)) != len(ids):
        raise ValueError("Annotation input must contain unique, nonempty trajectories")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Annotation output already exists; use --resume to continue it")
    completed_rows = read_records(args.output) if args.resume and args.output.exists() else []
    completed = {row["trajectory_id"]: row for row in completed_rows}
    if len(completed) != len(completed_rows):
        raise ValueError("The resume file contains duplicate trajectory IDs")
    if set(completed) - set(ids):
        raise ValueError("The resume file contains trajectories outside this input")
    for trajectory in trajectories:
        if trajectory["trajectory_id"] in completed:
            annotation = completed[trajectory["trajectory_id"]].get("annotation", {})
            expected = {
                "model": args.model, "tokenizer": args.tokenizer, "compression_budget": args.compression_budget,
                "source_sha256": source_digest(trajectory), "causal_prefix": True,
            }
            if annotation != expected:
                raise ValueError("The resume file does not match the input and annotation configuration")
            if source_digest(completed[trajectory["trajectory_id"]]) != source_digest(trajectory):
                raise ValueError("The resume file changed a trajectory observation or target action")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    for trajectory in completed.values():
        for step in trajectory["steps"]:
            decode_annotation(
                json.dumps({"compression": step["compression"], "think": step["think"]}),
                tokenizer, args.compression_budget, step["action"],
            )
    client = AnnotationClient(args.base_url, os.environ.get(args.api_key_env), args.model, args.timeout, args.retries)
    pending = [row for row in trajectories if row["trajectory_id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                annotate_trajectory, row, client, tokenizer, args.tokenizer, args.image_root,
                args.compression_budget, args.content_attempts, args.reuse_existing,
            ): row["trajectory_id"]
            for row in pending
        }
        for future in as_completed(futures):
            identifier = futures[future]
            completed[identifier] = future.result()
            write_records(args.output, (completed[key] for key in ids if key in completed))
            print(f"Annotated {len(completed)}/{len(trajectories)} trajectories", flush=True)
    if not pending:
        print(f"Annotations complete: {len(completed)} trajectories")


if __name__ == "__main__":
    main()
