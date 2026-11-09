from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from echoes.cta import parse_cta, truncate_memory
from echoes.inference import Policy
from echoes.prompts import student_messages


class CTAAgent:
    """A single-image policy retaining only its generated Compression between calls."""

    def __init__(self, policy: Policy, instruction: str):
        self.policy = policy
        self.reset(instruction)

    def reset(self, instruction: str) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty task instruction is required")
        self.instruction = instruction
        self.compression = ""
        self.step_index = 0

    def act(self, current_image) -> dict:
        generation = self.policy.generate(student_messages(self.instruction, current_image, self.compression))
        response = parse_cta(generation.text)
        self.compression = truncate_memory(response.compression, self.policy.tokenizer, 128)
        self.step_index += 1
        return {"step": self.step_index - 1, "compression": self.compression, "think": response.think,
                "action": response.action, "generation": asdict(generation)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    policy = Policy(args.model, device=args.device)
    agent = CTAAgent(policy, args.instruction)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not ({"reset", "image"} & set(request)):
                raise ValueError("Each request must contain image or reset")
            if "reset" in request:
                agent.reset(request["reset"])
            result = agent.act(request["image"]) if "image" in request else {"step": agent.step_index, "compression": agent.compression}
        except (ValueError, KeyError, TypeError, OSError) as error:
            result = {"error": str(error)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
