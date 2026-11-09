from __future__ import annotations

import json
from pathlib import Path

import torch.distributed as dist

from echoes.io import write_json
from echoes.training.config import resume_signature


def prepare_output(config: dict) -> Path:
    output = Path(config["training"]["output"])
    if not config["training"].get("resume") and (output / "metrics.jsonl").exists():
        raise ValueError("This output already contains a run; select a new output or specify --resume")
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
    dist.barrier()
    return output


def append_metrics(output: Path, record: dict) -> None:
    if dist.get_rank() == 0:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def save(engine, output: Path, config: dict, state: dict) -> Path:
    path = output / f"step_{state['global_step']:08d}"
    engine.save_checkpoint(str(path), global_step=state["global_step"])
    if dist.get_rank() == 0:
        write_json(path / "trainer_state.json", {**state, "configuration": resume_signature(config),
                                                "world_size": dist.get_world_size()})
        write_json(output / "latest.json", {"checkpoint": path.name})
    dist.barrier()
    return path


def restore(engine, config: dict, expected: dict) -> dict | None:
    path = config["training"].get("resume")
    if not path:
        return None
    path = Path(path)
    with (path / "trainer_state.json").open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state["configuration"] != resume_signature(config):
        raise ValueError("Resume configuration differs from the saved run")
    if state["world_size"] != dist.get_world_size():
        raise ValueError("Resume requires the checkpoint's distributed world size")
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Resume data mismatch: {key}")
    engine.load_checkpoint(str(path), del_local_after_load=False)
    return state
