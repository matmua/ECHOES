from __future__ import annotations

import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def initialize(seed: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    return rank, world, torch.device("cuda", local_rank)


def maximum(value: int, device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def summed(value: float, device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def average_metrics(metrics: dict[str, float], device) -> dict[str, float]:
    keys = sorted(metrics)
    tensor = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(keys, tensor.tolist()))


def timed(function, *args, device, **kwargs):
    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return result, float(elapsed.item())
