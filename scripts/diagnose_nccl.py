"""Minimal two-rank NCCL all-reduce diagnostic."""

from __future__ import annotations

from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    value = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    expected = dist.get_world_size() * (dist.get_world_size() - 1) / 2
    result = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": dist.get_world_size(),
        "device": torch.cuda.get_device_name(local_rank),
        "value": float(value.item()),
        "expected": expected,
        "success": float(value.item()) == expected,
    }
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
