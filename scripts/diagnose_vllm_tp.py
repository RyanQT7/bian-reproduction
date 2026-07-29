"""Load-only vLLM tensor-parallel diagnostic with no generation."""

from __future__ import annotations

import argparse
import json
import time

from vllm import LLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-format", choices=("dummy", "auto"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    started = time.perf_counter()
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=False,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        quantization=None,
        load_format=args.load_format,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=2,
        cpu_offload_gb=0,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        reasoning_parser="deepseek_r1",
        enable_prefix_caching=False,
    )
    tokenizer = engine.get_tokenizer()
    print(
        json.dumps(
            {
                "status": "loaded",
                "load_format": args.load_format,
                "tensor_parallel_size": args.tensor_parallel_size,
                "dtype": "bfloat16",
                "quantization": None,
                "elapsed_seconds": time.perf_counter() - started,
                "tokenizer": type(tokenizer).__name__,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
