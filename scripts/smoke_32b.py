"""Minimal offline multi-GPU 32B loading and structured-output smoke tests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError
from bian.methods.mixed_32b import validate_stage2_round, validate_type_result
from bian.models.dual_7b_backend import GenerationConfig
from bian.models.sharded_backend import Sharded32BBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=("bfloat16", "int8", "nf4"), default="bfloat16"
    )
    args = parser.parse_args()
    backend = Sharded32BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=4096,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            retries=1,
            seed=42,
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
        device_map={"": 0} if args.precision != "bfloat16" else "balanced",
        max_memory=(
            {0: "44GiB"}
            if args.precision != "bfloat16"
            else {0: "44GiB", 1: "44GiB"}
        ),
        precision=args.precision,
    )
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started
    short = backend.generate_json(
        role="32B",
        prompt_name="32b_short",
        prompt_version="32b-short-v1",
        payload={},
        validator=lambda value: value
        if value == {"status": "ok"}
        else (_ for _ in ()).throw(ValidationError("short response mismatch")),
    )
    evidence = {"E01"}
    stage2 = backend.generate_json(
        role="32B-Stage2",
        prompt_name="32b_stage2",
        prompt_version="32b-stage2-v1",
        payload={
            "round": 1,
            "candidates": [
                {
                    "candidate_id": "C01",
                    "device_role": "br",
                    "evidence": [
                        {
                            "evidence_id": "E01",
                            "metric_name": "bgp_session_state",
                            "status_transition": "up_to_down",
                        }
                    ],
                }
            ],
            "topology": {"nodes": ["C01"], "edges": []},
        },
        validator=lambda value: validate_stage2_round(
            value, {"C01"}, evidence
        ),
    )
    classification = backend.generate_json(
        role="32B-Classification",
        prompt_name="32b_type_ovr",
        prompt_version="32b-type-ovr-v1",
        payload={
            "type": {
                "type_id": "T01",
                "definition": "BGP session down",
                "required_monitoring_evidence": "BGP neighbor goes down",
            },
            "root_hypotheses": [
                {
                    "candidate_id": "C01",
                    "role": "br",
                    "root_weight": 1.0,
                    "evidence": [{"evidence_id": "E01", "metric": "BGP down"}],
                }
            ],
        },
        validator=lambda value: validate_type_result(
            value, "T01", {"C01"}, evidence
        ),
    )
    result = {
        "load_seconds": load_seconds,
        "short": short,
        "stage2_valid": bool(stage2["candidates"]),
        "classification_valid": bool(classification["root_hypotheses"]),
        "model": backend.model_manifest(),
        "calls": [asdict(call) for call in backend.calls],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"32B smoke passed; load={load_seconds:.3f}s; "
        f"calls={len(backend.calls)}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
