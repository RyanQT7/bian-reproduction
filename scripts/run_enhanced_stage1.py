"""Run Stage 1 only; this process has no ground-truth argument or access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.methods.enhanced_stage1 import rank_stage1, render_enhanced_evidence
from bian.models.dual_7b_backend import Dual7BBackend, GenerationConfig
from bian.models.structured_output import validate_device_analysis
from bian.real_inference import batched


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incident-id")
    args = parser.parse_args()
    if "evaluation" in str(args.input_root).lower():
        raise ValueError("truth-access guard: inference input cannot reference evaluation")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError("CUDA_VISIBLE_DEVICES is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    backend = Dual7BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=8192,
            max_new_tokens=768,
            retries=2,
            seed=config["seed"],
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
    )
    rankings_path = args.output_dir / "full_rankings.jsonl"
    shortlist_path = args.output_dir / "shortlist.jsonl"
    analyses_path = args.output_dir / "device_analyses.jsonl"
    input_paths = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if args.incident_id:
        input_paths = [
            path for path in input_paths if path.parent.name == args.incident_id
        ]
    if not input_paths:
        raise ValueError("no incident inputs selected")
    for path in input_paths:
        incident = json.loads(path.read_text())
        evidence = incident["engineering_evidence"]
        analyses = []
        for batch in batched(evidence, 8):
            requests = []
            for item in batch:
                node = item["node_id"]
                requests.append(
                    {
                        "role": "7B-A",
                        "prompt_name": "7b_a_device_analysis",
                        "prompt_version": "dual7b-a-device-engineering-v2",
                        "payload": {
                            "incident_id": incident["incident_id"],
                            "required_node_ids": [node],
                            "device_evidence_lines": [
                                render_enhanced_evidence(item)
                            ],
                        },
                        "validator": lambda value, expected=(node,): validate_device_analysis(
                            value, expected
                        ),
                    }
                )
            results = backend.generate_json_batch(requests, max_new_tokens=768)
            for result in results:
                analyses.extend(result["devices"])
        ranking, shortlist = rank_stage1(
            evidence, analyses, config["stage1"]
        )
        append(
            rankings_path,
            {"incident_id": incident["incident_id"], "ranking": ranking},
        )
        append(
            shortlist_path,
            {
                "incident_id": incident["incident_id"],
                "shortlist": shortlist,
                "candidate_map": {
                    item["candidate_id"]: item["node_id"] for item in shortlist
                },
            },
        )
        append(
            analyses_path,
            {"incident_id": incident["incident_id"], "device_analyses": analyses},
        )
        print(
            f"{incident['incident_id']}: shortlist={len(shortlist)} "
            f"top1={shortlist[0]['node_id']}",
            flush=True,
        )
    (args.output_dir / "model_calls.json").write_text(
        json.dumps(backend.call_manifest(), ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "truth_access_guard.json").write_text(
        json.dumps(
            {
                "passed": True,
                "ground_truth_available_to_inference": False,
                "evaluation_path_received": False,
            },
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "stage1_config.yaml").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "stage1_config.sha256").write_text(
        hashlib.sha256(args.config.read_bytes()).hexdigest() + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
