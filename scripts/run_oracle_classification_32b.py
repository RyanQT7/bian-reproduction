"""Post-freeze oracle-root classification diagnostic using no fault labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bian.models.dual_7b_backend import GenerationConfig
from bian.models.sharded_backend import Sharded32BBackend
from run_mixed_7b32b import load_jsonl, run_classification


def oracle_root_payload(evidence: dict) -> tuple[list[dict], set[str]]:
    ranked = sorted(
        evidence["evidence"],
        key=lambda item: (
            -float(item["stable_change_score"]),
            item["evidence_id"],
        ),
    )[:3]
    evidence_ids = {item["evidence_id"] for item in ranked}
    compact = [
        {
            field: item[field]
            for field in (
                "evidence_id",
                "metric_name",
                "source_type",
                "first_change_time",
                "pre_value",
                "fault_value",
                "post_value",
                "stable_change_score",
                "direction",
                "status_transition",
                "data_quality_status",
                "direct_fault_evidence",
            )
        }
        for item in ranked
    ]
    return [
        {
            "candidate_id": "C01",
            "device_role": evidence["device_role"],
            "region_id": "-".join(evidence["node_id"].split("-")[:2]),
            "root_weight": 1.0,
            "evidence": compact,
            "data_quality_statuses": sorted(
                {item["data_quality_status"] for item in evidence["evidence"]}
            ),
        }
    ], evidence_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    # Reading the frozen prediction is an explicit guard that oracle runs only later.
    frozen_ids = {
        item["incident_id"] for item in load_jsonl(args.frozen_predictions)
    }
    truths = {
        item["incident_id"]: item for item in load_jsonl(args.ground_truth)
    }
    if set(truths) != frozen_ids:
        raise ValueError("oracle truth IDs differ from frozen predictions")
    config = json.loads(args.config.read_text())
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    backend = Sharded32BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=config["stage2_max_input_tokens"],
            max_new_tokens=config["classification_max_new_tokens"],
            temperature=config["stage2_temperature"],
            top_p=config["stage2_top_p"],
            retries=config["structured_retries"],
            seed=42,
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
        device_map={"": 0},
        max_memory={0: "44GiB"},
        precision=config["precision"],
    )
    started = time.perf_counter()
    backend.load()
    predictions = []
    for incident_id in sorted(truths):
        incident = json.loads(
            (args.input_root / incident_id / "incident_input.json").read_text()
        )
        evidence_by_node = {
            item["node_id"]: item for item in incident["engineering_evidence"]
        }
        root_node = truths[incident_id]["root_node_id"]
        roots, evidence_ids = oracle_root_payload(evidence_by_node[root_node])
        top3, _ = run_classification(
            backend,
            incident_id,
            taxonomy,
            roots,
            evidence_ids,
            args.output_dir / "oracle_type_scores.jsonl",
        )
        record = {
            "incident_id": incident_id,
            "diagnostic_only": True,
            "oracle_root_node_id": root_node,
            "predicted_fault_type": top3[0]["fault_type"],
            "predicted_fault_category": top3[0]["fault_category"],
            "fault_type_top3": top3,
        }
        predictions.append(record)
        with (args.output_dir / "oracle_top3.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    calls = [asdict(call) for call in backend.calls]
    (args.output_dir / "model_calls.json").write_text(
        json.dumps(calls, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = {
        "diagnostic_only": True,
        "ground_truth_root_used": True,
        "ground_truth_fault_type_used_for_inference": False,
        "ground_truth_fault_category_used_for_inference": False,
        "elapsed_seconds": time.perf_counter() - started,
        "model": backend.model_manifest(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Oracle classification completed: {len(predictions)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
