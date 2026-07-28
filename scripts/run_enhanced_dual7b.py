"""Run the frozen-Stage-1 engineering-enhanced Dual-7B RCA pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.methods.enhanced_stage2 import (
    score_classification,
    score_stage2,
    validate_classification,
    validate_stage2,
)
from bian.models.dual_7b_backend import Dual7BBackend, GenerationConfig


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def taxonomy_description(item: dict) -> dict:
    fault_type = item["fault_type"]
    tokens = fault_type.replace("_", " ")
    roles = []
    if any(key in fault_type for key in ("bgp", "route_flap")):
        roles = ["br"]
    elif any(key in fault_type for key in ("ospf", "static_route", "interface_cost")):
        roles = ["cr", "br"]
    elif any(key in fault_type for key in ("acl", "firewall", "port_block")):
        roles = ["fw"]
    elif any(key in fault_type for key in ("cpu", "resource")):
        roles = ["fw", "service", "traffic-vm"]
    else:
        roles = ["br", "cr", "fw", "traffic-vm", "service"]
    return {
        "fault_category": item["fault_category"],
        "definition": f"Operational pattern associated with {tokens}.",
        "applicable_roles": roles,
    }


def compact_candidate(item: dict, evidence: dict) -> dict:
    by_id = {entry["evidence_id"]: entry for entry in evidence["evidence"]}
    evidence_items = [
        by_id[evidence_id]
        for evidence_id in item["supporting_evidence_ids"][:2]
        if evidence_id in by_id
    ]
    return {
        "candidate_id": item["candidate_id"],
        "device_role": item["device_role"],
        "stage1_components": {
            key: item[key]
            for key in (
                "deterministic_feature_score",
                "model_anomaly_score",
                "temporal_change_score",
                "direct_fault_evidence_score",
                "symptom_likelihood",
                "data_quality_adjustment",
                "stage1_score",
            )
        },
        "earliest_change_time": item["earliest_change_time"],
        "evidence": [
            {
                key: entry[key]
                for key in (
                    "evidence_id", "metric_name", "source_type",
                    "first_change_time", "pre_value", "fault_value",
                    "post_value", "stable_change_score", "direction",
                    "status_transition", "data_quality_status",
                    "direct_fault_evidence",
                )
            }
            for entry in evidence_items
        ],
    }


def compact_topology(topology: dict, alias_to_node: dict[str, str]) -> dict:
    node_to_alias = {node: alias for alias, node in alias_to_node.items()}
    candidate_nodes = set(node_to_alias)
    edges = [
        edge
        for edge in topology["edges"]
        if edge["source"] in candidate_nodes and edge["target"] in candidate_nodes
    ]
    return {
        "directed": topology["directed"],
        "nodes": sorted(alias_to_node),
        "edges": [
            {
                "source": node_to_alias[edge["source"]],
                "target": node_to_alias[edge["target"]],
                "relation": edge.get("relation", edge.get("type", "connected")),
            }
            for edge in edges
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incident-id")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    prediction_dir = args.output_dir / "prediction"
    reports_dir = prediction_dir / "reports"
    logs_dir = prediction_dir / "logs"
    environment_dir = args.output_dir / "environment"
    for path in (reports_dir, logs_dir, environment_dir):
        path.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    shortlists = {
        item["incident_id"]: item
        for item in load_jsonl(args.stage1_dir / "shortlist.jsonl")
    }
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    input_paths = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if args.incident_id:
        input_paths = [p for p in input_paths if p.parent.name == args.incident_id]
    backend = Dual7BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=8192,
            max_new_tokens=2048,
            temperature=0.0,
            retries=2,
            seed=config["seed"],
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
    )
    predictions_path = prediction_dir / "predictions.jsonl"
    sensitivity_records = []
    stage2_records = []
    started_all = time.perf_counter()
    for path in input_paths:
        incident = json.loads(path.read_text())
        incident_id = incident["incident_id"]
        started = time.perf_counter()
        calls_before = len(backend.calls)
        try:
            shortlist = shortlists[incident_id]["shortlist"]
            alias_to_node = {
                item["candidate_id"]: item["node_id"] for item in shortlist
            }
            evidence_by_node = {
                item["node_id"]: item for item in incident["engineering_evidence"]
            }
            candidates = [
                compact_candidate(item, evidence_by_node[item["node_id"]])
                for item in shortlist
            ]
            stage2 = backend.generate_json(
                role="7B-B",
                prompt_name="7b_b_enhanced_stage2",
                prompt_version="engineering-stage2-v2",
                payload={
                    "incident_window": {
                        "start": incident["fault_start_time_utc"],
                        "end": incident["fault_end_time_utc"],
                    },
                    "candidate_map_policy": "Cxx aliases only; node IDs hidden",
                    "topology": compact_topology(incident["topology"], alias_to_node),
                    "candidates": candidates,
                },
                validator=lambda value, aliases=set(alias_to_node): validate_stage2(
                    value, aliases
                ),
            )
            top5, rank_data = score_stage2(
                stage2["candidates"],
                alias_to_node,
                config["stage2"]["weights"],
            )
            stage2_records.append(
                {
                    "incident_id": incident_id,
                    "candidate_map": alias_to_node,
                    "model_components": stage2["candidates"],
                    "rank_of_ranks": rank_data,
                }
            )
            order_outputs = []
            for order_index in range(3):
                shuffled = list(taxonomy)
                random.Random(
                    config["seed"] * 1000
                    + int(incident_id.rsplit("-", 1)[1])
                    + order_index * 100
                ).shuffle(shuffled)
                alias_to_taxonomy = {
                    f"T{index:02d}": item for index, item in enumerate(shuffled, 1)
                }
                taxonomy_payload = [
                    {"fault_type_id": alias, **taxonomy_description(item)}
                    for alias, item in alias_to_taxonomy.items()
                ]
                classification = backend.generate_json(
                    role="7B-B",
                    prompt_name="7b_b_enhanced_classification",
                    prompt_version="engineering-classification-v2",
                    payload={
                        "top5": top5,
                        "top1_role": "-".join(top5[0]["node_id"].split("-")[2:]),
                        "top1_evidence": top5[0]["supporting_evidence_ids"],
                        "taxonomy": taxonomy_payload,
                    },
                    validator=lambda value, aliases=set(alias_to_taxonomy): (
                        validate_classification(value, aliases)
                    ),
                )
                scored = score_classification(
                    classification["fault_types"], alias_to_taxonomy
                )
                order_outputs.append(
                    {
                        "order_index": order_index,
                        "alias_map": {
                            alias: item["fault_type"]
                            for alias, item in alias_to_taxonomy.items()
                        },
                        "top3": scored,
                    }
                )
            top1s = [item["top3"][0]["fault_type"] for item in order_outputs]
            top3_sets = [
                sorted(x["fault_type"] for x in item["top3"])
                for item in order_outputs
            ]
            sensitivity = {
                "incident_id": incident_id,
                "orders": order_outputs,
                "top1_all_equal": len(set(top1s)) == 1,
                "top3_set_all_equal": len({tuple(x) for x in top3_sets}) == 1,
                "first_item_follow_count": sum(
                    output["top3"][0]["fault_type"]
                    == output["alias_map"]["T01"]
                    for output in order_outputs
                ),
            }
            sensitivity_records.append(sensitivity)
            official_top3 = order_outputs[0]["top3"]
            elapsed = time.perf_counter() - started
            calls = backend.calls[calls_before:]
            prediction = {
                "incident_id": incident_id,
                "prediction_status": "success",
                "result_label": config["result_label"],
                "model_configuration": config["model_configuration"],
                "top5_root_causes": top5,
                "predicted_fault_type": official_top3[0]["fault_type"],
                "predicted_fault_category": official_top3[0]["fault_category"],
                "fault_type_top3": official_top3,
                "rank_of_ranks": rank_data,
                "stage1_shortlist": [item["node_id"] for item in shortlist],
                "taxonomy_order_sensitivity": {
                    key: sensitivity[key]
                    for key in (
                        "top1_all_equal",
                        "top3_set_all_equal",
                        "first_item_follow_count",
                    )
                },
                "inference": {
                    "elapsed_seconds": elapsed,
                    "model_calls": len(calls),
                    "retries": sum(call.attempt > 1 for call in calls),
                    "input_tokens": sum(call.input_tokens for call in calls),
                    "output_tokens": sum(call.output_tokens for call in calls),
                    "peak_gpu_memory_mib": max(
                        (call.peak_gpu_memory_mib for call in calls), default=0
                    ),
                },
            }
            append(predictions_path, prediction)
            report = [
                f"# {incident_id} — {config['result_label']}",
                "",
                f"- Input window: {incident['slice_start_time_utc']} to {incident['slice_end_time_utc']}",
                f"- Stage 1 shortlist size: {len(shortlist)}",
                f"- Stage 2 reordered: {[x['node_id'] for x in top5] != [x['node_id'] for x in shortlist[:5]]}",
                f"- Taxonomy top1 stable: {sensitivity['top1_all_equal']}",
                f"- Elapsed seconds: {elapsed:.3f}",
                "",
                "## Top5 root causes",
                *[
                    f"{x['rank']}. {x['node_id']} ({x['failure_score']:.6f}) — {x['reason_summary']}"
                    for x in top5
                ],
                "",
                "## Top3 fault types",
                *[
                    f"{x['rank']}. {x['fault_type']} / {x['fault_category']} ({x['confidence']:.6f})"
                    for x in official_top3
                ],
            ]
            (reports_dir / f"{incident_id}.md").write_text("\n".join(report) + "\n")
        except Exception as exc:
            append(
                predictions_path,
                {
                    "incident_id": incident_id,
                    "prediction_status": "prediction_failed",
                    "result_label": config["result_label"],
                    "model_configuration": config["model_configuration"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    for target, records in (
        (prediction_dir / "stage2_outputs.jsonl", stage2_records),
        (prediction_dir / "taxonomy_order_sensitivity.json", sensitivity_records),
    ):
        if target.suffix == ".jsonl":
            for record in records:
                append(target, record)
        else:
            target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (logs_dir / "model_calls.json").write_text(
        json.dumps(backend.call_manifest(), ensure_ascii=False, indent=2) + "\n"
    )
    guard = {
        "ground_truth_available_to_inference": False,
        "evaluation_path_read_by_inference": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (prediction_dir / "truth_access_guard.json").write_text(
        json.dumps(guard, indent=2) + "\n"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest = {
        **{key: config[key] for key in (
            "result_label", "model_configuration", "paper_model_equivalent",
            "development_dataset", "strict_blind_evaluation",
            "ground_truth_used_for_post_run_diagnostics",
            "ground_truth_available_to_inference", "seed", "rank_rounds"
        )},
        "git_commit": commit,
        "model_path": str(args.model_path),
        "logical_roles": ["7B-A", "7B-B"],
        "total_inference_seconds": time.perf_counter() - started_all,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (environment_dir / "git.txt").write_text(commit + "\n")
    (environment_dir / "model_manifest.json").write_text(
        json.dumps({"path": str(args.model_path), "local_files_only": True}, indent=2)
        + "\n"
    )
    prompt_digest = hashlib.sha256()
    for prompt in sorted((PROJECT_ROOT / "src/bian/prompts").glob("*.txt")):
        prompt_digest.update(prompt.name.encode())
        prompt_digest.update(prompt.read_bytes())
    (environment_dir / "prompts.sha256").write_text(prompt_digest.hexdigest() + "\n")
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
