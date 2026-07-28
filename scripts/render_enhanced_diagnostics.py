"""Render post-freeze development diagnostics for enhanced Dual-7B runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def region(node: str) -> str:
    return "-".join(node.split("-")[:2])


def role(node: str) -> str:
    return "-".join(node.split("-")[2:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    predictions = {
        x["incident_id"]: x
        for x in load_jsonl(args.run_dir / "prediction/predictions_frozen.jsonl")
    }
    truths = {x["incident_id"]: x for x in load_jsonl(args.ground_truth)}
    stage1 = {
        x["incident_id"]: x
        for x in load_jsonl(args.run_dir / "stage1/full_rankings.jsonl")
    }
    calls = json.loads(
        (args.run_dir / "prediction/logs/model_calls.json").read_text()
    )
    sensitivity = json.loads(
        (args.run_dir / "prediction/taxonomy_order_sensitivity.json").read_text()
    )
    per_case = []
    changed = 0
    region_top1 = region_top5 = role_top1 = role_top5 = 0
    stage1_ranks = []
    stage2_ranks = []
    for incident_id in sorted(truths):
        truth_node = truths[incident_id]["root_node_id"]
        s1_nodes = [x["node_id"] for x in stage1[incident_id]["ranking"]]
        s2_nodes = [
            x["node_id"]
            for x in predictions[incident_id].get("top5_root_causes", [])
        ]
        s1_rank = s1_nodes.index(truth_node) + 1
        s2_rank = s2_nodes.index(truth_node) + 1 if truth_node in s2_nodes else None
        stage1_ranks.append(s1_rank)
        if s2_rank:
            stage2_ranks.append(s2_rank)
        changed += s2_nodes != s1_nodes[:5]
        region_top1 += bool(s2_nodes and region(s2_nodes[0]) == region(truth_node))
        region_top5 += any(region(node) == region(truth_node) for node in s2_nodes)
        role_top1 += bool(s2_nodes and role(s2_nodes[0]) == role(truth_node))
        role_top5 += any(role(node) == role(truth_node) for node in s2_nodes)
        per_case.append(
            {
                "incident_id": incident_id,
                "true_root_stage1_rank": s1_rank,
                "true_root_stage2_rank": s2_rank,
                "stage2_reordered_top5": s2_nodes != s1_nodes[:5],
            }
        )
    count = len(truths)
    retries = sum(call["attempt"] > 1 for call in calls)
    errors = sum(call["error"] is not None for call in calls)
    repaired = sum(call["json_repair_used"] for call in calls)
    metrics = {
        "stage1": {
            **{
                key: json.loads(
                    (args.run_dir / "stage1/stage1_metrics.json").read_text()
                )[key]
                for key in (
                    "root_recall_at_5",
                    "root_recall_at_10",
                    "root_recall_at_15",
                    "true_root_mean_rank",
                    "shortlist_mean_size",
                    "unavailable_by_role_anomaly_contribution_count",
                )
            },
            "true_root_ranks": {
                item["incident_id"]: item["true_root_stage1_rank"] for item in per_case
            },
        },
        "stage2": {
            "region_top1_accuracy": region_top1 / count,
            "region_top5_accuracy": region_top5 / count,
            "role_top1_accuracy": role_top1 / count,
            "role_top5_accuracy": role_top5 / count,
            "ranking_change_rate": changed / count,
            "true_root_mean_rank_when_in_top5": (
                statistics.mean(stage2_ranks) if stage2_ranks else None
            ),
            "per_case": per_case,
        },
        "engineering": {
            "successful_predictions": sum(
                x["prediction_status"] == "success" for x in predictions.values()
            ),
            "failed_predictions": sum(
                x["prediction_status"] != "success" for x in predictions.values()
            ),
            "model_calls": len(calls),
            "structured_call_success_rate": (len(calls) - errors) / len(calls),
            "first_json_success_rate": sum(
                call["attempt"] == 1 and call["error"] is None for call in calls
            )
            / len(calls),
            "retry_attempt_records": retries,
            "json_repair_count": repaired,
            "peak_gpu_memory_mib": max(
                call["peak_gpu_memory_mib"] for call in calls
            ),
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": sum(call["output_tokens"] for call in calls),
        },
    }
    (args.run_dir / "evaluation/stage_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    top1_equal = sum(x["top1_all_equal"] for x in sensitivity)
    top3_equal = sum(x["top3_set_all_equal"] for x in sensitivity)
    first_follow = sum(x["first_item_follow_count"] for x in sensitivity)
    sensitivity_summary = {
        "evaluated_cases": len(sensitivity),
        "orders_per_case": 3,
        "top1_three_order_consistency_rate": top1_equal / len(sensitivity),
        "top3_set_three_order_consistency_rate": top3_equal / len(sensitivity),
        "top1_followed_first_taxonomy_item_rate": first_follow
        / (len(sensitivity) * 3),
        "order_collapse_detected": first_follow == len(sensitivity) * 3,
        "per_case": sensitivity,
    }
    (args.run_dir / "evaluation/taxonomy_order_sensitivity.json").write_text(
        json.dumps(sensitivity_summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"stage_metrics": metrics, "taxonomy": sensitivity_summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
