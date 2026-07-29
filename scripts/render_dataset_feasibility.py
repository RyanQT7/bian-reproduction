"""Generate post-freeze dataset-feasibility diagnostics and recommendations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def node_role(node: str) -> str:
    return "-".join(node.split("-")[2:])


def node_region(node: str) -> str:
    return "-".join(node.split("-")[:2])


def classification_metrics(records: list[dict], truths: dict[str, dict]) -> dict:
    top1 = top3 = category = 0
    per_case = []
    for record in records:
        truth = truths[record["incident_id"]]
        predicted = record["predicted_fault_type"]
        predicted_category = record["predicted_fault_category"]
        types = [item["fault_type"] for item in record["fault_type_top3"]]
        top1 += predicted == truth["fault_type"]
        top3 += truth["fault_type"] in types
        category += predicted_category == truth["fault_category"]
        per_case.append(
            {
                "incident_id": record["incident_id"],
                "true_fault_type": truth["fault_type"],
                "predicted_fault_type": predicted,
                "true_fault_category": truth["fault_category"],
                "predicted_fault_category": predicted_category,
                "top3_hit": truth["fault_type"] in types,
            }
        )
    count = len(records)
    return {
        "evaluated_cases": count,
        "fault_type_top1_accuracy": top1 / count,
        "fault_type_top3_hit_rate": top3 / count,
        "fault_category_accuracy": category / count,
        "per_case": per_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    args = parser.parse_args()
    feasibility = args.run_dir / "feasibility"
    feasibility.mkdir(exist_ok=False)
    evaluation = args.run_dir / "evaluation"
    truths = {
        item["incident_id"]: item for item in load_jsonl(args.ground_truth)
    }
    predictions = {
        item["incident_id"]: item
        for item in load_jsonl(
            args.run_dir / "prediction/predictions_frozen.jsonl"
        )
    }
    stage1 = {
        item["incident_id"]: item
        for item in load_jsonl(
            args.run_dir / "stage1_reference/full_rankings.jsonl"
        )
    }
    shortlist = {
        item["incident_id"]: item
        for item in load_jsonl(args.run_dir / "stage1_reference/shortlists.jsonl")
    }
    formal_score = json.loads((evaluation / "score_summary.json").read_text())
    baseline_score = json.loads(
        (args.baseline_run / "evaluation/score_summary.json").read_text()
    )
    oracle_records = load_jsonl(
        args.run_dir / "oracle_classification/oracle_top3.jsonl"
    )
    oracle_metrics = classification_metrics(oracle_records, truths)
    dump(
        args.run_dir / "oracle_classification/oracle_metrics.json",
        oracle_metrics,
    )

    root_audit = []
    stage1_credit = 0.0
    stage2_up = stage2_down = stage2_same = 0
    stage1_weights = {1: 1, 2: .8, 3: .6, 4: .4, 5: .2}
    role_stats: dict[str, list[dict]] = defaultdict(list)
    for incident_id, truth in sorted(truths.items()):
        root = truth["root_node_id"]
        ranking = [item["node_id"] for item in stage1[incident_id]["ranking"]]
        s1_rank = ranking.index(root) + 1
        stage1_credit += stage1_weights.get(s1_rank, 0)
        top5 = [
            item["node_id"]
            for item in predictions[incident_id].get("top5_root_causes", [])
        ]
        s2_rank = top5.index(root) + 1 if root in top5 else None
        comparable = s2_rank if s2_rank is not None else 73
        if comparable < s1_rank:
            stage2_up += 1
        elif comparable > s1_rank:
            stage2_down += 1
        else:
            stage2_same += 1
        incident = json.loads(
            (
                args.input_root / incident_id / "incident_input.json"
            ).read_text()
        )
        root_evidence = next(
            item
            for item in incident["engineering_evidence"]
            if item["node_id"] == root
        )
        direct = [
            item
            for item in root_evidence["evidence"]
            if item["direct_fault_evidence"]
        ]
        record = {
            "incident_id": incident_id,
            "root_node_id": root,
            "root_role": truth["root_node_role"],
            "root_region_id": truth["root_region_id"],
            "fault_type": truth["fault_type"],
            "stage1_rank": s1_rank,
            "stage1_shortlist_retained": any(
                item["node_id"] == root
                for item in shortlist[incident_id]["shortlist"]
            ),
            "stage2_rank": s2_rank,
            "direct_evidence_count": len(direct),
            "direct_metric_names": sorted(
                {item["metric_name"] for item in direct}
            ),
            "earliest_direct_change": min(
                (
                    item["first_change_time"]
                    for item in direct
                    if item["first_change_time"]
                ),
                default=None,
            ),
            "data_quality_statuses": sorted(
                {
                    item["data_quality_status"]
                    for item in root_evidence["evidence"]
                }
            ),
        }
        root_audit.append(record)
        role_stats[truth["root_node_role"]].append(record)
    with (feasibility / "root_signal_audit.jsonl").open("w") as handle:
        for item in root_audit:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    taxonomy = json.loads(args.taxonomy.read_text())
    evidence_rows = []
    for item in taxonomy:
        cases = [
            record
            for record in root_audit
            if record["fault_type"] == item["fault_type"]
        ]
        evidence_rows.append(
            {
                "fault_type": item["fault_type"],
                "fault_category": item["fault_category"],
                "observed_case_count": len(cases),
                "cases_with_direct_evidence": sum(
                    record["direct_evidence_count"] > 0 for record in cases
                ),
                "direct_metric_names": ";".join(
                    sorted(
                        {
                            metric
                            for record in cases
                            for metric in record["direct_metric_names"]
                        }
                    )
                ),
            }
        )
    with (feasibility / "evidence_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_rows[0])
        writer.writeheader()
        writer.writerows(evidence_rows)

    oracle_scores = load_jsonl(
        args.run_dir / "oracle_classification/oracle_type_scores.jsonl"
    )
    by_incident: dict[str, list[dict]] = defaultdict(list)
    for item in oracle_scores:
        by_incident[item["incident_id"]].append(item)
    separability_rows = []
    for incident_id, items in sorted(by_incident.items()):
        truth_type = truths[incident_id]["fault_type"]
        scores = {
            item["fault_type"]: item["aggregated_raw_score"] for item in items
        }
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        true_score = scores[truth_type]
        best_other = max(
            score for fault_type, score in scores.items()
            if fault_type != truth_type
        )
        separability_rows.append(
            {
                "incident_id": incident_id,
                "fault_type": truth_type,
                "true_type_rank": ranked.index(truth_type) + 1,
                "true_type_score": true_score,
                "best_other_score": best_other,
                "score_margin": true_score - best_other,
            }
        )
    with (feasibility / "fault_type_separability.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=separability_rows[0])
        writer.writeheader()
        writer.writerows(separability_rows)

    role_rows = []
    for role, records in sorted(role_stats.items()):
        role_rows.append(
            {
                "root_role": role,
                "case_count": len(records),
                "stage1_recall_at_15": sum(
                    item["stage1_rank"] <= 15 for item in records
                )
                / len(records),
                "stage2_top5_accuracy": sum(
                    item["stage2_rank"] is not None for item in records
                )
                / len(records),
                "mean_direct_evidence_count": statistics.mean(
                    item["direct_evidence_count"] for item in records
                ),
            }
        )
    with (feasibility / "role_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=role_rows[0])
        writer.writeheader()
        writer.writerows(role_rows)

    inventory_count = len(json.loads(args.candidate_inventory.read_text()))
    category_counts = Counter(
        truth["fault_category"] for truth in truths.values()
    )
    stage1_metrics = json.loads(
        (args.run_dir / "stage1_reference/metrics.json").read_text()
    )
    baseline = {
        "random_global_top5": 5 / inventory_count,
        "random_shortlist_top5_conditional": statistics.mean(
            min(5 / len(item["shortlist"]), 1)
            for item in shortlist.values()
        ),
        "random_fault_type_top1": 1 / len(taxonomy),
        "random_fault_type_top3": 3 / len(taxonomy),
        "random_fault_category_top1": 1 / len(category_counts),
        "majority_fault_category_accuracy": max(category_counts.values())
        / len(truths),
    }
    comparison = {
        "random": baseline,
        "stage1_only": {
            "root_recall_at_5": stage1_metrics["root_recall_at_5"],
            "root_recall_at_10": stage1_metrics["root_recall_at_10"],
            "root_recall_at_15": stage1_metrics["root_recall_at_15"],
            "localization_score_40_if_top5_used": stage1_credit
            / len(truths)
            * 40,
        },
        "dual_7b_v3": {
            "localization_score_40": baseline_score["localization"]["score_40"],
            "classification_score_30": baseline_score["classification"]["score_30"],
            "total_score_70": baseline_score["total_score_70"],
            "root_top1": baseline_score["localization"]["top1_accuracy"],
            "root_top5": baseline_score["localization"]["top5_accuracy"],
            "classification_top1": baseline_score["classification"][
                "fault_type_top1_accuracy"
            ],
            "classification_top3": baseline_score["classification"][
                "fault_type_top3_hit_rate"
            ],
        },
        "frozen_7b_plus_32b": {
            "localization_score_40": formal_score["localization"]["score_40"],
            "classification_score_30": formal_score["classification"]["score_30"],
            "total_score_70": formal_score["total_score_70"],
            "root_top1": formal_score["localization"]["top1_accuracy"],
            "root_top5": formal_score["localization"]["top5_accuracy"],
            "classification_top1": formal_score["classification"][
                "fault_type_top1_accuracy"
            ],
            "classification_top3": formal_score["classification"][
                "fault_type_top3_hit_rate"
            ],
            "oracle_classification": oracle_metrics,
            "stage2_true_root_promoted_cases": stage2_up,
            "stage2_true_root_demoted_cases": stage2_down,
            "stage2_true_root_unchanged_cases": stage2_same,
        },
    }
    dump(evaluation / "baseline_comparison.json", comparison)
    dump(
        evaluation / "stage_comparison.json",
        {
            "stage1_true_root_ranks": {
                item["incident_id"]: item["stage1_rank"] for item in root_audit
            },
            "stage2_true_root_ranks": {
                item["incident_id"]: item["stage2_rank"] for item in root_audit
            },
            "promoted": stage2_up,
            "demoted": stage2_down,
            "unchanged": stage2_same,
        },
    )

    direct_cases = sum(item["direct_evidence_count"] > 0 for item in root_audit)
    localization_support = formal_score["localization"]["top5_accuracy"]
    oracle_top1 = oracle_metrics["fault_type_top1_accuracy"]
    oracle_category = oracle_metrics["fault_category_accuracy"]
    report = f"""# Dataset feasibility report

This is a development-set diagnostic over only {len(truths)} incidents. It is not
a paper reproduction or a conclusive dataset validation.

## Root-cause localization

- Direct root-device evidence exists in {direct_cases}/{len(truths)} cases.
- Frozen Stage 1 recall@15 is {stage1_metrics['root_recall_at_15']:.1%}.
- 7B+32B exact-node Top5 is {localization_support:.1%}, versus the global random
  Top5 baseline of {baseline['random_global_top5']:.1%}.
- Stage 2 promoted/demoted/unchanged the true root in
  {stage2_up}/{stage2_down}/{stage2_same} cases.

Interpretation: use exact-node results together with region and role metrics. Cases
with shortlist exclusion cannot be recovered by Stage 2; cases with only downstream
service/traffic changes support propagation analysis more strongly than exact root
identification.

## Fault classification

- Formal exact type Top1/Top3:
  {formal_score['classification']['fault_type_top1_accuracy']:.1%}/
  {formal_score['classification']['fault_type_top3_hit_rate']:.1%}.
- Formal category accuracy:
  {formal_score['classification']['fault_category_accuracy']:.1%}.
- Oracle-root exact type Top1/Top3:
  {oracle_top1:.1%}/{oracle_metrics['fault_type_top3_hit_rate']:.1%}.
- Oracle-root category accuracy: {oracle_category:.1%}.

If oracle remains weak, the principal limitation is evidence/taxonomy separability,
not only root-ranking error. See `fault_type_separability.csv` for per-case margins.

## Monitoring sufficiency

BR requires explicit BGP neighbor state, route announce/withdraw counters, selected
route/next-hop changes and timestamped protocol logs. CR requires OSPFv3 neighbor,
cost, route-table and static-route state. FW requires effective rule order, ACL hit/
drop counters, port-policy state, rate-limit counters, default-route next hop and
resource pressure. Traffic and service VM signals remain valuable propagation
evidence but should not substitute for these direct controls.

Keep aggregated traffic_flow_metrics and timestamped protocol/policy state. Raw
five-tuple flow rows may remain excluded when equivalent deterministic aggregates
are retained.

## Sample-size limitation

Ten development incidents are insufficient to establish general validity. Add at
least 20 independent incidents per fault type across multiple regions, loads and
topologies, with held-out injection variants and truly unseen final cases.
"""
    (feasibility / "dataset_feasibility_report.md").write_text(report)
    (feasibility / "label_granularity_analysis.md").write_text(
        "# Label granularity analysis\n\n"
        "Use oracle-root confusion and separability margins before retaining fine labels. "
        "Pairs such as route_blackhole/wrong_static_route, wrong_default_route/"
        "firewall_default_route_error, and ACL drop/port block/rule-order error require "
        "direct route or effective-policy state; downstream reachability alone cannot "
        "reliably distinguish them. Where those fields are absent, score the broader "
        "category or merge observationally equivalent labels. Preserve exact labels only "
        "when their defining control-plane/policy metric is collected.\n"
    )
    (feasibility / "competition_recommendations.md").write_text(
        "# Competition recommendations\n\n"
        "- Retain exact-node Top5 and hierarchical type/category scoring.\n"
        "- Add region-level and device-role auxiliary metrics without changing the main score.\n"
        "- Publish data-quality flags and identify cases whose defining metric was not collected.\n"
        "- Add at least 20 independent, region-diverse examples per fine fault type.\n"
        "- Keep protocol logs, route state, effective firewall policy and aggregate traffic.\n"
        "- Continue excluding raw five-tuples when aggregate direction/rate/connectivity features exist.\n"
        "- Report data-insufficient cases separately rather than silently excluding them.\n"
    )
    print(
        f"Feasibility report written; formal={formal_score['total_score_70']:.3f}/70; "
        f"oracle_top1={oracle_top1:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
