"""Auditable Stage 1 fusion for the engineering-enhanced pipeline."""

from __future__ import annotations

import math
from typing import Any

from bian.data.validators import normalize_scores


def rank_stage1(
    evidence: list[dict[str, Any]],
    model_analyses: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_node = {item["node_id"]: item for item in model_analyses}
    weights = config["weights"]
    records = []
    for item in evidence:
        node = item["node_id"]
        summary = item["feature_summary"]
        model = by_node[node]
        model_score = float(model["anomaly_score"])
        raw = (
            weights["deterministic_feature_score"]
            * summary["deterministic_feature_score"]
            + weights["model_anomaly_score"] * model_score
            + weights["temporal_change_score"] * summary["temporal_change_score"]
            + weights["direct_fault_evidence_score"]
            * summary["direct_fault_evidence_score"]
            + weights["data_quality_adjustment"]
            * summary["data_quality_adjustment"]
            - weights["symptom_likelihood_penalty"]
            * summary["symptom_likelihood"]
        )
        raw = max(0.0, raw)
        supporting = [
            evidence_item["evidence_id"]
            for evidence_item in item["evidence"]
            if evidence_item["stable_change_score"] >= 0.3
        ][:12]
        counter = [
            evidence_item["evidence_id"]
            for evidence_item in item["evidence"]
            if evidence_item["data_quality_status"]
            in {"partial", "collection_failed", "empty_but_expected"}
        ][:6]
        records.append(
            {
                "candidate_id": None,
                "node_id": node,
                "device_role": item["device_role"],
                "deterministic_feature_score": summary["deterministic_feature_score"],
                "model_anomaly_score": model_score,
                "temporal_change_score": summary["temporal_change_score"],
                "direct_fault_evidence_score": summary[
                    "direct_fault_evidence_score"
                ],
                "symptom_likelihood": summary["symptom_likelihood"],
                "data_quality_adjustment": summary["data_quality_adjustment"],
                "earliest_change_time": summary["earliest_change_time"],
                "raw_stage1_score": raw,
                "stage1_score": 0.0,
                "supporting_evidence_ids": supporting,
                "counter_evidence_ids": counter,
                "shortlist_reason": (
                    "direct role-specific evidence"
                    if summary["direct_fault_evidence_score"] >= 0.3
                    else "temporal/model evidence with symptom penalty"
                ),
            }
        )
    raw_scores = {item["node_id"]: item["raw_stage1_score"] for item in records}
    if sum(raw_scores.values()) <= 0:
        raise ValueError("Stage 1 produced no positive score")
    normalized = normalize_scores(raw_scores)
    for item in records:
        item["stage1_score"] = normalized[item["node_id"]]
    records.sort(key=lambda item: (-item["stage1_score"], item["node_id"]))
    cumulative = 0.0
    shortlist = []
    for item in records:
        shortlist.append(item)
        cumulative += item["stage1_score"]
        if (
            len(shortlist) >= config["min_candidates"]
            and cumulative >= config["top_p"]
        ):
            break
        if len(shortlist) >= config["max_candidates"]:
            break
    shortlist = shortlist[: config["max_candidates"]]
    while len(shortlist) < config["min_candidates"]:
        shortlist.append(records[len(shortlist)])
    for index, item in enumerate(shortlist, start=1):
        item["candidate_id"] = f"C{index:02d}"
    return records, shortlist


def render_enhanced_evidence(item: dict[str, Any], max_items: int = 8) -> str:
    states = ",".join(
        f"{name}={status}"
        for name, status in sorted(item["source_semantics"].items())
    )
    evidence = ";".join(
        f"{entry['evidence_id']}:{entry['source_type']}.{entry['metric_name']}:"
        f"{entry['pre_value']}->{entry['fault_value']}->post:{entry['post_value']},"
        f"stable:{entry['stable_change_score']:.3f},"
        f"direct:{str(entry['direct_fault_evidence']).lower()},"
        f"transition:{entry['status_transition']}"
        for entry in item["evidence"][:max_items]
    )
    return (
        f"node={item['node_id']}|role={item['device_role']}|states={states}|"
        f"evidence={evidence or 'none'}"
    )
