"""Auditable Stage 2 and taxonomy scoring for the engineering-enhanced run."""

from __future__ import annotations

import math
from typing import Any

from bian.data.validators import ValidationError, normalize_scores
from bian.methods.rank_of_ranks import aggregate_rankings


COMPONENTS = (
    "local_anomaly_score",
    "temporal_precedence_score",
    "topology_upstream_score",
    "fault_pattern_compatibility_score",
    "symptom_likelihood",
)
CLASS_COMPONENTS = (
    "root_role_compatibility",
    "metric_pattern_compatibility",
    "protocol_state_compatibility",
    "temporal_pattern_compatibility",
    "topology_context_compatibility",
    "counter_evidence_penalty",
)


def _score(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be finite")
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be within [0,1]")
    return float(value)


def validate_stage2(value: dict[str, Any], aliases: set[str]) -> dict[str, Any]:
    items = value.get("candidates")
    if not isinstance(items, list) or len(items) != len(aliases):
        raise ValidationError("candidates must contain every shortlist alias")
    seen = set()
    result = []
    for item in items:
        alias = item.get("candidate_id")
        if alias not in aliases or alias in seen:
            raise ValidationError("illegal or duplicate candidate_id")
        seen.add(alias)
        rendered = {"candidate_id": alias}
        for name in COMPONENTS:
            rendered[name] = _score(item.get(name), name)
        for name in ("supporting_evidence_ids", "counter_evidence_ids"):
            ids = item.get(name)
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                raise ValidationError(f"{name} must be a string list")
            rendered[name] = ids[:8]
        if not isinstance(item.get("reason"), str):
            raise ValidationError("reason must be a string")
        rendered["reason"] = item["reason"]
        result.append(rendered)
    return {"candidates": result}


def score_stage2(
    items: list[dict[str, Any]],
    alias_to_node: dict[str, str],
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = {}
    by_node = {}
    for item in items:
        node = alias_to_node[item["candidate_id"]]
        value = (
            weights["local_anomaly_score"] * item["local_anomaly_score"]
            + weights["temporal_precedence_score"] * item["temporal_precedence_score"]
            + weights["topology_upstream_score"] * item["topology_upstream_score"]
            + weights["fault_pattern_compatibility_score"]
            * item["fault_pattern_compatibility_score"]
            - weights["symptom_likelihood_penalty"] * item["symptom_likelihood"]
        )
        raw[node] = max(0.0, value)
        by_node[node] = {**item, "node_id": node, "raw_final_score": raw[node]}
    if not any(raw.values()):
        raise ValidationError("Stage 2 produced no positive score")
    combined = normalize_scores(raw)
    # Three transparent perspectives provide Rank-of-Ranks without pretending that
    # rank weights are model probabilities.
    perspectives = []
    for fields in (
        ("local_anomaly_score", "temporal_precedence_score"),
        ("topology_upstream_score", "fault_pattern_compatibility_score"),
        tuple(),
    ):
        perspective_raw = {}
        for node, item in by_node.items():
            perspective_raw[node] = (
                combined[node]
                if not fields
                else max(
                    0.0,
                    sum(item[field] for field in fields)
                    - 0.15 * item["symptom_likelihood"],
                )
            )
        if not any(perspective_raw.values()):
            perspective_raw = dict(combined)
        perspectives.append(normalize_scores(perspective_raw))
    candidates = tuple(sorted(raw))
    ranking, averages = aggregate_rankings(perspectives, candidates)
    top5_raw = {node: 1.0 / averages[node] for node in ranking[:5]}
    top5_scores = normalize_scores(top5_raw)
    output = []
    for rank, node in enumerate(ranking[:5], 1):
        item = by_node[node]
        output.append(
            {
                "rank": rank,
                "node_id": node,
                "failure_score": top5_scores[node],
                "reason_summary": item["reason"],
                "stage2_components": {
                    name: item[name] for name in COMPONENTS
                },
                "raw_final_score": item["raw_final_score"],
                "supporting_evidence_ids": item["supporting_evidence_ids"],
                "counter_evidence_ids": item["counter_evidence_ids"],
            }
        )
    return output, {
        "rounds": 3,
        "raw_rankings": [
            sorted(candidates, key=lambda node: (-scores[node], node))
            for scores in perspectives
        ],
        "average_ranks": averages,
    }


def validate_classification(
    value: dict[str, Any], aliases: set[str]
) -> dict[str, Any]:
    items = value.get("fault_types")
    if not isinstance(items, list) or len(items) != 3:
        raise ValidationError("fault_types must contain exactly three entries")
    seen = set()
    result = []
    for item in items:
        alias = item.get("fault_type_id")
        if alias not in aliases or alias in seen:
            raise ValidationError("illegal or duplicate fault_type_id")
        seen.add(alias)
        rendered = {"fault_type_id": alias}
        for name in CLASS_COMPONENTS:
            rendered[name] = _score(item.get(name), name)
        ids = item.get("evidence_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValidationError("evidence_ids must be a string list")
        rendered["evidence_ids"] = ids[:8]
        if not isinstance(item.get("reason"), str):
            raise ValidationError("classification reason must be a string")
        rendered["reason"] = item["reason"]
        result.append(rendered)
    return {"fault_types": result}


def score_classification(
    items: list[dict[str, Any]], alias_to_taxonomy: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    raw = {}
    by_type = {}
    for item in items:
        taxonomy = alias_to_taxonomy[item["fault_type_id"]]
        fault_type = taxonomy["fault_type"]
        raw[fault_type] = max(
            0.0,
            sum(item[name] for name in CLASS_COMPONENTS[:-1]) / 5
            - 0.2 * item["counter_evidence_penalty"],
        )
        by_type[fault_type] = (item, taxonomy)
    if not any(raw.values()):
        raise ValidationError("classification produced no positive score")
    normalized = normalize_scores(raw)
    ranked = sorted(raw, key=lambda key: (-raw[key], key))
    return [
        {
            "rank": rank,
            "fault_type": fault_type,
            "fault_category": by_type[fault_type][1]["fault_category"],
            "confidence": normalized[fault_type],
            "score_kind": "normalized_model_score",
            "reason_summary": by_type[fault_type][0]["reason"],
            "component_scores": {
                name: by_type[fault_type][0][name] for name in CLASS_COMPONENTS
            },
            "evidence_ids": by_type[fault_type][0]["evidence_ids"],
        }
        for rank, fault_type in enumerate(ranked, 1)
    ]
