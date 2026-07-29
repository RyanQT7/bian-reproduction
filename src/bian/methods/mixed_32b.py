"""Strict schemas and fixed aggregation for 32B feasibility validation."""

from __future__ import annotations

import math
from typing import Any

from bian.data.validators import ValidationError, normalize_scores
from bian.methods.rank_of_ranks import aggregate_rankings


STAGE2_COMPONENTS = (
    "local_anomaly_score",
    "temporal_precedence_score",
    "topology_upstream_score",
    "fault_pattern_compatibility_score",
    "symptom_likelihood",
)
TYPE_COMPONENTS = (
    "root_role_compatibility",
    "metric_pattern_compatibility",
    "protocol_state_compatibility",
    "temporal_pattern_compatibility",
    "topology_context_compatibility",
    "counter_evidence_penalty",
)


def bounded(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{field} must be finite")
    if not 0 <= value <= 1:
        raise ValidationError(f"{field} must be in [0,1]")
    return float(value)


def validate_stage2_round(
    value: dict[str, Any],
    aliases: set[str],
    evidence_ids: set[str],
) -> dict[str, Any]:
    items = value.get("candidates")
    if set(value) != {"candidates"} or not isinstance(items, list):
        raise ValidationError("Stage 2 output must contain only candidates")
    if len(items) != len(aliases):
        raise ValidationError("Stage 2 must score every candidate exactly once")
    seen = set()
    clean = []
    for item in items:
        expected = {
            "candidate_id",
            *STAGE2_COMPONENTS,
            "supporting_evidence_ids",
            "counter_evidence_ids",
            "concise_reason",
        }
        if not isinstance(item, dict) or set(item) != expected:
            raise ValidationError("Stage 2 candidate schema mismatch")
        alias = item["candidate_id"]
        if alias not in aliases or alias in seen:
            raise ValidationError("illegal or duplicate Cxx")
        seen.add(alias)
        rendered = {"candidate_id": alias}
        for field in STAGE2_COMPONENTS:
            rendered[field] = bounded(item[field], field)
        for field in ("supporting_evidence_ids", "counter_evidence_ids"):
            ids = item[field]
            if not isinstance(ids, list) or not all(
                isinstance(item_id, str) and item_id in evidence_ids for item_id in ids
            ):
                raise ValidationError(f"{field} contains unknown evidence ID")
            rendered[field] = ids[:10]
        if not isinstance(item["concise_reason"], str):
            raise ValidationError("concise_reason must be a string")
        rendered["concise_reason"] = item["concise_reason"]
        clean.append(rendered)
    if not any(
        item[field] > 0
        for item in clean
        for field in STAGE2_COMPONENTS[:-1]
    ):
        raise ValidationError("all Stage 2 positive components are zero")
    return {"candidates": clean}


def score_stage2_round(
    items: list[dict[str, Any]],
    alias_to_node: dict[str, str],
    weights: dict[str, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    raw = {}
    audit = []
    for item in items:
        node = alias_to_node[item["candidate_id"]]
        value = max(
            0.0,
            weights["local_anomaly_score"] * item["local_anomaly_score"]
            + weights["temporal_precedence_score"]
            * item["temporal_precedence_score"]
            + weights["topology_upstream_score"] * item["topology_upstream_score"]
            + weights["fault_pattern_compatibility_score"]
            * item["fault_pattern_compatibility_score"]
            - weights["symptom_likelihood_penalty"] * item["symptom_likelihood"],
        )
        raw[node] = value
        audit.append({**item, "node_id": node, "raw_final_score": value})
    if not any(raw.values()):
        raise ValidationError("all Stage 2 final scores are zero")
    return normalize_scores(raw), audit


def aggregate_stage2_rounds(
    rounds: list[dict[str, float]], candidates: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rounds) < 2:
        raise ValidationError("at least two valid Stage 2 rounds are required")
    ranking, averages = aggregate_rankings(rounds, candidates)
    top5_raw = {node: 1 / averages[node] for node in ranking[:5]}
    weights = normalize_scores(top5_raw)
    top5 = [
        {
            "rank": index,
            "node_id": node,
            "failure_score": weights[node],
            "rank_weight": 1 / averages[node],
            "reason_summary": "32B three-round auditable causal aggregation",
        }
        for index, node in enumerate(ranking[:5], 1)
    ]
    return top5, {
        "rounds": len(rounds),
        "raw_rankings": [
            sorted(candidates, key=lambda node: (-scores[node], node))
            for scores in rounds
        ],
        "average_ranks": averages,
    }


def validate_type_result(
    value: dict[str, Any],
    type_id: str,
    root_aliases: set[str],
    evidence_ids: set[str],
) -> dict[str, Any]:
    if set(value) != {"type_id", "root_hypotheses"} or value["type_id"] != type_id:
        raise ValidationError("classification type_id mismatch")
    items = value["root_hypotheses"]
    if not isinstance(items, list) or len(items) != len(root_aliases):
        raise ValidationError("classification must score every Top5 root")
    seen = set()
    clean = []
    for item in items:
        expected = {
            "candidate_id",
            *TYPE_COMPONENTS,
            "supporting_evidence_ids",
            "counter_evidence_ids",
        }
        if not isinstance(item, dict) or set(item) != expected:
            raise ValidationError("root hypothesis schema mismatch")
        alias = item["candidate_id"]
        if alias not in root_aliases or alias in seen:
            raise ValidationError("illegal or duplicate classification Cxx")
        seen.add(alias)
        rendered = {"candidate_id": alias}
        for field in TYPE_COMPONENTS:
            rendered[field] = bounded(item[field], field)
        for field in ("supporting_evidence_ids", "counter_evidence_ids"):
            ids = item[field]
            if not isinstance(ids, list) or not all(
                isinstance(item_id, str) and item_id in evidence_ids for item_id in ids
            ):
                raise ValidationError(f"{field} contains unknown evidence ID")
            rendered[field] = ids[:10]
        clean.append(rendered)
    return {"type_id": type_id, "root_hypotheses": clean}


def aggregate_type_result(
    result: dict[str, Any], root_weights: dict[str, float]
) -> tuple[float, list[dict[str, Any]]]:
    contributions = []
    total = 0.0
    for item in result["root_hypotheses"]:
        compatibility = max(
            0.0,
            sum(item[field] for field in TYPE_COMPONENTS[:-1]) / 5
            - 0.2 * item["counter_evidence_penalty"],
        )
        weighted = root_weights[item["candidate_id"]] * compatibility
        total += weighted
        contributions.append(
            {
                **item,
                "type_score_given_root": compatibility,
                "root_weight": root_weights[item["candidate_id"]],
                "weighted_contribution": weighted,
            }
        )
    return total, contributions
