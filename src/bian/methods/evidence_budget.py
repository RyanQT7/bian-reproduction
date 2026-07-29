"""Deterministic, role-neutral evidence budgeting for 32B RCA prompts."""

from __future__ import annotations

from typing import Any


EVIDENCE_FIELDS = (
    "evidence_id",
    "metric_name",
    "source_type",
    "first_change_time",
    "peak_time",
    "recovery_time",
    "pre_value",
    "fault_value",
    "post_value",
    "absolute_delta",
    "stable_change_score",
    "direction",
    "status_transition",
    "data_quality_status",
    "direct_fault_evidence",
)


def _number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _has_transition(item: dict[str, Any]) -> bool:
    value = item.get("status_transition")
    return value not in (None, "", "none", "stable", "unchanged", False)


def select_evidence(
    stage1: dict[str, Any],
    device_evidence: dict[str, Any],
    *,
    max_evidence: int,
) -> list[dict[str, Any]]:
    """Select the same bounded evidence profile for every shortlisted candidate.

    Stage-1 references are retained first, followed by direct control-plane or
    policy evidence, timestamped state transitions, and the strongest remaining
    stable changes. Role-inapplicable data is never promoted as anomaly evidence.
    """
    if max_evidence <= 0:
        raise ValueError("max_evidence must be positive")
    supporting = set(stage1.get("supporting_evidence_ids", []))
    counter = set(stage1.get("counter_evidence_ids", []))
    ranked = []
    for item in device_evidence.get("evidence", []):
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str):
            continue
        quality = item.get("data_quality_status")
        unavailable = quality == "unavailable_by_role"
        if evidence_id in supporting:
            tier = 0
        elif evidence_id in counter:
            tier = 1
        elif item.get("direct_fault_evidence") and not unavailable:
            tier = 2
        elif _has_transition(item) and not unavailable:
            tier = 3
        elif item.get("first_change_time") and not unavailable:
            tier = 4
        else:
            tier = 5
        ranked.append(
            (
                tier,
                -int(bool(item.get("direct_fault_evidence")) and not unavailable),
                -int(bool(item.get("first_change_time"))),
                -_number(item.get("stable_change_score")),
                evidence_id,
                item,
            )
        )
    selected = [item[-1] for item in sorted(ranked)[:max_evidence]]
    return [
        {field: item.get(field) for field in EVIDENCE_FIELDS}
        for item in selected
    ]
