"""Dynamic JSON Schemas for vLLM-constrained BiAn generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _score() -> dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def _evidence_ids(values: Iterable[str]) -> dict[str, Any]:
    allowed = sorted(set(values))
    items: dict[str, Any] = {"type": "string"}
    if allowed:
        items["enum"] = allowed
    return {
        "type": "array",
        "items": items,
        "maxItems": min(10, len(allowed)),
    }


def stage2_schema(
    candidate_ids: Iterable[str], evidence_ids: Iterable[str]
) -> dict[str, Any]:
    candidates = sorted(set(candidate_ids))
    if not candidates:
        raise ValueError("Stage 2 schema requires candidates")
    item_properties = {
        "candidate_id": {"type": "string", "enum": candidates},
        "local_anomaly_score": _score(),
        "temporal_precedence_score": _score(),
        "topology_upstream_score": _score(),
        "fault_pattern_compatibility_score": _score(),
        "symptom_likelihood": _score(),
        "supporting_evidence_ids": _evidence_ids(evidence_ids),
        "counter_evidence_ids": _evidence_ids(evidence_ids),
        "concise_reason": {"type": "string", "maxLength": 400},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": len(candidates),
                "maxItems": len(candidates),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(item_properties),
                    "properties": item_properties,
                },
            }
        },
    }


def classification_schema(
    type_id: str,
    candidate_ids: Iterable[str],
    evidence_ids: Iterable[str],
) -> dict[str, Any]:
    candidates = sorted(set(candidate_ids))
    if not candidates:
        raise ValueError("classification schema requires root hypotheses")
    item_properties = {
        "candidate_id": {"type": "string", "enum": candidates},
        "root_role_compatibility": _score(),
        "metric_pattern_compatibility": _score(),
        "protocol_state_compatibility": _score(),
        "temporal_pattern_compatibility": _score(),
        "topology_context_compatibility": _score(),
        "counter_evidence_penalty": _score(),
        "supporting_evidence_ids": _evidence_ids(evidence_ids),
        "counter_evidence_ids": _evidence_ids(evidence_ids),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type_id", "root_hypotheses"],
        "properties": {
            "type_id": {"type": "string", "const": type_id},
            "root_hypotheses": {
                "type": "array",
                "minItems": len(candidates),
                "maxItems": len(candidates),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(item_properties),
                    "properties": item_properties,
                },
            },
        },
    }
