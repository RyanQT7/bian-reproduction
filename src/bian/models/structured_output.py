"""Strict JSON extraction and validation for local language-model calls."""

from __future__ import annotations

import json
import math
from typing import Any, Callable

from bian.data.validators import ValidationError, normalize_scores


def strip_thinking(text: str) -> tuple[str, str]:
    """Separate complete DeepSeek-style thinking blocks without guessing JSON."""
    remaining = text.strip()
    thoughts: list[str] = []
    while "<think>" in remaining:
        start = remaining.index("<think>")
        end = remaining.find("</think>", start + len("<think>"))
        if end < 0:
            raise ValidationError("unterminated <think> block")
        thoughts.append(remaining[start + len("<think>") : end].strip())
        remaining = (remaining[:start] + remaining[end + len("</think>") :]).strip()
    return "\n\n".join(thoughts), remaining


def parse_strict_json(text: str) -> tuple[dict[str, Any], str]:
    """Parse one complete JSON object after removing explicit wrappers."""
    thoughts, answer = strip_thinking(text)
    if answer.startswith("```json") and answer.endswith("```"):
        answer = answer[len("```json") : -len("```")].strip()
    elif answer.startswith("```") and answer.endswith("```"):
        answer = answer[len("```") : -len("```")].strip()
    try:
        value = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"model answer is not strict JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValidationError("model JSON answer must be an object")
    return value, thoughts


def _require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{path} keys differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def validate_device_analysis(
    value: dict[str, Any], expected_nodes: tuple[str, ...]
) -> dict[str, Any]:
    _require_exact_keys(value, {"devices"}, "$")
    devices = value["devices"]
    if not isinstance(devices, list) or len(devices) != len(expected_nodes):
        raise ValidationError("devices must contain one item per requested node")
    by_node: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(devices):
        if not isinstance(item, dict):
            raise ValidationError(f"devices[{index}] must be an object")
        _require_exact_keys(
            item,
            {
                "node_id",
                "alert_summary",
                "is_anomalous",
                "anomaly_score",
                "anomaly_evidence",
                "uncertainty",
            },
            f"devices[{index}]",
        )
        node = item["node_id"]
        if node not in expected_nodes or node in by_node:
            raise ValidationError(f"illegal or duplicate device node {node!r}")
        if not isinstance(item["is_anomalous"], bool):
            raise ValidationError(f"{node}: is_anomalous must be boolean")
        score = item["anomaly_score"]
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValidationError(f"{node}: anomaly_score must be finite")
        if not 0 <= score <= 1:
            raise ValidationError(f"{node}: anomaly_score must be in [0, 1]")
        for field in ("alert_summary", "anomaly_evidence", "uncertainty"):
            if not isinstance(item[field], str):
                raise ValidationError(f"{node}: {field} must be a string")
        by_node[node] = item
    if set(by_node) != set(expected_nodes):
        raise ValidationError("device analysis node set differs from request")
    return {"devices": [by_node[node] for node in expected_nodes]}


def validate_stage1(
    value: dict[str, Any], expected_nodes: tuple[str, ...]
) -> dict[str, Any]:
    _require_exact_keys(value, {"scores"}, "$")
    if not isinstance(value["scores"], list):
        raise ValidationError("scores must be a list")
    raw: dict[str, float] = {}
    for index, item in enumerate(value["scores"]):
        if not isinstance(item, dict):
            raise ValidationError(f"scores[{index}] must be an object")
        _require_exact_keys(item, {"node_id", "score"}, f"scores[{index}]")
        node = item["node_id"]
        score = item["score"]
        if node not in expected_nodes or node in raw:
            raise ValidationError(f"illegal or duplicate Stage 1 node {node!r}")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValidationError(f"{node}: Stage 1 score must be finite and non-negative")
        raw[node] = float(score)
    if set(raw) != set(expected_nodes):
        raise ValidationError("Stage 1 scores must cover exactly all candidates")
    normalized = normalize_scores(raw)
    return {"scores": normalized}


def validate_stage2(
    value: dict[str, Any],
    allowed_nodes: tuple[str, ...],
    taxonomy: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    _require_exact_keys(value, {"root_causes", "fault_types", "analysis"}, "$")
    roots = value["root_causes"]
    faults = value["fault_types"]
    if not isinstance(roots, list) or len(roots) < 5:
        raise ValidationError("root_causes must contain at least five items")
    root_nodes: set[str] = set()
    clean_roots = []
    for index, item in enumerate(roots):
        if not isinstance(item, dict):
            raise ValidationError(f"root_causes[{index}] must be an object")
        _require_exact_keys(item, {"node_id", "score", "reason"}, f"root_causes[{index}]")
        node, score = item["node_id"], item["score"]
        if node not in allowed_nodes or node in root_nodes:
            raise ValidationError(f"illegal or duplicate root node {node!r}")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValidationError(f"{node}: root score must be finite and non-negative")
        if not isinstance(item["reason"], str):
            raise ValidationError(f"{node}: root reason must be a string")
        root_nodes.add(node)
        clean_roots.append({**item, "score": float(score)})
    taxonomy_map = {item["fault_type"]: item["fault_category"] for item in taxonomy}
    if not isinstance(faults, list) or len(faults) < 3:
        raise ValidationError("fault_types must contain at least three items")
    fault_names: set[str] = set()
    clean_faults = []
    for index, item in enumerate(faults):
        if not isinstance(item, dict):
            raise ValidationError(f"fault_types[{index}] must be an object")
        _require_exact_keys(
            item, {"fault_type", "fault_category", "confidence", "reason"},
            f"fault_types[{index}]",
        )
        fault_type = item["fault_type"]
        if fault_type not in taxonomy_map or fault_type in fault_names:
            raise ValidationError(f"illegal or duplicate fault type {fault_type!r}")
        if item["fault_category"] != taxonomy_map[fault_type]:
            raise ValidationError(f"category mismatch for {fault_type!r}")
        confidence = item["confidence"]
        if (
            not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or confidence < 0
        ):
            raise ValidationError(f"{fault_type}: confidence must be finite and non-negative")
        if not isinstance(item["reason"], str):
            raise ValidationError(f"{fault_type}: reason must be a string")
        fault_names.add(fault_type)
        clean_faults.append({**item, "confidence": float(confidence)})
    if not isinstance(value["analysis"], str):
        raise ValidationError("analysis must be a string")
    return {
        "root_causes": clean_roots,
        "fault_types": clean_faults,
        "analysis": value["analysis"],
    }


Validator = Callable[[dict[str, Any]], dict[str, Any]]
