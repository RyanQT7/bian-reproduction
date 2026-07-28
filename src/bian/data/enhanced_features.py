"""Deterministic, role-aware evidence extraction for engineering-enhanced RCA."""

from __future__ import annotations

import hashlib
import math
from typing import Any


DIRECT_TERMS = {
    "br": ("bgp", "route", "prefix", "nexthop", "blackhole", "default_route"),
    "cr": ("ospf", "route", "cost", "neighbor", "static_route"),
    "fw": ("acl", "rule", "port", "default_route", "drop", "cpu", "route"),
    "traffic-vm": ("connection", "flow"),
    "service": ("success", "error", "latency", "throughput"),
}
UPSTREAM_ROLES = {"br", "cr", "fw"}
SYMPTOM_ROLES = {"service", "traffic-vm"}


def source_semantic_status(source: dict[str, Any]) -> str:
    status = source["status"]
    if status == "unavailable_by_role":
        return status
    phases = source.get("phases", {})
    pre_rows = phases.get("pre_fault", {}).get("row_count", 0)
    fault_rows = phases.get("fault", {}).get("row_count", 0)
    post_rows = phases.get("post_fault", {}).get("row_count", 0)
    if pre_rows > 0 and fault_rows == 0:
        return "sudden_missing_during_fault"
    if status == "empty" and source.get("applicable", True):
        return "empty_but_expected"
    if status in {"partial", "collection_failed"}:
        return status
    if pre_rows == fault_rows == post_rows == 0 and source.get("applicable", True):
        return "empty_but_expected"
    return status


def stable_change(pre: float, fault: float, minimum_baseline: float = 1e-3) -> float:
    """Bounded change score that cannot create epsilon-division artifacts."""
    absolute = abs(fault - pre)
    scale = max(abs(pre), abs(fault), minimum_baseline)
    symmetric = absolute / scale
    log_delta = abs(math.log1p(abs(fault)) - math.log1p(abs(pre)))
    return min(1.0, 0.65 * symmetric + 0.35 * min(1.0, log_delta))


def _evidence_id(node: str, source: str, metric: str) -> str:
    suffix = hashlib.sha256(f"{node}|{source}|{metric}".encode()).hexdigest()[:10]
    return f"E-{suffix}"


def extract_device_evidence(device: dict[str, Any]) -> dict[str, Any]:
    role = device["device_family"]
    evidence = []
    source_states = {}
    unavailable_contribution = 0.0
    data_quality_adjustment = 0.0
    for source_name, source in sorted(device["sources"].items()):
        semantic = source_semantic_status(source)
        source_states[source_name] = semantic
        if semantic == "unavailable_by_role":
            continue
        if semantic == "collection_failed":
            data_quality_adjustment -= 0.10
        elif semantic == "partial":
            data_quality_adjustment -= 0.04
        elif semantic == "empty_but_expected":
            data_quality_adjustment -= 0.02
        elif semantic == "sudden_missing_during_fault":
            data_quality_adjustment -= 0.03
        phases = source.get("phases", {})
        pre_phase = phases.get("pre_fault", {})
        fault_phase = phases.get("fault", {})
        post_phase = phases.get("post_fault", {})
        metrics = sorted(
            set(pre_phase.get("metrics", {}))
            | set(fault_phase.get("metrics", {}))
            | set(post_phase.get("metrics", {}))
        )
        for metric in metrics:
            pre_item = pre_phase.get("metrics", {}).get(metric, {})
            fault_item = fault_phase.get("metrics", {}).get(metric, {})
            post_item = post_phase.get("metrics", {}).get(metric, {})
            pre = pre_item.get("mean")
            fault = fault_item.get("mean")
            post = post_item.get("mean")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in (pre, fault)
            ):
                continue
            pre_f, fault_f = float(pre), float(fault)
            post_f = (
                float(post)
                if isinstance(post, (int, float)) and math.isfinite(post)
                else None
            )
            change = stable_change(pre_f, fault_f)
            direct = any(term in metric.lower() for term in DIRECT_TERMS[role])
            transition = (
                "zero_to_nonzero"
                if pre_f == 0 and fault_f != 0
                else "nonzero_to_zero"
                if pre_f != 0 and fault_f == 0
                else "changed"
                if fault_f != pre_f
                else "stable"
            )
            recovered = (
                post_f is not None
                and abs(post_f - pre_f) <= max(1e-6, 0.1 * max(abs(pre_f), 1e-3))
            )
            evidence.append(
                {
                    "evidence_id": _evidence_id(device["node_id"], source_name, metric),
                    "node_id": device["node_id"],
                    "metric_name": metric,
                    "source_type": source_name,
                    "device_role": role,
                    "observed_at": fault_phase.get("first_timestamp_utc"),
                    "first_change_time": fault_phase.get("first_timestamp_utc"),
                    "peak_time": fault_phase.get("last_timestamp_utc"),
                    "recovery_time": (
                        post_phase.get("first_timestamp_utc") if recovered else None
                    ),
                    "pre_value": pre_f,
                    "fault_value": fault_f,
                    "post_value": post_f,
                    "fault_peak": fault_item.get("max"),
                    "fault_min": fault_item.get("min"),
                    "absolute_delta": abs(fault_f - pre_f),
                    "stable_change_score": change,
                    "direction": (
                        "increase"
                        if fault_f > pre_f
                        else "decrease"
                        if fault_f < pre_f
                        else "stable"
                    ),
                    "status_transition": transition,
                    "recovered": recovered,
                    "data_quality_status": semantic,
                    "direct_fault_evidence": direct,
                }
            )
    evidence.sort(
        key=lambda item: (
            -int(item["direct_fault_evidence"]),
            -item["stable_change_score"],
            item["source_type"],
            item["metric_name"],
        )
    )
    direct_scores = [
        item["stable_change_score"] for item in evidence if item["direct_fault_evidence"]
    ]
    general_scores = [item["stable_change_score"] for item in evidence]
    temporal = max(general_scores, default=0.0)
    direct_score = max(direct_scores, default=0.0)
    deterministic = (
        0.65 * direct_score + 0.35 * temporal
        if role in UPSTREAM_ROLES
        else 0.25 * direct_score + 0.35 * temporal
    )
    symptom_likelihood = (
        min(1.0, 0.35 + 0.65 * temporal) if role in SYMPTOM_ROLES else 0.05
    )
    return {
        "node_id": device["node_id"],
        "device_role": role,
        "source_semantics": source_states,
        "evidence": evidence,
        "feature_summary": {
            "deterministic_feature_score": min(1.0, deterministic),
            "temporal_change_score": temporal,
            "direct_fault_evidence_score": direct_score,
            "symptom_likelihood": symptom_likelihood,
            "data_quality_adjustment": max(-0.25, data_quality_adjustment),
            "unavailable_by_role_anomaly_contribution": unavailable_contribution,
            "earliest_change_time": min(
                (
                    item["first_change_time"]
                    for item in evidence
                    if item["stable_change_score"] > 0.15
                    and item["first_change_time"] is not None
                ),
                default=None,
            ),
        },
    }


def validate_role_schema(devices: list[dict[str, Any]]) -> dict[str, list[str]]:
    schemas: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    fields_by_role: dict[str, list[str]] = {}
    for device in devices:
        role = device["device_family"]
        signature = tuple(
            (name, tuple(sorted(source["phases"]["fault"]["metrics"])))
            for name, source in sorted(device["sources"].items())
        )
        schemas.setdefault(role, set()).add(signature)
        fields_by_role.setdefault(
            role,
            [
                f"{source}.{metric}"
                for source, metrics in signature
                for metric in metrics
            ],
        )
    inconsistent = {role: len(values) for role, values in schemas.items() if len(values) > 1}
    if inconsistent:
        raise ValueError(f"inconsistent within-role schemas: {inconsistent}")
    return fields_by_role
