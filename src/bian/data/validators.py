"""Input and score validation."""

from __future__ import annotations

import math

from .schema import IncidentRecord


class ValidationError(ValueError):
    """Raised when pipeline input violates its contract."""


def validate_incident(incident: IncidentRecord) -> None:
    if not incident.incident_id:
        raise ValidationError("incident_id is required")
    if not incident.candidate_devices:
        raise ValidationError("candidate_devices must not be empty")
    if len(set(incident.candidate_devices)) != len(incident.candidate_devices):
        raise ValidationError("candidate_devices must be unique")
    nodes = set(incident.topology.nodes)
    missing = set(incident.candidate_devices) - nodes
    if missing:
        raise ValidationError(f"candidate devices absent from topology: {sorted(missing)}")
    for left, right in incident.topology.edges:
        if left not in nodes or right not in nodes:
            raise ValidationError(f"topology edge has unknown endpoint: {(left, right)}")
    candidates = set(incident.candidate_devices)
    for alert in incident.alerts:
        if not alert.alert_id or not alert.source_type or not alert.device_ids:
            raise ValidationError("alert_id, source_type, and device_ids are required")
        if alert.end_time is not None and alert.end_time < alert.start_time:
            raise ValidationError(f"alert {alert.alert_id} ends before it starts")
        if not set(alert.device_ids) & candidates:
            raise ValidationError(f"alert {alert.alert_id} is unrelated to candidates")
    if incident.ground_truth is not None and not set(incident.ground_truth) <= candidates:
        raise ValidationError("ground truth must be within candidate devices")


def validate_scores(scores: dict[str, float], candidates: tuple[str, ...]) -> None:
    if set(scores) != set(candidates):
        raise ValidationError("score devices must exactly match candidates")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValidationError("failure scores must be finite")
    if any(value < 0 for value in scores.values()):
        raise ValidationError("failure scores must be non-negative")
    if not math.isclose(sum(scores.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError("failure scores must sum to 1")


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        raise ValidationError("scores must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise ValidationError("raw scores must be finite and non-negative")
    total = sum(scores.values())
    if total <= 0:
        raise ValidationError("at least one raw score must be positive")
    return {device: value / total for device, value in scores.items()}
