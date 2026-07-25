"""Deterministic stand-in for LLM agents."""

from __future__ import annotations

import hashlib

from bian.data.schema import IncidentRecord
from bian.data.validators import normalize_scores


class MockModelBackend:
    """Produces reproducible scores without loading any model."""

    def __init__(self, seed: int):
        self.seed = seed

    def _jitter(self, incident_id: str, device: str, round_index: int) -> float:
        value = f"{self.seed}:{incident_id}:{device}:{round_index}".encode()
        digest = hashlib.sha256(value).digest()
        return int.from_bytes(digest[:4], "big") / (2**32) * 0.02

    def score(
        self,
        incident: IncidentRecord,
        candidates: tuple[str, ...],
        round_index: int,
        include_context: bool,
    ) -> dict[str, float]:
        first_time = min(alert.start_time for alert in incident.alerts)
        raw = {}
        for device in candidates:
            related = [alert for alert in incident.alerts if device in alert.device_ids]
            alert_evidence = sum(1.0 + alert.severity for alert in related)
            chronology = sum(
                1.0 / (1.0 + (alert.start_time - first_time).total_seconds())
                for alert in related
            )
            context_bonus = 0.15 * len(related) if include_context else 0.0
            raw[device] = alert_evidence + chronology + context_bonus
            raw[device] += self._jitter(incident.incident_id, device, round_index)
        return normalize_scores(raw)
