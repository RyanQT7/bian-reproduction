"""Standard-library data contracts for pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str
    device_ids: tuple[str, ...]
    source_type: str
    start_time: datetime
    end_time: datetime | None = None
    severity: int = 0
    text: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyRecord:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    device_groups: dict[str, str] = field(default_factory=dict)
    directed: bool = False


@dataclass(frozen=True)
class IncidentRecord:
    incident_id: str
    start_time: datetime
    candidate_devices: tuple[str, ...]
    alerts: tuple[AlertRecord, ...]
    topology: TopologyRecord
    ground_truth: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
