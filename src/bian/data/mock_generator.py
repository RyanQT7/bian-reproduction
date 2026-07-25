"""Deterministic synthetic incidents for pipeline validation only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random

from .schema import AlertRecord, IncidentRecord, TopologyRecord


def generate_mock_incidents(
    seed: int, incident_count: int = 3, candidate_count: int = 4
) -> tuple[IncidentRecord, ...]:
    if incident_count <= 0 or candidate_count < 2:
        raise ValueError("incident_count must be positive and candidate_count at least 2")
    rng = random.Random(seed)
    base = datetime(2026, 7, 25, 0, 0, tzinfo=timezone.utc)
    incidents = []
    source_types = ("dashboard_alarm", "ping_log", "change_history")
    for incident_index in range(incident_count):
        candidates = tuple(f"I{incident_index + 1}-D{index + 1}" for index in range(candidate_count))
        transit = f"I{incident_index + 1}-X"
        nodes = candidates + (transit,)
        edges = tuple((candidates[index], candidates[index + 1]) for index in range(candidate_count - 1))
        edges += ((candidates[0], transit),)
        root_index = rng.randrange(candidate_count)
        start = base + timedelta(hours=incident_index)
        alerts = []
        sequence = 0
        for device_index, device in enumerate(candidates):
            count = 3 if device_index == root_index else 1 + rng.randrange(2)
            for item_index in range(count):
                sequence += 1
                offset = item_index if device_index == root_index else 5 + item_index + device_index
                alerts.append(
                    AlertRecord(
                        alert_id=f"I{incident_index + 1}-A{sequence}",
                        device_ids=(device,),
                        source_type=source_types[(device_index + item_index) % len(source_types)],
                        start_time=start + timedelta(seconds=offset),
                        severity=3 if device_index == root_index else 1,
                        text="Synthetic monitoring event for pipeline validation.",
                        attributes={"synthetic": True},
                    )
                )
        incidents.append(
            IncidentRecord(
                incident_id=f"MOCK-{incident_index + 1:03d}",
                start_time=start,
                candidate_devices=candidates,
                alerts=tuple(alerts),
                topology=TopologyRecord(nodes, edges),
                ground_truth=(candidates[root_index],),
                metadata={
                    "result_label": "流程验证结果 / Pipeline Validation Result",
                    "synthetic": True,
                },
            )
        )
    return tuple(incidents)
