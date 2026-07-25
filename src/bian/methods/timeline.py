"""Strict global event timeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from bian.data.schema import AlertRecord


@dataclass(frozen=True)
class TimelineEvent:
    timestamp: datetime
    alert_id: str
    device_id: str
    source_type: str


def build_timeline(alerts: tuple[AlertRecord, ...]) -> tuple[TimelineEvent, ...]:
    ordered = sorted(
        (
            TimelineEvent(alert.start_time, alert.alert_id, device, alert.source_type)
            for alert in alerts
            for device in alert.device_ids
        ),
        key=lambda event: (event.timestamp, event.alert_id, event.device_id),
    )
    result: list[TimelineEvent] = []
    previous: datetime | None = None
    for event in ordered:
        timestamp = event.timestamp
        if previous is not None and timestamp <= previous:
            timestamp = previous + timedelta(microseconds=1)
        result.append(TimelineEvent(timestamp, event.alert_id, event.device_id, event.source_type))
        previous = timestamp
    return tuple(result)
