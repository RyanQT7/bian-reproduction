"""Alert-count Hot Device baseline."""

from __future__ import annotations

from bian.data.schema import AlertRecord


def hot_device_ranking(
    alerts: tuple[AlertRecord, ...], candidates: tuple[str, ...]
) -> tuple[str, ...]:
    counts = {device: 0 for device in candidates}
    for alert in alerts:
        for device in alert.device_ids:
            if device in counts:
                counts[device] += 1
    return tuple(sorted(candidates, key=lambda device: (-counts[device], device)))
