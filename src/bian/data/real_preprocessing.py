"""Leakage-safe, deterministic preprocessing for known incident windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .validators import ValidationError


ROLE_ALIASES = {
    "br-1": "br-1",
    "br-2": "br-2",
    "cr-1": "cr-1",
    "cr-2": "cr-2",
    "fw": "fw",
    "traffic-vm": "traffic-vm",
    "service-vm-1": "service-1",
    "service-vm-2": "service-2",
    "service-vm-3": "service-3",
}
ROLE_FAMILIES = {
    "br-1": "br",
    "br-2": "br",
    "cr-1": "cr",
    "cr-2": "cr",
    "fw": "fw",
    "traffic-vm": "traffic-vm",
    "service-1": "service",
    "service-2": "service",
    "service-3": "service",
}
NEIGHBOR_ID_RE = re.compile(r'neighbor_id="([0-9.]+)"')


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValidationError("UTC timestamp is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"invalid timestamp: {value!r}") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    if result.utcoffset() != timezone.utc.utcoffset(result):
        result = result.astimezone(timezone.utc)
    return result


def phase_for(timestamp: datetime, incident: dict[str, Any]) -> str | None:
    slice_start = parse_utc(incident["slice_start_time_utc"])
    fault_start = parse_utc(incident["fault_start_time_utc"])
    fault_end = parse_utc(incident["fault_end_time_utc"])
    slice_end = parse_utc(incident["slice_end_time_utc"])
    if timestamp < slice_start or timestamp > slice_end:
        return None
    if timestamp < fault_start:
        return "pre_fault"
    if timestamp <= fault_end:
        return "fault"
    return "post_fault"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def validate_experiment_incidents(records: list[dict[str, Any]]) -> None:
    if len(records) != 10:
        raise ValidationError(f"expected exactly 10 experiment incidents, got {len(records)}")
    ids = [record.get("incident_id") for record in records]
    if len(set(ids)) != len(ids):
        raise ValidationError("experiment incident IDs must be unique")
    for record in records:
        if record.get("dataset_timezone") != "UTC":
            raise ValidationError("all experiment incidents must use UTC")
        if record.get("ground_truth_included") is not False:
            raise ValidationError("experiment incident unexpectedly includes ground truth")
        expected_start = parse_utc(record["fault_start_time_utc"])
        expected_end = parse_utc(record["fault_end_time_utc"])
        slice_start = parse_utc(record["slice_start_time_utc"])
        slice_end = parse_utc(record["slice_end_time_utc"])
        if (expected_start - slice_start).total_seconds() != 300:
            raise ValidationError("pre-fault interval must be exactly 300 seconds")
        if (slice_end - expected_end).total_seconds() != 300:
            raise ValidationError("post-fault interval must be exactly 300 seconds")


def infer_region_mapping(raw_root: Path, topology: dict[str, Any]) -> dict[Path, str]:
    router_prefix_to_region: dict[str, str] = {}
    for node in topology["nodes"]:
        router_id = node.get("router_id")
        if node.get("candidate") and router_id:
            prefix = router_id.split(".", 1)[0]
            current = router_prefix_to_region.setdefault(prefix, node["region_id"])
            if current != node["region_id"]:
                raise ValidationError("router ID prefix is not unique by region")
    result: dict[Path, str] = {}
    for region_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        files = sorted((region_dir / "processed").glob("routing_metrics_*.csv"))
        if len(files) != 1:
            raise ValidationError(f"expected one routing metrics file under {region_dir}")
        matched: set[str] = set()
        with files[0].open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                match = NEIGHBOR_ID_RE.search(row.get("label", ""))
                if match:
                    prefix = match.group(1).split(".", 1)[0]
                    if prefix in router_prefix_to_region:
                        matched.add(router_prefix_to_region[prefix])
                if len(matched) > 1:
                    raise ValidationError(f"ambiguous region mapping for {region_dir}")
        if len(matched) != 1:
            raise ValidationError(f"could not infer region mapping for {region_dir}")
        result[region_dir] = matched.pop()
    if len(result) != 8 or set(result.values()) != {f"region-{index}" for index in range(1, 9)}:
        raise ValidationError("raw region mapping must be a complete 8-region bijection")
    return result


def _finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class NumericAccumulator:
    count: int = 0
    missing_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    first: float | None = None
    last: float | None = None

    def add(self, value: str) -> None:
        number = _finite_float(value)
        if number is None:
            self.missing_count += 1
            return
        if self.first is None:
            self.first = number
        self.last = number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        self.total += number
        self.count += 1

    def render(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "missing_count": self.missing_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count if self.count else None,
            "first": self.first,
            "last": self.last,
            "delta": (
                self.last - self.first
                if self.first is not None and self.last is not None
                else None
            ),
        }


@dataclass
class SourceAccumulator:
    row_count: int = 0
    metrics: dict[str, NumericAccumulator] = field(default_factory=dict)
    dimensions: dict[str, set[str]] = field(default_factory=dict)
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    def add(
        self,
        row: dict[str, str],
        timestamp: datetime,
        numeric_fields: Iterable[str],
        dimensions: dict[str, str],
    ) -> None:
        self.row_count += 1
        rendered_time = timestamp.isoformat().replace("+00:00", "Z")
        if self.first_timestamp is None:
            self.first_timestamp = rendered_time
        self.last_timestamp = rendered_time
        for name in numeric_fields:
            self.metrics.setdefault(name, NumericAccumulator()).add(row.get(name, ""))
        for name, value in dimensions.items():
            if value:
                self.dimensions.setdefault(name, set()).add(value)

    def render(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "first_timestamp_utc": self.first_timestamp,
            "last_timestamp_utc": self.last_timestamp,
            "metrics": {key: self.metrics[key].render() for key in sorted(self.metrics)},
            "dimensions": {
                key: sorted(values) for key, values in sorted(self.dimensions.items())
            },
        }


def assert_no_leakage(value: Any, forbidden_keys: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            if normalized in forbidden_keys:
                raise ValidationError(f"forbidden leakage key found: {key}")
            assert_no_leakage(child, forbidden_keys)
    elif isinstance(value, list):
        for child in value:
            assert_no_leakage(child, forbidden_keys)


def node_id_for_row(source: str, row: dict[str, str], region_id: str) -> str | None:
    if source == "traffic_flow_metrics":
        role = "traffic-vm"
    elif source == "frr_syslog_events":
        hostname = row.get("hostname", "")
        role = next((raw for raw in ROLE_ALIASES if hostname.startswith(raw + "-")), "")
    else:
        role = row.get("node", "")
    mapped = ROLE_ALIASES.get(role)
    return f"{region_id}-{mapped}" if mapped else None


def dimensions_for(source: str, row: dict[str, str]) -> dict[str, str]:
    if source == "interface_metrics":
        return {"interface_id": row.get("interface_id", ""), "if_role": row.get("if_role", "")}
    if source == "routing_metrics":
        return {"metric_name": row.get("metric_name", ""), "label": row.get("label", "")}
    if source == "scrape_health":
        return {"exporter_type": row.get("exporter_type", "")}
    if source == "traffic_flow_metrics":
        return {"flow_type": row.get("flow_type", ""), "protocol": row.get("protocol", "")}
    if source == "frr_syslog_events":
        return {
            "program": row.get("program", ""),
            "severity": row.get("severity", ""),
        }
    return {}
