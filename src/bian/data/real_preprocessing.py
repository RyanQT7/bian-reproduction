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

    def render(
        self, expected_metrics: Iterable[str] = (), max_dimension_values: int = 64
    ) -> dict[str, Any]:
        metric_names = sorted(set(expected_metrics) | set(self.metrics))
        return {
            "row_count": self.row_count,
            "first_timestamp_utc": self.first_timestamp,
            "last_timestamp_utc": self.last_timestamp,
            "metrics": {
                key: self.metrics.get(key, NumericAccumulator()).render()
                for key in metric_names
            },
            "dimensions": {
                key: {
                    "values": sorted(values)[:max_dimension_values],
                    "unique_count": len(values),
                    "truncated": len(values) > max_dimension_values,
                }
                for key, values in sorted(self.dimensions.items())
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


def count_csv_records(path: Path) -> int:
    if path.stat().st_size == 0:
        return 0
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    line_count = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    return max(0, line_count - 1)


def _empty_phase(expected_metrics: Iterable[str]) -> dict[str, Any]:
    return SourceAccumulator().render(expected_metrics)


def preprocess_dataset(
    *,
    raw_root: Path,
    bundle_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    experiment_path = bundle_root / "experiment" / "incidents.jsonl"
    topology_path = bundle_root / "topology" / "full_device_topology.json"
    inventory_path = bundle_root / "topology" / "candidate_inventory.json"
    taxonomy_path = bundle_root / "schemas" / "fault_taxonomy.json"
    incidents = load_jsonl(experiment_path)
    validate_experiment_incidents(incidents)
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    if len(inventory) != 72:
        raise ValidationError(f"expected 72 candidate nodes, got {len(inventory)}")
    candidate_ids = tuple(sorted(item["node_id"] for item in inventory))
    if any(not item.get("candidate") for item in inventory):
        raise ValidationError("candidate inventory contains a non-candidate node")
    region_mapping = infer_region_mapping(raw_root, topology)
    source_files: dict[str, str] = config["source_files"]
    time_fields: dict[str, str] = config["time_fields"]
    numeric_fields: dict[str, list[str]] = config["numeric_fields"]
    applicability: dict[str, list[str]] = config["source_applicability"]
    phases = tuple(config["aggregation"]["phases"])
    max_dimension_values = int(config["aggregation"].get("max_dimension_values", 64))
    forbidden_keys = {item.lower() for item in config["leakage_forbidden_keys"]}

    accumulators: dict[
        str, dict[str, dict[str, dict[str, SourceAccumulator]]]
    ] = {incident["incident_id"]: {} for incident in incidents}
    registry: dict[str, dict[str, set[str]]] = {
        family: {source: set(fields) for source, fields in numeric_fields.items()}
        for family in {"br", "cr", "fw", "traffic-vm", "service"}
    }
    source_stats: dict[str, dict[str, int]] = {
        source: {
            "input_files": 0,
            "input_bytes": 0,
            "input_rows": 0,
            "selected_rows": 0,
        }
        for source in source_files
    }
    skipped_asset_rows = 0
    parse_error_rows = 0
    file_coverage: dict[tuple[str, str], list[datetime | None]] = {}

    for region_dir, region_id in sorted(region_mapping.items(), key=lambda item: item[1]):
        processed_dir = region_dir / "processed"
        for source, pattern in source_files.items():
            paths = sorted(processed_dir.glob(pattern))
            if len(paths) != 1:
                raise ValidationError(
                    f"expected exactly one {source} file under {processed_dir}, got {len(paths)}"
                )
            path = paths[0]
            stats = source_stats[source]
            stats["input_files"] += 1
            stats["input_bytes"] += path.stat().st_size
            if path.stat().st_size == 0:
                continue
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or time_fields[source] not in reader.fieldnames:
                    raise ValidationError(f"{path} lacks configured time field")
                for row in reader:
                    stats["input_rows"] += 1
                    timestamp_text = row.get(time_fields[source], "")
                    if source == "frr_syslog_events" and not timestamp_text:
                        timestamp_text = row.get("received_at", "")
                    try:
                        timestamp = parse_utc(timestamp_text)
                    except ValidationError:
                        parse_error_rows += 1
                        continue
                    coverage = file_coverage.setdefault(
                        (region_id, source), [None, None]
                    )
                    coverage[0] = (
                        timestamp
                        if coverage[0] is None
                        else min(coverage[0], timestamp)
                    )
                    coverage[1] = (
                        timestamp
                        if coverage[1] is None
                        else max(coverage[1], timestamp)
                    )
                    node_id = node_id_for_row(source, row, region_id)
                    if node_id is None:
                        skipped_asset_rows += 1
                        continue
                    if node_id not in candidate_ids:
                        raise ValidationError(f"mapped node is outside candidate inventory: {node_id}")
                    role = node_id.split("-", 2)[2]
                    family = ROLE_FAMILIES[role]
                    if family not in applicability[source]:
                        raise ValidationError(
                            f"source {source} unexpectedly mapped to inapplicable role {role}"
                        )
                    row_numeric_fields = numeric_fields[source]
                    numeric_row = row
                    if source == "routing_metrics":
                        metric_name = row.get("metric_name", "")
                        if not metric_name:
                            parse_error_rows += 1
                            continue
                        numeric_row = {metric_name: row.get("value", "")}
                        row_numeric_fields = [metric_name]
                        registry[family][source].add(metric_name)
                    for incident in incidents:
                        phase = phase_for(timestamp, incident)
                        if phase is None:
                            continue
                        accumulator = (
                            accumulators[incident["incident_id"]]
                            .setdefault(node_id, {})
                            .setdefault(source, {})
                            .setdefault(phase, SourceAccumulator())
                        )
                        accumulator.add(
                            numeric_row,
                            timestamp,
                            row_numeric_fields,
                            dimensions_for(source, row),
                        )
                        stats["selected_rows"] += 1

    excluded_stats: dict[str, dict[str, int | str]] = {}
    for source, rule in config.get("excluded_sources", {}).items():
        total_files = 0
        total_bytes = 0
        total_rows = 0
        for region_dir in region_mapping:
            paths = sorted((region_dir / "processed").glob(f"{source}.csv"))
            if len(paths) != 1:
                raise ValidationError(f"expected one excluded source file for {source}")
            total_files += 1
            total_bytes += paths[0].stat().st_size
            total_rows += count_csv_records(paths[0])
        excluded_stats[source] = {
            "action": rule["action"],
            "reason": rule["reason"],
            "input_files": total_files,
            "input_bytes": total_bytes,
            "input_rows": total_rows,
            "output_bytes": 0,
            "output_rows": 0,
        }

    output_root.mkdir(parents=True, exist_ok=False)
    schema_root = output_root / "schema_registry"
    schema_root.mkdir()
    for family in sorted(registry):
        rendered_registry = {
            "device_family": family,
            "sources": {
                source: {
                    "applicable": family in applicability[source],
                    "metric_fields": sorted(registry[family][source]),
                    "phases": list(phases),
                }
                for source in sorted(source_files)
            },
        }
        (schema_root / f"{family.replace('-', '_')}.json").write_text(
            json.dumps(rendered_registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    coverage_failures: list[dict[str, str]] = []
    periodic_sources = set(config["periodic_sources"])
    for incident in incidents:
        slice_start = parse_utc(incident["slice_start_time_utc"])
        slice_end = parse_utc(incident["slice_end_time_utc"])
        for region_id in sorted(region_mapping.values()):
            for source in sorted(periodic_sources):
                coverage = file_coverage.get((region_id, source), [None, None])
                if (
                    coverage[0] is None
                    or coverage[1] is None
                    or coverage[0] > slice_start
                    or coverage[1] < slice_end
                ):
                    coverage_failures.append(
                        {
                            "incident_id": incident["incident_id"],
                            "region_id": region_id,
                            "source": source,
                            "required_start_utc": incident["slice_start_time_utc"],
                            "required_end_utc": incident["slice_end_time_utc"],
                            "available_start_utc": (
                                coverage[0].isoformat().replace("+00:00", "Z")
                                if coverage[0]
                                else ""
                            ),
                            "available_end_utc": (
                                coverage[1].isoformat().replace("+00:00", "Z")
                                if coverage[1]
                                else ""
                            ),
                        }
                    )
    incident_summaries = []
    for incident in incidents:
        incident_id = incident["incident_id"]
        incident_dir = output_root / incident_id
        incident_dir.mkdir()
        devices = []
        for item in sorted(inventory, key=lambda candidate: candidate["node_id"]):
            node_id = item["node_id"]
            role = item["role"]
            family = ROLE_FAMILIES[role]
            sources = {}
            for source in sorted(source_files):
                applicable = family in applicability[source]
                phase_records = {}
                total_rows = 0
                for phase in phases:
                    accumulator = (
                        accumulators[incident_id]
                        .get(node_id, {})
                        .get(source, {})
                        .get(phase)
                    )
                    rendered = (
                        accumulator.render(
                            registry[family][source], max_dimension_values
                        )
                        if accumulator
                        else _empty_phase(registry[family][source])
                    )
                    phase_records[phase] = rendered
                    total_rows += rendered["row_count"]
                if not applicable:
                    status = "unavailable_by_role"
                elif total_rows == 0:
                    status = "empty"
                elif any(
                    phase_records[phase]["row_count"] == 0 for phase in phases
                ):
                    status = "partial"
                else:
                    status = "available"
                missing_phases = [
                    phase for phase in phases if phase_records[phase]["row_count"] == 0
                ]
                sources[source] = {
                    "status": status,
                    "applicable": applicable,
                    "missing_phases": missing_phases,
                    "phases": phase_records,
                }
            devices.append(
                {
                    "node_id": node_id,
                    "region_index": item["region_index"],
                    "region_id": item["region_id"],
                    "role": role,
                    "device_family": family,
                    "candidate": True,
                    "sources": sources,
                }
            )
        model_input = {
            "incident_id": incident_id,
            "dataset_id": incident["dataset_id"],
            "dataset_timezone": "UTC",
            "fault_start_time_utc": incident["fault_start_time_utc"],
            "fault_end_time_utc": incident["fault_end_time_utc"],
            "slice_start_time_utc": incident["slice_start_time_utc"],
            "slice_end_time_utc": incident["slice_end_time_utc"],
            "candidate_scope": "global_72_nodes",
            "candidate_node_ids": list(candidate_ids),
            "topology": {
                "directed": topology["directed"],
                "nodes": [
                    {
                        "node_id": node["node_id"],
                        "region_id": node["region_id"],
                        "region_index": node["region_index"],
                        "role": node["role"],
                        "candidate": node["candidate"],
                    }
                    for node in topology["nodes"]
                ],
                "edges": [
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "edge_type": edge["edge_type"],
                        "protocol": edge.get("protocol"),
                    }
                    for edge in topology["edges"]
                ],
            },
            "fault_taxonomy": taxonomy,
            "devices": devices,
        }
        assert_no_leakage(
            {key: value for key, value in model_input.items() if key != "fault_taxonomy"},
            forbidden_keys,
        )
        path = incident_dir / "incident_input.json"
        path.write_text(
            json.dumps(model_input, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        incident_summaries.append(
            {
                "incident_id": incident_id,
                "candidate_count": len(candidate_ids),
                "device_count": len(devices),
                "output_bytes": path.stat().st_size,
            }
        )

    audit = {
        "status": "failed_coverage" if coverage_failures else "success",
        "evaluated_cases": len(incidents),
        "candidate_count": len(candidate_ids),
        "topology_node_count": len(topology["nodes"]),
        "topology_edge_count": len(topology["edges"]),
        "source_stats": source_stats,
        "excluded_source_stats": excluded_stats,
        "skipped_non_candidate_asset_rows": skipped_asset_rows,
        "timestamp_parse_error_rows": parse_error_rows,
        "coverage_failure_count": len(coverage_failures),
        "coverage_failures": coverage_failures,
        "incidents": incident_summaries,
    }
    assert_no_leakage(audit, forbidden_keys)
    (output_root / "preprocessing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if coverage_failures and config.get("reject_incomplete_periodic_coverage", True):
        raise ValidationError(
            f"{len(coverage_failures)} periodic source/device/phase coverage failures"
        )
    return audit
