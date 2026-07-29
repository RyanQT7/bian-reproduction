"""Read-only raw-data compatibility audit for blind incident windows."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .real_preprocessing import parse_utc
from .validators import ValidationError


def _source_name(path: Path) -> str:
    name = path.name
    for prefix in (
        "interface_metrics",
        "node_metrics",
        "routing_metrics",
        "scrape_health",
        "frr_syslog_events",
    ):
        if name.startswith(prefix + "_"):
            return prefix
    if name == "traffic_flow_metrics.csv":
        return "traffic_flow_metrics"
    if name == "netflow_5tuple_minute_readable.csv":
        return "netflow_5tuple_minute_readable"
    return path.stem


def audit_dataset(
    *,
    raw_root: Path,
    experiment_path: Path,
    region_mapping: dict[Path, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    incidents = [
        json.loads(line)
        for line in experiment_path.read_text().splitlines()
        if line.strip()
    ]
    if len(incidents) != 33 or len({x["incident_id"] for x in incidents}) != 33:
        raise ValidationError("blind audit requires 33 unique incidents")
    required_start = min(parse_utc(x["analysis_window_start"]) for x in incidents)
    required_end = max(parse_utc(x["analysis_window_end"]) for x in incidents)
    source_patterns = {
        **config["source_files"],
        **{name: f"{name}.csv" for name in config["excluded_sources"]},
    }
    time_fields = config["time_fields"]
    schemas: dict[str, set[tuple[str, ...]]] = {}
    source_metric_names: dict[str, set[str]] = {}
    files = []
    coverage_failures = []
    for region_dir, region_id in sorted(region_mapping.items(), key=lambda x: x[1]):
        processed = region_dir / "processed"
        for source, pattern in source_patterns.items():
            matches = sorted(processed.glob(pattern))
            if len(matches) != 1:
                files.append(
                    {
                        "region_id": region_id,
                        "source": source,
                        "status": "missing",
                        "match_count": len(matches),
                    }
                )
                continue
            path = matches[0]
            size = path.stat().st_size
            if source in config["excluded_sources"]:
                files.append(
                    {
                        "region_id": region_id,
                        "source": source,
                        "status": "excluded_five_tuple",
                        "bytes": size,
                        "rows_scanned": 0,
                    }
                )
                continue
            if size == 0:
                files.append(
                    {
                        "region_id": region_id,
                        "source": source,
                        "status": "empty",
                        "bytes": 0,
                        "rows": 0,
                    }
                )
                continue
            first: datetime | None = None
            last: datetime | None = None
            rows = 0
            parse_errors = 0
            non_monotonic = 0
            duplicate_full_rows = 0
            previous_timestamp: datetime | None = None
            previous_row: tuple[str, ...] | None = None
            with path.open(
                newline="", encoding="utf-8-sig", errors="replace"
            ) as handle:
                reader = csv.DictReader(handle)
                header = tuple(reader.fieldnames or ())
                schemas.setdefault(source, set()).add(header)
                timestamp_field = time_fields[source]
                if timestamp_field not in header:
                    raise ValidationError(f"{path} lacks {timestamp_field}")
                for row in reader:
                    rows += 1
                    raw_timestamp = row.get(timestamp_field, "")
                    if source == "frr_syslog_events" and not raw_timestamp:
                        raw_timestamp = row.get("received_at", "")
                    try:
                        timestamp = parse_utc(raw_timestamp)
                    except ValidationError:
                        parse_errors += 1
                        continue
                    first = timestamp if first is None else min(first, timestamp)
                    last = timestamp if last is None else max(last, timestamp)
                    if previous_timestamp and timestamp < previous_timestamp:
                        non_monotonic += 1
                    rendered = tuple(row.get(field, "") for field in header)
                    if rendered == previous_row:
                        duplicate_full_rows += 1
                    previous_timestamp = timestamp
                    previous_row = rendered
                    if source == "routing_metrics" and row.get("metric_name"):
                        source_metric_names.setdefault(region_id, set()).add(
                            row["metric_name"]
                        )
            status = "available"
            if source in config["periodic_sources"] and (
                first is None
                or last is None
                or first > required_start
                or last < required_end
            ):
                status = "coverage_failed"
                coverage_failures.append(
                    {
                        "region_id": region_id,
                        "source": source,
                        "required_start": required_start.isoformat(),
                        "required_end": required_end.isoformat(),
                        "available_start": first.isoformat() if first else None,
                        "available_end": last.isoformat() if last else None,
                    }
                )
            files.append(
                {
                    "region_id": region_id,
                    "source": source,
                    "status": status,
                    "bytes": size,
                    "rows": rows,
                    "timestamp_start": first.isoformat() if first else None,
                    "timestamp_end": last.isoformat() if last else None,
                    "timestamp_parse_errors": parse_errors,
                    "non_monotonic_rows": non_monotonic,
                    "adjacent_duplicate_full_rows": duplicate_full_rows,
                    "schema": list(header),
                }
            )
    schema_consistency = {
        source: {
            "consistent_across_regions": len(values) == 1,
            "schema_variants": [list(value) for value in sorted(values)],
        }
        for source, values in sorted(schemas.items())
    }
    routing_sets = list(source_metric_names.values())
    routing_intersection = set.intersection(*routing_sets) if routing_sets else set()
    routing_union = set.union(*routing_sets) if routing_sets else set()
    return {
        "audit_kind": "read_only_blind_schema_coverage",
        "incident_count": len(incidents),
        "region_count": len(region_mapping),
        "timezone_values": sorted({x["timezone"] for x in incidents}),
        "required_time_range_utc": {
            "start": required_start.isoformat(),
            "end": required_end.isoformat(),
        },
        "files": files,
        "schema_consistency": schema_consistency,
        "routing_metric_intersection": sorted(routing_intersection),
        "routing_metric_union": sorted(routing_union),
        "routing_metrics_consistent": routing_intersection == routing_union,
        "coverage_failure_count": len(coverage_failures),
        "coverage_failures": coverage_failures,
        "five_tuple_rule": {
            "source": "netflow_5tuple_minute_readable",
            "action": "excluded_without_reading_rows",
            "uniform_across_regions_and_cases": True,
            "aggregated_traffic_flow_metrics_retained": True,
        },
    }
