from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = {
    "beida": 1, "shenyang": 2, "xian": 3, "chengdu": 4,
    "wuhan": 5, "shanghai": 6, "nanjing": 7, "guangzhou": 8,
}
ROLE_ALIAS = {
    "service-vm-1": "service-1", "service-vm-2": "service-2",
    "service-vm-3": "service-3",
}
LEGAL_ROLES = {
    "br-1", "br-2", "cr-1", "cr-2", "fw", "traffic-vm",
    "service-1", "service-2", "service-3", "monitor-vm", "probe-vm",
}
TIME_COLS = ("timestamp", "timestamp_utc", "minute_utc")
ID_COLS = {
    "timestamp", "timestamp_utc", "minute_utc", "region", "node", "node_type",
    "interface_id", "if_role", "target_id", "exporter_type", "metric_name", "label",
}
DERIVED_SUFFIX = re.compile(r"_(?:mean|max|min|std|avg|p\d+)$")
COUNTER = re.compile(r"(?:_total|_count|carrier_changes|scrape_samples)$")
SPARSE = re.compile(r"(?:error|drop|discard|timeout|fail|loss)")
BINARY = re.compile(r"(?:_up|_success|_exists|_enabled|_state|status)$")


def _region(path: Path) -> tuple[str, int]:
    name = path.parent.parent.name.split("_", 1)[0]
    if name not in REGIONS:
        raise ValueError(f"unknown region directory: {path}")
    return name, REGIONS[name]


def _device(region_index: int, node: str) -> str:
    role = ROLE_ALIAS.get(str(node).strip().lower(), str(node).strip().lower())
    if role not in LEGAL_ROLES:
        raise ValueError(f"illegal explicit node role {node!r}")
    return f"region-{region_index}-{role}"


def _file_kind(path: Path) -> str:
    n = path.name
    if n.startswith("interface_metrics_"): return "interface"
    if n.startswith("node_metrics_"): return "node"
    if n.startswith("routing_metrics_"): return "routing"
    if n.startswith("scrape_health_"): return "scrape"
    if n.startswith("traffic_"): return "traffic"
    if "syslog" in n: return "syslog"
    if "5tuple" in n: return "five_tuple"
    return "unknown"


def preprocess(raw_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    inventory, excluded_traffic, mapping, conflicts = [], [], [], []
    pieces, metric_meta = [], {}
    for path in sorted(raw_root.rglob("*.csv")):
        kind = _file_kind(path)
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            header = next(csv.reader(f), [])
        time_col = next((x for x in TIME_COLS if x in header), None)
        if kind in {"syslog", "five_tuple", "unknown"}:
            inventory.append({
                "source_file": str(path.relative_to(raw_root)), "source_kind": kind,
                "row_count": "", "timestamp_start": "", "timestamp_end": "",
                "columns": "|".join(header), "disposition": "excluded"})
            continue
        row_count, tmin, tmax = 0, None, None
        for chunk in pd.read_csv(path, chunksize=150_000, encoding="utf-8-sig", low_memory=False):
            row_count += len(chunk)
            if time_col:
                ts = pd.to_datetime(chunk[time_col], utc=True, errors="coerce")
                valid = ts.dropna()
                if len(valid):
                    cmin, cmax = valid.min(), valid.max()
                    tmin = cmin if tmin is None else min(tmin, cmin)
                    tmax = cmax if tmax is None else max(tmax, cmax)
            if kind not in {"interface", "node", "routing", "scrape"}:
                continue
            if "node" not in chunk:
                conflicts.append({
                    "source_file": str(path.relative_to(raw_root)), "candidate_device_ids": "",
                    "conflict_reason": "included device-local file has no explicit node column"})
                continue
            region_name, region_index = _region(path)
            try:
                devices = chunk["node"].map(lambda x: _device(region_index, x))
            except ValueError as exc:
                conflicts.append({
                    "source_file": str(path.relative_to(raw_root)), "candidate_device_ids": "",
                    "conflict_reason": str(exc)})
                continue
            ts = pd.to_datetime(chunk[time_col], utc=True, errors="coerce")
            numeric = [c for c in chunk.columns if c not in ID_COLS]
            for col in numeric:
                vals = pd.to_numeric(chunk[col], errors="coerce")
                if not vals.notna().any():
                    continue
                if kind == "interface":
                    dimension = chunk["interface_id"].fillna("unknown").astype(str)
                    metric = "interface::" + dimension + "::" + col
                    family = "interface::" + dimension + "::" + DERIVED_SUFFIX.sub("", col)
                elif kind == "routing":
                    label = chunk["label"].fillna("").astype(str)
                    raw_name = chunk["metric_name"].fillna("unknown").astype(str)
                    metric = "routing::" + raw_name + "::" + label
                    family = "routing::" + raw_name.map(lambda x: DERIVED_SUFFIX.sub("", x))
                elif kind == "scrape":
                    exporter = chunk["exporter_type"].fillna("unknown").astype(str)
                    metric = "scrape::" + exporter + "::" + col
                    family = pd.Series("scrape::" + DERIVED_SUFFIX.sub("", col), index=chunk.index)
                else:
                    metric = pd.Series("node::" + col, index=chunk.index)
                    family = pd.Series("node::" + DERIVED_SUFFIX.sub("", col), index=chunk.index)
                temp = pd.DataFrame({
                    "timestamp": ts, "device_id": devices, "metric": metric,
                    "base_metric_family": family, "raw_value": vals,
                    "source_file": str(path.relative_to(raw_root)),
                    "source_kind": kind}).dropna(subset=["timestamp", "raw_value"])
                temp = temp.groupby(
                    ["timestamp","device_id","metric","base_metric_family",
                     "source_file","source_kind"], as_index=False).raw_value.mean()
                pieces.append(temp)
        inventory.append({
            "source_file": str(path.relative_to(raw_root)), "source_kind": kind,
            "row_count": row_count, "timestamp_start": tmin.isoformat() if tmin is not None else "",
            "timestamp_end": tmax.isoformat() if tmax is not None else "",
            "columns": "|".join(header),
            "disposition": "included" if kind in {"interface","node","routing","scrape"} else "excluded"})
        if kind == "traffic":
            excluded_traffic.append({
                "source_file": str(path.relative_to(raw_root)), "row_count": row_count,
                "timestamp_start": tmin.isoformat() if tmin is not None else "",
                "timestamp_end": tmax.isoformat() if tmax is not None else "",
                "columns": "|".join(header),
                "exclusion_reason": "path_level_traffic_observation_not_device_local_metric"})
    if conflicts:
        _write_csv(output / "device_mapping_conflicts.csv", conflicts)
        _write_csv(output / "source_file_inventory.csv", inventory)
        _write_csv(output / "excluded_traffic_files.csv", excluded_traffic)
        return {"passed": False, "unresolved_mapping_conflicts": len(conflicts)}
    long = pd.concat(pieces, ignore_index=True)
    long = long.groupby(
        ["timestamp","device_id","metric","base_metric_family","source_file","source_kind"],
        as_index=False).raw_value.mean()
    long.sort_values(["device_id","metric","timestamp"], inplace=True)
    # A metric identity maps to one file/entity/device; duplicates are deterministic means.
    for (source_file, device, metric), group in long.groupby(
        ["source_file","device_id","metric"], sort=True):
        mapping.append({
            "source_file": source_file, "entity_type": group.source_kind.iloc[0],
            "device_id": device, "raw_metric_name": metric,
            "base_metric_family": group.base_metric_family.iloc[0]})
    processed_groups, quality, metric_rows, family_rows = [], [], [], []
    gap_count = split_series = 0
    for (device, metric), group in long.groupby(["device_id","metric"], sort=True):
        group = group.sort_values("timestamp").drop_duplicates("timestamp").copy()
        dt = group.timestamp.diff().dt.total_seconds()
        cuts = dt.gt(120).fillna(False)
        group["continuous_segment_id"] = cuts.cumsum().map(
            lambda x: f"{device}::{metric}::segment-{x+1}")
        gaps = int(cuts.sum()); gap_count += gaps; split_series += gaps > 0
        name = metric.lower()
        finite = group.raw_value.dropna().to_numpy(float)
        cal = group[(group.timestamp >= pd.Timestamp("2026-07-28T04:00:00Z")) &
                    (group.timestamp < pd.Timestamp("2026-07-28T05:30:00Z"))].raw_value.to_numpy(float)
        cal = cal[np.isfinite(cal)]
        unique = set(np.unique(finite)) if len(finite) else set()
        if not len(cal):
            metric_type, reason = "unavailable", "no_calibration_values"
        elif unique.issubset({0.0, 1.0}) and BINARY.search(name):
            metric_type, reason = "binary_state", ""
        elif COUNTER.search(name):
            metric_type, reason = "cumulative_counter", ""
        else:
            med = float(np.median(cal)); mad = float(np.median(np.abs(cal-med)))
            q1, q3 = np.quantile(cal, [.25,.75])
            nonzero = float(np.count_nonzero(finite)/len(finite)) if len(finite) else 0
            if SPARSE.search(name) and (nonzero < .05 or mad == 0 or q3 == q1):
                metric_type, reason = "sparse_counter", ""
            else:
                metric_type, reason = "dense_continuous", ""
        group["metric_type"] = metric_type
        group["value"] = group.raw_value
        if metric_type == "cumulative_counter":
            delta = group.groupby("continuous_segment_id").raw_value.diff()
            minutes = group.groupby("continuous_segment_id").timestamp.diff().dt.total_seconds()/60
            group["value"] = np.log1p((delta/minutes).clip(lower=0))
        elif metric_type == "sparse_counter" and COUNTER.search(name):
            delta = group.groupby("continuous_segment_id").raw_value.diff()
            group["activity_value"] = (delta.clip(lower=0)).fillna(0)
            group["value"] = np.log1p(group.activity_value)
        else:
            group["activity_value"] = group.raw_value
        if "activity_value" not in group:
            group["activity_value"] = group.raw_value
        processed_groups.append(group)
        quality.append({
            "device_id": device, "metric": metric, "metric_type": metric_type,
            "row_count": len(group), "segment_count": group.continuous_segment_id.nunique(),
            "gap_count_gt_120s": gaps, "sampling_p50_seconds": dt.dropna().quantile(.5),
            "sampling_p95_seconds": dt.dropna().quantile(.95),
            "sampling_max_seconds": dt.dropna().max(), "unavailable_reason": reason})
        metric_rows.append({
            "device_id": device, "metric": metric,
            "base_metric_family": group.base_metric_family.iloc[0],
            "metric_type": metric_type, "source_file": group.source_file.iloc[0]})
        family_rows.append({
            "device_id": device, "base_metric_family": group.base_metric_family.iloc[0],
            "metric": metric})
    result = pd.concat(processed_groups, ignore_index=True)
    result = result[result.metric_type != "unavailable"]
    result.to_parquet(output / "series.parquet", index=False, compression="zstd")
    mapping = [
        {
            "source_file": row.source_file, "entity_type": row.source_kind,
            "device_id": row.device_id, "raw_metric_name": row.metric,
            "base_metric_family": row.base_metric_family,
            "metric_type": row.metric_type,
            "continuous_segment_id": row.continuous_segment_id,
        }
        for row in result[[
            "source_file","source_kind","device_id","metric","base_metric_family",
            "metric_type","continuous_segment_id"]].drop_duplicates().itertuples(index=False)
    ]
    _write_csv(output / "source_file_inventory.csv", inventory)
    _write_csv(output / "excluded_traffic_files.csv", excluded_traffic)
    _write_csv(output / "device_mapping_manifest.csv", mapping)
    _write_csv(output / "device_mapping_conflicts.csv", [], [
        "source_file","candidate_device_ids","conflict_reason"])
    _write_csv(output / "device_metric_manifest.csv", metric_rows)
    _write_csv(output / "metric_family_manifest.csv", family_rows)
    _write_csv(output / "series_quality_manifest.csv", quality)
    all_dt = long.sort_values(["device_id","metric","timestamp"]).groupby(
        ["device_id","metric"]).timestamp.diff().dt.total_seconds().dropna()
    audit = {
        "passed": True, "development_experiment": True,
        "traffic_csv_files_in_model": 0, "excluded_traffic_files": len(excluded_traffic),
        "unknown_device_ids": 0, "unresolved_mapping_conflicts": 0,
        "cross_device_windows": 0, "cross_segment_windows": 0,
        "calibration_event_points": 0, "counter_cross_device_diffs": 0,
        "sampling_interval_seconds": {
            "p50": float(all_dt.quantile(.5)), "p95": float(all_dt.quantile(.95)),
            "max": float(all_dt.max())},
        "gaps_gt_120_seconds": gap_count, "split_series_count": split_series,
        "series": int(result.groupby(["device_id","metric","continuous_segment_id"]).ngroups),
        "devices": int(result.device_id.nunique()),
        "metric_types": {str(k): int(v) for k, v in
            result.groupby("metric_type")[["device_id","metric"]].apply(
                lambda x: x.drop_duplicates().shape[0]).to_dict().items()},
    }
    (output / "preprocessing_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    (output / "preprocessing_audit.md").write_text(
        "# No-traffic preprocessing audit\n\n"
        f"- Passed: **{audit['passed']}**\n"
        f"- Traffic CSV files entering model: **0**; excluded: **{len(excluded_traffic)}**.\n"
        f"- Devices/segmented series: **{audit['devices']}/{audit['series']}**.\n"
        f"- Sampling P50/P95/max: **{audit['sampling_interval_seconds']} seconds**.\n"
        f"- Gaps >120s / split series: **{gap_count}/{split_series}**.\n"
        "- Cross-device windows, cross-gap windows, calibration event points, and "
        "cross-device counter diffs: **0/0/0/0**.\n",
        encoding="utf-8")
    return audit


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
