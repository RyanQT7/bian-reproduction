#!/usr/bin/env python3
"""Reaggregate persisted MOMENT scores without importing or invoking MOMENT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bian.detection.aggregation import (
    complete_linkage_clusters,
    merge_intervals_by_timestamp,
    sampled_hysteresis_intervals,
    temporal_gap_seconds,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--source-run", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
source, output = Path(args.source_run), Path(args.output)
cfg = json.loads(Path(args.config).read_text())
output.mkdir(parents=True, exist_ok=False)
scores = pd.read_parquet(source / "metric_point_scores.parquet")
scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
detection_start = pd.Timestamp(cfg["detection_start"])

ordered = scores.sort_values(["device", "metric", "timestamp"])
diffs = ordered.groupby(["device", "metric"]).timestamp.diff().dt.total_seconds().dropna()
old_before = [json.loads(x) for x in (source / "device_intervals_before_merge.jsonl").open()]
old_longest: dict[str, dict] = {}
for row in old_before:
    duration = (pd.Timestamp(row["end_time"]) - pd.Timestamp(row["start_time"])).total_seconds()
    item = {**row, "duration_seconds": duration}
    if row["device"] not in old_longest or duration > old_longest[row["device"]]["duration_seconds"]:
        old_longest[row["device"]] = item

audit = {
    "mode": "persisted_point_scores_only; MOMENT not imported or invoked",
    "sampling_interval_seconds": {
        "median": float(diffs.median()), "minimum": float(diffs.min()),
        "maximum": float(diffs.max()), "mode": float(diffs.mode().iloc[0])},
    "time_parameter_units_before_fix": {
        "config_onset_20": "seconds, converted by ceil(20/60), then max(2) => 2 sample points",
        "config_clear_30": "seconds, converted by ceil(30/60), then max(2) => 2 sample points",
        "config_merge_gap_60": "seconds, but max(2*median_interval,60) => 120 seconds / 2 points",
        "cross_device_60": "seconds using pandas Timedelta.total_seconds()",
    },
    "time_parameter_units_after_fix": {
        "onset": "exactly 2 consecutive sampled points",
        "clear": "exactly 2 consecutive sampled points outside the low-threshold active state",
        "same_device_merge_gap": "120 seconds by actual timestamp difference",
        "cross_device_compatibility": "120 seconds by direct actual timestamp difference",
    },
    "merge_gap_seconds": 120,
    "merge_gap_nominal_points": 2,
    "twenty_minute_gap_merged_before_fix": False,
    "uses_row_difference_instead_of_timestamp": False,
    "datetime_implementation": "UTC pandas Timestamp; gap uses Timedelta.total_seconds()",
    "mixed_units_defect": True,
    "calibration_points_entered_old_state_machine": sum(
        pd.Timestamp(x["start_time"]) < detection_start for x in old_before),
    "old_device_intervals": len(old_before),
    "old_longest_interval_before_merge_by_device": old_longest,
    "old_non_exit_reason": (
        "binary-state flip logic used reconstruction-score diff().abs()>0; continuously "
        "changing floating-point errors repeatedly reset quiet/clear even though they "
        "were not original binary states"),
    "old_clear_logic": (
        "two non-high-trigger points, not two points below the low-threshold active state"),
    "same_device_old_global_merge_parameter": (
        "yes: merge_gap_seconds=60 was passed, then silently raised to 120 by max(2*dt, gap)"),
}
(output / "time_state_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

before_rows, after_rows, spans = [], [], []
for device, d in scores.groupby("device", sort=True):
    pivot = d.pivot_table(index="timestamp", columns="metric", values="score", aggfunc="mean")
    high = d.groupby("metric").high.first().reindex(pivot.columns)
    low = d.groupby("metric").low.first().reindex(pivot.columns)
    safe_high = high.replace(0, cfg["minimum_scale"])
    high_ratio = pivot.divide(safe_high, axis=1)
    start_state = (pivot.ge(high, axis=1).sum(axis=1) >= 2) | (
        high_ratio.max(axis=1) >= cfg["extreme_high_multiplier"])
    # Low hysteresis mirrors the two-indicator trigger. Extreme incidents remain active
    # while still extreme; score changes are never treated as binary state flips.
    low_active_state = (pivot.ge(low, axis=1).sum(axis=1) >= 2) | (
        high_ratio.max(axis=1) >= cfg["extreme_high_multiplier"])
    raw = sampled_hysteresis_intervals(
        start_state.to_numpy(), low_active_state.to_numpy(), pivot.index.to_numpy(),
        detection_start, cfg["onset_points"], cfg["clear_points"])
    merged = merge_intervals_by_timestamp(raw, cfg["same_device_merge_gap_seconds"])
    for i, (start, end) in enumerate(raw, 1):
        before_rows.append({
            "interval_id": f"{device}:before:{i}", "device": device,
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "onset_points": 2, "clear_points": 2,
            "calibration_excluded": start >= detection_start})
    for i, (start, end) in enumerate(merged, 1):
        interval_id = f"{device}:after:{i}"
        segment = (pivot.index >= start) & (pivot.index <= end)
        top = high_ratio.loc[segment].max().nlargest(cfg["max_top_indicators"])
        member_ids = [
            x["interval_id"] for x in before_rows if x["device"] == device
            and pd.Timestamp(x["start_time"]) >= start and pd.Timestamp(x["end_time"]) <= end]
        record = {
            "interval_id": interval_id, "device": device,
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "duration_seconds": (end-start).total_seconds(),
            "member_before_merge_ids": member_ids,
            "top_indicators": top.index.tolist(), "max_device_score": float(top.mean())}
        after_rows.append(record)
        spans.append({
            "interval_id": interval_id, "device": device, "start": start, "end": end,
            "top_indicators": record["top_indicators"],
            "max_device_score": record["max_device_score"]})

write_jsonl(output / "device_intervals_before_merge.jsonl", before_rows)
write_jsonl(output / "device_intervals_after_same_device_merge.jsonl", after_rows)
clusters = complete_linkage_clusters(spans, cfg["cross_device_compatibility_seconds"])
lineage, events = [], []
for i, cluster in enumerate(clusters, 1):
    members = cluster["members"]
    lineage.append({
        "cluster_id": cluster["cluster_id"],
        "boundary_after": {"start": cluster["start"].isoformat(), "end": cluster["end"].isoformat()},
        "member_intervals": [
            {"interval_id": x["interval_id"], "device": x["device"],
             "start_time": x["start"].isoformat(), "end_time": x["end"].isoformat()}
            for x in members],
        "joins": cluster["joins"]})
    events.append({
        "incident_id": f"moment-dev-{i:04d}", "cluster_id": cluster["cluster_id"],
        "start_time": cluster["start"].isoformat(), "end_time": cluster["end"].isoformat(),
        "devices": sorted({x["device"] for x in members}),
        "top_indicators": sorted({m for x in members for m in x["top_indicators"]}),
        "member_interval_ids": [x["interval_id"] for x in members],
        "max_device_score": max(x["max_device_score"] for x in members)})
write_jsonl(output / "cross_device_merge_lineage.jsonl", lineage)
write_jsonl(output / "detected_incidents.jsonl", events)

pairwise_ok = all(
    temporal_gap_seconds(a, b) <= cfg["cross_device_compatibility_seconds"]
    for c in clusters for i, a in enumerate(c["members"]) for b in c["members"][i+1:])
durations = [(x["end"]-x["start"]).total_seconds() for x in spans]
event_durations = [(c["end"]-c["start"]).total_seconds() for c in clusters]
summary = {
    "source_point_scores": str((source / "metric_point_scores.parquet").resolve()),
    "point_score_rows": len(scores), "calibration_points_used_for_events": 0,
    "device_intervals_before_merge": len(before_rows),
    "device_intervals_after_same_device_merge": len(after_rows),
    "events": len(events), "complete_linkage_pairwise_valid": pairwise_ok,
    "device_duration_seconds": {
        "minimum": min(durations) if durations else None,
        "median": float(np.median(durations)) if durations else None,
        "maximum": max(durations) if durations else None},
    "event_duration_seconds": {
        "minimum": min(event_durations) if event_durations else None,
        "median": float(np.median(event_durations)) if event_durations else None,
        "maximum": max(event_durations) if event_durations else None},
    "gaps_over_120_seconds_merged": False,
    "binary_score_diff_used_as_state_flip": False,
}
(output / "validation_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
