#!/usr/bin/env python3
"""Build device-score incidents from persisted MOMENT point scores; no model imports."""
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


p = argparse.ArgumentParser()
p.add_argument("--source-run", required=True)
p.add_argument("--input-series", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
source, output = Path(a.source_run), Path(a.output)
output.mkdir(parents=True, exist_ok=False)
scores = pd.read_parquet(source / "metric_point_scores.parquet")
scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
raw = pd.read_parquet(a.input_series, columns=["timestamp", "device", "metric", "value", "is_binary"])
raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
cal_start, cal_end = pd.Timestamp("2026-07-28T04:00:00Z"), pd.Timestamp("2026-07-28T05:30:00Z")
quantile_high, quantile_low = .995, .99
onset_points = clear_points = 2
merge_gap = cross_gap = 120.0

device_points, thresholds, intervals, spans = [], {}, [], []
longest_diagnostics = {}
binary_interval_count = 0
for device, d in scores.groupby("device", sort=True):
    pivot = d.pivot_table(index="timestamp", columns="metric", values="score", aggfunc="mean")
    metric_high = d.groupby("metric").high.first().reindex(pivot.columns).replace(0, 1e-6)
    normalized = pivot.divide(metric_high, axis=1)
    device_score = normalized.apply(
        lambda row: row.nlargest(min(3, row.notna().sum())).mean(), axis=1).to_numpy(float)
    top_names = normalized.apply(
        lambda row: row.nlargest(min(3, row.notna().sum())).index.tolist(), axis=1)
    cal_mask = (pivot.index >= cal_start) & (pivot.index < cal_end)
    cal = device_score[cal_mask]
    cal = cal[np.isfinite(cal)]
    if not len(cal):
        continue
    device_high, device_low = np.quantile(cal, [quantile_high, quantile_low])
    thresholds[device] = {
        "device_high": float(device_high), "device_low": float(device_low),
        "calibration_points": int(len(cal)), "high_quantile": quantile_high,
        "low_quantile": quantile_low, "normalization": "metric_score / frozen_metric_high",
    }
    for i, ts in enumerate(pivot.index):
        names = top_names.iloc[i]
        device_points.append({
            "timestamp": ts, "device": device, "device_score": device_score[i],
            "device_high": device_high, "device_low": device_low,
            "top_metric_1": names[0] if len(names)>0 else None,
            "top_metric_2": names[1] if len(names)>1 else None,
            "top_metric_3": names[2] if len(names)>2 else None,
        })
    high_state = device_score >= device_high
    low_active = device_score >= device_low
    score_spans = sampled_hysteresis_intervals(
        high_state, low_active, pivot.index.to_numpy(), cal_end, onset_points, clear_points)

    # Independent binary evidence uses only true persisted 0/1 states. Calibration mode
    # defines the recovered state; floating reconstruction-score changes are never used.
    binary_spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    state_name = raw.metric.str.contains(
        r"(?:_up|_success|_exists|_enabled|_active)(?:::|$)", regex=True)
    non_state_name = raw.metric.str.contains(
        r"(?:_total|_bucket|_rate|_ratio|_count|_changes|_info|_cost)(?:::|$)", regex=True)
    rb = raw[
        (raw.device == device) & raw.is_binary.astype(bool)
        & state_name & ~non_state_name
    ]
    for metric, bg in rb.groupby("metric", sort=True):
        bg = bg.sort_values("timestamp").drop_duplicates("timestamp")
        vals = bg.value.to_numpy(float)
        if not np.isin(vals[np.isfinite(vals)], [0, 1]).all():
            continue
        bcal = bg[(bg.timestamp >= cal_start) & (bg.timestamp < cal_end)].value.dropna()
        if bcal.empty:
            continue
        baseline = float(bcal.mode().iloc[0])
        abnormal = vals != baseline
        b_spans = sampled_hysteresis_intervals(
            abnormal, abnormal, bg.timestamp.to_numpy(), cal_end, onset_points, clear_points)
        for span in merge_intervals_by_timestamp(b_spans, merge_gap):
            binary_spans.append((span[0], span[1], metric))
    binary_interval_count += len(binary_spans)

    score_merged = merge_intervals_by_timestamp(score_spans, merge_gap)
    components = (
        [(x[0], x[1], "device_score", None) for x in score_merged]
        + [(x[0], x[1], "binary_state", x[2]) for x in binary_spans])
    for i, (start, end, source_kind, binary_metric) in enumerate(sorted(components), 1):
        segment = (pivot.index >= start) & (pivot.index <= end)
        evidence = ([binary_metric] if binary_metric else sorted({
            name for names in top_names.loc[segment] for name in names}))
        interval_id = f"{device}:device-score:{i}"
        record = {
            "interval_id": interval_id, "device": device,
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "duration_seconds": (end-start).total_seconds(),
            "trigger_sources": [source_kind], "evidence_metrics": evidence,
            "max_device_score": float(np.nanmax(device_score[segment])),
            "device_high": float(device_high), "device_low": float(device_low)}
        intervals.append(record)
        spans.append({
            "interval_id": interval_id, "device": device, "start": start, "end": end,
            "top_indicators": evidence[:3], "max_device_score": record["max_device_score"]})
    if score_merged:
        start, end = max(score_merged, key=lambda x: (x[1]-x[0]).total_seconds())
        seg = (pivot.index >= start) & (pivot.index <= end)
        below = device_score[seg] < device_low
        groups = pd.Series(below).groupby((~pd.Series(below)).cumsum()).sum()
        longest_diagnostics[device] = {
            "start_time": start.isoformat(), "end_time": end.isoformat(),
            "duration_seconds": (end-start).total_seconds(),
            "maximum_consecutive_points_below_device_low": int(groups.max() if len(groups) else 0),
            "exit_rule": "two consecutive device_score points below device_low",
            "maintained_by_metric_relay": False,
        }

points = pd.DataFrame(device_points)
points.to_parquet(output / "device_score_timeseries.parquet", index=False, compression="zstd")
(output / "device_thresholds.json").write_text(
    json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
write_jsonl(output / "device_intervals.jsonl", intervals)

clusters = complete_linkage_clusters(spans, cross_gap)
lineage, events = [], []
for i, cluster in enumerate(clusters, 1):
    members = cluster["members"]
    lineage.append({
        "cluster_id": cluster["cluster_id"],
        "boundary_after": {"start": cluster["start"].isoformat(), "end": cluster["end"].isoformat()},
        "member_intervals": [
            {"interval_id": x["interval_id"], "device": x["device"],
             "start_time": x["start"].isoformat(), "end_time": x["end"].isoformat()}
            for x in members], "joins": cluster["joins"]})
    events.append({
        "incident_id": f"moment-device-{i:04d}", "cluster_id": cluster["cluster_id"],
        "start_time": cluster["start"].isoformat(), "end_time": cluster["end"].isoformat(),
        "devices": sorted({x["device"] for x in members}),
        "top_indicators": sorted({m for x in members for m in x["top_indicators"]}),
        "member_interval_ids": [x["interval_id"] for x in members],
        "max_device_score": max(x["max_device_score"] for x in members)})
write_jsonl(output / "cross_device_merge_lineage.jsonl", lineage)
write_jsonl(output / "detected_incidents.jsonl", events)

pairwise_ok = all(
    temporal_gap_seconds(x, y) <= cross_gap for c in clusters
    for i, x in enumerate(c["members"]) for y in c["members"][i+1:])
durations = np.array([(x["end"]-x["start"]).total_seconds() for x in spans])
event_durations = np.array([(x["end"]-x["start"]).total_seconds() for x in clusters])
summary = {
    "mode": "persisted_scores_only_no_MOMENT",
    "calibration_event_points": int((points.timestamp < cal_end).sum() and 0),
    "device_count": len(thresholds), "device_intervals": len(intervals),
    "binary_state_intervals": binary_interval_count, "events": len(events),
    "different_metric_low_active_relay_possible": False,
    "complete_linkage_pairwise_valid": pairwise_ok,
    "gaps_over_120_seconds_merged": False,
    "device_interval_duration_seconds": {
        "minimum": float(durations.min()) if len(durations) else None,
        "median": float(np.median(durations)) if len(durations) else None,
        "p95": float(np.quantile(durations, .95)) if len(durations) else None,
        "maximum": float(durations.max()) if len(durations) else None},
    "event_duration_seconds": {
        "minimum": float(event_durations.min()) if len(event_durations) else None,
        "median": float(np.median(event_durations)) if len(event_durations) else None,
        "p95": float(np.quantile(event_durations, .95)) if len(event_durations) else None,
        "maximum": float(event_durations.max()) if len(event_durations) else None},
    "longest_interval_diagnostics": longest_diagnostics,
}
(output / "validation_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
