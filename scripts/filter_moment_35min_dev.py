#!/usr/bin/env python3
"""Development-only 35-minute filtering over persisted MOMENT artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from bian.detection.guard import sha256
from bian.detection.scoring import score


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--source-validation", required=True)
parser.add_argument("--point-scores", required=True)
parser.add_argument("--input-series", required=True)
parser.add_argument("--truth-times", required=True)
parser.add_argument("--truth-labels", required=True)
parser.add_argument("--output", required=True)
a = parser.parse_args()
src, out = Path(a.source_validation), Path(a.output)
out.mkdir(parents=True, exist_ok=False)
limit_s, gap_s = 35 * 60.0, 120.0
cal_start, cal_end = pd.Timestamp("2026-07-28T04:00:00Z"), pd.Timestamp("2026-07-28T05:30:00Z")
filter_config = {
    "classification": "development_validation_after_truth_access",
    "maximum_short_interval_seconds": limit_s,
    "network_or_merge_gap_seconds": gap_s,
    "device_rule": "duration <= 35 minutes retained; duration > 35 minutes isolated",
    "network_rule": "UTC timeline OR; overlap/gap <=120s merged; resulting >35m isolated",
}
config_bytes = json.dumps(filter_config, sort_keys=True, separators=(",", ":")).encode()
config_sha = hashlib.sha256(config_bytes).hexdigest()

device_intervals = load_jsonl(src / "device_intervals.jsonl")
device_scores = pd.read_parquet(src / "device_score_timeseries.parquet")
device_scores["timestamp"] = pd.to_datetime(device_scores.timestamp, utc=True)
points_path = Path(a.point_scores)
points = pd.read_parquet(points_path)
points["timestamp"] = pd.to_datetime(points.timestamp, utc=True)
raw = pd.read_parquet(a.input_series)
raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
calibration_stats = {}
raw_cal = raw[(raw.timestamp >= cal_start) & (raw.timestamp < cal_end)]
for key, group in raw_cal.groupby(["device", "metric"], sort=False):
    cal = group.value.dropna().to_numpy(float)
    if len(cal):
        med = float(np.median(cal))
        mad = float(np.median(np.abs(cal-med)))
        q1, q3 = np.quantile(cal, [.25, .75])
        calibration_stats[key] = (med, mad, float(q3-q1), int(np.unique(cal).size))

long_device, short_device = [], []
metric_frequency = Counter()
device_long_counts = Counter()
role_long_counts = Counter()
source_long_counts = Counter()
near_constant_metric_occurrences = mad0_occurrences = iqr0_occurrences = 0
for interval in device_intervals:
    start, end = pd.Timestamp(interval["start_time"]), pd.Timestamp(interval["end_time"])
    duration = (end-start).total_seconds()
    if duration <= limit_s:
        short_device.append({**interval, "_start": start, "_end": end})
        continue
    device = interval["device"]
    ds = device_scores[
        (device_scores.device == device) & (device_scores.timestamp >= start)
        & (device_scores.timestamp <= end)].device_score.dropna().to_numpy(float)
    candidates = []
    dp = points[
        (points.device == device) & (points.timestamp >= start) & (points.timestamp <= end)]
    for metric, group in dp.groupby("metric", sort=False):
        vals = group.score.dropna().to_numpy(float)
        if not len(vals):
            continue
        threshold_high = float(group.high.iloc[0])
        threshold_low = float(group.low.iloc[0])
        contribution = float(np.quantile(vals / max(threshold_high, 1e-6), .95))
        med, mad, iqr, unique = calibration_stats.get(
            (device, metric), (float("nan"), float("nan"), float("nan"), 0))
        expected = max(1, int(duration // 60) + 1)
        missing_ratio = max(0.0, 1.0 - group.timestamp.nunique()/expected)
        binary = bool(group.binary.iloc[0])
        near_constant = bool(unique and (mad == 0 or iqr == 0 or unique <= 1))
        candidates.append({
            "metric_name": metric, "metric_type": "binary_state" if binary else "continuous",
            "contribution": contribution,
            "calibration_median": med, "calibration_mad": mad, "calibration_iqr": iqr,
            "high_threshold": threshold_high, "low_threshold": threshold_low,
            "score_p50": float(np.quantile(vals, .50)),
            "score_p95": float(np.quantile(vals, .95)),
            "score_p99": float(np.quantile(vals, .99)),
            "score_max": float(np.max(vals)), "missing_ratio": float(missing_ratio),
            "near_constant": near_constant})
    top = sorted(candidates, key=lambda x: (-x["contribution"], x["metric_name"]))[:5]
    for metric in top:
        metric_frequency[metric["metric_name"]] += 1
        near_constant_metric_occurrences += metric["near_constant"]
        mad0_occurrences += metric["calibration_mad"] == 0
        iqr0_occurrences += metric["calibration_iqr"] == 0
        source_long_counts[metric["metric_name"].split("::", 1)[0]] += 1
    role = device.split(":", 1)[0] if ":" not in device else "traffic-source"
    if device in {"br-1","br-2","cr-1","cr-2","fw","monitor-vm","probe-vm",
                  "service-vm-1","service-vm-2","service-vm-3","traffic-vm"}:
        role = device
    role_long_counts[role] += 1
    device_long_counts[device] += 1
    long_device.append({
        "device_id": device, "start_time": start.isoformat(), "end_time": end.isoformat(),
        "duration_minutes": duration/60,
        "device_high": float(interval["device_high"]), "device_low": float(interval["device_low"]),
        "device_score_min": float(np.min(ds)) if len(ds) else None,
        "device_score_median": float(np.median(ds)) if len(ds) else None,
        "device_score_p95": float(np.quantile(ds, .95)) if len(ds) else None,
        "device_score_max": float(np.max(ds)) if len(ds) else None,
        "top_contributing_metrics": top})
write_jsonl(out / "persistent_device_intervals.jsonl", long_device)

# Network OR over only short device intervals. This is an actual UTC timeline union:
# every admitted join has direct gap <=120 seconds; any larger blank starts a new segment.
ordered = sorted(short_device, key=lambda x: (x["_start"], x["_end"], x["device"]))
network = []
for item in ordered:
    if network and (item["_start"] - network[-1]["end"]).total_seconds() <= gap_s:
        network[-1]["end"] = max(network[-1]["end"], item["_end"])
        network[-1]["members"].append(item)
    else:
        network.append({"start": item["_start"], "end": item["_end"], "members": [item]})

long_network, short_network = [], []
for segment in network:
    duration = (segment["end"]-segment["start"]).total_seconds()
    members = segment["members"]
    devices = sorted({x["device"] for x in members})
    metrics = Counter(m for x in members for m in x.get("evidence_metrics", []))
    record = {
        "start_time": segment["start"].isoformat(), "end_time": segment["end"].isoformat(),
        "duration_minutes": duration/60, "devices": devices,
        "member_device_interval_ids": [x["interval_id"] for x in members],
        "formed_by_multiple_short_device_interval_relay": len(members) > 1,
        "primary_contributing_devices": [
            x for x, _ in Counter(y["device"] for y in members).most_common(5)],
        "primary_contributing_metrics": [x for x, _ in metrics.most_common(5)],
        "_members": members}
    (long_network if duration > limit_s else short_network).append(record)
write_jsonl(out / "persistent_network_intervals.jsonl", [
    {k:v for k,v in x.items() if k != "_members"} for x in long_network])

predictions = []
for i, segment in enumerate(short_network, 1):
    members = segment["_members"]
    predictions.append({
        "incident_id": f"detected-dev-{i:04d}",
        "start_time": segment["start_time"], "end_time": segment["end_time"],
        "score": float(max(x.get("max_device_score", 0.0) for x in members)),
        "detector": "moment_1_large_zero_shot_reconstruction_dev_filtered",
        "config_sha256": config_sha})
write_jsonl(out / "detected_incidents.jsonl", predictions)
prediction_sha = sha256(out / "detected_incidents.jsonl")
(out / "detected_incidents.sha256").write_text(
    f"{prediction_sha}  {out/'detected_incidents.jsonl'}\n", encoding="utf-8")

truth_times, truth_labels = load_jsonl(Path(a.truth_times)), load_jsonl(Path(a.truth_labels))
result = score(predictions, truth_times, 33)
result["classification"] = "development_validation_not_blind"
(out / "detection_score.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

label_by_index = {i:x for i,x in enumerate(truth_labels)}
unmatched_types = Counter(label_by_index[i]["fault_type"] for i in result["unmatched_truth"])
unmatched_prediction_ids = set(result["unmatched_predictions"])
fp_device_counts, fp_metric_counts = Counter(), Counter()
for index in unmatched_prediction_ids:
    for member in short_network[index]["_members"]:
        fp_device_counts[member["device"]] += 1
        fp_metric_counts.update(member.get("evidence_metrics", []))

with (out / "persistent_interval_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "level","id","start_time","end_time","duration_minutes","devices"])
    writer.writeheader()
    for i, x in enumerate(long_device, 1):
        writer.writerow({"level":"device","id":f"persistent-device-{i:04d}",
                         "start_time":x["start_time"],"end_time":x["end_time"],
                         "duration_minutes":x["duration_minutes"],"devices":x["device_id"]})
    for i, x in enumerate(long_network, 1):
        writer.writerow({"level":"network","id":f"persistent-network-{i:04d}",
                         "start_time":x["start_time"],"end_time":x["end_time"],
                         "duration_minutes":x["duration_minutes"],
                         "devices":"|".join(x["devices"])})

manifest = {
    **filter_config,
    "point_scores_source": str(points_path.resolve()),
    "point_scores_sha256": sha256(points_path),
    "source_validation": str(src.resolve()),
    "device_long_interval_count": len(long_device),
    "network_long_interval_count": len(long_network),
    "excluded_device_time_ranges": [
        {"device_id":x["device_id"],"start_time":x["start_time"],"end_time":x["end_time"]}
        for x in long_device],
    "remaining_short_event_count": len(predictions),
    "MOMENT_invoked": False, "GPU_used": False,
    "thresholds_modified": False, "point_scores_modified": False,
    "prediction_sha256": prediction_sha, "config_sha256": config_sha,
}
(out / "filtering_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

report = f"""# MOMENT 35-minute filtering — development validation

This is **development validation after truth access**, not a new blind result. MOMENT and
GPU were not used; frozen point scores and thresholds were not modified.

## Filtering and data quality

- Persistent device intervals (>35m): **{len(long_device)}**, across
  **{len(device_long_counts)} devices** and **{len(role_long_counts)} roles**.
- Repeated persistent devices: {json.dumps(dict(device_long_counts.most_common()), ensure_ascii=False)}.
- Persistent network intervals after UTC OR: **{len(long_network)}**.
- Top 20 contributing metrics: {json.dumps(metric_frequency.most_common(20), ensure_ascii=False)}.
- Top-metric occurrences near-constant/MAD=0/IQR=0:
  **{near_constant_metric_occurrences}/{mad0_occurrences}/{iqr0_occurrences}**.
- Role concentration: {json.dumps(dict(role_long_counts.most_common()), ensure_ascii=False)}.
- Data-source concentration: {json.dumps(dict(source_long_counts.most_common()), ensure_ascii=False)}.
- All joins use real UTC timestamps and direct blank gaps <=120s. A >120s blank always
  starts a new network segment; no row indices are used.

## Filtered detection

- Remaining short predictions: **{len(predictions)}**.
- TP/FP/FN: **{result['TP']}/{result['FP']}/{result['FN']}**.
- Precision/Recall/F1: **{result['Precision']:.6f}/{result['Recall']:.6f}/{result['F1']:.6f}**.
- alpha: **{result['alpha']:.6f}**; detection/30: **{result['detection_score_30']:.6f}**.
- Unmatched truth types: {json.dumps(dict(unmatched_types.most_common()), ensure_ascii=False)}.
- FP device concentration: {json.dumps(fp_device_counts.most_common(20), ensure_ascii=False)}.
- FP metric concentration: {json.dumps(fp_metric_counts.most_common(20), ensure_ascii=False)}.

Long intervals are excluded from short-event FP accounting exactly as specified and remain
fully traceable in the persistent interval artifacts.
"""
(out / "report.md").write_text(report, encoding="utf-8")
