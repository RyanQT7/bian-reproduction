from __future__ import annotations

import numpy as np
import pandas as pd


def robust_scale(values: np.ndarray, binary: bool, minimum: float = 1e-6) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    center = float(np.median(finite)) if len(finite) else 0.0
    if binary:
        return center, 1.0
    mad = float(np.median(np.abs(finite - center))) * 1.4826
    if mad <= minimum:
        q1, q3 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 0.0)
        mad = max(float((q3 - q1) / 1.349), minimum)
    return center, mad


def intervals(mask: np.ndarray, timestamps: np.ndarray, onset_s: float, clear_s: float,
              merge_s: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    raw = hysteresis_intervals(mask, timestamps, onset_s, clear_s)
    return merge_same_device_intervals(raw, timestamps, merge_s)


def hysteresis_intervals(mask: np.ndarray, timestamps: np.ndarray, onset_s: float,
                         clear_s: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not len(mask):
        return []
    dt = max(float(np.median(np.diff(timestamps).astype("timedelta64[s]").astype(float))), 1.0)
    onset, clear = max(2, int(np.ceil(onset_s / dt))), max(2, int(np.ceil(clear_s / dt)))
    spans, active, run, start, quiet = [], False, 0, None, 0
    for i, hit in enumerate(mask):
        if hit:
            run += 1; quiet = 0
            if not active and run >= onset:
                active, start = True, i - run + 1
        else:
            run = 0
            if active:
                quiet += 1
                if quiet >= clear:
                    spans.append((pd.Timestamp(timestamps[start]), pd.Timestamp(timestamps[i - quiet])))
                    active, quiet = False, 0
    if active:
        spans.append((pd.Timestamp(timestamps[start]), pd.Timestamp(timestamps[-1])))
    return spans


def merge_same_device_intervals(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]], timestamps: np.ndarray, merge_s: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not spans:
        return []
    dt = max(float(np.median(np.diff(timestamps).astype("timedelta64[s]").astype(float))), 1.0)
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for span in spans:
        if merged and (span[0] - merged[-1][1]).total_seconds() <= max(2 * dt, merge_s):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged


def temporal_gap_seconds(a: dict, b: dict) -> float:
    """Zero for overlap, otherwise the direct boundary-to-boundary gap."""
    if a["end"] < b["start"]:
        return float((b["start"] - a["end"]).total_seconds())
    if b["end"] < a["start"]:
        return float((a["start"] - b["end"]).total_seconds())
    return 0.0


def complete_linkage_clusters(items: list[dict], compatibility_seconds: float) -> list[dict]:
    """Greedy deterministic complete-linkage; every pair must be directly compatible."""
    clusters: list[dict] = []
    ordered = sorted(items, key=lambda x: (x["start"], x["end"], x["device"]))
    for item in ordered:
        compatible = [
            cluster for cluster in clusters
            if all(temporal_gap_seconds(item, member) <= compatibility_seconds
                   for member in cluster["members"])
        ]
        if compatible:
            cluster = compatible[0]
            relations = [
                {"member_interval_id": member["interval_id"],
                 "gap_seconds": temporal_gap_seconds(item, member),
                 "compatible": True}
                for member in cluster["members"]
            ]
            cluster["joins"].append({
                "joining_interval_id": item["interval_id"],
                "join_reason": f"complete_linkage_all_pair_gaps_le_{compatibility_seconds:g}s",
                "direct_temporal_relations": relations,
                "boundary_before": {
                    "start": cluster["start"].isoformat(), "end": cluster["end"].isoformat()},
                "boundary_after": {
                    "start": min(cluster["start"], item["start"]).isoformat(),
                    "end": max(cluster["end"], item["end"]).isoformat()},
            })
            cluster["members"].append(item)
            cluster["start"] = min(cluster["start"], item["start"])
            cluster["end"] = max(cluster["end"], item["end"])
        else:
            clusters.append({
                "cluster_id": f"cluster-{len(clusters)+1:05d}",
                "start": item["start"], "end": item["end"], "members": [item],
                "joins": [{
                    "joining_interval_id": item["interval_id"],
                    "join_reason": "new_cluster_no_complete_linkage_compatible_cluster",
                    "direct_temporal_relations": [],
                    "boundary_before": None,
                    "boundary_after": {
                        "start": item["start"].isoformat(), "end": item["end"].isoformat()},
                }],
            })
    return clusters
