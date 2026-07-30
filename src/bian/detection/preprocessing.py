from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .guard import assert_blind_path

TIME_NAMES = ("timestamp", "timestamp_utc", "minute_utc")
ID_COLUMNS = {
    "timestamp", "timestamp_utc", "minute_utc", "prometheus_sample_time_utc", "region",
    "region_code", "node", "node_key", "node_type", "interface_id", "if_role", "target_id",
    "exporter_type", "metric_name", "label", "id", "series_key", "flow_type", "source_region",
    "source_ip", "target_region", "target_domain", "protocol",
}
EXCLUDED_FILES = ("syslog", "5tuple")
COUNTER_RE = re.compile(r"(?:_total|_count|_bytes|_packets|_errors?|_drops?|carrier_changes)$")
BINARY_RE = re.compile(r"(?:_up|_success|_active|_state|status)$")


def audit_inputs(input_dir: str | Path) -> dict:
    root = assert_blind_path(input_dir)
    included, excluded = [], []
    for path in sorted(root.rglob("*.csv")):
        rel = str(path.relative_to(root))
        if any(x in path.name.lower() for x in EXCLUDED_FILES):
            excluded.append({"file": rel, "reason": "text/syslog or raw five-tuple"})
            continue
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            header = next(csv.reader(f), [])
        time_col = next((x for x in TIME_NAMES if x in header), None)
        numeric_candidates = [x for x in header if x not in ID_COLUMNS]
        included.append({"file": rel, "time_column": time_col, "candidate_fields": numeric_candidates})
    return {"input_root": str(root), "included": included, "excluded": excluded}


def _device(frame: pd.DataFrame, filename: str) -> pd.Series:
    if "node" in frame:
        return frame["node"].astype(str)
    if "node_key" in frame:
        return frame["node_key"].astype(str)
    if "source_ip" in frame:
        region = frame.get("source_region", pd.Series("", index=frame.index)).astype(str)
        return region + ":" + frame["source_ip"].astype(str)
    if "target_id" in frame:
        return frame["target_id"].astype(str)
    return pd.Series(filename, index=frame.index)


def preprocess(input_dir: str | Path, output_dir: str | Path, manifest: dict,
               start: str | None = None, end: str | None = None) -> dict:
    root, out = assert_blind_path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    field_manifest = []
    for spec in manifest["included"]:
        path = root / spec["file"]
        tcol = spec["time_column"]
        if not tcol:
            continue
        for chunk in pd.read_csv(path, chunksize=150_000, low_memory=False, encoding="utf-8-sig"):
            ts = pd.to_datetime(chunk[tcol], utc=True, errors="coerce")
            keep = ts.notna()
            if start:
                keep &= ts >= pd.Timestamp(start)
            if end:
                keep &= ts < pd.Timestamp(end)
            chunk, ts = chunk.loc[keep], ts.loc[keep]
            if chunk.empty:
                continue
            dev = _device(chunk, path.name)
            group_ids = [ts.rename("timestamp"), dev.rename("device")]
            for col in spec["candidate_fields"]:
                if col not in chunk:
                    continue
                vals = pd.to_numeric(chunk[col], errors="coerce")
                if vals.notna().sum() == 0:
                    continue
                base_metric = f"{path.stem.split('_2026')[0]}::{col}"
                if col == "value" and "metric_name" in chunk:
                    suffix = chunk["metric_name"].astype(str)
                    if "label" in chunk:
                        suffix = suffix + "::" + chunk["label"].astype(str)
                    metric_series = base_metric + "::" + suffix
                else:
                    metric_series = pd.Series(base_metric, index=chunk.index)
                temp = pd.DataFrame({"timestamp": ts, "device": dev, "value": vals,
                                     "metric": metric_series}).dropna()
                # Multiple interfaces/flows at a device/time are deterministically aggregated.
                temp = temp.groupby(["timestamp", "device", "metric"], as_index=False)["value"].mean()
                temp["is_counter"] = bool(COUNTER_RE.search(col))
                temp["is_binary_hint"] = bool(BINARY_RE.search(col))
                pieces.append(temp)
                field_manifest.append((spec["file"], col, base_metric))
    if not pieces:
        raise ValueError("no numeric time series parsed")
    long = pd.concat(pieces, ignore_index=True)
    long = long.groupby(["timestamp", "device", "metric"], as_index=False).agg(
        value=("value", "mean"), is_counter=("is_counter", "max"),
        is_binary_hint=("is_binary_hint", "max"))
    long.sort_values(["device", "metric", "timestamp"], inplace=True)
    counters = long["is_counter"]
    diffs = long.groupby(["device", "metric"], sort=False)["value"].diff()
    dt = long.groupby(["device", "metric"], sort=False)["timestamp"].diff().dt.total_seconds()
    rate = diffs / dt.replace(0, np.nan)
    long.loc[counters, "value"] = rate.loc[counters].clip(lower=0)
    long = long.dropna(subset=["value"])
    binary_observed = long.groupby(["device", "metric"])["value"].transform(
        lambda x: x.dropna().isin([0, 1]).all() and x.nunique() <= 2)
    long["is_binary"] = long["is_binary_hint"] | binary_observed
    long.to_parquet(out / "series.parquet", index=False)
    fields = [{"file": a, "field": b, "metric": c} for a, b, c in sorted(set(field_manifest))]
    (out / "field_manifest.json").write_text(json.dumps(fields, indent=2), encoding="utf-8")
    summary = {"rows": len(long), "series": int(long.groupby(["device", "metric"]).ngroups),
               "devices": int(long.device.nunique()), "start": str(long.timestamp.min()),
               "end": str(long.timestamp.max())}
    (out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
