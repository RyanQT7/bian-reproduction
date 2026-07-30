from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .aggregation import (
    complete_linkage_clusters,
    merge_intervals_by_timestamp,
    robust_scale,
    sampled_hysteresis_intervals,
    temporal_gap_seconds,
)


def load_model(model_id: str, revision: str, device: str):
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        model_id, revision=revision, model_kwargs={"task_name": "reconstruction"})
    model.init()
    model.to(device)
    model.eval()
    seq_len = int(getattr(model.config, "seq_len", getattr(model.config, "context_length", 512)))
    return model, seq_len


def _reconstruct(model, seq_len: int, arrays: list[np.ndarray], batch_size: int,
                 device: str) -> list[np.ndarray]:
    stride = max(1, seq_len // 2)
    jobs, meta = [], []
    sums = [np.zeros(len(x), dtype=np.float64) for x in arrays]
    counts = [np.zeros(len(x), dtype=np.int32) for x in arrays]
    for si, x in enumerate(arrays):
        starts = list(range(0, max(1, len(x)-seq_len+1), stride))
        last = max(0, len(x)-seq_len)
        if not starts or starts[-1] != last:
            starts.append(last)
        for start in starts:
            n = min(seq_len, len(x)-start)
            data = np.zeros(seq_len, np.float32)
            mask = np.zeros(seq_len, np.int64)
            data[:n], mask[:n] = x[start:start+n], np.isfinite(x[start:start+n])
            data[~np.isfinite(data)] = 0
            jobs.append((data, mask)); meta.append((si, start, n))
    with torch.no_grad():
        for off in range(0, len(jobs), batch_size):
            block = jobs[off:off+batch_size]
            x = torch.tensor(np.stack([z[0] for z in block])[:, None, :], device=device)
            mask = torch.tensor(np.stack([z[1] for z in block]), device=device)
            output = model(x_enc=x, input_mask=mask)
            recon = output.reconstruction.detach().float().cpu().numpy()[:, 0, :]
            for k, prediction in enumerate(recon):
                si, start, n = meta[off+k]
                valid = block[k][1][:n].astype(bool)
                err = np.abs(prediction[:n] - block[k][0][:n])
                ix = np.arange(start, start+n)[valid]
                sums[si][ix] += err[valid]; counts[si][ix] += 1
    return [np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0) for s, c in zip(sums, counts)]


def detect(parquet: str | Path, output: str | Path, config: dict, device: str,
           smoke_end: str | None = None) -> dict:
    torch.cuda.reset_peak_memory_stats()
    frame = pd.read_parquet(parquet)
    if smoke_end:
        frame = frame[frame.timestamp < pd.Timestamp(smoke_end)]
    calibration_end = pd.Timestamp(config["calibration_end"])
    series, values, metadata = [], [], []
    for (dev, metric), group in frame.groupby(["device", "metric"], sort=True):
        group = group.sort_values("timestamp").drop_duplicates("timestamp")
        cal = group.loc[group.timestamp < calibration_end, "value"].to_numpy(float)
        binary = bool(group.is_binary.max())
        center, scale = robust_scale(cal, binary, config["minimum_scale"])
        series.append(group.timestamp.to_numpy())
        values.append((group.value.to_numpy(float)-center)/scale)
        metadata.append((dev, metric, binary))
    model, seq_len = load_model(config["model_id"], config["revision"], device)
    errors = _reconstruct(model, seq_len, values, config["batch_size"], device)
    records = []
    for ts, err, (dev, metric, binary) in zip(series, errors, metadata):
        cal_mask = pd.DatetimeIndex(ts) < calibration_end
        cal = err[np.asarray(cal_mask)]
        finite = cal[np.isfinite(cal)]
        if not len(finite):
            continue
        high, low = np.quantile(finite, [config["high_quantile"], config["low_quantile"]])
        for t, e in zip(ts, err):
            if np.isfinite(e):
                records.append((t, dev, metric, e, high, low, binary))
    scores = pd.DataFrame(records, columns=["timestamp","device","metric","score","high","low","binary"])
    artifact_dir = Path(output).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(artifact_dir / "metric_point_scores.parquet", index=False,
                      compression="zstd")
    metric_interval_records = []
    for (dev, metric), group in scores.groupby(["device", "metric"], sort=True):
        group = group.sort_values("timestamp")
        hit = group.score.to_numpy() >= group.high.to_numpy()
        start = None
        for i, value in enumerate(hit):
            if value and start is None:
                start = i
            if start is not None and (not value or i == len(hit)-1):
                stop = i-1 if not value else i
                metric_interval_records.append({
                    "device": dev, "metric": metric,
                    "start_time": pd.Timestamp(group.timestamp.iloc[start]).isoformat(),
                    "end_time": pd.Timestamp(group.timestamp.iloc[stop]).isoformat(),
                    "point_count": stop-start+1,
                    "max_score": float(group.score.iloc[start:stop+1].max()),
                    "high": float(group.high.iloc[0]), "formation": "contiguous_points_score_ge_high",
                })
                start = None
    with (artifact_dir / "metric_intervals.jsonl").open("w", encoding="utf-8") as f:
        for record in metric_interval_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    device_spans, before_records, after_records = [], [], []
    detection_start = pd.Timestamp(config.get("detection_start", config["calibration_end"]))
    same_device_gap = float(config.get("same_device_merge_gap_seconds", 120))
    cross_device_gap = float(config.get("cross_device_compatibility_seconds", 120))
    for dev, d in scores.groupby("device"):
        pivot = d.pivot_table(index="timestamp", columns="metric", values="score", aggfunc="mean")
        high = d.groupby("metric").high.first().reindex(pivot.columns)
        low = d.groupby("metric").low.first().reindex(pivot.columns)
        ratios = pivot.divide(high.replace(0, config["minimum_scale"]), axis=1)
        above = (ratios >= 1).sum(axis=1)
        extreme = ratios.max(axis=1) >= 2.5
        trigger = (above >= 2) | extreme
        low_active = (pivot.ge(low, axis=1).sum(axis=1) >= 2) | extreme
        top3 = ratios.apply(lambda row: row.nlargest(min(3, row.notna().sum())).mean(), axis=1)
        before = sampled_hysteresis_intervals(
            trigger.to_numpy(), low_active.to_numpy(), pivot.index.to_numpy(),
            detection_start, onset_points=2, clear_points=2)
        after = merge_intervals_by_timestamp(before, same_device_gap)
        for i, (start, end) in enumerate(before, 1):
            before_records.append({
                "interval_id": f"{dev}:before:{i}", "device": dev,
                "start_time": start.isoformat(), "end_time": end.isoformat(),
                "stage": "after_onset_and_clear_before_same_device_merge"})
        for i, (start, end) in enumerate(after, 1):
            segment = (pivot.index >= start) & (pivot.index <= end)
            indicators = ratios.loc[segment].max().nlargest(3).index.tolist()
            interval_id = f"{dev}:after:{i}"
            member_before = [
                x["interval_id"] for x in before_records
                if x["device"] == dev and pd.Timestamp(x["start_time"]) >= start
                and pd.Timestamp(x["end_time"]) <= end]
            span = {"interval_id": interval_id, "device": dev, "start": start, "end": end,
                                 "max_device_score": float(top3.loc[segment].max()),
                                 "top_indicators": indicators}
            device_spans.append(span)
            after_records.append({
                "interval_id": interval_id, "device": dev, "start_time": start.isoformat(),
                "end_time": end.isoformat(), "member_before_merge_ids": member_before,
                "max_device_score": span["max_device_score"], "top_indicators": indicators})
    for name, rows in (
        ("device_intervals_before_merge.jsonl", before_records),
        ("device_intervals_after_same_device_merge.jsonl", after_records),
    ):
        with (artifact_dir / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    clusters = complete_linkage_clusters(device_spans, cross_device_gap)
    with (artifact_dir / "cross_device_merge_lineage.jsonl").open("w", encoding="utf-8") as f:
        for cluster in clusters:
            lineage = {
                "cluster_id": cluster["cluster_id"],
                "boundary_after": {
                    "start": cluster["start"].isoformat(), "end": cluster["end"].isoformat()},
                "member_intervals": [
                    {"interval_id": x["interval_id"], "device": x["device"],
                     "start_time": x["start"].isoformat(), "end_time": x["end"].isoformat()}
                    for x in cluster["members"]],
                "joins": cluster["joins"],
            }
            f.write(json.dumps(lineage, ensure_ascii=False) + "\n")
    no_transitive_chain = all(
        temporal_gap_seconds(a, b) <= cross_device_gap
        for cluster in clusters for i, a in enumerate(cluster["members"])
        for b in cluster["members"][i+1:])
    output = Path(output)
    with output.open("w", encoding="utf-8") as f:
        for i, cluster in enumerate(clusters, 1):
            members = cluster["members"]
            record = {"incident_id": f"moment-{i:04d}", "cluster_id": cluster["cluster_id"],
                      "start_time": cluster["start"].isoformat(), "end_time": cluster["end"].isoformat(),
                      "devices": sorted({x["device"] for x in members}),
                      "top_indicators": sorted({m for x in members for m in x["top_indicators"]}),
                      "member_interval_ids": [x["interval_id"] for x in members],
                      "max_device_score": max(x["max_device_score"] for x in members)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    peak = int(torch.cuda.max_memory_allocated())
    del model
    torch.cuda.empty_cache()
    result = {"predictions": len(clusters), "series": len(series), "window_length": seq_len,
              "metric_intervals": len(metric_interval_records),
              "device_intervals_before_merge": len(before_records),
              "device_intervals_after_same_device_merge": len(after_records),
              "no_cross_device_transitive_chain": no_transitive_chain,
              "peak_gpu_memory_bytes": peak, "model_eval": True}
    if smoke_end:
        Path(output).with_name("smoke_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result
