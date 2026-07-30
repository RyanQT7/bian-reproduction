from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .aggregation import intervals, robust_scale


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
    if smoke_end:
        result = {"series": len(series), "window_length": seq_len,
                  "nan_errors": int(sum(np.isnan(x).sum() for x in errors)),
                  "model_eval": not model.training, "device": device}
        del model
        torch.cuda.empty_cache()
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
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
    device_spans = []
    for dev, d in scores.groupby("device"):
        pivot = d.pivot_table(index="timestamp", columns="metric", values="score", aggfunc="mean")
        high = d.groupby("metric").high.first().reindex(pivot.columns)
        binary = d.groupby("metric").binary.first().reindex(pivot.columns)
        ratios = pivot.divide(high.replace(0, config["minimum_scale"]), axis=1)
        above = (ratios >= 1).sum(axis=1)
        extreme = ratios.max(axis=1) >= 2.5
        flips = pivot.loc[:, binary].diff().abs().gt(0).rolling(2, min_periods=2).sum().ge(2).any(axis=1) if binary.any() else pd.Series(False, index=pivot.index)
        trigger = (above >= 2) | extreme | flips
        top3 = ratios.apply(lambda row: row.nlargest(min(3, row.notna().sum())).mean(), axis=1)
        for start, end in intervals(trigger.to_numpy(), pivot.index.to_numpy(),
                                    config["onset_seconds"], config["clear_seconds"],
                                    config["merge_gap_seconds"]):
            segment = (pivot.index >= start) & (pivot.index <= end)
            indicators = ratios.loc[segment].max().nlargest(3).index.tolist()
            device_spans.append({"device": dev, "start": start, "end": end,
                                 "max_device_score": float(top3.loc[segment].max()),
                                 "top_indicators": indicators})
    # Deterministic temporal union: overlapping/nearby device incidents form network events.
    events = []
    for span in sorted(device_spans, key=lambda x: (x["start"], x["end"], x["device"])):
        if events and span["start"] <= events[-1]["end"] + pd.Timedelta(seconds=60):
            events[-1]["end"] = max(events[-1]["end"], span["end"])
            events[-1]["devices"].append(span["device"])
            events[-1]["top_indicators"] = sorted(set(events[-1]["top_indicators"] + span["top_indicators"]))
        else:
            events.append({"start": span["start"], "end": span["end"], "devices": [span["device"]],
                           "top_indicators": span["top_indicators"]})
    output = Path(output)
    with output.open("w", encoding="utf-8") as f:
        for i, event in enumerate(events, 1):
            record = {"incident_id": f"moment-{i:04d}", "start_time": event["start"].isoformat(),
                      "end_time": event["end"].isoformat(), "devices": sorted(set(event["devices"])),
                      "top_indicators": event["top_indicators"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()
    return {"predictions": len(events), "series": len(series), "window_length": seq_len}
