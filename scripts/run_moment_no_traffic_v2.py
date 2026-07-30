#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bian.detection.aggregation import (
    merge_intervals_by_timestamp, sampled_hysteresis_intervals)
from bian.detection.moment import _reconstruct, load_model

CAL_START = pd.Timestamp("2026-07-28T04:00:00Z")
CAL_END = pd.Timestamp("2026-07-28T05:30:00Z")

p=argparse.ArgumentParser()
p.add_argument("--input",required=True)
p.add_argument("--output",required=True)
p.add_argument("--config",required=True)
p.add_argument("--smoke-end")
a=p.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0","1","4","5","6"}:
    raise SystemExit("one allowed GPU must be selected")
cfg=json.loads(Path(a.config).read_text())
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
frame=pd.read_parquet(a.input)
frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
if a.smoke_end:
    frame=frame[frame.timestamp < pd.Timestamp(a.smoke_end)]

# Per-device+metric calibration normalization, while reconstruction arrays remain
# strictly separated by continuous_segment_id.
calibration={}
for key,g in frame.groupby(["device_id","metric"],sort=False):
    cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].value.dropna().to_numpy(float)
    if not len(cal): continue
    center=float(np.median(cal)); mad=float(np.median(np.abs(cal-center)))*1.4826
    q1,q3=np.quantile(cal,[.25,.75]); iqr=float((q3-q1)/1.349)
    metric_type=g.metric_type.iloc[0]
    scale=mad if mad>0 else iqr
    if scale<=0 or not np.isfinite(scale):
        scale=1.0 if metric_type=="sparse_counter" else max(abs(center)*.01,1.0)
    calibration[key]=(center,scale,float(np.median(g[
        (g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].activity_value.dropna())))

groups=[]; arrays=[]; binary_groups=[]
for key,g in frame.groupby(["device_id","metric","continuous_segment_id"],sort=True):
    g=g.sort_values("timestamp")
    dm=key[:2]
    if dm not in calibration: continue
    if g.metric_type.iloc[0]=="binary_state":
        binary_groups.append((key,g))
        continue
    center,scale,_=calibration[dm]
    arrays.append((g.value.to_numpy(float)-center)/scale)
    groups.append((key,g))

model,seq_len=load_model(cfg["model_id"],cfg["revision"],"cuda:0")
patch_len=int(getattr(model.config,"patch_len",8))
torch.cuda.reset_peak_memory_stats()
errors=_reconstruct(model,seq_len,arrays,cfg["batch_size"],"cuda:0")
peak=int(torch.cuda.max_memory_allocated())

score_frames=[]
for (key,g),err in zip(groups,errors):
    dev,metric,_=key; mtype=g.metric_type.iloc[0]
    gate=np.ones(len(g),dtype=bool)
    if mtype=="sparse_counter":
        baseline=calibration[(dev,metric)][2]
        gate=np.abs(g.activity_value.to_numpy(float)-baseline)>0
        err=np.where(gate,err,0.0)
    block=g[["timestamp","base_metric_family","raw_value"]].copy()
    block["device_id"]=dev; block["metric"]=metric; block["metric_type"]=mtype
    block["score"]=err; block["activity_gate"]=gate
    block["continuous_segment_id"]=key[2]
    score_frames.append(block[np.isfinite(block.score)])
for key,g in binary_groups:
    dev,metric,_=key
    baseline=float(pd.Series(g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].raw_value).mode().iloc[0])
    block=g[["timestamp","base_metric_family","raw_value"]].copy()
    block["device_id"]=dev; block["metric"]=metric; block["metric_type"]="binary_state"
    block["score"]=0.0; block["activity_gate"]=block.raw_value!=baseline
    block["continuous_segment_id"]=key[2]
    score_frames.append(block)
scores=pd.concat(score_frames,ignore_index=True)
threshold_rows=[]
for key,g in scores[scores.metric_type!="binary_state"].groupby(["device_id","metric"]):
    cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].score.to_numpy(float)
    if len(cal):
        threshold_rows.append({"device_id":key[0],"metric":key[1],
                               "high":float(np.quantile(cal,.995)),
                               "low":float(np.quantile(cal,.99))})
scores=scores.merge(pd.DataFrame(threshold_rows),on=["device_id","metric"],how="left")
scores.to_parquet(out/"metric_point_scores.parquet",index=False,compression="zstd")

device_intervals=[]; persistent=[]
for dev,d in scores.groupby("device_id",sort=True):
    timestamps=np.array(sorted(d.timestamp.unique()))
    trigger=pd.Series(False,index=pd.DatetimeIndex(timestamps))
    # A: at least two distinct base families exceed high.
    cont=d[d.metric_type.isin(["dense_continuous","cumulative_counter"])]
    if len(cont):
        hit=cont[cont.score>cont.high].groupby("timestamp").base_metric_family.nunique()
        trigger |= hit.reindex(trigger.index,fill_value=0)>=2
    # C: sparse activity gate plus high score.
    sparse=d[(d.metric_type=="sparse_counter")&d.activity_gate&(d.score>d.high)]
    if len(sparse):
        trigger.loc[trigger.index.intersection(sparse.timestamp)] = True
    # B: only true raw binary departure from calibration mode.
    binary=d[d.metric_type=="binary_state"]
    if len(binary):
        abnormal=binary[binary.activity_gate].groupby("timestamp").size()>0
        trigger.loc[trigger.index.intersection(abnormal.index)] = True
    spans=sampled_hysteresis_intervals(
        trigger.to_numpy(),trigger.to_numpy(),trigger.index.to_numpy(),CAL_END,2,2)
    spans=merge_intervals_by_timestamp(spans,120)
    for i,(start,end) in enumerate(spans,1):
        rec={"interval_id":f"{dev}:{i}","device_id":dev,
             "start_time":start.isoformat(),"end_time":end.isoformat(),
             "duration_minutes":(end-start).total_seconds()/60}
        (persistent if rec["duration_minutes"]>35 else device_intervals).append(
            {**rec,"level":"device"} if rec["duration_minutes"]>35 else rec)

# Network OR of short device intervals on actual UTC timestamps.
network=[]
for item in sorted(device_intervals,key=lambda x:(x["start_time"],x["end_time"],x["device_id"])):
    start,end=pd.Timestamp(item["start_time"]),pd.Timestamp(item["end_time"])
    if network and (start-network[-1]["end"]).total_seconds()<=120:
        network[-1]["end"]=max(network[-1]["end"],end)
    else: network.append({"start":start,"end":end})
events=[]
for x in network:
    duration=(x["end"]-x["start"]).total_seconds()/60
    if duration>35:
        persistent.append({"level":"network","device_id":"","start_time":x["start"].isoformat(),
                           "end_time":x["end"].isoformat(),"duration_minutes":duration})
    else:
        events.append({"incident_id":f"moment-no-traffic-{len(events)+1:04d}",
                       "start_time":x["start"].isoformat(),"end_time":x["end"].isoformat()})
with (out/"device_intervals.jsonl").open("w") as f:
    for x in device_intervals:f.write(json.dumps(x)+"\n")
with (out/"persistent_long_intervals.jsonl").open("w") as f:
    for x in persistent:f.write(json.dumps(x)+"\n")
with (out/"detected_incidents.jsonl").open("w") as f:
    for x in events:f.write(json.dumps(x)+"\n")
summary={
    "development_experiment":True,"model_eval":not model.training,
    "window_length":seq_len,"patch_length":patch_len,"stride":seq_len//2,
    "mask":"1 for finite observed positions; 0 for right padding/missing",
    "error_positions":"finite observed positions only",
    "overlap_aggregation":"arithmetic mean in deterministic window order",
    "nan_scores":int(scores.score.isna().sum()),"inf_scores":int(np.isinf(scores.score).sum()),
    "near_constant_protection":"IQR fallback; fixed scale 1 for sparse, max(1,1% center) otherwise",
    "traffic_series":0,"metric_types":scores.groupby("metric_type")[["device_id","metric"]].apply(
        lambda x:int(x.drop_duplicates().shape[0])).to_dict(),
    "point_scores":len(scores),"device_intervals":len(device_intervals),
    "persistent_long_intervals":len(persistent),"events":len(events),
    "calibration_event_points":0,"cross_device_windows":0,"cross_segment_windows":0,
    "peak_gpu_memory_bytes":peak}
(out/"run_summary.json").write_text(json.dumps(summary,indent=2))
del model
torch.cuda.empty_cache()
print(json.dumps(summary,indent=2))
