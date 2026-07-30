#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from bian.detection.aggregation import merge_intervals_by_timestamp,sampled_hysteresis_intervals
from bian.detection.moment import _reconstruct,load_model
from bian.detection.screening_v3 import robust_score_threshold

CAL_START=pd.Timestamp("2026-07-28T04:00:00Z"); CAL_END=pd.Timestamp("2026-07-28T05:30:00Z")
p=argparse.ArgumentParser(); p.add_argument("--input-dir",required=True); p.add_argument("--output",required=True)
p.add_argument("--config",required=True); p.add_argument("--smoke-end"); a=p.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0","1","4","5","6"}: raise SystemExit("select one allowed GPU")
inp,out=Path(a.input_dir),Path(a.output); out.mkdir(parents=True,exist_ok=True)
cfg=json.loads(Path(a.config).read_text()); frame=pd.read_parquet(inp/"screened_series.parquet")
frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
binary=pd.read_parquet(inp/"rule_based_series.parquet"); binary.timestamp=pd.to_datetime(binary.timestamp,utc=True)
if a.smoke_end:
    frame=frame[frame.timestamp<pd.Timestamp(a.smoke_end)]
    binary=binary[binary.timestamp<pd.Timestamp(a.smoke_end)]
calibration={}
for key,g in frame.groupby(["device_id","metric"],sort=False):
    cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].value.dropna().to_numpy(float)
    if len(cal)<60: continue
    center=float(np.median(cal)); mad=1.4826*float(np.median(np.abs(cal-center)))
    q1,q3=np.quantile(cal,[.25,.75]); scale=max(mad,float((q3-q1)/1.349))
    if g.metric_type.iloc[0]=="sparse_counter" and scale<=0: scale=1.0
    if scale<=0: continue
    calibration[key]=(center,scale)
groups=[]; arrays=[]
for key,g in frame.groupby(["device_id","metric","continuous_segment_id"],sort=True):
    if key[:2] not in calibration: continue
    g=g.sort_values("timestamp"); center,scale=calibration[key[:2]]
    groups.append((key,g)); arrays.append((g.value.to_numpy(float)-center)/scale)
model,seq_len=load_model(cfg["model_id"],cfg["revision"],"cuda:0")
patch_len=int(getattr(model.config,"patch_len",8)); torch.cuda.reset_peak_memory_stats()
errors=_reconstruct(model,seq_len,arrays,cfg["batch_size"],"cuda:0")
peak=int(torch.cuda.max_memory_allocated()); blocks=[]
for (key,g),err in zip(groups,errors):
    block=g[["timestamp","device_id","metric","base_metric_family","metric_type",
             "activity_value","raw_value","continuous_segment_id"]].copy()
    if g.metric_type.iloc[0]=="sparse_counter":
        gate=g.activity_value.to_numpy(float)>0; err=np.where(gate,err,0.0)
    else: gate=np.ones(len(g),bool)
    block["score"]=err; block["activity_gate"]=gate
    blocks.append(block[np.isfinite(block.score)])
scores=pd.concat(blocks,ignore_index=True)
threshold_rows=[]
for key,g in scores.groupby(["device_id","metric"],sort=True):
    cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].score.to_numpy(float)
    result=robust_score_threshold(cal)
    row={"device_id":key[0],"metric":key[1],"trigger_eligible":result is not None,
         "exclusion_reason":"" if result else "unreliable_score_scale_or_insufficient_calibration"}
    if result:
        high,low,stats=result; row.update(stats); row.update({"high_threshold":high,"low_threshold":low})
    threshold_rows.append(row)
thresholds=pd.DataFrame(threshold_rows)
thresholds.to_csv(out/"metric_thresholds.csv",index=False)
scores=scores.merge(thresholds[["device_id","metric","trigger_eligible","high_threshold","low_threshold"]],
                    on=["device_id","metric"],how="left")
scores.to_parquet(out/"metric_point_scores.parquet",index=False,compression="zstd")
eligible=scores[(scores.trigger_eligible==True)&(scores.metric_type!="sparse_counter")]
device_intervals=[]; persistent=[]
devices=sorted(set(scores.device_id)|set(binary.device_id))
for dev in devices:
    times=pd.date_range(CAL_START, pd.Timestamp(a.smoke_end) if a.smoke_end else pd.Timestamp(
        "2026-07-29T04:00:00Z"),freq="min",inclusive="left")
    trigger=pd.Series(False,index=times)
    d=eligible[eligible.device_id==dev]
    if len(d):
        hit=d[d.score>d.high_threshold].groupby("timestamp").base_metric_family.nunique()
        trigger |= hit.reindex(times,fill_value=0)>=2
    b=binary[binary.device_id==dev]
    for metric,g in b.groupby("metric",sort=False):
        cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].raw_value.dropna()
        if cal.empty: continue
        normal=float(cal.mode().iloc[0]); abnormal=g[g.raw_value!=normal].groupby("timestamp").size()>0
        trigger |= abnormal.reindex(times,fill_value=False)
    spans=sampled_hysteresis_intervals(trigger.to_numpy(),trigger.to_numpy(),times.to_numpy(),CAL_END,2,2)
    spans=merge_intervals_by_timestamp(spans,120)
    for i,(start,end) in enumerate(spans,1):
        rec={"interval_id":f"{dev}:{i}","device_id":dev,"start_time":start.isoformat(),
             "end_time":end.isoformat(),"duration_minutes":(end-start).total_seconds()/60}
        if rec["duration_minutes"]>35: persistent.append({"level":"device",**rec})
        else: device_intervals.append(rec)
network=[]
for x in sorted(device_intervals,key=lambda z:(z["start_time"],z["end_time"],z["device_id"])):
    start,end=pd.Timestamp(x["start_time"]),pd.Timestamp(x["end_time"])
    if network and (start-network[-1][1]).total_seconds()<=120: network[-1]=(network[-1][0],max(network[-1][1],end))
    else: network.append((start,end))
events=[]
for start,end in network:
    duration=(end-start).total_seconds()/60
    if duration>35: persistent.append({"level":"network","device_id":"","start_time":start.isoformat(),
                                       "end_time":end.isoformat(),"duration_minutes":duration})
    else: events.append({"incident_id":f"moment-screened-{len(events)+1:04d}",
                         "start_time":start.isoformat(),"end_time":end.isoformat()})
for name,rows in [("device_intervals.jsonl",device_intervals),("detected_incidents.jsonl",events),
                  ("persistent_long_intervals.jsonl",persistent)]:
    with (out/name).open("w") as f:
        for x in rows:f.write(json.dumps(x)+"\n")
summary={"development_experiment":True,"model_eval":not model.training,"window_length":seq_len,
 "patch_length":patch_len,"stride":seq_len//2,"point_scores":len(scores),
 "threshold_eligible_metrics":int(thresholds.trigger_eligible.sum()),"threshold_excluded_metrics":int((~thresholds.trigger_eligible).sum()),
 "device_events":len(device_intervals),"network_short_events":len(events),"persistent_long_intervals":len(persistent),
 "nan_scores":int(scores.score.isna().sum()),"inf_scores":int(np.isinf(scores.score).sum()),
 "calibration_event_points":0,"traffic_series":0,"peak_gpu_memory_bytes":peak}
(out/"run_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
del model; torch.cuda.empty_cache()
