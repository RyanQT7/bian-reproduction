"""Raw-data minute series and causal 5-sigma event extraction.

This detector is deliberately independent of ground truth.  Ground truth is read
only by the downstream gate after event extraction.
"""
from __future__ import annotations
import argparse, csv, json, math, statistics, hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

IDENT = {"timestamp", "region", "node", "node_type", "interface_id", "if_role", "metric", "entity"}

def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def minute(s: str) -> datetime:
    t = ts(s); return t.replace(second=0, microsecond=0)

def num(s):
    try:
        x=float(s); return x if math.isfinite(x) else None
    except Exception: return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--gt",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    out=args.output; (out/"timeseries").mkdir(parents=True,exist_ok=True); (out/"five_sigma_detection").mkdir(exist_ok=True)
    # series[(entity, metric)] -> minute -> values. Same parsing is applied to every region.
    series=defaultdict(lambda: defaultdict(list)); manifest=[]; files=[]
    for f in sorted(args.data_root.glob("*/processed/*.csv")):
        if "netflow_5tuple" in f.name: continue
        files.append(str(f));
        with f.open(newline="", errors="replace") as h:
            rd=csv.DictReader(h); fields=rd.fieldnames or []; idcols=[x for x in ("region","node","interface_id") if x in fields]
            metric_cols=[x for x in fields if x not in IDENT and x not in idcols]
            for row in rd:
                if not row.get("timestamp"): continue
                try: m=minute(row["timestamp"])
                except Exception: continue
                ent="|".join(row.get(x,"") for x in idcols) or f.stem
                for metric in metric_cols:
                    v=num(row.get(metric,""));
                    if v is not None: series[(ent,metric)][m].append(v)
    point_path=out/"five_sigma_detection/point_anomalies.jsonl"; interval_path=out/"five_sigma_detection/metric_anomaly_intervals.jsonl"
    points=[]; intervals=[]; all_anoms=[]; seq_count=0; missing=[]
    cfg={"window_minutes":120,"minimum_valid_samples":60,"threshold_sigma":5.0,"strict_historical":True,"exclude_anomalies_from_baseline":True,"event_freeze_during_anomaly":True,"aggregation":"minute_mean_numeric","excluded_sources":["netflow_5tuple_minute_readable"]}
    for (ent,metric), buckets in sorted(series.items()):
        vals={k:statistics.mean(v) for k,v in buckets.items() if v}; times=sorted(vals); hist=deque(maxlen=120); cur=[]; seq_count+=1
        for t in times:
            x=vals[t]
            if len(hist)>=60:
                mu=statistics.mean(hist); sd=statistics.pstdev(hist); z=(abs(x-mu)/sd if sd>0 else (math.inf if x!=mu else 0.0)); is_anom=(z>5.0)
            else: mu=sd=z=None; is_anom=False
            rec={"entity":ent,"metric":metric,"timestamp":t.isoformat().replace("+00:00","Z"),"value":x,"mean":mu,"std":sd,"zscore":z,"is_anomaly":is_anom}
            if is_anom:
                points.append(rec); all_anoms.append(rec); cur.append(rec)
            else:
                if cur: intervals.append({"entity":ent,"metric":metric,"start_time":cur[0]["timestamp"],"end_time":cur[-1]["timestamp"],"point_count":len(cur),"points":cur}); cur=[]
                hist.append(x)
        if cur: intervals.append({"entity":ent,"metric":metric,"start_time":cur[0]["timestamp"],"end_time":cur[-1]["timestamp"],"point_count":len(cur),"points":cur})
    with point_path.open("w") as h:
        for x in points: h.write(json.dumps(x,ensure_ascii=False)+"\n")
    with interval_path.open("w") as h:
        for x in intervals: h.write(json.dumps(x,ensure_ascii=False)+"\n")
    # Merge anomalies in a 60-second adjacency window into detection events.
    all_anoms.sort(key=lambda x:x["timestamp"]); events=[]; cur=[]; last=None
    for p in all_anoms:
        t=ts(p["timestamp"])
        if last is None or (t-last).total_seconds()<=60: cur.append(p)
        else:
            events.append(cur); cur=[p]
        last=t
    if cur: events.append(cur)
    merged=[]
    for i,ev in enumerate(events,1):
        merged.append({"event_id":f"five-sigma-{i:04d}","start_time":min(x["timestamp"] for x in ev),"end_time":max(x["timestamp"] for x in ev),"duration_seconds":(ts(max(x["timestamp"] for x in ev))-ts(min(x["timestamp"] for x in ev))).total_seconds()+60,"trigger_entities":sorted({x["entity"] for x in ev}),"trigger_metrics":sorted({x["metric"] for x in ev}),"point_count":len(ev),"source":"five_sigma"})
    with (out/"five_sigma_detection/merged_detection_events.jsonl").open("w") as h:
        for x in merged: h.write(json.dumps(x,ensure_ascii=False)+"\n")
    with (out/"five_sigma_detection/merged_detection_events.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["event_id","start_time","end_time","duration_seconds","point_count","trigger_entities","trigger_metrics"]); w.writeheader()
        for x in merged: w.writerow({k:x[k] for k in w.fieldnames if k in x} | {"trigger_entities":";".join(x["trigger_entities"]),"trigger_metrics":";".join(x["trigger_metrics"])})
    ser_manifest=[]
    for (ent,metric), buckets in sorted(series.items()):
        n=sum(len(v) for v in buckets.values()); ser_manifest.append({"entity":ent,"metric":metric,"aggregation":"minute_mean_numeric","valid_points":len(buckets),"raw_values":n,"missing_ratio":None})
    (out/"timeseries/timeseries_manifest.json").write_text(json.dumps(ser_manifest,indent=2,ensure_ascii=False)+"\n")
    (out/"five_sigma_detection/five_sigma_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (out/"five_sigma_detection/anomaly_scoring_report.json").write_text(json.dumps({"series_count":seq_count,"point_anomalies":len(points),"intervals":len(intervals),"merged_events":len(merged),"input_files":len(files),"excluded_files":files.count("netflow_5tuple_minute_readable")},indent=2)+"\n")
    print(json.dumps({"series":seq_count,"points":len(points),"intervals":len(intervals),"events":len(merged)}))
if __name__=="__main__": main()
