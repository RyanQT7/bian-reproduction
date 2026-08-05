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
    # Hierarchical aggregation: metric -> entity only.  The old implementation
    # sorted all points globally, allowing unrelated regions to form one
    # transitive connected component.  Keep cross-entity/path propagation out of
    # this detector; downstream topology reasoning handles relationships.
    event_evidence=[]
    direct_tokens=("cpu","disk_io","filesystem","process_count","ospf","bgp","route","prefix","peer","failed","timeout","latency","drop","state","loss","error")
    for (ent,metric), buckets in sorted(series.items()):
        seq=[x for x in intervals if x["entity"]==ent and x["metric"]==metric]
        for iv in seq:
            pts=iv.get("points",[])
            # Isolated ordinary continuous-metric spikes are retained only for
            # audit unless they are >=8 sigma; discrete/state indicators remain.
            is_direct=any(tok in metric.lower() for tok in direct_tokens)
            max_z=max((abs(float(p.get("zscore") or 0)) for p in pts),default=0)
            if len(pts)==1 and not is_direct and max_z<8.0:
                continue
            event_evidence.append({"entity":ent,"metric":metric,"start_time":iv["start_time"],"end_time":iv["end_time"],"point_count":iv["point_count"],"points":pts,"direct":is_direct})
    # Merge only intervals belonging to the same entity; break at >=3 quiet
    # minutes and hard-cap every final event at 40 minutes.
    by_entity=defaultdict(list)
    for iv in event_evidence: by_entity[iv["entity"]].append(iv)
    events=[]
    for ent,ivs in by_entity.items():
        ivs.sort(key=lambda x:x["start_time"]); cur=[]; cur_end=None
        for iv in ivs:
            st, en=ts(iv["start_time"]), ts(iv["end_time"])
            if not cur or (st-cur_end).total_seconds()>120:
                if cur: events.append(cur)
                cur=[iv]; cur_end=en
            else:
                cur.append(iv); cur_end=max(cur_end,en)
        if cur: events.append(cur)
    # Split long entity events at 40-minute boundaries. This is a detector
    # safety bound, not a GT-derived operation.
    bounded=[]
    for ev in events:
        start=min(ts(x["start_time"]) for x in ev); end=max(ts(x["end_time"]) for x in ev)
        while (end-start).total_seconds()+60>2400:
            cut=start+timedelta(minutes=40)
            left=[x for x in ev if ts(x["start_time"])<cut]
            right=[x for x in ev if ts(x["end_time"])>=cut]
            if not left: left=[ev[0]]
            bounded.append(left); ev=right; start=cut
            if not ev: break
            end=max(ts(x["end_time"]) for x in ev)
        if ev: bounded.append(ev)
    events=bounded
    merged=[]
    for i,ev in enumerate(events,1):
        st=min(x["start_time"] for x in ev); en=max(x["end_time"] for x in ev)
        merged.append({"event_id":f"five-sigma-{i:04d}","start_time":st,"end_time":en,"duration_seconds":(ts(en)-ts(st)).total_seconds()+60,"trigger_entities":sorted({x["entity"] for x in ev}),"trigger_metrics":sorted({x["metric"] for x in ev}),"point_count":sum(x["point_count"] for x in ev),"source":"five_sigma"})
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
    if merged:
        longest=max(merged,key=lambda x:x["duration_seconds"])
        audit=("# Five-sigma event merge audit\n\n"
                "## Root cause of the previous all-day event\n"
                "The previous implementation globally sorted every entity/metric point and merged adjacent timestamps. "
                "That created a transitive cross-region connected component; baseline state was per series, but event aggregation was global.\n\n"
                f"- point anomalies: {len(points)}\n- metric intervals: {len(intervals)}\n- retained intervals after singleton policy: {len(event_evidence)}\n- final entity-level events: {len(merged)}\n"
                f"- longest event: {longest['event_id']} {longest['start_time']}..{longest['end_time']} ({longest['duration_seconds']}s)\n"
                f"- longest trigger entities: {', '.join(longest['trigger_entities'][:20])}\n"
                f"- longest trigger metrics: {', '.join(longest['trigger_metrics'][:40])}\n\n"
                "## Merge rules now enforced\n"
                "1. Baselines are maintained independently for each entity+metric.\n"
                "2. Metric intervals merge only within the same entity+metric.\n"
                "3. Entity events merge only within the same entity and a 2-minute gap.\n"
                "4. Cross-entity transitive/path merging is disabled; topology is left to RCA.\n"
                "5. Isolated ordinary spikes below 8-sigma are audit-only.\n"
                "6. Events are split at 3-minute evidence gaps and hard-capped at 40 minutes.\n\n"
                "## Full chain counts\n"
                f"point anomalies -> metric intervals -> retained intervals -> entity events: {len(points)} -> {len(intervals)} -> {len(event_evidence)} -> {len(merged)}\n")
    else:
        audit=f"# Five-sigma event merge audit\n\npoint anomalies: {len(points)}\nmetric intervals: {len(intervals)}\nretained intervals: {len(event_evidence)}\nfinal events: 0\n"
    (out/"five_sigma_detection/five_sigma_event_merge_audit.md").write_text(audit)
    print(json.dumps({"series":seq_count,"points":len(points),"intervals":len(intervals),"events":len(merged)}))
if __name__=="__main__": main()
