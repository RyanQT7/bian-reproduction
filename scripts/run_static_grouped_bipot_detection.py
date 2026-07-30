#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genpareto

CAL_START = pd.Timestamp('2026-07-28T04:00:00Z')
CAL_END = pd.Timestamp('2026-07-28T05:30:00Z')
GAP = 120.0

def intervals(mask, ts):
    out=[]; active=False; run=quiet=0; start=None
    for i, hit in enumerate(mask):
        if ts[i] < CAL_END: continue
        if hit:
            run += 1; quiet=0
            if not active and run >= 2: active=True; start=i-1
        elif active:
            run=0; quiet += 1
            if quiet >= 2:
                out.append((ts[start], ts[i-2])); active=False; quiet=0; start=None
        else: run=0
    if active and start is not None: out.append((ts[start], ts[-1]))
    return out

def merge(spans):
    out=[]
    for s,e in sorted(spans):
        if out and (s-out[-1][1]).total_seconds() <= GAP: out[-1]=(out[-1][0],max(out[-1][1],e))
        else: out.append((s,e))
    return out

def fit_tail(x, side):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return None
    u=float(np.quantile(x,0.98)); ex=x[x>u]-u
    if len(ex)<30: return None
    frac=len(ex)/len(x); cp=1-5e-5/frac
    if not (0<cp<1): return None
    try: shape,loc,scale=genpareto.fit(ex,floc=0)
    except Exception: return None
    th=float(u+genpareto.ppf(cp,shape,loc=0,scale=scale))
    if not np.isfinite(th) or th<=u or scale<=0: return None
    return dict(side=side,u=u,tail_count=len(ex),sample_count=len(x),tail_fraction=frac,conditional_p=cp,shape=float(shape),scale=float(scale),threshold=th)

def self_check():
    rng=np.random.default_rng(7); x=rng.normal(size=3000); x[:500]+=8; x[500:1000]-=8
    assert fit_tail(x,'upper') and fit_tail(-x,'lower')
    ts=pd.date_range(CAL_END,periods=8,freq='min'); m=np.array([0,1,1,0,0,1,1,0],bool)
    assert intervals(m,ts)==[(ts[1],ts[2]),(ts[5],ts[7])]
    assert merge([(ts[0],ts[0]),(ts[-1]+pd.Timedelta(minutes=20),ts[-1]+pd.Timedelta(minutes=21))]) == [(ts[0],ts[0]),(ts[-1]+pd.Timedelta(minutes=20),ts[-1]+pd.Timedelta(minutes=21))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-root',required=True); ap.add_argument('--manifest-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--start-time',default=str(CAL_START)); ap.add_argument('--end-time',default='2026-07-29T04:00:00Z')
    a=ap.parse_args(); self_check()
    inp=Path(a.input_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
    man=pd.read_csv(Path(a.manifest_root)/'metric_screening_manifest.csv')
    role_map=man.drop_duplicates(['device_id','raw_metric_name']).set_index(['device_id','raw_metric_name'])['device_role'].to_dict()
    sm=pd.read_parquet(inp/'screened_series.parquet'); sm['timestamp']=pd.to_datetime(sm.timestamp,utc=True)
    sm=sm[sm.metric_type.isin(['dense_continuous','cumulative_counter'])].copy(); sm['device_role']=[role_map.get((d,m),str(d).split('-')[-1]) for d,m in zip(sm.device_id,sm.metric)]
    records=[]; seq_cal={}; seq_meta={}; skipped=0
    for key,g in sm.groupby(['device_id','metric','continuous_segment_id'],sort=False):
        g=g.sort_values('timestamp'); vals=pd.to_numeric(g.value,errors='coerce').to_numpy(float); ts=g.timestamp.to_numpy()
        cal=(g.timestamp>=CAL_START)&(g.timestamp<CAL_END); x=vals[cal.to_numpy()]; x=x[np.isfinite(x)]
        if len(x)<2: skipped+=1; continue
        c=float(np.median(x)); mad=float(np.median(np.abs(x-c))*1.4826); iqr=float(np.quantile(x,.75)-np.quantile(x,.25)/1.349) if False else float((np.quantile(x,.75)-np.quantile(x,.25))/1.349)
        scale=mad if mad>0 else iqr
        if not np.isfinite(scale) or scale<=0: skipped+=1; continue
        z=np.clip((vals-c)/scale,-20,20); rec=g[['timestamp','device_id','metric','base_metric_family','metric_type','continuous_segment_id','device_role']].copy(); rec['z']=z; rec['center']=c; rec['scale']=scale
        records.append(rec); seq_cal[key]=rec.loc[cal.to_numpy(),'z'].to_numpy(); seq_meta[key]=(rec.device_role.iloc[0],rec.base_metric_family.iloc[0])
    dat=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    groups={}
    for key,z in seq_cal.items(): groups.setdefault(seq_meta[key],[]).extend(z.tolist())
    thresholds=[]
    for (role,fam), arr in groups.items():
        up=fit_tail(arr,'upper'); lo=fit_tail(-np.asarray(arr),'lower'); fallback='role_family'
        if not (up and lo):
            arr2=[]
            for (r,f),v in groups.items():
                if f==fam: arr2.extend(v)
            up=fit_tail(arr2,'upper'); lo=fit_tail(-np.asarray(arr2),'lower'); fallback='family'
        if not (up and lo):
            allz=np.concatenate([np.asarray(v) for v in groups.values()]); up=fit_tail(allz,'upper'); lo=fit_tail(-allz,'lower'); fallback='all'
        if up and lo: thresholds.append({'device_role':role,'base_metric_family':fam,'fallback_level':fallback,'upper_threshold':up['threshold'],'lower_threshold':lo['threshold'],'u_upper':up['u'],'u_lower':lo['u'],'upper_shape':up['shape'],'lower_shape':lo['shape'],'upper_scale':up['scale'],'lower_scale':lo['scale'],'upper_tail_count':up['tail_count'],'lower_tail_count':lo['tail_count'],'sample_count':up['sample_count']})
    th=pd.DataFrame(thresholds); th.to_csv(out/'group_bipot_thresholds.csv',index=False)
    tmap={(r['device_role'],r['base_metric_family']):r for r in thresholds}
    if not dat.empty:
        keys=pd.MultiIndex.from_arrays([dat.device_role,dat.base_metric_family]);
        dat['upper_threshold']=[tmap.get(k,{}).get('upper_threshold',np.nan) for k in keys]
        dat['lower_threshold']=[tmap.get(k,{}).get('lower_threshold',np.nan) for k in keys]
        dat['upper_hit']=dat.z.to_numpy()>dat.upper_threshold.to_numpy(); dat['lower_hit']=(-dat.z.to_numpy())>dat.lower_threshold.to_numpy()
        dat['outlier']=dat.upper_hit|dat.lower_hit
    dat.to_parquet(out/'metric_anomaly_points.parquet',index=False)
    metric_spans=[]
    for (dev,met,seg),g in dat.groupby(['device_id','metric','continuous_segment_id']):
        for s,e in intervals(g.outlier.to_numpy(bool),g.timestamp.to_numpy()): metric_spans.append({'device_id':dev,'metric':met,'base_metric_family':g.base_metric_family.iloc[0],'start':s,'end':e})
    # binary states
    bi=pd.read_parquet(inp/'rule_based_series.parquet'); bi['timestamp']=pd.to_datetime(bi.timestamp,utc=True); binary_spans=[]
    for (dev,met,seg),g in bi.groupby(['device_id','metric','continuous_segment_id']):
        g=g.sort_values('timestamp'); cal=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)];
        if cal.empty: continue
        base=float(cal.value.mode().iloc[0]); mask=(g.value!=base).to_numpy(bool)
        for s,e in intervals(mask,g.timestamp.to_numpy()): binary_spans.append({'device_id':dev,'metric':met,'base_metric_family':g.base_metric_family.iloc[0],'start':s,'end':e})
    # device trigger from simultaneous distinct families, plus binary
    bydev={}
    for x in metric_spans: bydev.setdefault(x['device_id'],[]).append(x)
    dev_intervals=[]
    for dev, spans in bydev.items():
        points=sorted(set([x['start'] for x in spans]+[x['end'] for x in spans]));
        # use minute grid from all actual anomaly timestamps for deterministic overlap
        ts=pd.date_range(CAL_END,pd.Timestamp(a.end_time),freq='min',inclusive='left'); mask=[]
        for t in ts: mask.append(len({x['base_metric_family'] for x in spans if x['start']<=t<=x['end']})>=2)
        for s,e in intervals(np.array(mask),ts.to_numpy()): dev_intervals.append({'device_id':dev,'start':s,'end':e,'families':sorted({x['base_metric_family'] for x in spans if x['start']<=e and x['end']>=s})})
    for x in binary_spans: dev_intervals.append({'device_id':x['device_id'],'start':x['start'],'end':x['end'],'families':[x['base_metric_family']],'binary':True})
    device_out=[]; persistent=[]
    for dev,arr in pd.DataFrame(dev_intervals).groupby('device_id') if dev_intervals else []:
        spans=merge([(r.start,r.end) for r in arr.itertuples()])
        for s,e in spans:
            rec={'level':'device','device_id':dev,'start_time':s.isoformat(),'end_time':e.isoformat(),'duration_minutes':(e-s).total_seconds()/60}
            (persistent if rec['duration_minutes']>35 else device_out).append(rec)
    # network OR
    network=[]
    for s,e in merge([(pd.Timestamp(x['start_time']),pd.Timestamp(x['end_time'])) for x in device_out]): network.append({'start':s,'end':e})
    incidents=[]
    for i,x in enumerate(network,1):
        dur=(x['end']-x['start']).total_seconds()/60; rec={'level':'network','device_id':None,'start_time':x['start'].isoformat(),'end_time':x['end'].isoformat(),'duration_minutes':dur}
        if dur>35: persistent.append(rec)
        else: incidents.append({'incident_id':f'detected-bipot-{i:04d}','start_time':x['start'].isoformat(),'end_time':x['end'].isoformat(),'score':1.0,'detector':'static_grouped_bipot'})
    for name,rows in [('device_intervals.jsonl',device_out),('detected_incidents.jsonl',incidents),('persistent_long_intervals.jsonl',persistent)]:
        with (out/name).open('w') as f:
            for r in rows: f.write(json.dumps(r,default=str)+'\n')
    cfg={'algorithm':'static_grouped_two_sided_pot','calibration_start':CAL_START.isoformat(),'calibration_end':CAL_END.isoformat(),'u_quantile':.98,'q_total':1e-4,'q_side':5e-5,'min_exceedances':30,'merge_gap_seconds':120,'onset_points':2,'clear_points':2,'max_duration_minutes':35,'input_root':str(inp)}
    (out/'bipot_configuration.json').write_text(json.dumps(cfg,indent=2))
    pred=out/'detected_incidents.jsonl'; sha=hashlib.sha256(pred.read_bytes()).hexdigest(); (out/'detected_incidents.sha256').write_text(sha+'  '+pred.name+'\n')
    truth='/home/xieqitong/AIOps_2026_processed_data/failures/bian_new_cases_20260728_20260729_final/experiment/incidents.jsonl'; score_dir=out/'score'; score_dir.mkdir()
    import subprocess,sys
    env=dict(**__import__('os').environ); env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    subprocess.run([sys.executable,'scripts/score_moment_detection.py','--predictions',str(pred),'--truth',truth,'--frozen-sha',str(out/'detected_incidents.sha256'),'--output',str(out/'detection_score.json')],check=True,env=env)
    (out/'truth_access_audit.jsonl').write_text(json.dumps({'event':'truth_access_after_prediction_freeze','truth_path':truth})+'\n')
    result=json.loads((out/'detection_score.json').read_text()); (out/'report.md').write_text('# Static grouped biPOT development run\n\n'+json.dumps({'events':len(incidents),'persistent_intervals':len(persistent),'score':result},indent=2))
    print(json.dumps({'events':len(incidents),'persistent_intervals':len(persistent),'sha256':sha,'score':result},default=str))
if __name__=='__main__': main()
