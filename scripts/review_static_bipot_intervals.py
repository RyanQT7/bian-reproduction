#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

def load_jsonl(p):
    with Path(p).open() as f: return [json.loads(x) for x in f if x.strip()]
def iso(x): return pd.Timestamp(x).tz_convert('UTC') if pd.Timestamp(x).tzinfo else pd.Timestamp(x,tz='UTC')
def iv(item): return iso(item['start_time']), iso(item['end_time'])
def iou(a,b):
    s=max(a[0],b[0]); e=min(a[1],b[1]); inter=max(0,(e-s).total_seconds())
    union=max((a[1]-a[0]).total_seconds(),0)+max((b[1]-b[0]).total_seconds(),0)-inter
    return inter/union if union else 0.0
def safe(x): return str(x).replace('/','_').replace(' ','_')
def band(ax, spans, y, label, color):
    for s,e in spans: ax.plot([s,e],[y,y],lw=10,color=color,solid_capstyle='butt',label=label)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-dir',required=True); ap.add_argument('--ground-truth-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--start-time',required=True); ap.add_argument('--end-time',required=True); a=ap.parse_args()
    rdir=Path(a.result_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
    short=load_jsonl(rdir/'detected_incidents.jsonl'); long_all=load_jsonl(rdir/'persistent_long_intervals.jsonl'); long=[x for x in long_all if x.get('level')=='network']
    evdir=Path(a.ground_truth_dir); gtmeta={x['incident_id']:x for x in load_jsonl(evdir/'ground_truth.jsonl')}
    times=evdir.parent/'experiment'/'incidents.jsonl'; timed=load_jsonl(times) if times.exists() else []
    gt=[]
    for x in timed:
        m=gtmeta.get(x['incident_id'],{}); z=dict(x); z.update({k:v for k,v in m.items() if k not in z}); gt.append(z)
    shorts=[(x['incident_id'],iv(x)) for x in short]; longs_all=[(f'long-{i+1:04d}',iv(x)) for i,x in enumerate(long_all)]; longs=[(f'long-network-{i+1:04d}',iv(x)) for i,x in enumerate(long)]
    all_det=shorts+longs
    rows=[]
    for lid,lv in longs_all:
        overlaps=[g for g in gt if iou(lv,iv(g))>0]; nearest=max(gt,key=lambda g:iou(lv,iv(g))) if gt else None
        rows.append({'long_interval_id':lid,'start_time':lv[0].isoformat(),'end_time':lv[1].isoformat(),'duration_minutes':(lv[1]-lv[0]).total_seconds()/60,'covered_truth_count':len(overlaps),'covered_incident_ids':';'.join(g['incident_id'] for g in overlaps),'nearest_incident_id':nearest['incident_id'] if nearest else '','nearest_iou':iou(lv,iv(nearest)) if nearest else 0})
    pd.DataFrame(rows).to_csv(out/'long_interval_coverage_audit.csv',index=False)
    summ=[]
    for g in gt:
        gv=iv(g); os=[(n,v) for n,v in shorts if iou(v,gv)>0]; ol=[(n,v) for n,v in longs_all if iou(v,gv)>0]; near=max(all_det,key=lambda z:iou(z[1],gv)) if all_det else ('', (gv[0],gv[0])); best=max([iou(v,gv) for _,v in all_det] or [0]); summ.append({'incident_id':g['incident_id'],'start_time':g['start_time'],'end_time':g['end_time'],'covered_by_short_detected':'Y' if os else 'N','covered_by_long_interval':'Y' if ol else 'N','nearest_detected_interval_type':'short_detected' if near[0].startswith('detected') else 'long_persistent','nearest_detected_interval_id':near[0],'best_IoU':best,'predicted_earlier_than_truth':'Y' if any(v[0]<gv[0] for _,v in all_det if iou(v,gv)==best) else 'N'})
    pd.DataFrame(summ).to_csv(out/'ground_truth_coverage_summary.csv',index=False)
    score=json.loads((rdir/'detection_score.json').read_text()); match=score.get('matches',[{}])[0]
    audit={'ground_truth_count':len(gt),'short_count':len(short),'long_count':len(long_all),'long_network_count':len(long),'truth_covered_by_long':sum(x['covered_by_long_interval']=='Y' for x in summ),'truth_covered_by_short':sum(x['covered_by_short_detected']=='Y' for x in summ),'score':score}
    (out/'audit_summary.json').write_text(json.dumps(audit,indent=2,default=str))
    zero=[m for m in score.get('matches',[]) if m.get('S_time',1)==0]; lines=['# Scoring zero-time audit','',f"Formal matches: {len(score.get('matches',[]))}; matches with zero S_time: {len(zero)}."]
    for m in score.get('matches',[]): lines.append(f"- prediction {m.get('prediction')} vs truth {m.get('truth')}: IoU={m.get('iou')}, start_delta={m.get('start_delta_seconds')}s, end_delta={m.get('end_delta_seconds')}s, S_time={m.get('S_time')}.")
    lines.append(''); lines.append('A zero S_time occurs when the frozen timing-accuracy rule yields zero, including an early prediction or sufficiently large boundary error.')
    (out/'scoring_zero_tp_audit.md').write_text('\n'.join(lines)+'\n')
    start=iso(a.start_time); end=iso(a.end_time); pdf=out/'static_bipot_interval_review.pdf'
    with PdfPages(pdf) as pp:
        fig,axs=plt.subplots(3,1,figsize=(14,8),sharex=True); labels=['Ground Truth All Cases','Detected Short Intervals','Persistent Long Intervals (>35 min)']; sets=[[iv(g) for g in gt],[v for _,v in shorts],[v for _,v in longs]]; cols=['tab:red','tab:blue','tab:orange']
        for ax,sp,l,c in zip(axs,sets,labels,cols): band(ax,sp,1,l,c); ax.set_ylim(.5,1.5); ax.set_yticks([1]); ax.set_yticklabels([l]); ax.grid(alpha=.2)
        axs[-1].set_xlim(start,end); axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=start.tz)); fig.suptitle(f'Static grouped biPOT review | GT={len(gt)} short={len(short)} long={len(long)} TP/FP/FN={score["TP"]}/{score["FP"]}/{score["FN"]} score={score["detection_score_30"]}/30\nSHA: '+next(iter((rdir/'detected_incidents.sha256').read_text().split()),'')); fig.tight_layout(); pp.savefig(fig); plt.close(fig)
        for iid,v in all_det:
            lo=max(start,v[0]-pd.Timedelta(minutes=30)); hi=min(end,v[1]+pd.Timedelta(minutes=30)); fig,ax=plt.subplots(figsize=(14,6)); ax.set_xlim(lo,hi); ax.set_ylim(.5,3.5)
            band(ax,[iv(g) for g in gt if iv(g)[1]>=lo and iv(g)[0]<=hi],3,'Ground truth','tab:red'); band(ax,[v],2,'This interval','tab:blue' if iid.startswith('detected') else 'tab:orange'); band(ax,[w for j,w in all_det if j!=iid and w[1]>=lo and w[0]<=hi],1,'Other detected','0.6')
            hits=[g for g in gt if iou(v,iv(g))>0]; nearest=max(gt,key=lambda g:iou(v,iv(g))) if gt else None; txt=f'{iid} | {"short_detected" if iid.startswith("detected") else "long_persistent"}\n{v[0].isoformat()} to {v[1].isoformat()} ({(v[1]-v[0]).total_seconds()/60:.1f} min)\nOverlapping GT: '+(', '.join(g['incident_id'] for g in hits) if hits else 'No overlapping ground-truth case.')
            if nearest: txt+=f'\nNearest: {nearest["incident_id"]}, IoU={iou(v,iv(nearest)):.3f}, predicted earlier={v[0]<iv(nearest)[0]}'
            ax.text(.01,.02,txt,transform=ax.transAxes,va='bottom',fontsize=9); ax.set_yticks([1,2,3]); ax.set_yticklabels(['Other detected','This interval','Ground truth']); ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=start.tz)); ax.grid(alpha=.2); fig.tight_layout(); pp.savefig(fig); plt.close(fig)
        for k in range(0,len(summ),18):
            fig,ax=plt.subplots(figsize=(14,8)); ax.axis('off'); sub=pd.DataFrame(summ[k:k+18]); tbl=ax.table(cellText=sub.values,colLabels=sub.columns,loc='center',cellLoc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1,1.5); ax.set_title('Ground Truth Coverage Summary'); pp.savefig(fig); plt.close(fig)
    pd.DataFrame([{'interval_id':i,'interval_type':'short_detected' if i.startswith('detected') else 'long_persistent','start_time':v[0].isoformat(),'end_time':v[1].isoformat()} for i,v in all_det]).to_csv(out/'interval_match_index.csv',index=False)
    (out/'report.md').write_text('# Static biPOT interval review\n\n'+json.dumps(audit,indent=2,default=str))
    print(json.dumps({'short':len(short),'long':len(long),'truth_long':audit['truth_covered_by_long'],'truth_short':audit['truth_covered_by_short'],'pdf':str(pdf)}))
if __name__=='__main__': main()
