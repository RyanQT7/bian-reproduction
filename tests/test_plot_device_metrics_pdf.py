from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader

from scripts.plot_device_metrics_pdf import _plot_metric, render


def test_multpage_binary_gap_traffic_and_timeline(tmp_path):
    data_root=tmp_path/"data"/"dataset"; data_root.mkdir(parents=True)
    manifest=tmp_path/"manifest"; manifest.mkdir()
    device="region-1-fw"; rows=[]
    for i in range(7):
        rows.append({"device_id":device,"metric":f"metric-{i}",
                     "base_metric_family":f"family-{i}","metric_type":
                     "binary_state" if i==0 else "dense_continuous",
                     "source_file":"region/processed/node_metrics_x.csv"})
    rows.append({"device_id":device,"metric":"traffic",
                 "base_metric_family":"traffic","metric_type":"dense_continuous",
                 "source_file":"region/processed/traffic_flow_metrics.csv"})
    pd.DataFrame(rows).to_csv(manifest/"device_metric_manifest.csv",index=False)
    pd.DataFrame([{"source_file":"region/processed/node_metrics_x.csv",
                   "device_id":device}]).to_csv(
        manifest/"device_mapping_manifest.csv",index=False)
    (manifest/"device_intervals.jsonl").write_text(
        '{"device_id":"region-1-fw","start_time":"2026-07-28T05:00:00Z",'
        '"end_time":"2026-07-28T05:02:00Z"}\n')
    (manifest/"persistent_long_intervals.jsonl").write_text("")
    gt=tmp_path/"ground_truth.jsonl"
    gt.write_text('{"incident_id":"incident-1","root_device_ids":["region-1-fw"]}\n')
    loc=tmp_path/"localization.jsonl"
    loc.write_text('{"incident_id":"incident-1","root_device_ids":["region-1-fw"]}\n')
    cls=tmp_path/"classification.jsonl"
    cls.write_text('{"incident_id":"incident-1","fault_type":"test_fault"}\n')
    times=tmp_path/"times.jsonl"
    times.write_text('{"incident_id":"incident-1","start_time":"2026-07-28T05:00:00Z",'
                     '"end_time":"2026-07-28T05:02:00Z"}\n')
    points=[]
    for i in range(7):
        for minute,segment in ((0,"a"),(1,"a"),(10,"b")):
            points.append({"timestamp":f"2026-07-28T04:{minute:02d}:00Z",
                           "device_id":device,"metric":f"metric-{i}",
                           "raw_value":minute%2,"continuous_segment_id":segment,
                           "metric_type":"binary_state" if i==0 else "dense_continuous"})
    series=tmp_path/"series.parquet"; pd.DataFrame(points).to_parquet(series,index=False)
    output=tmp_path/"pdfs"
    args=SimpleNamespace(data_root=str(data_root),manifest_root=str(manifest),
        output_dir=str(output),start_time="2026-07-28T04:00:00Z",
        end_time="2026-07-29T04:00:00Z",plots_per_page=6,
        series_parquet=str(series),ground_truth_file=str(gt),
        localization_ground_truth_file=str(loc),
        classification_ground_truth_file=str(cls),incident_time_file=str(times))
    index,errors=render(args)
    assert not errors and index[0]["metric_count"]==7 and index[0]["page_count"]==3
    assert len(PdfReader(index[0]["pdf_path"]).pages)==3
    assert index[0]["detected_interval_count"]==1
    assert index[0]["ground_truth_fault_count_on_device"]==1
    normalized=pd.read_csv(output/"ground_truth_intervals_normalized.csv")
    assert normalized.iloc[0].normalized_device_id=="region-1-fw"

    fig,ax=plt.subplots()
    binary=pd.DataFrame(points[:3])
    binary["timestamp"]=pd.to_datetime(binary.timestamp,utc=True)
    _plot_metric(ax,binary,"binary_state")
    assert len(ax.lines)==2  # no line across the 9-minute gap
    assert all(line.get_drawstyle()=="steps-post" for line in ax.lines)
    plt.close(fig)
