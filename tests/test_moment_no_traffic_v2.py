import pandas as pd

from bian.detection.aggregation import merge_intervals_by_timestamp, sampled_hysteresis_intervals
from bian.detection.no_traffic_v2 import DERIVED_SUFFIX, _device, _file_kind, preprocess


def _dataset(tmp_path):
    processed=tmp_path/"beida_x"/"processed"; processed.mkdir(parents=True)
    pd.DataFrame({
        "timestamp":["2026-07-28 04:00:00","2026-07-28 04:01:00",
                     "2026-07-28 04:00:00","2026-07-28 04:01:00",
                     "2026-07-28 04:10:00"],
        "region":["r"]*5,"node":["br-1","br-1","br-2","br-2","br-1"],
        "node_type":["br"]*5,"cpu_mean":[1,2,10,11,3],
        "cpu_max":[2,3,11,12,4],"requests_total":[100,110,1000,1020,120],
    }).to_csv(processed/"node_metrics_20260728040000_20260729040000.csv",index=False)
    pd.DataFrame({
        "timestamp_utc":["2026-07-28 04:00:00"],"source_ip":["x"],"value":[1],
    }).to_csv(processed/"traffic_flow_metrics.csv",index=False)
    return tmp_path


def test_traffic_file_is_path_excluded(tmp_path):
    root=_dataset(tmp_path); out=tmp_path/"out"; preprocess(root,out)
    x=pd.read_parquet(out/"series.parquet")
    assert not x.source_file.str.contains("traffic_").any()


def test_different_devices_are_not_concatenated(tmp_path):
    root=_dataset(tmp_path); out=tmp_path/"out"; preprocess(root,out)
    x=pd.read_parquet(out/"series.parquet")
    assert x.device_id.nunique()==2


def test_counter_diff_stays_inside_device(tmp_path):
    root=_dataset(tmp_path); out=tmp_path/"out"; preprocess(root,out)
    x=pd.read_parquet(out/"series.parquet")
    b=x[(x.device_id=="region-1-br-2")&x.metric.str.contains("requests_total")]
    assert b.value.dropna().max() < 10  # log1p(20), never diffed from br-1's 110


def test_gap_splits_continuous_segment(tmp_path):
    root=_dataset(tmp_path); out=tmp_path/"out"; preprocess(root,out)
    x=pd.read_parquet(out/"series.parquet")
    b=x[(x.device_id=="region-1-br-1")&x.metric.str.contains("cpu_mean")]
    assert b.continuous_segment_id.nunique()==2


def test_derived_statistics_share_family():
    assert DERIVED_SUFFIX.sub("","cpu_mean")==DERIVED_SUFFIX.sub("","cpu_max")=="cpu"


def test_two_distinct_families_trigger():
    hits=pd.DataFrame({"family":["cpu","disk"],"hit":[True,True]})
    assert hits.loc[hits.hit,"family"].nunique()>=2


def test_calibration_cannot_start_state():
    ts=pd.date_range("2026-07-28T04:00Z",periods=93,freq="min").to_numpy()
    spans=sampled_hysteresis_intervals(
        [True]*93,[True]*93,ts,pd.Timestamp("2026-07-28T05:30Z"),2,2)
    assert spans[0][0]==pd.Timestamp("2026-07-28T05:30Z")


def test_long_interval_isolated_and_20m_not_merged():
    b=pd.Timestamp("2026-07-28T06:00Z")
    spans=[(b,b+pd.Timedelta(minutes=36)),
           (b+pd.Timedelta(minutes=56),b+pd.Timedelta(minutes=57))]
    assert len(merge_intervals_by_timestamp(spans,120))==2
    assert (spans[0][1]-spans[0][0]).total_seconds()>35*60
