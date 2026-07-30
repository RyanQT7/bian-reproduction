import numpy as np
import pandas as pd
import pytest

from bian.detection.aggregation import merge_intervals_by_timestamp, sampled_hysteresis_intervals
from bian.detection.screening_v3 import different_family_trigger, robust_score_threshold, screen


@pytest.fixture(scope="module")
def screened(tmp_path_factory):
    root=tmp_path_factory.mktemp("screen"); rows=[]; ts=pd.date_range("2026-07-28T04:00Z",periods=1440,freq="min")
    def add(dev,metric,values,family=None,segment=None):
        for i,(t,v) in enumerate(zip(ts,values)):
            rows.append({"timestamp":t,"device_id":dev,"metric":metric,
                "base_metric_family":family or metric,"raw_value":v,"value":v,
                "activity_value":v,"source_file":"node.csv","source_kind":"node",
                "metric_type":"dense_continuous",
                "continuous_segment_id":segment[i] if segment else f"{dev}:{metric}:s1"})
    add("region-1-fw","node::constant",np.ones(1440))
    tiny=np.ones(1440); tiny[300]=1+1e-9; add("region-1-fw","node::near",tiny)
    pulse=np.zeros(1440); pulse[300:]=1; add("region-1-fw","node::crc_errors",pulse)
    counter=np.arange(1440.); counter[400:]-=390; add("region-1-fw","node::bytes_total",counter)
    add("region-2-fw","node::bytes_total",np.arange(1000,2440.))
    add("region-1-fw","node::process_count",np.tile([1,3,2],480))
    add("region-1-fw","routing::peer_up",np.ones(1440))
    add("region-1-fw","node::cpu_mean",np.sin(np.arange(1440)/20),family="node::cpu")
    add("region-1-fw","node::cpu_max",np.sin(np.arange(1440)/20)+1,family="node::cpu")
    source=root/"source.parquet"; pd.DataFrame(rows).to_parquet(source,index=False)
    out=root/"out"; screen(source,out,512)
    return out


def manifest(screened): return pd.read_csv(screened/"metric_screening_manifest.csv")

def test_constant_excluded(screened):
    assert manifest(screened).query("raw_metric_name=='node::constant'").screening_action.iloc[0]=="exclude"

def test_near_constant_excluded(screened):
    assert "constant" in manifest(screened).query("raw_metric_name=='node::near'").exclusion_reason.iloc[0]

def test_sparse_pulse_not_plain_constant(screened):
    assert manifest(screened).query("raw_metric_name=='node::crc_errors'").detected_metric_type.iloc[0]=="sparse_counter"

def test_counter_diff_inside_device(screened):
    x=pd.read_parquet(screened/"screened_series.parquet")
    a=x[(x.device_id=="region-2-fw")&(x.metric=="node::bytes_total")]
    assert np.isclose(a.value.dropna().median(),np.log1p(1))

def test_counter_reset_not_negative_peak(screened):
    x=pd.read_parquet(screened/"screened_series.parquet")
    a=x[(x.device_id=="region-1-fw")&(x.metric=="node::bytes_total")]
    assert a.value.dropna().min()>=0

def test_diff_does_not_cross_device(screened):
    x=pd.read_parquet(screened/"screened_series.parquet")
    x=x[x.metric.str.contains("bytes_total|crc_errors")]
    assert x.groupby(["device_id","metric","continuous_segment_id"]).value.nth(0).isna().all()

def test_ambiguous_counter_excluded(screened):
    assert "node::process_count" in pd.read_csv(screened/"ambiguous_metrics.csv").raw_metric_name.tolist()

def test_zero_score_scale_rejected():
    assert robust_score_threshold(np.ones(90)) is None

def test_conservative_threshold_formula():
    x=np.arange(90,dtype=float); high,low,s=robust_score_threshold(x)
    assert high==max(np.quantile(x,.999),np.median(x)+8*s["robust_scale"])
    assert low==max(np.quantile(x,.995),np.median(x)+5*s["robust_scale"])

def test_same_family_derivatives_not_double_trigger():
    x=pd.DataFrame({"above_high":[True,True],"base_metric_family":["cpu","cpu"]})
    assert not different_family_trigger(x)

def test_sparse_cannot_satisfy_continuous_family_trigger():
    x=pd.DataFrame({"above_high":[True],"base_metric_family":["errors"],"metric_type":["sparse_counter"]})
    assert not different_family_trigger(x[x.metric_type!="sparse_counter"])

def test_two_families_trigger():
    x=pd.DataFrame({"above_high":[True,True],"base_metric_family":["cpu","disk"]})
    assert different_family_trigger(x)

def test_calibration_excluded():
    ts=pd.date_range("2026-07-28T04:00Z",periods=93,freq="min").to_numpy()
    got=sampled_hysteresis_intervals([1]*93,[1]*93,ts,pd.Timestamp("2026-07-28T05:30Z"),2,2)
    assert got[0][0]==pd.Timestamp("2026-07-28T05:30Z")

def test_long_interval_isolated():
    start=pd.Timestamp("2026-07-28T06:00Z"); end=start+pd.Timedelta(minutes=36)
    assert (end-start).total_seconds()>35*60
    assert len(merge_intervals_by_timestamp([(start,end),(end+pd.Timedelta(minutes=20),end+pd.Timedelta(minutes=21))],120))==2
