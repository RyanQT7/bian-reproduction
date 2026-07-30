import numpy as np
import pandas as pd
from bian.detection.aggregation import robust_scale, intervals

def test_robust_scale_fallback_and_binary():
    assert robust_scale(np.ones(4), False)[1] > 0
    assert robust_scale(np.array([0, 1]), True)[1] == 1

def test_interval_onset_clear():
    ts = pd.date_range("2026-01-01", periods=8, freq="10s").to_numpy()
    got = intervals(np.array([0,1,1,1,0,0,0,0], bool), ts, 20, 20, 60)
    assert got == [(pd.Timestamp(ts[1]), pd.Timestamp(ts[3]))]

