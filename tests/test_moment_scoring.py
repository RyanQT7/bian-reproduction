import pytest
from bian.detection.scoring import score

def test_early_prediction_match_scores_zero():
    p=[{"start":"2026-01-01T00:00:00Z","end":"2026-01-01T00:01:00Z"}]
    t=[{"start":"2026-01-01T00:00:10Z","end":"2026-01-01T00:01:00Z"}]
    got=score(p,t)
    assert got["TP"] == 1 and got["matches"][0]["S_time"] == 0
    assert got["Precision"] == 1 and got["alpha"] == 1
