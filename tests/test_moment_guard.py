from pathlib import Path
import pytest
from bian.detection.guard import assert_blind_path

def test_truth_guard():
    with pytest.raises(PermissionError):
        assert_blind_path("/tmp/bian_new_cases_20260728_20260729_final/experiment/x")
    assert assert_blind_path("/tmp/data").name == "data"

