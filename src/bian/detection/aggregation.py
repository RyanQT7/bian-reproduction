from __future__ import annotations

import numpy as np
import pandas as pd


def robust_scale(values: np.ndarray, binary: bool, minimum: float = 1e-6) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    center = float(np.median(finite)) if len(finite) else 0.0
    if binary:
        return center, 1.0
    mad = float(np.median(np.abs(finite - center))) * 1.4826
    if mad <= minimum:
        q1, q3 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 0.0)
        mad = max(float((q3 - q1) / 1.349), minimum)
    return center, mad


def intervals(mask: np.ndarray, timestamps: np.ndarray, onset_s: float, clear_s: float,
              merge_s: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not len(mask):
        return []
    dt = max(float(np.median(np.diff(timestamps).astype("timedelta64[s]").astype(float))), 1.0)
    onset, clear = max(2, int(np.ceil(onset_s / dt))), max(2, int(np.ceil(clear_s / dt)))
    spans, active, run, start, quiet = [], False, 0, None, 0
    for i, hit in enumerate(mask):
        if hit:
            run += 1; quiet = 0
            if not active and run >= onset:
                active, start = True, i - run + 1
        else:
            run = 0
            if active:
                quiet += 1
                if quiet >= clear:
                    spans.append((pd.Timestamp(timestamps[start]), pd.Timestamp(timestamps[i - quiet])))
                    active, quiet = False, 0
    if active:
        spans.append((pd.Timestamp(timestamps[start]), pd.Timestamp(timestamps[-1])))
    merged = []
    for span in spans:
        if merged and (span[0] - merged[-1][1]).total_seconds() <= max(2 * dt, merge_s):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged

