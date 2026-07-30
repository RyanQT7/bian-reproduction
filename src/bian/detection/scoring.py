from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def score(predictions: list[dict], truth: list[dict], denominator: int = 33) -> dict:
    def bounds(x):
        import pandas as pd
        keys = (("start_time", "end_time"), ("start", "end"), ("begin", "end"))
        for a, b in keys:
            if a in x and b in x:
                return pd.Timestamp(x[a]), pd.Timestamp(x[b])
        raise ValueError(f"missing interval fields: {sorted(x)}")
    pb, tb = list(map(bounds, predictions)), list(map(bounds, truth))
    weights = np.zeros((len(pb), len(tb)))
    for i, (ps, pe) in enumerate(pb):
        for j, (ts, te) in enumerate(tb):
            inter = max(0.0, (min(pe, te) - max(ps, ts)).total_seconds())
            union = max(1e-9, (max(pe, te) - min(ps, ts)).total_seconds())
            weights[i, j] = inter / union
    rows, cols = linear_sum_assignment(-weights) if weights.size else ([], [])
    matches, used_p, used_t, total = [], set(), set(), 0.0
    for i, j in zip(rows, cols):
        if weights[i, j] < .5:
            continue
        ps, pe = pb[i]; ts, te = tb[j]
        ds, de = (ps-ts).total_seconds(), (pe-te).total_seconds()
        s_acc = max(0.0, 1.0 - (abs(ds)+abs(de))/120.0)
        s_time = 0.0 if ps < ts else .6 + .4*s_acc
        total += s_time; used_p.add(i); used_t.add(j)
        matches.append({"prediction": i, "truth": j, "iou": weights[i,j],
                        "start_delta_seconds": ds, "end_delta_seconds": de, "S_time": s_time})
    tp, fp, fn = len(matches), len(pb)-len(matches), len(tb)-len(matches)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/denominator
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    alpha = 1.0 if precision == 1 else precision**1.2
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall,
            "F1": f1, "alpha": alpha, "detection_score_30": total/denominator*alpha*30,
            "matches": matches, "unmatched_predictions": sorted(set(range(len(pb)))-used_p),
            "unmatched_truth": sorted(set(range(len(tb)))-used_t)}
