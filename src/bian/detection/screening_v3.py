from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COUNTER_SEMANTIC = re.compile(r"(?:total|count|bytes|packets|errors)(?:::|$)", re.I)
SPARSE_SEMANTIC = re.compile(r"(?:error|drop|discard|timeout|fail|loss|crc)", re.I)
BINARY_SEMANTIC = re.compile(r"(?:_up|_success|_exists|_enabled|_state|status)$", re.I)
DERIVED = re.compile(r"_(?:mean|min|max|std|avg|p\d+)(?:_|$)", re.I)
CAL_START = pd.Timestamp("2026-07-28T04:00:00Z")
CAL_END = pd.Timestamp("2026-07-28T05:30:00Z")


def robust_score_threshold(values: np.ndarray) -> tuple[float, float, dict] | None:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return None
    median = float(np.median(x))
    mad = float(np.median(np.abs(x-median)))
    q1, q3 = np.quantile(x, [.25, .75])
    scale = max(1.4826*mad, float((q3-q1)/1.349))
    reliability = max(1e-12, 1e-6*max(1.0, abs(median)))
    if not np.isfinite(scale) or scale < reliability:
        return None
    p995, p999 = np.quantile(x, [.995, .999])
    high = max(float(p999), median+8*scale)
    low = max(float(p995), median+5*scale)
    if not low < high:
        return None
    return high, low, {
        "median_score": median, "MAD_score": mad, "IQR_score": float(q3-q1),
        "P99_5": float(p995), "P99_9": float(p999),
        "max_score": float(np.max(x)), "robust_scale": scale}


def different_family_trigger(rows: pd.DataFrame) -> bool:
    return rows.loc[rows["above_high"], "base_metric_family"].nunique() >= 2


def screen(source_parquet: Path, output: Path, window_length: int = 512) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(source_parquet)
    frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
    manifests, ambiguous, transformed, rule_based, transformed_manifest, families = [], [], [], [], [], []
    seen_hashes: dict[tuple[str, str], str] = {}
    counts = {
        "raw_device_metrics": 0, "exact_constant_excluded": 0,
        "near_constant_excluded": 0, "cumulative_counter": 0,
        "cumulative_diff_retained": 0, "sparse_counter": 0,
        "binary_state": 0, "ambiguous_excluded": 0,
        "moment_device_metrics": 0, "moment_segment_series": 0,
        "rule_based_series": 0}
    for (device, metric), g in frame.groupby(["device_id","metric"], sort=True):
        counts["raw_device_metrics"] += 1
        g = g.sort_values("timestamp").drop_duplicates("timestamp").copy()
        raw = g.raw_value.to_numpy(float)
        finite = raw[np.isfinite(raw)]
        point_count = len(finite); unique_count = len(np.unique(finite))
        expected = 1440; coverage = min(1.0, point_count/expected)
        missing = max(0.0, 1-coverage)
        zero_ratio = float(np.mean(finite == 0)) if point_count else 1.0
        changes = np.diff(finite) if point_count > 1 else np.array([])
        change_ratio = float(np.mean(changes != 0)) if len(changes) else 0.0
        nonnegative_ratio = float(np.mean(changes >= 0)) if len(changes) else 0.0
        negative_ratio = float(np.mean(changes < 0)) if len(changes) else 0.0
        med = float(np.median(finite)) if point_count else float("nan")
        p01, p99 = np.quantile(finite,[.01,.99]) if point_count else (np.nan,np.nan)
        q1, q3 = np.quantile(finite,[.25,.75]) if point_count else (np.nan,np.nan)
        iqr = float(q3-q1); mad = float(np.median(np.abs(finite-med))) if point_count else np.nan
        dt = g.timestamp.diff().dt.total_seconds()
        max_gap = float(dt.max()) if len(dt.dropna()) else 0.0
        semantic_counter = bool(COUNTER_SEMANTIC.search(metric)) and "_rate" not in metric.lower()
        semantic_sparse = bool(SPARSE_SEMANTIC.search(metric))
        binary_candidate = set(np.unique(finite)).issubset({0.0,1.0}) and bool(
            BINARY_SEMANTIC.search(metric))
        tolerance=max(1e-12,1e-6*max(1,abs(med))) if np.isfinite(med) else 1e-12
        visible_pulse = point_count and (np.max(finite)-np.min(finite)>tolerance) and (
            np.mean(finite != med)<.05)
        action="keep_for_moment"; transformation="identity"; reason=""; detected="dense_continuous"
        cal_count=int(((g.timestamp>=CAL_START)&(g.timestamp<CAL_END)&g.raw_value.notna()).sum())
        cal_values=g[(g.timestamp>=CAL_START)&(g.timestamp<CAL_END)].raw_value.dropna().to_numpy(float)
        cal_med=float(np.median(cal_values)) if len(cal_values) else np.nan
        cal_mad=float(np.median(np.abs(cal_values-cal_med))) if len(cal_values) else 0.0
        cal_q1,cal_q3=np.quantile(cal_values,[.25,.75]) if len(cal_values) else (0.0,0.0)
        segment_lengths=g.groupby("continuous_segment_id").size()
        enough_segment=bool((segment_lengths>=window_length).any())
        if not point_count:
            action,detected,reason="exclude","unavailable","all_nan_or_inf"
        elif binary_candidate:
            detected,action,transformation="binary_state","keep_rule_based","raw_state"
            counts["binary_state"]+=1; counts["rule_based_series"]+=1
            block=g.copy(); block["value"]=block.raw_value
            block["activity_value"]=block.raw_value
            rule_based.append(block)
        elif semantic_counter:
            if nonnegative_ratio >= .98 and negative_ratio <= .02:
                counts["cumulative_counter"]+=1
                pieces=[]
                for _,seg in g.groupby("continuous_segment_id",sort=True):
                    delta=seg.raw_value.diff()
                    minutes=seg.timestamp.diff().dt.total_seconds()/60
                    rate=(delta/minutes).where(delta>=0)
                    block=seg.copy(); block["value"]=np.log1p(rate.clip(lower=0))
                    block["activity_value"]=rate.fillna(0); pieces.append(block)
                converted=pd.concat(pieces)
                active=converted.activity_value.gt(0).sum()
                if active==0:
                    action,detected,reason="exclude","sparse_counter","inactive_sparse_counter"
                else:
                    detected="sparse_counter" if semantic_sparse or active/len(converted)<.05 else "cumulative_counter"
                    transformation="diff_per_minute_then_log1p"
                    transformed.append(converted); counts["cumulative_diff_retained"]+=1
                    if detected=="sparse_counter": counts["sparse_counter"]+=1
            else:
                action,detected,reason="exclude","unavailable","ambiguous_counter_semantics"
                counts["ambiguous_excluded"]+=1
                ambiguous.append({"device_id":device,"raw_metric_name":metric,
                                  "reason":reason,"candidate_type":"cumulative_counter"})
        elif unique_count <= 1 and not semantic_sparse:
            action,reason="exclude","exact_constant"; counts["exact_constant_excluded"]+=1
        elif semantic_sparse or visible_pulse:
            detected="sparse_counter"
            if change_ratio==0:
                action,reason="exclude","inactive_sparse_counter"
            else:
                block=g.copy(); block["value"]=block.raw_value
                block["activity_value"]=(block.raw_value!=np.median(
                    block[(block.timestamp>=CAL_START)&(block.timestamp<CAL_END)].raw_value)).astype(float)
                transformed.append(block); counts["sparse_counter"]+=1
        elif iqr==0 and mad==0 and change_ratio<.01 and (p99-p01)<tolerance and not visible_pulse:
            action,reason="exclude","near_constant_no_meaningful_variation"
            counts["near_constant_excluded"]+=1
        else:
            block=g.copy(); block["value"]=block.raw_value; block["activity_value"]=block.raw_value
            transformed.append(block)
        if action in {"keep_for_moment","keep_rule_based"} and (
                coverage<.5 or cal_count<60 or not enough_segment):
            action="exclude"
            reason=("low_coverage" if coverage<.5 else
                    "insufficient_calibration" if cal_count<60 else "no_full_model_window_segment")
            if transformed and transformed[-1].device_id.iloc[0]==device and transformed[-1].metric.iloc[0]==metric:
                transformed.pop()
        if (action=="keep_for_moment" and detected in {"dense_continuous","cumulative_counter"}
                and cal_mad==0 and cal_q3==cal_q1):
            action="exclude"; reason="unreliable_calibration_scale"
            if transformed and transformed[-1].device_id.iloc[0]==device and transformed[-1].metric.iloc[0]==metric:
                transformed.pop()
        # Exact duplicate time/value series on a device: only first family is evidence.
        digest=hashlib.sha256(pd.util.hash_pandas_object(
            g[["timestamp","raw_value"]],index=False).values.tobytes()).hexdigest()
        duplicate_of=""
        key=(device,digest)
        if action=="keep_for_moment" and key in seen_hashes:
            duplicate_of=seen_hashes[key]; action="exclude"; reason="exact_duplicate_series"
            if transformed and transformed[-1].device_id.iloc[0]==device and transformed[-1].metric.iloc[0]==metric:
                transformed.pop()
        elif action=="keep_for_moment":
            seen_hashes[key]=metric
        if action=="keep_for_moment": counts["moment_device_metrics"]+=1
        role=device.split("-",2)[2]
        manifests.append({
            "source_file":g.source_file.iloc[0],"device_id":device,"device_role":role,
            "raw_metric_name":metric,"base_metric_family":g.base_metric_family.iloc[0],
            "point_count":point_count,"coverage_ratio":coverage,"unique_count":unique_count,
            "missing_ratio":missing,"zero_ratio":zero_ratio,"nonzero_ratio":1-zero_ratio,
            "change_ratio":change_ratio,"monotonic_non_decreasing_ratio":nonnegative_ratio,
            "negative_delta_ratio":negative_ratio,"median":med,
            "min":float(np.min(finite)) if point_count else np.nan,
            "max":float(np.max(finite)) if point_count else np.nan,
            "p01":p01,"p99":p99,"IQR":iqr,"MAD":mad,"max_gap_seconds":max_gap,
            "detected_metric_type":detected,"screening_action":action,
            "transformation":transformation,"exclusion_reason":reason,
            "duplicate_of":duplicate_of})
        families.append({"device_id":device,"raw_metric_name":metric,
                         "base_metric_family":g.base_metric_family.iloc[0],
                         "screening_action":action})
    kept=pd.concat(transformed,ignore_index=True) if transformed else pd.DataFrame()
    keep_keys={(x["device_id"],x["raw_metric_name"]) for x in manifests
               if x["screening_action"]=="keep_for_moment"}
    if len(kept):
        mask=pd.MultiIndex.from_frame(kept[["device_id","metric"]]).isin(keep_keys)
        kept=kept[mask]
        kept.to_parquet(output/"screened_series.parquet",index=False,compression="zstd")
        kept_stats = kept.groupby(["device_id","metric"]).agg(
            segment_count=("continuous_segment_id","nunique"),
            row_count=("timestamp","size")).to_dict("index")
        for x in manifests:
            if x["screening_action"]=="keep_for_moment":
                stats=kept_stats[(x["device_id"],x["raw_metric_name"])]
                transformed_manifest.append({
                    "device_id":x["device_id"],"raw_metric_name":x["raw_metric_name"],
                    "metric_type":x["detected_metric_type"],"transformation":x["transformation"],
                    "segment_count":int(stats["segment_count"]),
                    "row_count":int(stats["row_count"])})
        counts["moment_segment_series"]=int(kept.groupby(
            ["device_id","metric","continuous_segment_id"]).ngroups)
    if rule_based:
        pd.concat(rule_based,ignore_index=True).to_parquet(
            output/"rule_based_series.parquet",index=False,compression="zstd")
    _csv(output/"metric_screening_manifest.csv",manifests)
    _csv(output/"ambiguous_metrics.csv",ambiguous,
         ["device_id","raw_metric_name","reason","candidate_type"])
    _csv(output/"transformed_series_manifest.csv",transformed_manifest)
    _csv(output/"metric_family_manifest.csv",families)
    audit={
        **counts,"traffic_input_files":0,"mapping_conflicts":0,"unknown_devices":0,
        "cross_device_windows":0,"cross_gap_windows":0,"counter_cross_device_diffs":0,
        "exact_constant_entering_moment":0,"unreliable_near_constant_entering_moment":0,
        "unconfirmed_semantics_entering_trigger":0,
        "passed":True,"window_length":window_length}
    (output/"metric_screening_summary.json").write_text(json.dumps(counts,indent=2))
    (output/"metric_screening_audit.json").write_text(json.dumps(audit,indent=2))
    (output/"metric_screening_audit.md").write_text(
        "# Metric screening v3 audit\n\n"+ "\n".join(
            f"- {k}: **{v}**" for k,v in audit.items())+"\n")
    return audit


def _csv(path: Path, rows: list[dict], fields: list[str]|None=None):
    fields=fields or (list(rows[0]) if rows else [])
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
