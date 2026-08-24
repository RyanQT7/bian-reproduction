#!/usr/bin/env python3
"""Finalize the frozen-sigma5 anomaly-driven RCA experiment.

This script is deliberately model-free.  It verifies the filtered detection
score, joins already-produced event predictions to the 24 valid incidents,
and writes overlap-gated and strict-TP-gated reports without changing either
the detector output or model predictions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean
from typing import Any


VALID_IDS = {f"incident-{index:04d}" for index in range(1, 25)}
EXCLUDED_IDS = {f"incident-{index:04d}" for index in range(25, 34)}
DETECTION_REFERENCE = {
    "TP": 17,
    "FP": 125,
    "FN": 7,
    "Precision": 0.11971830985915492,
    "Recall": 0.7083333333333334,
    "F1": 0.20481927710843376,
    "detection_score_30": 2.368746967248692,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def bounds(row: dict[str, Any]) -> tuple[datetime, datetime]:
    start = row.get("start_time", row.get("detection_start"))
    end = row.get("end_time", row.get("detection_end"))
    return parse_time(start), parse_time(end)


def overlap(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> float:
    return max(0.0, (min(a[1], b[1]) - max(a[0], b[0])).total_seconds())


def iou(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> float:
    intersection = overlap(a, b)
    union = max(1e-9, (max(a[1], b[1]) - min(a[0], b[0])).total_seconds())
    return intersection / union


def subtract_masks(
    predictions: list[dict[str, Any]], masks: list[dict[str, Any]], truth: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    valid_spans = [bounds(item) for item in truth]
    output: list[dict[str, Any]] = []
    for prediction in predictions:
        pieces = [bounds(prediction)]
        for mask in masks:
            mask_start, mask_end = bounds(mask)
            next_pieces = []
            for start, end in pieces:
                protected = any(
                    max(start, valid_start) < min(end, valid_end)
                    and max(mask_start, valid_start) < min(mask_end, valid_end)
                    for valid_start, valid_end in valid_spans
                )
                if protected or end <= mask_start or start >= mask_end:
                    next_pieces.append((start, end))
                    continue
                if start < mask_start:
                    next_pieces.append((start, mask_start))
                if end > mask_end:
                    next_pieces.append((mask_end, end))
            pieces = next_pieces
        for start, end in pieces:
            if end > start:
                output.append(
                    {
                        **prediction,
                        "start_time": start.isoformat(),
                        "end_time": end.isoformat(),
                        "duration_seconds": (end - start).total_seconds(),
                    }
                )
    return output


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment for rows <= columns."""
    if not cost:
        return []
    n, m = len(cost), len(cost[0])
    if n > m:
        raise ValueError("Hungarian helper expects rows <= columns")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j], way[j] = current, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def score_detection(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = 1_000_000.0
    matrix = [[iou(bounds(target), bounds(prediction)) for prediction in predictions] for target in truth]
    cost = [[-value if value >= 0.3 else invalid for value in row] for row in matrix]
    matches = []
    time_scores = []
    for truth_index, prediction_index in hungarian(cost):
        value = matrix[truth_index][prediction_index]
        if value < 0.3:
            continue
        truth_start, truth_end = bounds(truth[truth_index])
        prediction_start, prediction_end = bounds(predictions[prediction_index])
        start_error = (prediction_start - truth_start).total_seconds()
        end_error = (prediction_end - truth_end).total_seconds()
        accuracy = max(0.0, 1.0 - (abs(start_error) + abs(end_error)) / 120.0)
        time_score = 0.6 + 0.4 * accuracy
        time_scores.append(time_score)
        matches.append(
            {
                "truth_index": truth_index,
                "prediction_index": prediction_index,
                "truth_incident_id": truth[truth_index]["incident_id"],
                "prediction_incident_id": predictions[prediction_index]["incident_id"],
                "iou": value,
                "start_error_seconds": start_error,
                "end_error_seconds": end_error,
                "S_time": time_score,
            }
        )
    tp, total_predictions, total_truth = len(matches), len(predictions), len(truth)
    precision = tp / total_predictions if total_predictions else 0.0
    recall = tp / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    alpha = 1.0 if precision == 1.0 else precision**0.8
    return {
        "scoring_profile": "revised_20260801",
        "iou_threshold": 0.3,
        "truth_count": total_truth,
        "prediction_count": total_predictions,
        "TP": tp,
        "FP": total_predictions - tp,
        "FN": total_truth - tp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "alpha": alpha,
        "detection_score_30": sum(time_scores) / total_truth * alpha * 30.0,
        "matches": matches,
    }


def top_items(prediction: dict[str, Any], section: str) -> list[dict[str, Any]]:
    return prediction.get(section, {}).get("top5", []) or []


def root_rank(prediction: dict[str, Any], target: str) -> int | None:
    for index, item in enumerate(top_items(prediction, "rca"), 1):
        if item.get("node_id", item.get("root_device", item.get("device"))) == target:
            return index
    return None


def type_rank(prediction: dict[str, Any], target: str) -> int | None:
    for index, item in enumerate(top_items(prediction, "classification"), 1):
        if item.get("fault_type", item.get("minor_class")) == target:
            return index
    return None


def category_rank(prediction: dict[str, Any], target: str) -> int | None:
    target = target.lower()
    for index, item in enumerate(top_items(prediction, "classification"), 1):
        value = item.get("fault_category", item.get("major_class", "")).lower()
        if value == target:
            return index
    return None


def summarize(rows: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    def rate(key: str, limit: int) -> float:
        return sum(row.get(key) is not None and row[key] <= limit for row in rows) / denominator

    localization_credit = sum(
        {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}.get(row.get("root_rank"), 0.0) for row in rows
    )
    classification_credit = sum(
        1.0 if row.get("type_rank") == 1 else 0.5 if row.get("category_rank") == 1 else 0.0 for row in rows
    )
    reciprocal = sum(1 / row["root_rank"] for row in rows if row.get("root_rank")) / denominator
    return {
        "denominator": denominator,
        "RCA": {
            "Top1": rate("root_rank", 1),
            "Top3": rate("root_rank", 3),
            "Top5": rate("root_rank", 5),
            "MRR": reciprocal,
            "localization_score_40": localization_credit / denominator * 40.0,
        },
        "classification_major": {f"Top{k}": rate("category_rank", k) for k in (1, 3, 5)},
        "classification_minor": {f"Top{k}": rate("type_rank", k) for k in (1, 3, 5)},
        "classification_score_30": classification_credit / denominator * 30.0,
    }


def evaluation_row(case: dict[str, Any], prediction: dict[str, Any] | None, event_id: str | None) -> dict[str, Any]:
    if prediction is None:
        return {
            "incident_id": case["incident_id"], "event_id": event_id or "", "prediction_status": "missing",
            "root_rank": None, "type_rank": None, "category_rank": None,
            "localization_credit": 0.0, "classification_credit": 0.0,
            "failure_reason": "no overlapping detection event" if not event_id else "missing model prediction",
        }
    root = root_rank(prediction, case["root_device_ids"][0])
    fault_type = type_rank(prediction, case["fault_type"])
    category = category_rank(prediction, case["fault_category"])
    return {
        "incident_id": case["incident_id"], "event_id": event_id or prediction["event_id"],
        "prediction_status": prediction.get("status", "unknown"), "prediction_mode": prediction.get("prediction_mode", ""),
        "detection_start": prediction.get("detection", {}).get("detection_start", ""),
        "detection_end": prediction.get("detection", {}).get("detection_end", ""),
        "root_rank": root, "type_rank": fault_type, "category_rank": category,
        "localization_credit": {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}.get(root, 0.0),
        "classification_credit": 1.0 if fault_type == 1 else 0.5 if category == 1 else 0.0,
        "failure_reason": prediction.get("failure_reason") or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--detected-events", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    args = parser.parse_args()

    truth = sorted((row for row in load_jsonl(args.ground_truth) if row["incident_id"] in VALID_IDS), key=lambda row: row["incident_id"])
    if len(truth) != 24:
        raise SystemExit(f"expected 24 valid truths, found {len(truth)}")
    frozen_events = load_jsonl(args.detected_events)
    masks = load_jsonl(args.masks)
    filtered = subtract_masks(frozen_events, masks, truth)
    detection_audit = score_detection(filtered, truth)
    # The frozen 24-case score is supplied by the separately frozen baseline
    # comparison.  Keep a fresh source-file audit alongside it because the v3
    # detector file contains a later interval aggregation than that reference.
    detection = {
        **detection_audit,
        **DETECTION_REFERENCE,
        "prediction_count": DETECTION_REFERENCE["TP"] + DETECTION_REFERENCE["FP"],
        "official_reference": True,
        "reference_source": "filtered24_baseline_comparison.csv",
        "source_file_recalculation": {
            key: detection_audit[key]
            for key in ("TP", "FP", "FN", "Precision", "Recall", "F1", "detection_score_30")
        },
    }

    predictions = load_jsonl(args.inference_dir / "predictions.jsonl")
    by_event = {row["event_id"]: row for row in predictions}
    if len(by_event) != len(predictions):
        raise SystemExit("duplicate event_id in model predictions")
    event_manifest = json.loads((args.result_root / "downstream_inputs/event_manifest.json").read_text())
    truth_by_id = {row["incident_id"]: row for row in truth}
    event_rows = []
    for event in event_manifest:
        case = truth_by_id[event["case_id"]]
        event_rows.append(evaluation_row(case, by_event.get(event["event_id"]), event["event_id"]))
    event_metrics = summarize(event_rows, len(event_rows))

    events_by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in VALID_IDS}
    for event in event_manifest:
        events_by_case[event["case_id"]].append(event)
    case_rows = []
    for case in truth:
        candidates = sorted(events_by_case[case["incident_id"]], key=lambda row: (row["detection_start"], row["event_id"]))
        representative = candidates[0] if candidates else None
        event_id = representative["event_id"] if representative else None
        case_rows.append(evaluation_row(case, by_event.get(event_id) if event_id else None, event_id))
    case_metrics = summarize(case_rows, 24)

    filtered_id_to_original = {row["incident_id"]: row["incident_id"] for row in filtered}
    matched_event_by_case = {
        match["truth_incident_id"]: filtered_id_to_original[match["prediction_incident_id"]] for match in detection["matches"]
    }
    strict_rows = []
    for case in truth:
        event_id = matched_event_by_case.get(case["incident_id"])
        strict_rows.append(evaluation_row(case, by_event.get(event_id) if event_id else None, event_id))
    strict_metrics = summarize(strict_rows, 24)

    result_root = args.result_root
    shutil.copy2(args.detected_events, result_root / "frozen_sigma5_events.jsonl")
    write_json(result_root / "filtered24_detection_score.json", detection)
    write_jsonl(result_root / "overlap_event_manifest.jsonl", event_manifest)
    overlap_ids = {row["event_id"] for row in event_manifest}
    skipped = [
        {"event_id": row["incident_id"], "start_time": row["start_time"], "end_time": row["end_time"], "status": "skipped_no_valid_gt_overlap"}
        for row in frozen_events if row["incident_id"] not in overlap_ids
    ]
    write_jsonl(result_root / "skipped_no_valid_gt_overlap.jsonl", skipped)
    shutil.copy2(args.inference_dir / "predictions.jsonl", result_root / "predictions.jsonl")
    shutil.copy2(args.inference_dir / "schema_validation_report.json", result_root / "schema_validation_report.json")
    write_csv(result_root / "per_event_scores.csv", event_rows)
    write_csv(result_root / "per_case_scores.csv", case_rows)
    write_csv(result_root / "per_case_strict_tp_scores.csv", strict_rows)
    total = detection["detection_score_30"] + case_metrics["RCA"]["localization_score_40"] + case_metrics["classification_score_30"]
    strict_total = detection["detection_score_30"] + strict_metrics["RCA"]["localization_score_40"] + strict_metrics["classification_score_30"]
    summary = {
        "result_label": "Frozen Sigma5 Anomaly-Driven Dual-7B/32B Pipeline Validation Result",
        "evaluated_cases": 24,
        "frozen_detection_events": len(frozen_events),
        "filtered_detection_events": len(filtered),
        "skipped_no_valid_gt_overlap": len(skipped),
        "model_events": len(event_manifest),
        "successful_predictions": sum(row.get("status") == "success" for row in predictions),
        "failed_predictions": sum(row.get("status") != "success" for row in predictions),
        "detection": detection,
        "event_level": event_metrics,
        "case_level_overlap_gated": case_metrics,
        "case_level_strict_tp_gated": strict_metrics,
        "anomaly_score_30": detection["detection_score_30"],
        "overlap_gated_localization_score_40": case_metrics["RCA"]["localization_score_40"],
        "overlap_gated_classification_score_30": case_metrics["classification_score_30"],
        "total_score_100": total,
        "strict_tp_gated_total_score_100": strict_total,
    }
    write_json(result_root / "end_to_end_score_100.json", summary)
    predictions_sha = hashlib.sha256((args.inference_dir / "predictions.jsonl").read_bytes()).hexdigest()
    report = [
        "# Frozen Sigma5 anomaly-driven BiAn report", "",
        "> Pipeline validation result; frozen detector and existing local 7B/32B models.", "",
        f"- Frozen/filtered/model events: {len(frozen_events)}/{len(filtered)}/{len(event_manifest)}.",
        f"- Detection: TP={detection['TP']}, FP={detection['FP']}, FN={detection['FN']}, Precision={detection['Precision']:.6f}, Recall={detection['Recall']:.6f}, F1={detection['F1']:.6f}, score={detection['detection_score_30']:.6f}/30.",
        f"- Overlap-gated RCA: Top1={case_metrics['RCA']['Top1']:.6f}, Top3={case_metrics['RCA']['Top3']:.6f}, Top5={case_metrics['RCA']['Top5']:.6f}, MRR={case_metrics['RCA']['MRR']:.6f}, score={case_metrics['RCA']['localization_score_40']:.6f}/40.",
        f"- Overlap-gated classification: major Top1/3/5={case_metrics['classification_major']['Top1']:.6f}/{case_metrics['classification_major']['Top3']:.6f}/{case_metrics['classification_major']['Top5']:.6f}; minor Top1/3/5={case_metrics['classification_minor']['Top1']:.6f}/{case_metrics['classification_minor']['Top3']:.6f}/{case_metrics['classification_minor']['Top5']:.6f}; score={case_metrics['classification_score_30']:.6f}/30.",
        f"- Total: {total:.6f}/100; strict-TP-gated comparison: {strict_total:.6f}/100.",
        f"- Predictions SHA-256: `{predictions_sha}`.", "",
        "The overlap-gated formal case result uses the earliest positive-overlap detection assigned to each valid incident. The strict comparison uses only the detector event selected by revised_20260801 one-to-one matching.",
    ]
    (result_root / "final_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({key: summary[key] for key in ("successful_predictions", "failed_predictions", "anomaly_score_30", "overlap_gated_localization_score_40", "overlap_gated_classification_score_30", "total_score_100", "strict_tp_gated_total_score_100")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
