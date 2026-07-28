"""Seventy-point scoring for frozen, one-record-per-incident RCA output."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from bian.data.validators import ValidationError


LOCALIZATION_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
MISSING_LABEL = "__missing__"


def _index_unique(
    records: list[dict[str, Any]], name: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        incident_id = record.get("incident_id")
        if not incident_id:
            raise ValidationError(f"{name} record lacks incident_id")
        if incident_id in result:
            raise ValidationError(f"{name} contains duplicate incident_id {incident_id}")
        result[incident_id] = record
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _precision_recall_f1(
    truths: list[str], predictions: list[str]
) -> dict[str, dict[str, float | int]]:
    labels = sorted(set(truths) | set(predictions))
    result = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = sum(t == label and p != label for t, p in zip(truths, predictions))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        result[label] = {
            "support": sum(t == label for t in truths),
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
        }
    return result


def score_predictions(
    *,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    root_node_field: str,
    fault_type_field: str = "fault_type",
    fault_category_field: str = "fault_category",
    expected_case_count: int = 10,
) -> dict[str, Any]:
    truth_by_id = _index_unique(ground_truth, "ground truth")
    prediction_by_id = _index_unique(predictions, "predictions")
    if len(truth_by_id) != expected_case_count:
        raise ValidationError(
            f"expected {expected_case_count} ground-truth cases, got {len(truth_by_id)}"
        )
    unknown_predictions = sorted(set(prediction_by_id) - set(truth_by_id))
    if unknown_predictions:
        raise ValidationError(f"predictions contain unknown incidents: {unknown_predictions}")

    per_case = []
    top_hits = Counter()
    reciprocal_rank_sum = 0.0
    localization_credit_sum = 0.0
    classification_credit_sum = 0.0
    classification_top3_hits = 0
    true_types: list[str] = []
    predicted_types: list[str] = []
    true_categories: list[str] = []
    predicted_categories: list[str] = []
    type_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    category_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    localization_by_role: dict[str, list[float]] = defaultdict(list)
    localization_by_region: dict[str, list[float]] = defaultdict(list)

    for incident_id in sorted(truth_by_id):
        truth = truth_by_id[incident_id]
        prediction = prediction_by_id.get(incident_id)
        root_node = truth.get(root_node_field)
        true_type = truth.get(fault_type_field)
        true_category = truth.get(fault_category_field)
        if not all(isinstance(value, str) and value for value in (root_node, true_type, true_category)):
            raise ValidationError(f"{incident_id}: incomplete ground truth")
        parts = root_node.split("-")
        if len(parts) < 4:
            raise ValidationError(f"{incident_id}: invalid root node ID {root_node!r}")
        region_id = "-".join(parts[:2])
        role = "-".join(parts[2:])
        hit_rank = None
        localization_credit = 0.0
        predicted_type = MISSING_LABEL
        predicted_category = MISSING_LABEL
        top3_types: list[str] = []
        format_error = None

        if prediction is not None:
            top5 = prediction.get("top5_root_causes")
            if not isinstance(top5, list) or len(top5) != 5:
                format_error = "invalid_top5_length"
            else:
                node_ids = [
                    item.get("node_id") for item in top5 if isinstance(item, dict)
                ]
                if len(node_ids) != 5 or len(set(node_ids)) != 5:
                    format_error = "duplicate_or_malformed_top5"
                elif root_node in node_ids:
                    hit_rank = node_ids.index(root_node) + 1
                    localization_credit = LOCALIZATION_WEIGHTS[hit_rank]
            predicted_type = prediction.get("predicted_fault_type") or MISSING_LABEL
            predicted_category = (
                prediction.get("predicted_fault_category") or MISSING_LABEL
            )
            top3 = prediction.get("fault_type_top3")
            if isinstance(top3, list):
                top3_types = [
                    item.get("fault_type")
                    for item in top3
                    if isinstance(item, dict) and item.get("fault_type")
                ]

        if hit_rank is not None:
            reciprocal_rank_sum += 1.0 / hit_rank
            for k in (1, 2, 3, 5):
                if hit_rank <= k:
                    top_hits[k] += 1
        localization_credit_sum += localization_credit
        localization_by_role[role].append(localization_credit)
        localization_by_region[region_id].append(localization_credit)

        if predicted_type == true_type:
            classification_credit = 1.0
        elif predicted_category == true_category:
            classification_credit = 0.5
        else:
            classification_credit = 0.0
        classification_credit_sum += classification_credit
        if true_type in top3_types:
            classification_top3_hits += 1
        true_types.append(true_type)
        predicted_types.append(predicted_type)
        true_categories.append(true_category)
        predicted_categories.append(predicted_category)
        type_confusion[true_type][predicted_type] += 1
        category_confusion[true_category][predicted_category] += 1
        per_case.append(
            {
                "incident_id": incident_id,
                "root_node": root_node,
                "hit_rank": hit_rank,
                "localization_credit": localization_credit,
                "true_fault_type": true_type,
                "predicted_fault_type": predicted_type,
                "true_fault_category": true_category,
                "predicted_fault_category": predicted_category,
                "classification_credit": classification_credit,
                "classification_top3_hit": true_type in top3_types,
                "prediction_missing": prediction is None,
                "format_error": format_error,
            }
        )

    case_count = len(truth_by_id)
    localization_score = localization_credit_sum / case_count * 40.0
    classification_score = classification_credit_sum / case_count * 30.0
    result = {
        "evaluated_cases": case_count,
        "score_scope": {
            "localization_max": 40,
            "classification_max": 30,
            "total_max": 70,
            "incident_detection_evaluated": False,
        },
        "localization": {
            "score_40": localization_score,
            "top1_accuracy": top_hits[1] / case_count,
            "top2_accuracy": top_hits[2] / case_count,
            "top3_accuracy": top_hits[3] / case_count,
            "top5_accuracy": top_hits[5] / case_count,
            "mean_reciprocal_rank": reciprocal_rank_sum / case_count,
            "by_root_role": {
                key: {
                    "case_count": len(values),
                    "mean_credit": sum(values) / len(values),
                }
                for key, values in sorted(localization_by_role.items())
            },
            "by_region": {
                key: {
                    "case_count": len(values),
                    "mean_credit": sum(values) / len(values),
                }
                for key, values in sorted(localization_by_region.items())
            },
        },
        "classification": {
            "score_30": classification_score,
            "fault_type_top1_accuracy": sum(
                truth == prediction
                for truth, prediction in zip(true_types, predicted_types)
            )
            / case_count,
            "fault_type_top3_hit_rate": classification_top3_hits / case_count,
            "fault_category_accuracy": sum(
                truth == prediction
                for truth, prediction in zip(true_categories, predicted_categories)
            )
            / case_count,
            "fault_type_confusion_matrix": {
                truth: dict(sorted(counts.items()))
                for truth, counts in sorted(type_confusion.items())
            },
            "fault_category_confusion_matrix": {
                truth: dict(sorted(counts.items()))
                for truth, counts in sorted(category_confusion.items())
            },
            "per_fault_type": _precision_recall_f1(true_types, predicted_types),
        },
        "total_score_70": localization_score + classification_score,
        "per_case": per_case,
        "failure_cases": [
            item
            for item in per_case
            if item["localization_credit"] < 1.0
            or item["classification_credit"] < 1.0
        ],
    }
    for value in (
        localization_score,
        classification_score,
        result["total_score_70"],
    ):
        if not math.isfinite(value):
            raise ValidationError("non-finite score generated")
    return result
