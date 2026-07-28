"""Validation and freezing for real RCA predictions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from typing import Any

from bian.data.validators import ValidationError


def load_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def _finite_nonnegative(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValidationError(f"{field} must be finite and non-negative")
    return float(value)


def validate_predictions(
    records: list[dict[str, Any]],
    *,
    expected_incident_ids: set[str],
    candidate_node_ids: set[str],
    taxonomy: list[dict[str, str]],
    expected_rank_rounds: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    ids = [record.get("incident_id") for record in records]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    unknown = sorted(set(ids) - expected_incident_ids)
    missing = sorted(expected_incident_ids - set(ids))
    if duplicates:
        errors.append(f"duplicate incident IDs: {duplicates}")
    if unknown:
        errors.append(f"unknown incident IDs: {unknown}")
    if missing:
        errors.append(f"missing incident IDs: {missing}")
    if len(records) != len(expected_incident_ids):
        errors.append(
            f"expected {len(expected_incident_ids)} records, got {len(records)}"
        )
    taxonomy_map = {item["fault_type"]: item["fault_category"] for item in taxonomy}

    for record in records:
        incident_id = str(record.get("incident_id", "<missing>"))
        status = record.get("prediction_status", "success")
        if status == "prediction_failed":
            forbidden = {
                "top5_root_causes",
                "predicted_fault_type",
                "predicted_fault_category",
                "fault_type_top3",
                "rank_of_ranks",
            } & set(record)
            if forbidden:
                errors.append(
                    f"{incident_id}: failed prediction contains answer fields "
                    f"{sorted(forbidden)}"
                )
            if not isinstance(record.get("error_type"), str) or not isinstance(
                record.get("error"), str
            ):
                errors.append(
                    f"{incident_id}: failed prediction requires error_type and error"
                )
            continue
        if status != "success":
            errors.append(f"{incident_id}: unknown prediction_status {status!r}")
            continue
        top5 = record.get("top5_root_causes")
        if not isinstance(top5, list) or len(top5) != 5:
            errors.append(f"{incident_id}: top5_root_causes must contain exactly 5 items")
            continue
        nodes = [item.get("node_id") for item in top5 if isinstance(item, dict)]
        if len(nodes) != 5 or len(set(nodes)) != 5:
            errors.append(f"{incident_id}: Top5 nodes must be five distinct values")
        illegal = sorted(set(nodes) - candidate_node_ids)
        if illegal:
            errors.append(f"{incident_id}: illegal Top5 nodes: {illegal}")
        ranks = [item.get("rank") for item in top5 if isinstance(item, dict)]
        if ranks != [1, 2, 3, 4, 5]:
            errors.append(f"{incident_id}: Top5 ranks must be 1..5 in order")
        try:
            scores = [
                _finite_nonnegative(item.get("failure_score"), "failure_score")
                for item in top5
            ]
            if not math.isclose(sum(scores), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"{incident_id}: normalized Top5 scores must sum to 1")
        except (ValidationError, AttributeError) as exc:
            errors.append(f"{incident_id}: {exc}")
        fault_top3 = record.get("fault_type_top3")
        if not isinstance(fault_top3, list) or len(fault_top3) != 3:
            errors.append(f"{incident_id}: fault_type_top3 must contain exactly 3 items")
            continue
        types = [item.get("fault_type") for item in fault_top3 if isinstance(item, dict)]
        if len(types) != 3 or len(set(types)) != 3:
            errors.append(f"{incident_id}: Top3 fault types must be distinct")
        ranks = [item.get("rank") for item in fault_top3 if isinstance(item, dict)]
        if ranks != [1, 2, 3]:
            errors.append(f"{incident_id}: fault type ranks must be 1..3 in order")
        for item in fault_top3:
            fault_type = item.get("fault_type")
            category = item.get("fault_category")
            if fault_type not in taxonomy_map:
                errors.append(f"{incident_id}: unknown fault type {fault_type!r}")
            elif taxonomy_map[fault_type] != category:
                errors.append(
                    f"{incident_id}: category {category!r} does not match {fault_type!r}"
                )
            try:
                _finite_nonnegative(item.get("confidence"), "confidence")
            except ValidationError as exc:
                errors.append(f"{incident_id}: {exc}")
        if fault_top3 and (
            record.get("predicted_fault_type") != fault_top3[0].get("fault_type")
            or record.get("predicted_fault_category")
            != fault_top3[0].get("fault_category")
        ):
            errors.append(f"{incident_id}: top-level classification must match Top1")
        rank_data = record.get("rank_of_ranks")
        if not isinstance(rank_data, dict):
            errors.append(f"{incident_id}: rank_of_ranks is required")
        else:
            rounds = rank_data.get("rounds")
            raw_rankings = rank_data.get("raw_rankings")
            if expected_rank_rounds is not None and rounds != expected_rank_rounds:
                errors.append(
                    f"{incident_id}: expected {expected_rank_rounds} rank rounds"
                )
            if not isinstance(raw_rankings, list) or len(raw_rankings) != rounds:
                errors.append(f"{incident_id}: raw ranking count must equal rounds")
            elif any(
                not isinstance(ranking, list)
                or len(ranking) < 5
                or len(set(ranking)) != len(ranking)
                or not set(ranking) <= candidate_node_ids
                for ranking in raw_rankings
            ):
                errors.append(f"{incident_id}: invalid raw Rank of Ranks ranking")
    return {
        "valid": not errors,
        "record_count": len(records),
        "expected_record_count": len(expected_incident_ids),
        "errors": errors,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_predictions(
    *,
    predictions_path: Path,
    output_dir: Path,
    validation: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not validation.get("valid"):
        raise ValidationError("cannot freeze invalid predictions")
    frozen_path = output_dir / "predictions_frozen.jsonl"
    checksum_path = output_dir / "predictions.sha256"
    validation_path = output_dir / "predictions.schema_validation.json"
    manifest_path = output_dir / "run_manifest.json"
    for path in (frozen_path, checksum_path, validation_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    shutil.copyfile(predictions_path, frozen_path)
    checksum = sha256_file(frozen_path)
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_path.write_text(
        f"{checksum}  {frozen_path.name}\n", encoding="utf-8"
    )
    rendered_manifest = {
        **manifest,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "predictions_file": frozen_path.name,
        "predictions_sha256": checksum,
        "python_version": platform.python_version(),
    }
    manifest_path.write_text(
        json.dumps(rendered_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(frozen_path, 0o444)
    return {
        "frozen_path": str(frozen_path),
        "sha256": checksum,
        "manifest_path": str(manifest_path),
    }
