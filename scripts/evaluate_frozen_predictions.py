"""Evaluate a previously frozen prediction file on the 70-point RCA rubric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.real_preprocessing import load_jsonl
from bian.predictions import load_prediction_jsonl, sha256_file
from bian.evaluation.real_scoring import score_predictions
from bian.data.validators import ValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--checksum-file", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-node-field", required=True)
    parser.add_argument("--fault-type-field", default="fault_type")
    parser.add_argument("--fault-category-field", default="fault_category")
    args = parser.parse_args()
    expected_checksum = args.checksum_file.read_text().split()[0]
    actual_checksum = sha256_file(args.frozen_predictions)
    if actual_checksum != expected_checksum:
        raise ValidationError("frozen prediction checksum mismatch")
    predictions = load_prediction_jsonl(args.frozen_predictions)
    truth = load_jsonl(args.ground_truth)
    result = score_predictions(
        predictions=predictions,
        ground_truth=truth,
        root_node_field=args.root_node_field,
        fault_type_field=args.fault_type_field,
        fault_category_field=args.fault_category_field,
        expected_case_count=10,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Evaluated cases: {result['evaluated_cases']}; "
        f"localization={result['localization']['score_40']:.3f}/40; "
        f"classification={result['classification']['score_30']:.3f}/30; "
        f"total={result['total_score_70']:.3f}/70"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
