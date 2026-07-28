"""Validate and freeze a complete ten-case prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.real_preprocessing import load_jsonl
from bian.data.validators import ValidationError
from bian.predictions import (
    freeze_predictions,
    load_prediction_jsonl,
    validate_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    args = parser.parse_args()
    incidents = load_jsonl(args.bundle_root / "experiment" / "incidents.jsonl")
    inventory = json.loads(
        (args.bundle_root / "topology" / "candidate_inventory.json").read_text()
    )
    taxonomy = json.loads(
        (args.bundle_root / "schemas" / "fault_taxonomy.json").read_text()
    )
    records = load_prediction_jsonl(args.predictions)
    validation = validate_predictions(
        records,
        expected_incident_ids={item["incident_id"] for item in incidents},
        candidate_node_ids={item["node_id"] for item in inventory},
        taxonomy=taxonomy,
        expected_rank_rounds=args.rank_rounds,
    )
    if not validation["valid"]:
        raise ValidationError("; ".join(validation["errors"]))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = freeze_predictions(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        validation=validation,
        manifest={
            "evaluated_cases": 10,
            "git_commit": commit,
            "random_seed": args.seed,
            "rank_rounds": args.rank_rounds,
            "small_model": args.small_model,
            "large_model": args.large_model,
        },
    )
    print(f"Frozen: {result['frozen_path']}")
    print(f"SHA-256: {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
