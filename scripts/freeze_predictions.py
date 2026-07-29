"""Validate and freeze a complete ten-case prediction file."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distributions
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
    parser.add_argument(
        "--allow-variable-rank-rounds",
        action="store_true",
        help="Accept each successful localization with its recorded valid round count.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--large-model", required=True)
    parser.add_argument(
        "--git-commit",
        help="Inference commit to record; defaults to the current repository HEAD.",
    )
    parser.add_argument(
        "--model-configuration", default="dual_7b_pipeline_validation"
    )
    parser.add_argument(
        "--result-label", default="Dual-7B Pipeline Validation Result"
    )
    parser.add_argument("--development-dataset", action="store_true")
    parser.add_argument(
        "--stage1-large-model-replaced-by-7b",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stage2-large-model-replaced-by-7b",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        expected_rank_rounds=(
            None if args.allow_variable_rank_rounds else args.rank_rounds
        ),
    )
    if not validation["valid"]:
        raise ValidationError("; ".join(validation["errors"]))
    commit = args.git_commit
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    prompt_digest = hashlib.sha256()
    for prompt_path in sorted((PROJECT_ROOT / "src/bian/prompts").glob("*.txt")):
        prompt_digest.update(prompt_path.name.encode())
        prompt_digest.update(prompt_path.read_bytes())
    dependencies = sorted(
        f"{item.metadata['Name']}=={item.version}"
        for item in distributions()
        if item.metadata.get("Name")
    )
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
            "model_configuration": args.model_configuration,
            "result_label": args.result_label,
            "paper_model_equivalent": False,
            "development_dataset": args.development_dataset,
            "strict_blind_evaluation": False if args.development_dataset else None,
            "ground_truth_used_for_post_run_diagnostics": args.development_dataset,
            "ground_truth_available_to_inference": False,
            "stage1_large_model_replaced_by_7b": (
                args.stage1_large_model_replaced_by_7b
            ),
            "stage2_large_model_replaced_by_7b": (
                args.stage2_large_model_replaced_by_7b
            ),
            "prompt_sha256": prompt_digest.hexdigest(),
            "dependencies": dependencies,
        },
    )
    print(f"Frozen: {result['frozen_path']}")
    print(f"SHA-256: {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
