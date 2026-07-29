"""Freeze an already completed blind run without invoking either model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.predictions import freeze_predictions, validate_predictions
from scripts.run_blind33_vllm import load_jsonl, normalize_taxonomy, prompt_manifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--taxonomy-ids-file", type=Path, required=True)
    parser.add_argument("--taxonomy-prompt-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-commit", required=True)
    args = parser.parse_args()
    for path in vars(args).values():
        lowered = str(path).lower()
        if "/evaluation/" in lowered or "/review/" in lowered:
            raise ValueError("blind finalizer refuses evaluation/review paths")

    predictions = load_jsonl(args.output_dir / "prediction/predictions.jsonl")
    taxonomy = normalize_taxonomy(
        json.loads(args.taxonomy_file.read_text()),
        json.loads(args.taxonomy_ids_file.read_text()),
    )
    incident_ids = {item["incident_id"] for item in predictions}
    if len(incident_ids) != 33:
        raise ValueError("formal finalizer requires 33 unique predictions")
    candidate_ids = set(
        json.loads(
            (args.input_root / sorted(incident_ids)[0] / "incident_input.json").read_text()
        )["candidate_node_ids"]
    )
    config = json.loads(args.config.read_text())
    validation = validate_predictions(
        predictions,
        expected_incident_ids=incident_ids,
        candidate_node_ids=candidate_ids,
        taxonomy=taxonomy,
        minimum_rank_rounds=config["minimum_valid_stage2_rounds"],
    )
    frozen = freeze_predictions(
        predictions_path=args.output_dir / "prediction/predictions.jsonl",
        output_dir=args.output_dir / "prediction",
        validation=validation,
        allow_identical_validation=True,
        manifest={
            "evaluated_cases": 33,
            "git_commit": args.inference_commit,
            "prediction_modes": {
                mode: sum(item["prediction_mode"] == mode for item in predictions)
                for mode in (
                    config["prediction_mode_success"],
                    config["prediction_mode_fallback"],
                )
            },
            "taxonomy_sha256": sha(args.taxonomy_file),
            "taxonomy_ids_sha256": sha(args.taxonomy_ids_file),
            "taxonomy_prompt_sha256": sha(args.taxonomy_prompt_file),
            "config_sha256": sha(args.config),
            "stage1_config_sha256": (
                args.stage1_source / "stage1_config.sha256"
            ).read_text().strip(),
            "prompt_sha256": prompt_manifest(),
            "formal_inference_read_ground_truth": False,
            "evaluation_or_review_path_received": False,
            "finalized_without_model_rerun": True,
            "finalizer_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
        },
    )
    print(json.dumps(frozen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
