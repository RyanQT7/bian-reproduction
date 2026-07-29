"""Post-freeze RCA-only /70 scoring for the 33-case blind experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.evaluation.real_scoring import score_predictions


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    args = parser.parse_args()
    prediction_dir = args.run_dir / "prediction"
    frozen = prediction_dir / "predictions_frozen.jsonl"
    recorded = (prediction_dir / "predictions.sha256").read_text().split()[0]
    actual = digest(frozen)
    if actual != recorded or frozen.stat().st_mode & 0o222:
        raise ValueError("predictions must be immutable and match recorded SHA")
    guard = json.loads((prediction_dir / "truth_access_guard.json").read_text())
    if not guard.get("passed") or guard.get("formal_inference_read_ground_truth"):
        raise ValueError("truth-access guard did not pass")
    predictions = load_jsonl(frozen)
    truths = load_jsonl(args.ground_truth_file)
    score = score_predictions(
        predictions=predictions,
        ground_truth=truths,
        root_node_field="root_node_id",
        expected_case_count=33,
    )
    truth_by_id = {item["incident_id"]: item for item in truths}
    top5_hits = 0
    per_case = []
    for prediction in predictions:
        truth = truth_by_id[prediction["incident_id"]]
        class_types = [
            item["fault_type"] for item in prediction.get("fault_type_top5", [])
        ]
        top5_hit = truth["fault_type"] in class_types
        top5_hits += top5_hit
        root_nodes = [
            item["node_id"] for item in prediction.get("top5_root_causes", [])
        ]
        root_rank = (
            root_nodes.index(truth["root_node_id"]) + 1
            if truth["root_node_id"] in root_nodes
            else None
        )
        per_case.append(
            {
                "incident_id": prediction["incident_id"],
                "prediction_mode": prediction.get("prediction_mode"),
                "root_rank": root_rank or "",
                "true_fault_type": truth["fault_type"],
                "predicted_fault_type": prediction.get("predicted_fault_type", ""),
                "classification_top5_hit": top5_hit,
            }
        )
    score["classification"]["fault_type_top5_hit_rate"] = top5_hits / 33
    score["rca_only_alpha"] = 1
    score["not_full_competition_score"] = True
    score["frozen_prediction_sha256"] = actual
    evaluation = args.run_dir / "evaluation"
    evaluation.mkdir(exist_ok=False)
    (evaluation / "score_summary.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n"
    )
    with (evaluation / "per_case_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    localization = score["localization"]
    classification = score["classification"]
    (evaluation / "score_summary.md").write_text(
        "\n".join(
            [
                "# BiAn 33-case blind RCA-only result",
                "",
                "This is RCA-only /70, not the full competition /100 score.",
                "",
                f"- Evaluated cases: {score['evaluated_cases']}",
                f"- Localization: {localization['score_40']:.6f}/40",
                f"- Classification: {classification['score_30']:.6f}/30",
                f"- Total: {score['total_score_70']:.6f}/70",
                f"- Localization Top1: {localization['top1_accuracy']:.6%}",
                f"- Localization Top5: {localization['top5_accuracy']:.6%}",
                f"- Classification Top1: {classification['fault_type_top1_accuracy']:.6%}",
                f"- Classification Top3: {classification['fault_type_top3_hit_rate']:.6%}",
                f"- Classification Top5: {classification['fault_type_top5_hit_rate']:.6%}",
                f"- Frozen SHA-256: `{actual}`",
            ]
        )
        + "\n"
    )
    after = digest(frozen)
    if after != actual:
        raise RuntimeError("frozen prediction changed during scoring")
    print(
        json.dumps(
            {
                "localization_score_40": localization["score_40"],
                "classification_score_30": classification["score_30"],
                "total_score_70": score["total_score_70"],
                "classification_top5": classification[
                    "fault_type_top5_hit_rate"
                ],
                "frozen_sha256": actual,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
