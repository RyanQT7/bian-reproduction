"""Render immutable evaluation JSON into the required local report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.score_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "localization_metrics.json", result["localization"])
    write_json(args.output_dir / "classification_metrics.json", result["classification"])
    write_json(
        args.output_dir / "confusion_matrices.json",
        {
            "fault_type": result["classification"]["fault_type_confusion_matrix"],
            "fault_category": result["classification"][
                "fault_category_confusion_matrix"
            ],
        },
    )
    per_case_path = args.output_dir / "per_case_scores.csv"
    with per_case_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "incident_id",
            "root_node",
            "hit_rank",
            "localization_credit",
            "true_fault_type",
            "predicted_fault_type",
            "true_fault_category",
            "predicted_fault_category",
            "classification_credit",
            "classification_top3_hit",
            "prediction_missing",
            "format_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: item.get(key) for key in fieldnames} for item in result["per_case"]
        )
    summary = [
        "# Dual-7B Pipeline Validation Result",
        "",
        "This is not a paper-equivalent BiAn result and is not directly comparable "
        "with the paper's reported accuracy.",
        "",
        f"- evaluated_cases: {result['evaluated_cases']}",
        f"- localization_score: {result['localization']['score_40']:.6f} / 40",
        f"- classification_score: {result['classification']['score_30']:.6f} / 30",
        f"- total_score_70: {result['total_score_70']:.6f} / 70",
        f"- localization Top1: {result['localization']['top1_accuracy']:.6f}",
        f"- localization Top5: {result['localization']['top5_accuracy']:.6f}",
        f"- localization MRR: {result['localization']['mean_reciprocal_rank']:.6f}",
        f"- fault type Top1: "
        f"{result['classification']['fault_type_top1_accuracy']:.6f}",
        f"- fault type Top3: "
        f"{result['classification']['fault_type_top3_hit_rate']:.6f}",
        f"- fault category accuracy: "
        f"{result['classification']['fault_category_accuracy']:.6f}",
    ]
    (args.output_dir / "score_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    failures = [
        "# Failure analysis",
        "",
        "Dual-7B Pipeline Validation Result; not a paper reproduction result.",
        "",
    ]
    for item in result["failure_cases"]:
        failures += [
            f"## {item['incident_id']}",
            "",
            f"- root truth: `{item['root_node']}`",
            f"- predicted hit rank: {item['hit_rank']}",
            f"- localization credit: {item['localization_credit']}",
            f"- true type/category: `{item['true_fault_type']}` / "
            f"`{item['true_fault_category']}`",
            f"- predicted type/category: `{item['predicted_fault_type']}` / "
            f"`{item['predicted_fault_category']}`",
            f"- classification credit: {item['classification_credit']}",
            f"- format error: {item['format_error']}",
            "",
        ]
    (args.output_dir / "failure_analysis.md").write_text(
        "\n".join(failures), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
