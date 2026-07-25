"""Run the deterministic mock pipeline and save local artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian import VALIDATION_LABEL
from bian.data.mock_generator import generate_mock_incidents
from bian.evaluation.metrics import top_k_accuracy
from bian.models.mock_backend import MockModelBackend
from bian.pipeline import run_incident


def _value(config: dict, key: str):
    item = config[key]
    return item["value"] if isinstance(item, dict) and "value" in item else item


def _git_changed_files() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="configs/data/mock.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    data_config = json.loads((PROJECT_ROOT / args.data_config).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    incidents = generate_mock_incidents(
        seed,
        int(data_config["incident_count"]),
        int(data_config["candidate_count"]),
    )
    backend = MockModelBackend(seed)
    predictions = [
        run_incident(
            incident,
            backend,
            rank_rounds=int(_value(config, "rank_rounds")),
            entropy_threshold=float(_value(config, "entropy_threshold")),
            entropy_mode=str(_value(config, "entropy_mode")),
            stage2_device_top_p=float(_value(config, "stage2_device_top_p")),
        )
        for incident in incidents
    ]
    rankings = [tuple(item["ranking"]) for item in predictions]
    truths = [incident.ground_truth for incident in incidents]
    metrics = {
        "result_label": VALIDATION_LABEL,
        "synthetic": True,
        "incident_count": len(incidents),
        "top_1_accuracy": top_k_accuracy(rankings, truths, 1),
        "top_2_accuracy": top_k_accuracy(rankings, truths, 2),
        "top_3_accuracy": top_k_accuracy(rankings, truths, 3),
        "note": "Mock metrics validate software flow only; they are not paper results.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / (args.output_dir or f"outputs/mock_run_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    resolved = {"experiment": config, "data": data_config}
    (output_dir / "resolved_config.yaml").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    summary = (
        f"# {VALIDATION_LABEL}\n\n"
        f"- Incidents: {len(incidents)}\n"
        f"- Top-1 mock metric: {metrics['top_1_accuracy']:.3f}\n"
        "- This is a synthetic software-flow check, not a paper result.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "run.log").write_text(
        f"{VALIDATION_LABEL}\nstatus=success\nseed={seed}\n", encoding="utf-8"
    )
    (output_dir / "tests.log").write_text(
        "Tests are run separately with unittest and saved under logs/.\n", encoding="utf-8"
    )
    (output_dir / "changed_files.txt").write_text(_git_changed_files(), encoding="utf-8")
    print(VALIDATION_LABEL)
    print(f"Incidents: {len(incidents)}; Top-1 mock metric: {metrics['top_1_accuracy']:.3f}")
    print(f"Output: {output_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
