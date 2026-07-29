"""Resume post-freeze evaluation after completed vLLM inference.

This command never regenerates formal predictions. It is intended for a run
that safely stopped after all prediction records were written but before the
prediction freeze opened ground truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError
from bian.models.vllm_32b_backend import VLLMCall
from bian.predictions import load_prediction_jsonl
from scripts import run_vllm_7b32b as runner


def load_formal_calls(raw_output_dir: Path) -> list[VLLMCall]:
    calls = []
    for path in sorted(raw_output_dir.glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("role") == "32B-OracleClassification":
            continue
        calls.append(VLLMCall(**value))
    return calls


def elapsed_formal_seconds(output_dir: Path, state: dict) -> float:
    timestamp = output_dir.name.rsplit("_", 2)[-2:]
    started = datetime.strptime("_".join(timestamp), "%Y%m%d_%H%M%S").replace(
        tzinfo=timezone.utc
    )
    completed = datetime.fromisoformat(state["updated_at_utc"])
    return max(0.0, (completed - started).total_seconds())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--int8-partial-manifest", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.output_dir / "prediction/predictions.jsonl"
    frozen_path = args.output_dir / "prediction/predictions_frozen.jsonl"
    if not args.output_dir.is_dir() or not predictions_path.is_file():
        raise FileNotFoundError("completed inference artifacts are missing")
    if frozen_path.exists():
        raise FileExistsError("prediction is already frozen; refusing to resume")
    state_path = args.output_dir / "run_state.json"
    state = json.loads(state_path.read_text())
    records = load_prediction_jsonl(predictions_path)
    if len(records) != 10 or state.get("completed_cases") != 10:
        raise ValidationError("post-freeze resume requires 10 completed cases")
    if state.get("ground_truth_opened") is True:
        raise ValidationError("truth-access guard indicates ground truth was opened")

    config = json.loads(args.config.read_text())
    if config["precision"] != "bfloat16" or config.get("quantization") is not None:
        raise ValidationError("post-freeze resume requires unquantized BF16")
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    incidents = runner.load_jsonl(args.bundle_root / "experiment/incidents.jsonl")
    formal_calls = load_formal_calls(
        args.output_dir / "prediction/logs/raw_outputs"
    )
    if not formal_calls:
        raise ValidationError("formal model call records are missing")

    backend = runner.build_backend(
        args, config, args.output_dir / "prediction/logs/raw_outputs"
    )
    backend.calls = formal_calls
    formal_call_count = len(formal_calls)
    inference_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    formal_seconds = elapsed_formal_seconds(args.output_dir, state)
    resumed = time.perf_counter()
    post_result = runner.post_freeze(
        args, config, backend, taxonomy, incidents, inference_commit
    )
    resume_seconds = time.perf_counter() - resumed
    total_seconds = formal_seconds + resume_seconds

    calls = backend.call_manifest()
    runner.write_json(
        args.output_dir / "prediction/logs/model_calls.json",
        calls[:formal_call_count],
    )
    runner.write_json(
        args.output_dir / "environment/model_manifest.json",
        backend.model_manifest(),
    )
    runner.write_json(
        args.output_dir / "environment/gpu_peak.json",
        backend.model_manifest()["peak_gpu_memory_mib"],
    )
    (args.output_dir / "environment/dependencies.txt").write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    )
    subprocess.run(
        ["nvidia-smi"],
        stdout=(args.output_dir / "environment/gpu_after.txt").open("w"),
        check=True,
    )
    prompt_sha, prompt_files = runner.prompt_digest()
    root_manifest = {
        **config,
        "git_commit": inference_commit,
        "config_sha256": runner.sha256_file(args.config),
        "prompt_sha256": prompt_sha,
        "prompt_file_sha256": prompt_files,
        "formal_inference_read_ground_truth": False,
        "model_manifest": backend.model_manifest(),
        "prediction_records": len(records),
        "formal_inference_seconds": formal_seconds,
        "post_freeze_resume_seconds": resume_seconds,
        "total_seconds": total_seconds,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_frozen": True,
        "frozen_sha256": post_result["frozen"]["sha256"],
        "resumed_post_freeze": True,
    }
    runner.write_json(args.output_dir / "run_manifest.json", root_manifest)
    performance = runner.performance_summary(
        args, backend, total_seconds, len(records)
    )
    runner.write_json(
        args.output_dir / "evaluation/performance_comparison.json",
        performance,
    )
    runner.write_json(
        state_path,
        {
            "status": "completed",
            "prediction_records": len(records),
            "prediction_frozen": True,
            "ground_truth_opened": True,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    print(
        "vLLM post-freeze resume completed: "
        f"cases={len(records)}; sha256={post_result['frozen']['sha256']}; "
        f"total={total_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
