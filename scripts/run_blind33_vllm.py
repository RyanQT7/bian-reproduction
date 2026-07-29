"""Blind 33-case inference with new Stage 1, v3 fusion, and explicit fallback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError, normalize_scores
from bian.methods.mixed_32b import (
    aggregate_stage2_rounds,
    aggregate_type_result,
    apply_v3_stage1_fusion,
    score_stage2_round,
    stage1_fallback_top5,
    validate_stage2_round,
    validate_type_result,
)
from bian.models.dual_7b_backend import GenerationConfig
from bian.models.vllm_32b_backend import VLLM32BBackend
from bian.models.vllm_schemas import classification_schema, stage2_schema
from bian.predictions import freeze_predictions, validate_predictions
from scripts.run_vllm_7b32b import (
    candidate_evidence_ids,
    compact_candidate,
    relevant_topology,
    root_payload,
    timeline,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_manifest() -> dict[str, str]:
    return {
        path.name: sha(path)
        for path in sorted(
            (PROJECT_ROOT / "src/bian/prompts").glob("blind33_*.txt")
        )
    }


def normalize_taxonomy(document: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    candidates = document.get("candidates", [])
    if (
        document.get("candidate_count") != 32
        or len(candidates) != 32
        or ids != [item.get("fault_type") for item in candidates]
    ):
        raise ValidationError("taxonomy files must define the same ordered 32 types")
    if set(document.get("categories", [])) != {
        "link", "firewall", "resource", "route", "service"
    }:
        raise ValidationError("taxonomy must contain the declared five categories")
    return [
        {
            **item,
            "fault_category": item["category"],
            "type_id": f"T{index:02d}",
        }
        for index, item in enumerate(candidates, 1)
    ]


def build_backend(
    model_path: Path, config: dict[str, Any], raw_output_dir: Path
) -> VLLM32BBackend:
    expected_sampler = "1" if config.get("use_flashinfer_sampler", True) else "0"
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != expected_sampler:
        raise ValidationError(
            "VLLM_USE_FLASHINFER_SAMPLER must match the frozen configuration"
        )
    return VLLM32BBackend(
        model_path,
        config=GenerationConfig(
            max_input_tokens=config["max_input_tokens"],
            max_new_tokens=config["stage2_max_new_tokens"],
            temperature=config["stage2_temperature"],
            top_p=config["stage2_top_p"],
            retries=config["structured_retries"],
            seed=config["stage2_seeds"][0],
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
        tensor_parallel_size=config["tensor_parallel_size"],
        pipeline_parallel_size=config["pipeline_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        enforce_eager=config["enforce_eager"],
        disable_custom_all_reduce=config["disable_custom_all_reduce"],
        raw_output_dir=raw_output_dir,
        physical_gpu_ids=config["physical_gpus"],
    )


def classification_requests(
    incident_id: str,
    taxonomy: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    evidence_ids: set[str],
    taxonomy_prompt_text: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    aliases = {item["candidate_id"] for item in roots}
    universe = [
        {
            "type_id": item["type_id"],
            "fault_type": item["fault_type"],
            "category": item["fault_category"],
            "description": item["description_zh"],
            "allowed_variants": item.get("allowed_variants", []),
            "legacy_aliases": item.get("legacy_aliases", []),
        }
        for item in taxonomy
    ]
    requests = []
    for item in taxonomy:
        seed = int(
            hashlib.sha256(
                f"{incident_id}:{item['type_id']}".encode()
            ).hexdigest()[:8],
            16,
        ) % (2**31)
        request = {
            "request_id": f"blind33_classification_{incident_id}_{item['type_id']}",
            "role": "32B-Classification",
            "prompt_name": "blind33_type_ovr",
            "prompt_version": "blind33-type-ovr-v1",
            "payload": {
                "candidate_set_instructions": taxonomy_prompt_text,
                "candidate_universe": universe,
                "target_type": universe[item["candidate_index"] - 1],
                "root_hypotheses": roots,
                "deterministic_seed": seed,
            },
            "seed": seed,
            "json_schema": classification_schema(
                item["type_id"], aliases, evidence_ids
            ),
            "validator": lambda value, expected=item["type_id"]: (
                validate_type_result(value, expected, aliases, evidence_ids)
            ),
        }
        requests.append((item, request))
    return requests


def classify(
    backend: VLLM32BBackend,
    incident_id: str,
    taxonomy: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    evidence_ids: set[str],
    output_dir: Path,
    max_new_tokens: int,
    taxonomy_prompt_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = classification_requests(
        incident_id, taxonomy, roots, evidence_ids, taxonomy_prompt_text
    )
    outputs = backend.generate_json_batch(
        [request for _, request in pending],
        max_new_tokens=max_new_tokens,
        allow_partial=True,
    )
    root_weights = {
        item["candidate_id"]: item["root_weight"] for item in roots
    }
    raw_scores: dict[str, float] = {}
    records = []
    failures = []
    for (item, request), result in zip(pending, outputs):
        if result is None:
            calls = [
                call
                for call in backend.calls
                if call.request_id == request["request_id"]
            ]
            error = calls[-1].error if calls else "missing structured result"
            failure = {
                "incident_id": incident_id,
                "type_id": item["type_id"],
                "fault_type": item["fault_type"],
                "status": "failed",
                "error": error,
            }
            failures.append(failure)
            append(output_dir / "classification/type_status.jsonl", failure)
            continue
        score, contributions = aggregate_type_result(result, root_weights)
        raw_scores[item["fault_type"]] = score
        record = {
            "incident_id": incident_id,
            "type_id": item["type_id"],
            "fault_type": item["fault_type"],
            "fault_category": item["fault_category"],
            "status": "success",
            "aggregated_raw_score": score,
            "root_hypotheses": contributions,
        }
        records.append(record)
        append(output_dir / "classification/type_scores.jsonl", record)
        append(
            output_dir / "classification/type_status.jsonl",
            {
                "incident_id": incident_id,
                "type_id": item["type_id"],
                "fault_type": item["fault_type"],
                "status": "success",
            },
        )
    if failures:
        raise ValidationError(
            f"{len(failures)}/32 one-vs-rest tasks failed"
        )
    if len(raw_scores) != 32 or not any(raw_scores.values()):
        raise ValidationError("classification requires 32 non-degenerate type results")
    normalized = normalize_scores(raw_scores)
    by_type = {item["fault_type"]: item for item in taxonomy}
    ranked = sorted(raw_scores, key=lambda key: (-raw_scores[key], key))
    top5 = [
        {
            "rank": rank,
            "fault_type": fault_type,
            "fault_category": by_type[fault_type]["fault_category"],
            "confidence": normalized[fault_type],
            "score_kind": "normalized_model_score",
            "reason_summary": "32B independent 32-type one-vs-rest score",
        }
        for rank, fault_type in enumerate(ranked[:5], 1)
    ]
    return top5, records


def run_case(
    *,
    backend: VLLM32BBackend,
    incident: dict[str, Any],
    stage1_ranking: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    taxonomy: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    taxonomy_prompt_text: str,
) -> dict[str, Any]:
    incident_id = incident["incident_id"]
    started = time.perf_counter()
    alias_to_node = {
        item["candidate_id"]: item["node_id"] for item in shortlist
    }
    shortlist_by_alias = {
        item["candidate_id"]: item for item in shortlist
    }
    evidence_by_node = {
        item["node_id"]: item for item in incident["engineering_evidence"]
    }
    candidates = [
        compact_candidate(
            item,
            evidence_by_node[item["node_id"]],
            max_evidence=config["evidence_per_candidate"],
        )
        for item in shortlist
    ]
    evidence_ids = candidate_evidence_ids(candidates)
    topology = relevant_topology(incident["topology"], alias_to_node)
    case_timeline = timeline(candidates)
    requests = []
    for round_index, seed in enumerate(config["stage2_seeds"], 1):
        requests.append(
            {
                "request_id": f"blind33_stage2_{incident_id}_round{round_index}",
                "role": "32B-Stage2",
                "prompt_name": f"blind33_stage2_round{round_index}",
                "prompt_version": f"blind33-stage2-round{round_index}-v1",
                "payload": {
                    "round": round_index,
                    "seed": seed,
                    "incident_window": {
                        "start": incident["fault_start_time_utc"],
                        "end": incident["fault_end_time_utc"],
                    },
                    "candidates": candidates,
                    "topology": topology,
                    "timeline": case_timeline,
                },
                "seed": seed,
                "temperature": config["stage2_temperature"],
                "top_p": config["stage2_top_p"],
                "json_schema": stage2_schema(alias_to_node, evidence_ids),
                "validator": lambda value: validate_stage2_round(
                    value, set(alias_to_node), evidence_ids
                ),
            }
        )
    outputs = backend.generate_json_batch(
        requests,
        max_new_tokens=config["stage2_max_new_tokens"],
        allow_partial=True,
    )
    fused_scores = []
    round_errors = []
    for round_index, (request, result) in enumerate(zip(requests, outputs), 1):
        if result is None:
            calls = [
                call
                for call in backend.calls
                if call.request_id == request["request_id"]
            ]
            error = calls[-1].error if calls else "missing structured result"
            round_errors.append({"round": round_index, "error": error})
            append(
                output_dir / "stage2/round_status.jsonl",
                {
                    "incident_id": incident_id,
                    "round": round_index,
                    "status": "failed_after_repair",
                    "error": error,
                },
            )
            continue
        fused = apply_v3_stage1_fusion(
            result["candidates"], shortlist_by_alias
        )
        scores, audit = score_stage2_round(
            fused, alias_to_node, config["v3_fusion"]["weights"]
        )
        fused_scores.append(scores)
        append(
            output_dir / f"stage2/round_{round_index}.jsonl",
            {
                "incident_id": incident_id,
                "round": round_index,
                "scores": scores,
                "components": audit,
            },
        )
        append(
            output_dir / "stage2/round_status.jsonl",
            {
                "incident_id": incident_id,
                "round": round_index,
                "status": "success",
                "attempts": len(
                    [
                        call
                        for call in backend.calls
                        if call.request_id == request["request_id"]
                    ]
                ),
            },
        )
    if len(fused_scores) >= config["minimum_valid_stage2_rounds"]:
        top5, rank_data = aggregate_stage2_rounds(
            fused_scores, tuple(sorted(alias_to_node.values()))
        )
        mode = config["prediction_mode_success"]
    else:
        top5 = stage1_fallback_top5(stage1_ranking)
        rank_data = {
            "rounds": 0,
            "raw_rankings": [],
            "average_ranks": {},
            "fallback": "stage1",
            "valid_stage2_rounds": len(fused_scores),
            "stage2_errors": round_errors,
        }
        mode = config["prediction_mode_fallback"]
    append(
        output_dir / "stage2/final_root_cause_ranking.jsonl",
        {
            "incident_id": incident_id,
            "prediction_mode": mode,
            "valid_stage2_rounds": len(fused_scores),
            "final_root_cause_ranking": top5,
            "rank_of_ranks": rank_data,
        },
    )
    shortlist_by_node = {item["node_id"]: item for item in shortlist}
    roots, root_evidence_ids = root_payload(
        top5,
        shortlist_by_node,
        evidence_by_node,
        max_evidence=config["evidence_per_candidate"],
    )
    classification_status = "success"
    classification_error = None
    class_top5 = None
    try:
        class_top5, _ = classify(
            backend,
            incident_id,
            taxonomy,
            roots,
            root_evidence_ids,
            output_dir,
            config["classification_max_new_tokens"],
            taxonomy_prompt_text,
        )
    except Exception as exc:
        classification_status = "classification_failed"
        classification_error = f"{type(exc).__name__}: {exc}"
    record = {
        "incident_id": incident_id,
        "prediction_status": "success",
        "classification_status": classification_status,
        "prediction_mode": mode,
        "result_label": config["result_label"],
        "model_configuration": config["model_configuration"],
        "top5_root_causes": top5,
        "final_root_cause_ranking": top5,
        "rank_of_ranks": rank_data,
        "valid_stage2_rounds": len(fused_scores),
        "inference_seconds": time.perf_counter() - started,
    }
    if class_top5:
        record.update(
            {
                "predicted_fault_type": class_top5[0]["fault_type"],
                "predicted_fault_category": class_top5[0]["fault_category"],
                "fault_type_top3": class_top5[:3],
                "fault_type_top5": class_top5,
            }
        )
        append(
            output_dir / "classification/predictions_top5.jsonl",
            {"incident_id": incident_id, "fault_type_top5": class_top5},
        )
    else:
        record["classification_error"] = classification_error
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--experiment-file", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--taxonomy-ids-file", type=Path, required=True)
    parser.add_argument("--taxonomy-prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incident-id")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    for path in (
        args.experiment_file,
        args.taxonomy_file,
        args.taxonomy_ids_file,
        args.taxonomy_prompt_file,
    ):
        lowered = str(path).lower()
        if "/evaluation/" in lowered or "/review/" in lowered:
            raise ValueError("blind inference refuses evaluation/review paths")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    config = json.loads(args.config.read_text())
    if config["precision"] != "bfloat16" or config["quantization"] is not None:
        raise ValidationError("blind runner requires unquantized BF16")
    taxonomy = normalize_taxonomy(
        json.loads(args.taxonomy_file.read_text()),
        json.loads(args.taxonomy_ids_file.read_text()),
    )
    taxonomy_prompt_text = args.taxonomy_prompt_file.read_text()
    incidents = load_jsonl(args.experiment_file)
    incident_ids = [item["incident_id"] for item in incidents]
    if len(incidents) != 33 or len(set(incident_ids)) != 33:
        raise ValidationError("formal experiment definition must contain 33 cases")
    selected = incident_ids
    if args.incident_id:
        selected = [args.incident_id]
    if args.freeze and args.incident_id:
        raise ValidationError("smoke output cannot be frozen as formal output")
    if args.freeze and len(selected) != 33:
        raise ValidationError("formal freeze requires all 33 incidents")
    stage1_rankings = {
        item["incident_id"]: item["ranking"]
        for item in load_jsonl(args.stage1_source / "full_rankings.jsonl")
    }
    shortlists = {
        item["incident_id"]: item["shortlist"]
        for item in load_jsonl(args.stage1_source / "shortlist.jsonl")
    }
    if set(selected) - set(stage1_rankings) or set(selected) - set(shortlists):
        raise ValidationError("new Stage 1 output is incomplete")
    args.output_dir.mkdir(parents=True)
    for relative in (
        "stage1_reference", "stage2", "classification",
        "prediction/reports", "prediction/logs/raw_outputs", "environment",
    ):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)
    for name in (
        "full_rankings.jsonl", "shortlist.jsonl", "stage1_config.yaml",
        "stage1_config.sha256", "truth_access_guard.json",
    ):
        shutil.copyfile(
            args.stage1_source / name,
            args.output_dir / "stage1_reference" / name,
        )
    inference_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    write_json(
        args.output_dir / "prediction/truth_access_guard.json",
        {
            "passed": True,
            "formal_inference_read_ground_truth": False,
            "evaluation_or_review_path_received": False,
            "prediction_process_accepts_ground_truth_argument": False,
        },
    )
    subprocess.run(
        ["nvidia-smi"],
        stdout=(args.output_dir / "environment/gpu_before.txt").open("w"),
        check=True,
    )
    backend = build_backend(
        args.model_path,
        config,
        args.output_dir / "prediction/logs/raw_outputs",
    )
    started = time.perf_counter()
    backend.load()
    predictions = []
    for incident_id in selected:
        incident = json.loads(
            (args.input_root / incident_id / "incident_input.json").read_text()
        )
        record = run_case(
            backend=backend,
            incident=incident,
            stage1_ranking=stage1_rankings[incident_id],
            shortlist=shortlists[incident_id],
            taxonomy=taxonomy,
            config=config,
            output_dir=args.output_dir,
            taxonomy_prompt_text=taxonomy_prompt_text,
        )
        append(args.output_dir / "prediction/predictions.jsonl", record)
        predictions.append(record)
        print(
            f"{incident_id}: mode={record['prediction_mode']} "
            f"classification={record['classification_status']}",
            flush=True,
        )
    candidate_ids = set(
        json.loads(
            (args.input_root / selected[0] / "incident_input.json").read_text()
        )["candidate_node_ids"]
    )
    validation = validate_predictions(
        predictions,
        expected_incident_ids=set(selected),
        candidate_node_ids=candidate_ids,
        taxonomy=taxonomy,
        minimum_rank_rounds=config["minimum_valid_stage2_rounds"],
    )
    write_json(
        args.output_dir / "prediction/predictions.schema_validation.json",
        validation,
    )
    if not validation["valid"]:
        raise ValidationError("; ".join(validation["errors"]))
    frozen = None
    if args.freeze:
        frozen = freeze_predictions(
            predictions_path=args.output_dir / "prediction/predictions.jsonl",
            output_dir=args.output_dir / "prediction",
            validation=validation,
            manifest={
                "evaluated_cases": 33,
                "git_commit": inference_commit,
                "prediction_modes": {
                    mode: sum(x["prediction_mode"] == mode for x in predictions)
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
                "backend": backend.model_manifest(),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        Path(frozen["frozen_path"]).chmod(0o444)
    total = time.perf_counter() - started
    write_json(
        args.output_dir / "prediction/logs/model_calls.json",
        backend.call_manifest(),
    )
    write_json(
        args.output_dir / "environment/model_manifest.json",
        backend.model_manifest(),
    )
    write_json(
        args.output_dir / "run_manifest.json",
        {
            **config,
            "git_commit": inference_commit,
            "incident_ids": selected,
            "formal": args.freeze,
            "prediction_frozen": bool(frozen),
            "frozen_sha256": frozen["sha256"] if frozen else None,
            "taxonomy_sha256": sha(args.taxonomy_file),
            "taxonomy_ids_sha256": sha(args.taxonomy_ids_file),
            "taxonomy_prompt_sha256": sha(args.taxonomy_prompt_file),
            "prompt_sha256": prompt_manifest(),
            "config_sha256": sha(args.config),
            "runtime_compatibility": {
                "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
                    "VLLM_USE_FLASHINFER_SAMPLER"
                ),
                "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
            },
            "total_seconds": total,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    subprocess.run(
        ["nvidia-smi"],
        stdout=(args.output_dir / "environment/gpu_after.txt").open("w"),
        check=True,
    )
    print(
        json.dumps(
            {
                "cases": len(predictions),
                "fused": sum(
                    x["prediction_mode"] == config["prediction_mode_success"]
                    for x in predictions
                ),
                "fallback": sum(
                    x["prediction_mode"] == config["prediction_mode_fallback"]
                    for x in predictions
                ),
                "frozen_sha256": frozen["sha256"] if frozen else None,
                "total_seconds": total,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
