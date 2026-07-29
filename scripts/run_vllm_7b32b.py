"""vLLM BF16 feasibility runner with frozen 7B Stage 1 and post-freeze diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError, normalize_scores
from bian.evaluation.real_scoring import score_predictions
from bian.methods.evidence_budget import select_evidence
from bian.methods.mixed_32b import (
    aggregate_stage2_rounds,
    aggregate_type_result,
    score_stage2_round,
    validate_stage2_round,
    validate_type_result,
)
from bian.models.dual_7b_backend import GenerationConfig
from bian.models.vllm_32b_backend import VLLM32BBackend
from bian.models.vllm_schemas import classification_schema, stage2_schema
from bian.predictions import (
    freeze_predictions,
    load_prediction_jsonl,
    validate_predictions,
)
from bian.taxonomy_profiles import profile


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest() -> tuple[str, dict[str, str]]:
    combined = hashlib.sha256()
    files = {}
    for path in sorted((PROJECT_ROOT / "src/bian/prompts").glob("32b_*.txt")):
        digest = sha256_file(path)
        files[path.name] = digest
        combined.update(path.name.encode())
        combined.update(path.read_bytes())
    return combined.hexdigest(), files


def copy_stage1_reference(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    mapping = {
        "full_rankings.jsonl": "full_rankings.jsonl",
        "shortlist.jsonl": "shortlists.jsonl",
        "stage1_config.yaml": "frozen_config.yaml",
        "stage1_config.sha256": "config.sha256",
        "stage1_metrics.json": "metrics.json",
    }
    missing = [name for name in mapping if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen Stage 1 lacks {missing}")
    for source_name, target_name in mapping.items():
        shutil.copyfile(source / source_name, output / target_name)
    write_json(
        output / "source_run.json",
        {
            "source_stage1": str(source),
            "stage1_frozen": True,
            "config_sha256": (source / "stage1_config.sha256").read_text().strip(),
        },
    )


def compact_candidate(
    stage1: dict[str, Any],
    evidence: dict[str, Any],
    *,
    max_evidence: int,
) -> dict[str, Any]:
    selected = select_evidence(stage1, evidence, max_evidence=max_evidence)
    return {
        "candidate_id": stage1["candidate_id"],
        "device_role": stage1["device_role"],
        "region_id": "-".join(stage1["node_id"].split("-")[:2]),
        "stage1_components": {
            field: stage1[field]
            for field in (
                "deterministic_feature_score",
                "model_anomaly_score",
                "temporal_change_score",
                "direct_fault_evidence_score",
                "symptom_likelihood",
                "data_quality_adjustment",
                "stage1_score",
            )
        },
        "earliest_change_time": stage1["earliest_change_time"],
        "data_quality_statuses": sorted(
            {item["data_quality_status"] for item in evidence["evidence"]}
        ),
        "evidence": selected,
    }


def relevant_topology(
    topology: dict[str, Any], alias_to_node: dict[str, str]
) -> dict[str, Any]:
    node_to_alias = {node: alias for alias, node in alias_to_node.items()}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in topology["edges"]:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    selected = set(node_to_alias)
    frontier = set(selected)
    for _ in range(2):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in adjacency.get(node, ())
            if neighbor not in selected
        }
        selected.update(frontier)
    extra = sorted(selected - set(node_to_alias))
    for index, node in enumerate(extra, 1):
        node_to_alias[node] = f"N{index:03d}"
    metadata = {item["node_id"]: item for item in topology["nodes"]}
    nodes = [
        {
            "topology_id": node_to_alias[node],
            "candidate_id": (
                node_to_alias[node] if node_to_alias[node].startswith("C") else None
            ),
            "region_id": metadata[node]["region_id"],
            "device_role": metadata[node]["role"],
            "candidate": metadata[node]["candidate"],
        }
        for node in sorted(selected, key=lambda item: node_to_alias[item])
    ]
    edges = [
        {
            "source": node_to_alias[edge["source"]],
            "target": node_to_alias[edge["target"]],
            "edge_type": edge.get("edge_type", "connected"),
            "protocol": edge.get("protocol"),
        }
        for edge in topology["edges"]
        if edge["source"] in selected and edge["target"] in selected
    ]
    return {"nodes": nodes, "edges": edges, "expansion_hops": 2}


def timeline(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [
        {
            "evidence_id": evidence["evidence_id"],
            "candidate_id": candidate["candidate_id"],
            "first_change_time": evidence["first_change_time"],
            "peak_time": evidence["peak_time"],
            "recovery_time": evidence["recovery_time"],
            "metric_name": evidence["metric_name"],
            "direction": evidence["direction"],
            "status_transition": evidence["status_transition"],
        }
        for candidate in candidates
        for evidence in candidate["evidence"]
        if evidence["first_change_time"]
    ]
    return sorted(
        values,
        key=lambda item: (
            item["first_change_time"],
            item["candidate_id"],
            item["evidence_id"],
        ),
    )


def candidate_evidence_ids(candidates: list[dict[str, Any]]) -> set[str]:
    return {
        evidence["evidence_id"]
        for candidate in candidates
        for evidence in candidate["evidence"]
    }


def root_payload(
    top5: list[dict[str, Any]],
    shortlist_by_node: dict[str, dict[str, Any]],
    evidence_by_node: dict[str, dict[str, Any]],
    *,
    max_evidence: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    roots = []
    evidence_ids = set()
    for index, top in enumerate(top5, 1):
        alias = f"C{index:02d}"
        node = top["node_id"]
        compact = compact_candidate(
            {**shortlist_by_node[node], "candidate_id": alias},
            evidence_by_node[node],
            max_evidence=max_evidence,
        )
        evidence_ids.update(item["evidence_id"] for item in compact["evidence"])
        roots.append(
            {
                "candidate_id": alias,
                "device_role": compact["device_role"],
                "region_id": compact["region_id"],
                "root_weight": top["failure_score"],
                "evidence": compact["evidence"],
                "data_quality_statuses": compact["data_quality_statuses"],
            }
        )
    return roots, evidence_ids


def classification_requests(
    incident_id: str,
    taxonomy: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    evidence_ids: set[str],
    *,
    role: str,
) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    aliases = {item["candidate_id"] for item in roots}
    requests = []
    for index, item in enumerate(taxonomy, 1):
        type_id = f"T{index:02d}"
        seed = int(
            hashlib.sha256(f"{incident_id}:{type_id}".encode()).hexdigest()[:8], 16
        ) % (2**31)
        requests.append(
            (
                item,
                seed,
                {
                    "request_id": f"{role.lower()}_{incident_id}_{type_id}",
                    "role": role,
                    "prompt_name": "32b_type_ovr",
                    "prompt_version": "32b-type-ovr-v1",
                    "payload": {
                        "type": {
                            "type_id": type_id,
                            "fault_category": item["fault_category"],
                            **profile(item["fault_type"]),
                        },
                        "root_hypotheses": roots,
                        "deterministic_seed": seed,
                    },
                    "seed": seed,
                    "json_schema": classification_schema(
                        type_id, aliases, evidence_ids
                    ),
                    "validator": lambda value, expected=type_id: validate_type_result(
                        value, expected, aliases, evidence_ids
                    ),
                },
            )
        )
    return sorted(
        requests,
        key=lambda value: hashlib.sha256(
            f"{incident_id}:{value[0]['fault_type']}:order".encode()
        ).hexdigest(),
    )


def run_classification(
    backend: VLLM32BBackend,
    incident_id: str,
    taxonomy: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    evidence_ids: set[str],
    output_path: Path,
    *,
    max_new_tokens: int,
    role: str = "32B-Classification",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_weights = {
        item["candidate_id"]: item["root_weight"] for item in roots
    }
    pending = classification_requests(
        incident_id, taxonomy, roots, evidence_ids, role=role
    )
    outputs = backend.generate_json_batch(
        [item[2] for item in pending], max_new_tokens=max_new_tokens
    )
    raw_scores = {}
    type_results = []
    for (taxonomy_item, seed, _request), result in zip(pending, outputs):
        score, contributions = aggregate_type_result(result, root_weights)
        raw_scores[taxonomy_item["fault_type"]] = score
        record = {
            "incident_id": incident_id,
            "type_id": result["type_id"],
            "fault_type": taxonomy_item["fault_type"],
            "fault_category": taxonomy_item["fault_category"],
            "seed": seed,
            "aggregated_raw_score": score,
            "root_hypotheses": contributions,
        }
        type_results.append(record)
        append_jsonl(output_path, record)
        append_jsonl(
            output_path.parent / "root_weighted_scores.jsonl",
            {
                "incident_id": incident_id,
                "type_id": result["type_id"],
                "fault_type": taxonomy_item["fault_type"],
                "aggregated_raw_score": score,
                "root_contributions": contributions,
            },
        )
    if not any(raw_scores.values()):
        raise ValidationError("all one-vs-rest type scores are zero")
    taxonomy_map = {
        item["fault_type"]: item["fault_category"] for item in taxonomy
    }
    normalized = normalize_scores(raw_scores)
    ranked = sorted(raw_scores, key=lambda key: (-raw_scores[key], key))
    top3 = [
        {
            "rank": rank,
            "fault_type": fault_type,
            "fault_category": taxonomy_map[fault_type],
            "confidence": normalized[fault_type],
            "score_kind": "normalized_model_score",
            "reason_summary": "32B BF16 one-vs-rest Top5-root weighted score",
        }
        for rank, fault_type in enumerate(ranked[:3], 1)
    ]
    reversed_scores = {
        item["fault_type"]: item["aggregated_raw_score"]
        for item in reversed(type_results)
    }
    reverse_ranked = sorted(
        reversed_scores, key=lambda key: (-reversed_scores[key], key)
    )
    if ranked[:3] != reverse_ranked[:3]:
        raise ValidationError("classification aggregation is order dependent")
    append_jsonl(
        output_path.parent / "order_invariance.jsonl",
        {
            "incident_id": incident_id,
            "forward_top3": ranked[:3],
            "reverse_top3": reverse_ranked[:3],
            "invariant": True,
        },
    )
    return top3, type_results


def build_backend(
    args: argparse.Namespace, config: dict[str, Any], raw_output_dir: Path
) -> VLLM32BBackend:
    return VLLM32BBackend(
        args.model_path,
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


def run_inference(
    args: argparse.Namespace,
    config: dict[str, Any],
    backend: VLLM32BBackend,
    shortlists: dict[str, dict[str, Any]],
    taxonomy: list[dict[str, Any]],
    incident_paths: list[Path],
) -> None:
    predictions_path = args.output_dir / "prediction/predictions.jsonl"
    state_path = args.output_dir / "run_state.json"
    completed = []
    for path in incident_paths:
        incident = json.loads(path.read_text())
        incident_id = incident["incident_id"]
        case_started = time.perf_counter()
        shortlist = shortlists[incident_id]["shortlist"]
        alias_to_node = {
            item["candidate_id"]: item["node_id"] for item in shortlist
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
        stage_requests = []
        for round_index, seed in enumerate(config["stage2_seeds"], 1):
            prompt = (
                "32b_stage2"
                if round_index == 1
                else f"32b_stage2_round{round_index}"
            )
            stage_requests.append(
                {
                    "request_id": f"stage2_{incident_id}_round{round_index}",
                    "role": "32B-Stage2",
                    "prompt_name": prompt,
                    "prompt_version": f"32b-stage2-round{round_index}-v1",
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
        round_errors = []
        try:
            stage_outputs = backend.generate_json_batch(
                stage_requests,
                max_new_tokens=config["stage2_max_new_tokens"],
                allow_partial=True,
            )
        except Exception as exc:
            stage_outputs = []
            round_errors.append(
                {"error_type": type(exc).__name__, "error": str(exc)}
            )
        round_scores = []
        round_audits = []
        for round_index, (request, output) in enumerate(
            zip(stage_requests, stage_outputs), 1
        ):
            if output is None:
                matching_calls = [
                    item
                    for item in backend.calls
                    if item.request_id == request["request_id"] and item.error
                ]
                round_errors.append(
                    {
                        "round": round_index,
                        "error_type": "StructuredOutputValidationError",
                        "error": (
                            matching_calls[-1].error
                            if matching_calls
                            else "structured output remained invalid after retry"
                        ),
                    }
                )
                continue
            scores, audit = score_stage2_round(
                output["candidates"],
                alias_to_node,
                config["stage2_weights"],
            )
            round_scores.append(scores)
            round_audits.append(audit)
            append_jsonl(
                args.output_dir / f"stage2/round_{round_index}.jsonl",
                {
                    "incident_id": incident_id,
                    "round": round_index,
                    "seed": request["seed"],
                    "candidate_map": alias_to_node,
                    "scores": scores,
                    "components": audit,
                },
            )
        if len(round_scores) < 2:
            append_jsonl(
                predictions_path,
                {
                    "incident_id": incident_id,
                    "prediction_status": "prediction_failed",
                    "error_type": "InsufficientValidStage2Rounds",
                    "error": json.dumps(round_errors),
                },
            )
            completed.append(incident_id)
            continue
        top5, rank_data = aggregate_stage2_rounds(
            round_scores, tuple(sorted(alias_to_node.values()))
        )
        append_jsonl(
            args.output_dir / "stage2/rank_of_ranks.jsonl",
            {
                "incident_id": incident_id,
                "valid_rounds": len(round_scores),
                "round_errors": round_errors,
                "rank_of_ranks": rank_data,
                "top5": top5,
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
        top3 = None
        try:
            top3, _ = run_classification(
                backend,
                incident_id,
                taxonomy,
                roots,
                root_evidence_ids,
                args.output_dir / "classification/type_scores.jsonl",
                max_new_tokens=config["classification_max_new_tokens"],
            )
            append_jsonl(
                args.output_dir / "classification/predictions_top3.jsonl",
                {"incident_id": incident_id, "fault_type_top3": top3},
            )
        except Exception as exc:
            classification_status = "classification_failed"
            classification_error = f"{type(exc).__name__}: {exc}"
        record = {
            "incident_id": incident_id,
            "prediction_status": "success",
            "classification_status": classification_status,
            "result_label": config["result_label"],
            "model_configuration": "frozen_7b_stage1_vllm_32b_bf16",
            "top5_root_causes": top5,
            "rank_of_ranks": rank_data,
            "valid_stage2_rounds": len(round_scores),
            "inference_seconds": time.perf_counter() - case_started,
        }
        if top3 is not None:
            record.update(
                {
                    "predicted_fault_type": top3[0]["fault_type"],
                    "predicted_fault_category": top3[0]["fault_category"],
                    "fault_type_top3": top3,
                }
            )
        else:
            record["classification_error"] = classification_error
        append_jsonl(predictions_path, record)
        report = [
            f"# {incident_id} — {config['result_label']}",
            "",
            f"- Stage 1 candidates: {len(shortlist)}",
            f"- Evidence budget per candidate: {config['evidence_per_candidate']}",
            f"- Valid Stage 2 rounds: {len(round_scores)}/3",
            f"- Classification: {classification_status}",
            f"- Elapsed seconds: {record['inference_seconds']:.3f}",
            "",
            "## Top5 root causes",
            *[
                f"{item['rank']}. {item['node_id']} ({item['failure_score']:.6f})"
                for item in top5
            ],
        ]
        if top3:
            report += ["", "## Top3 fault types"] + [
                f"{item['rank']}. {item['fault_type']} ({item['confidence']:.6f})"
                for item in top3
            ]
        (args.output_dir / f"prediction/reports/{incident_id}.md").write_text(
            "\n".join(report) + "\n"
        )
        completed.append(incident_id)
        write_json(
            state_path,
            {
                "status": "inference_running",
                "completed_cases": len(completed),
                "completed_incident_ids": completed,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=True,
        )


def oracle_root_payload(
    stage1: dict[str, Any],
    evidence: dict[str, Any],
    *,
    max_evidence: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    compact = compact_candidate(
        {**stage1, "candidate_id": "C01"},
        evidence,
        max_evidence=max_evidence,
    )
    ids = {item["evidence_id"] for item in compact["evidence"]}
    return [
        {
            "candidate_id": "C01",
            "device_role": compact["device_role"],
            "region_id": compact["region_id"],
            "root_weight": 1.0,
            "evidence": compact["evidence"],
            "data_quality_statuses": compact["data_quality_statuses"],
        }
    ], ids


def post_freeze(
    args: argparse.Namespace,
    config: dict[str, Any],
    backend: VLLM32BBackend,
    taxonomy: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    inference_commit: str,
) -> dict[str, Any]:
    predictions_path = args.output_dir / "prediction/predictions.jsonl"
    inventory = json.loads(
        (args.bundle_root / "topology/candidate_inventory.json").read_text()
    )
    records = load_prediction_jsonl(predictions_path)
    validation = validate_predictions(
        records,
        expected_incident_ids={item["incident_id"] for item in incidents},
        candidate_node_ids={item["node_id"] for item in inventory},
        taxonomy=taxonomy,
        expected_rank_rounds=3,
    )
    if not validation["valid"]:
        raise ValidationError("; ".join(validation["errors"]))
    prompt_sha, prompt_files = prompt_digest()
    frozen = freeze_predictions(
        predictions_path=predictions_path,
        output_dir=args.output_dir / "prediction",
        validation=validation,
        manifest={
            "evaluated_cases": 10,
            "git_commit": inference_commit,
            "random_seed": config["stage2_seeds"][0],
            "rank_rounds": 3,
            "small_model": "DeepSeek-R1-Distill-Qwen-7B (frozen Stage 1)",
            "large_model": "DeepSeek-R1-Distill-Qwen-32B",
            "model_configuration": "frozen_7b_stage1_vllm_32b_bf16",
            "result_label": config["result_label"],
            "paper_model_equivalent": False,
            "development_dataset": True,
            "strict_blind_evaluation": False,
            "ground_truth_used_for_post_run_diagnostics": True,
            "ground_truth_available_to_inference": False,
            "stage1_frozen": True,
            "stage2_large_model_replaced_by_7b": False,
            "backend": "vllm",
            "precision": "bfloat16",
            "quantization": None,
            "prompt_sha256": prompt_sha,
            "prompt_file_sha256": prompt_files,
            "config_sha256": sha256_file(args.config),
            "model_manifest": backend.model_manifest(),
        },
    )
    Path(frozen["frozen_path"]).chmod(0o444)
    write_json(
        args.output_dir / "prediction/truth_access_guard.json",
        {
            "formal_inference_read_ground_truth": False,
            "prediction_frozen_before_ground_truth_open": True,
            "frozen_sha256": frozen["sha256"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Ground truth is first opened below, after immutable prediction freezing.
    truth_path = args.bundle_root / "evaluation/ground_truth.jsonl"
    truths = load_jsonl(truth_path)
    score = score_predictions(
        predictions=records,
        ground_truth=truths,
        root_node_field="root_node_id",
        expected_case_count=10,
    )
    evaluation = args.output_dir / "evaluation"
    write_json(evaluation / "score_summary.json", score)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/render_real_evaluation.py"),
            "--score-summary",
            str(evaluation / "score_summary.json"),
            "--output-dir",
            str(evaluation),
            "--result-label",
            config["result_label"],
        ],
        check=True,
    )

    full_rankings = {
        item["incident_id"]: {
            row["node_id"]: row for row in item["ranking"]
        }
        for item in load_jsonl(
            args.output_dir / "stage1_reference/full_rankings.jsonl"
        )
    }
    truth_by_id = {item["incident_id"]: item for item in truths}
    oracle_dir = args.output_dir / "oracle_classification"
    oracle_dir.mkdir(exist_ok=False)
    (oracle_dir / "oracle_type_scores.jsonl").touch()
    (oracle_dir / "oracle_top3.jsonl").touch()
    oracle_call_start = len(backend.calls)
    oracle_failures = []
    for incident in incidents:
        incident_id = incident["incident_id"]
        prepared = json.loads(
            (args.input_root / incident_id / "incident_input.json").read_text()
        )
        evidence_by_node = {
            item["node_id"]: item for item in prepared["engineering_evidence"]
        }
        root_node = truth_by_id[incident_id]["root_node_id"]
        roots, evidence_ids = oracle_root_payload(
            full_rankings[incident_id][root_node],
            evidence_by_node[root_node],
            max_evidence=config["evidence_per_candidate"],
        )
        try:
            top3, _ = run_classification(
                backend,
                incident_id,
                taxonomy,
                roots,
                evidence_ids,
                oracle_dir / "oracle_type_scores.jsonl",
                max_new_tokens=config["classification_max_new_tokens"],
                role="32B-OracleClassification",
            )
            append_jsonl(
                oracle_dir / "oracle_top3.jsonl",
                {
                    "incident_id": incident_id,
                    "diagnostic_only": True,
                    "oracle_root_node_id": root_node,
                    "predicted_fault_type": top3[0]["fault_type"],
                    "predicted_fault_category": top3[0]["fault_category"],
                    "fault_type_top3": top3,
                },
            )
        except Exception as exc:
            oracle_failures.append(
                {
                    "incident_id": incident_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    write_json(
        oracle_dir / "model_calls.json",
        backend.call_manifest()[oracle_call_start:],
    )
    write_json(
        oracle_dir / "run_manifest.json",
        {
            "diagnostic_only": True,
            "ground_truth_root_used": True,
            "ground_truth_fault_type_used_for_inference": False,
            "ground_truth_fault_category_used_for_inference": False,
            "classification_failures": oracle_failures,
            "model": backend.model_manifest(),
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/render_dataset_feasibility.py"),
            "--run-dir",
            str(args.output_dir),
            "--baseline-run",
            str(args.baseline_run),
            "--input-root",
            str(args.input_root),
            "--ground-truth",
            str(truth_path),
            "--taxonomy",
            str(args.bundle_root / "schemas/fault_taxonomy.json"),
            "--candidate-inventory",
            str(args.bundle_root / "topology/candidate_inventory.json"),
        ],
        check=True,
    )
    return {"frozen": frozen, "score": score}


def performance_summary(
    args: argparse.Namespace,
    backend: VLLM32BBackend,
    total_seconds: float,
    case_count: int,
) -> dict[str, Any]:
    calls = backend.call_manifest()
    stage2 = [item for item in calls if item["role"] == "32B-Stage2"]
    classification = [
        item for item in calls if item["role"] == "32B-Classification"
    ]
    oracle = [
        item for item in calls if item["role"] == "32B-OracleClassification"
    ]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "request_count": len(items),
            "input_tokens": sum(item["input_tokens"] for item in items),
            "output_tokens": sum(item["output_tokens"] for item in items),
            "mean_request_batch_seconds": (
                statistics.mean(item["elapsed_seconds"] for item in items)
                if items
                else 0
            ),
            "mean_output_tokens_per_second": (
                statistics.mean(item["output_tokens_per_second"] for item in items)
                if items
                else 0
            ),
            "first_json_success_rate": (
                sum(item["attempt"] == 1 and item["error"] is None for item in items)
                / len(items)
                if items
                else 0
            ),
            "retry_records": sum(item["attempt"] > 1 for item in items),
            "failed_attempt_records": sum(item["error"] is not None for item in items),
        }

    partial = json.loads(args.int8_partial_manifest.read_text())
    return {
        "backend": backend.model_manifest(),
        "load_seconds": backend.load_seconds,
        "formal_plus_oracle_wall_seconds": total_seconds,
        "formal_case_count": case_count,
        "mean_wall_seconds_per_formal_case": total_seconds / case_count,
        "stage2": aggregate(stage2),
        "formal_classification": aggregate(classification),
        "oracle_classification": aggregate(oracle),
        "partial_int8_baseline": {
            "completed_cases": partial["completed_cases"],
            "elapsed_seconds": partial["elapsed_seconds"],
            "mean_wall_seconds_per_case": (
                partial["elapsed_seconds"] / partial["completed_cases"]
            ),
            "precision": partial["precision"],
            "backend": partial["backend"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incident-id")
    parser.add_argument("--complete-experiment", action="store_true")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--int8-partial-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.complete_experiment and (
        args.baseline_run is None or args.int8_partial_manifest is None
    ):
        parser.error("--complete-experiment requires baseline and INT8 manifest")
    config = json.loads(args.config.read_text())
    if config["precision"] != "bfloat16" or config.get("quantization") is not None:
        raise ValidationError("vLLM formal runner requires unquantized BF16")
    args.output_dir.mkdir(parents=True)
    for relative in (
        "stage2",
        "classification",
        "prediction/reports",
        "prediction/logs/raw_outputs",
        "environment",
    ):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)
    copy_stage1_reference(
        args.stage1_source, args.output_dir / "stage1_reference"
    )
    shortlists = {
        item["incident_id"]: item
        for item in load_jsonl(args.stage1_source / "shortlist.jsonl")
    }
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    incidents = load_jsonl(args.bundle_root / "experiment/incidents.jsonl")
    incident_paths = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if args.incident_id:
        incident_paths = [
            path for path in incident_paths if path.parent.name == args.incident_id
        ]
    if args.complete_experiment and len(incident_paths) != 10:
        raise ValidationError("formal vLLM run requires exactly 10 incidents")
    write_json(
        args.output_dir / "run_state.json",
        {
            "status": "initializing",
            "ground_truth_opened": False,
            "incident_count": len(incident_paths),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    subprocess.run(
        ["nvidia-smi"],
        stdout=(args.output_dir / "environment/gpu_before.txt").open("w"),
        check=True,
    )
    inference_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    backend = build_backend(
        args, config, args.output_dir / "prediction/logs/raw_outputs"
    )
    started = time.perf_counter()
    backend.load()
    run_inference(
        args, config, backend, shortlists, taxonomy, incident_paths
    )
    formal_call_count = len(backend.calls)
    formal_seconds = time.perf_counter() - started
    prediction_records = load_prediction_jsonl(
        args.output_dir / "prediction/predictions.jsonl"
    )
    post_result = None
    if args.complete_experiment:
        post_result = post_freeze(
            args, config, backend, taxonomy, incidents, inference_commit
        )
    total_seconds = time.perf_counter() - started
    calls = backend.call_manifest()
    write_json(
        args.output_dir / "prediction/logs/model_calls.json",
        calls[:formal_call_count],
    )
    write_json(
        args.output_dir / "environment/model_manifest.json",
        backend.model_manifest(),
    )
    write_json(
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
    prompt_sha, prompt_files = prompt_digest()
    manifest = {
        **config,
        "git_commit": inference_commit,
        "config_sha256": sha256_file(args.config),
        "prompt_sha256": prompt_sha,
        "prompt_file_sha256": prompt_files,
        "formal_inference_read_ground_truth": False,
        "model_manifest": backend.model_manifest(),
        "prediction_records": len(prediction_records),
        "formal_inference_seconds": formal_seconds,
        "total_seconds": total_seconds,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_frozen": bool(post_result),
        "frozen_sha256": (
            post_result["frozen"]["sha256"] if post_result else None
        ),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    if args.int8_partial_manifest is not None:
        performance = performance_summary(
            args, backend, total_seconds, len(incident_paths)
        )
        write_json(
            args.output_dir / "evaluation/performance_comparison.json",
            performance,
        )
    write_json(
        args.output_dir / "run_state.json",
        {
            "status": "completed",
            "prediction_records": len(prediction_records),
            "prediction_frozen": bool(post_result),
            "ground_truth_opened": bool(post_result),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    print(
        f"vLLM run completed: cases={len(prediction_records)}; "
        f"formal={formal_seconds:.3f}s; total={total_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
