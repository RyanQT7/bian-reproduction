"""Formal inference for frozen 7B Stage 1 plus sharded 32B Stage 2/classifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError, normalize_scores
from bian.methods.mixed_32b import (
    aggregate_stage2_rounds,
    aggregate_type_result,
    score_stage2_round,
    validate_stage2_round,
    validate_type_result,
)
from bian.models.dual_7b_backend import GenerationConfig
from bian.models.sharded_backend import Sharded32BBackend
from bian.taxonomy_profiles import profile


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def compact_candidate(stage1: dict, evidence: dict) -> dict:
    by_id = {item["evidence_id"]: item for item in evidence["evidence"]}
    selected_ids = list(
        dict.fromkeys(
            stage1.get("supporting_evidence_ids", [])
            + stage1.get("counter_evidence_ids", [])
        )
    )[:6]
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
        "evidence": [
            {
                field: by_id[evidence_id][field]
                for field in (
                    "evidence_id",
                    "metric_name",
                    "source_type",
                    "first_change_time",
                    "peak_time",
                    "recovery_time",
                    "pre_value",
                    "fault_value",
                    "post_value",
                    "absolute_delta",
                    "stable_change_score",
                    "direction",
                    "status_transition",
                    "data_quality_status",
                    "direct_fault_evidence",
                )
            }
            for evidence_id in selected_ids
            if evidence_id in by_id
        ],
    }


def compact_topology(topology: dict, alias_to_node: dict[str, str]) -> dict:
    node_to_alias = {node: alias for alias, node in alias_to_node.items()}
    edges = []
    for edge in topology["edges"]:
        source, target = edge["source"], edge["target"]
        if source in node_to_alias and target in node_to_alias:
            edges.append(
                {
                    "source": node_to_alias[source],
                    "target": node_to_alias[target],
                    "relation": edge.get("relation", edge.get("type", "connected")),
                }
            )
    return {"nodes": sorted(alias_to_node), "edges": edges}


def candidate_evidence_ids(candidates: list[dict]) -> set[str]:
    return {
        item["evidence_id"]
        for candidate in candidates
        for item in candidate["evidence"]
    }


def root_payload(
    top5: list[dict],
    shortlist_by_node: dict[str, dict],
    evidence_by_node: dict[str, dict],
) -> tuple[list[dict], dict[str, str], set[str]]:
    aliases = {f"C{index:02d}": item["node_id"] for index, item in enumerate(top5, 1)}
    roots = []
    evidence_ids = set()
    for index, top in enumerate(top5, 1):
        alias = f"C{index:02d}"
        node = top["node_id"]
        compact = compact_candidate(
            {**shortlist_by_node[node], "candidate_id": alias},
            evidence_by_node[node],
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
    return roots, aliases, evidence_ids


def run_classification(
    backend: Sharded32BBackend,
    incident_id: str,
    taxonomy: list[dict],
    roots: list[dict],
    evidence_ids: set[str],
    output_path: Path,
) -> tuple[list[dict], list[dict]]:
    root_aliases = {item["candidate_id"] for item in roots}
    root_weights = {
        item["candidate_id"]: item["root_weight"] for item in roots
    }
    type_results = []
    raw_scores = {}
    pending = []
    for index, taxonomy_item in enumerate(taxonomy, 1):
        type_id = f"T{index:02d}"
        seed = (
            int(hashlib.sha256(f"{incident_id}:{type_id}".encode()).hexdigest()[:8], 16)
            % (2**31)
        )
        pending.append(
            (
                taxonomy_item,
                seed,
                {
                    "role": "32B-Classification",
                    "prompt_name": "32b_type_ovr",
                    "prompt_version": "32b-type-ovr-v1",
                    "payload": {
                        "type": {
                            "type_id": type_id,
                            "fault_category": taxonomy_item["fault_category"],
                            **profile(taxonomy_item["fault_type"]),
                        },
                        "root_hypotheses": roots,
                        "deterministic_seed": seed,
                    },
                    "validator": lambda value, expected=type_id: validate_type_result(
                        value, expected, root_aliases, evidence_ids
                    ),
                },
            )
        )
    for offset in range(0, len(pending), 4):
        chunk = pending[offset : offset + 4]
        outputs = backend.generate_json_batch(
            [item[2] for item in chunk], max_new_tokens=2048
        )
        for (taxonomy_item, seed, _request), result in zip(chunk, outputs):
            type_id = result["type_id"]
            score, contributions = aggregate_type_result(result, root_weights)
            raw_scores[taxonomy_item["fault_type"]] = score
            record = {
                "incident_id": incident_id,
                "type_id": type_id,
                "fault_type": taxonomy_item["fault_type"],
                "fault_category": taxonomy_item["fault_category"],
                "seed": seed,
                "aggregated_raw_score": score,
                "root_hypotheses": contributions,
            }
            type_results.append(record)
            append(output_path, record)
    if not any(raw_scores.values()):
        raise ValidationError("all one-vs-rest type scores are zero")
    normalized = normalize_scores(raw_scores)
    ranked = sorted(raw_scores, key=lambda item: (-raw_scores[item], item))
    taxonomy_map = {
        item["fault_type"]: item["fault_category"] for item in taxonomy
    }
    top3 = [
        {
            "rank": rank,
            "fault_type": fault_type,
            "fault_category": taxonomy_map[fault_type],
            "confidence": normalized[fault_type],
            "score_kind": "normalized_model_score",
            "reason_summary": "32B independent one-vs-rest Top5-root weighted score",
        }
        for rank, fault_type in enumerate(ranked[:3], 1)
    ]
    # Aggregation order invariance: recompute from reverse cached results.
    reverse = {
        item["fault_type"]: item["aggregated_raw_score"]
        for item in reversed(type_results)
    }
    reverse_ranked = sorted(reverse, key=lambda item: (-reverse[item], item))
    if reverse_ranked[:3] != ranked[:3]:
        raise ValidationError("classification aggregation is order dependent")
    return top3, type_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--incident-id")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for relative in (
        "stage1_reference",
        "stage2",
        "classification",
        "prediction/reports",
        "prediction/logs",
        "environment",
    ):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    shortlists = {
        item["incident_id"]: item
        for item in load_jsonl(args.stage1_source / "shortlist.jsonl")
    }
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    paths = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if args.incident_id:
        paths = [path for path in paths if path.parent.name == args.incident_id]
    backend = Sharded32BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=config["stage2_max_input_tokens"],
            max_new_tokens=config["stage2_max_new_tokens"],
            temperature=config["stage2_temperature"],
            top_p=config["stage2_top_p"],
            retries=config["structured_retries"],
            seed=config["stage2_seeds"][0],
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
        device_map={"": 0} if config["precision"] != "bfloat16" else "balanced",
        max_memory=(
            {0: "44GiB"}
            if config["precision"] != "bfloat16"
            else {0: "44GiB", 1: "44GiB"}
        ),
        precision=config["precision"],
    )
    started = time.perf_counter()
    load_started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - load_started
    model_manifest = backend.model_manifest()
    predictions_path = args.output_dir / "prediction/predictions.jsonl"
    type_scores_path = args.output_dir / "classification/type_scores.jsonl"
    round_paths = [
        args.output_dir / f"stage2/round_{index}.jsonl" for index in (1, 2, 3)
    ]
    for path in paths:
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
            compact_candidate(item, evidence_by_node[item["node_id"]])
            for item in shortlist
        ]
        evidence_ids = candidate_evidence_ids(candidates)
        valid_round_scores = []
        round_audits = []
        round_errors = []
        for round_index, seed in enumerate(config["stage2_seeds"], 1):
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            prompt = "32b_stage2" if round_index == 1 else f"32b_stage2_round{round_index}"
            try:
                output = backend.generate_json(
                    role="32B-Stage2",
                    prompt_name=prompt,
                    prompt_version=f"32b-stage2-round{round_index}-v1",
                    payload={
                        "round": round_index,
                        "seed": seed,
                        "incident_window": {
                            "start": incident["fault_start_time_utc"],
                            "end": incident["fault_end_time_utc"],
                        },
                        "candidates": candidates,
                        "topology": compact_topology(
                            incident["topology"], alias_to_node
                        ),
                        "timeline": sorted(
                            [
                                {
                                    "evidence_id": evidence["evidence_id"],
                                    "candidate_id": candidate["candidate_id"],
                                    "first_change_time": evidence["first_change_time"],
                                    "metric_name": evidence["metric_name"],
                                    "direction": evidence["direction"],
                                }
                                for candidate in candidates
                                for evidence in candidate["evidence"]
                                if evidence["first_change_time"]
                            ],
                            key=lambda item: (
                                item["first_change_time"],
                                item["evidence_id"],
                            ),
                        ),
                    },
                    validator=lambda value: validate_stage2_round(
                        value, set(alias_to_node), evidence_ids
                    ),
                )
                scores, audit = score_stage2_round(
                    output["candidates"],
                    alias_to_node,
                    config["stage2_weights"],
                )
                valid_round_scores.append(scores)
                round_audits.append(audit)
                append(
                    round_paths[round_index - 1],
                    {
                        "incident_id": incident_id,
                        "round": round_index,
                        "seed": seed,
                        "candidate_map": alias_to_node,
                        "scores": scores,
                        "components": audit,
                    },
                )
            except Exception as exc:
                round_errors.append(
                    {"round": round_index, "error_type": type(exc).__name__, "error": str(exc)}
                )
        if len(valid_round_scores) < 2:
            append(
                predictions_path,
                {
                    "incident_id": incident_id,
                    "prediction_status": "prediction_failed",
                    "error_type": "InsufficientValidStage2Rounds",
                    "error": json.dumps(round_errors),
                },
            )
            continue
        top5, rank_data = aggregate_stage2_rounds(
            valid_round_scores, tuple(sorted(alias_to_node.values()))
        )
        append(
            args.output_dir / "stage2/rank_of_ranks.jsonl",
            {
                "incident_id": incident_id,
                "valid_rounds": len(valid_round_scores),
                "round_errors": round_errors,
                "rank_of_ranks": rank_data,
                "top5": top5,
            },
        )
        shortlist_by_node = {item["node_id"]: item for item in shortlist}
        roots, root_aliases, root_evidence_ids = root_payload(
            top5, shortlist_by_node, evidence_by_node
        )
        classification_status = "success"
        classification_error = None
        top3 = None
        try:
            top3, _type_results = run_classification(
                backend,
                incident_id,
                taxonomy,
                roots,
                root_evidence_ids,
                type_scores_path,
            )
            append(
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
            "model_configuration": "frozen_7b_stage1_32b_stage2_classification",
            "top5_root_causes": top5,
            "rank_of_ranks": rank_data,
            "inference_seconds": time.perf_counter() - case_started,
            "valid_stage2_rounds": len(valid_round_scores),
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
        append(predictions_path, record)
        report = [
            f"# {incident_id} — {config['result_label']}",
            "",
            f"- Valid Stage 2 rounds: {len(valid_round_scores)}/3",
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
    total_seconds = time.perf_counter() - started
    (args.output_dir / "prediction/logs/model_calls.json").write_text(
        json.dumps([asdict(call) for call in backend.calls], ensure_ascii=False, indent=2)
        + "\n"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest = {
        **config,
        "git_commit": commit,
        "load_seconds": load_seconds,
        "total_inference_seconds": total_seconds,
        "formal_inference_read_ground_truth": False,
        "model_manifest": model_manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "environment/model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Mixed inference completed: cases={len(paths)}; "
        f"load={load_seconds:.3f}s; total={total_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
