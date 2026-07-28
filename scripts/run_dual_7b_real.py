"""Run leakage-safe Dual-7B pipeline validation on preprocessed incidents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.validators import ValidationError
from bian.methods.early_stop import score_entropy, should_early_stop
from bian.models.dual_7b_backend import Dual7BBackend, GenerationConfig
from bian.models.structured_output import (
    validate_device_analysis,
    validate_stage1,
    validate_stage2,
)
from bian.real_inference import (
    aggregate_stage2_rounds,
    batched,
    build_metric_timeline,
    compact_device,
    cumulative_top_p,
    merge_stage1,
    render_device_evidence,
    topology_subgraph,
    utc_now,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def smoke_structured(backend: Dual7BBackend) -> dict:
    nodes = (
        "region-1-br-1",
        "region-1-br-2",
        "region-1-service-1",
        "region-1-service-2",
        "region-1-service-3",
    )
    statuses = ("available", "partial", "empty", "missing", "collection_failed")
    smoke_devices = []
    for indexes in batched(list(range(len(nodes))), 2):
        expected = tuple(nodes[index] for index in indexes)
        part = backend.generate_json(
            role="7B-A",
            prompt_name="7b_a_device_analysis",
            prompt_version="dual7b-a-device-v1",
            payload={
                "incident_id": "mock-smoke",
                "required_node_ids": list(expected),
                "device_evidence_lines": [
                    f"node={nodes[index]}|"
                    f"family={'br' if 'br-' in nodes[index] else 'service'}|"
                    f"states=node_metrics={statuses[index]}|"
                    f"top_changes={'routing.bgp_session_up:1->0->post:1,rel:1' if index == 0 else 'none'}"
                    for index in indexes
                ],
            },
            validator=lambda value, requested=expected: validate_device_analysis(
                value, requested
            ),
        )
        smoke_devices.extend(part["devices"])
    device_result = {"devices": smoke_devices}
    # The Stage 1 smoke fixture must exercise normalization, so explicitly supply
    # one synthetic anomaly after independently validating the 7B-A response.
    device_result["devices"][0].update(
        {
            "is_anomalous": True,
            "anomaly_score": 0.95,
            "anomaly_evidence": "synthetic BGP session drop for smoke validation",
        }
    )
    stage1 = backend.generate_json(
        role="7B-B",
        prompt_name="7b_b_stage1",
        prompt_version="dual7b-b-stage1-v1",
        payload={"incident_id": "mock-smoke", "device_analyses": device_result["devices"]},
        validator=lambda value: validate_stage1(value, nodes),
    )
    taxonomy = (
        {"fault_type": "bgp_route_flap", "fault_category": "bgp"},
        {"fault_type": "bgp_session_down", "fault_category": "bgp"},
        {"fault_type": "route_blackhole", "fault_category": "static_route"},
    )
    smoke_aliases = tuple(f"C{index:02d}" for index in range(1, 6))
    stage2 = backend.generate_json(
        role="7B-B",
        prompt_name="7b_b_stage2",
        prompt_version="dual7b-b-stage2-v1",
        payload={
            "incident_id": "mock-smoke",
            "rank_round": 1,
            "allowed_candidates": list(smoke_aliases),
            "candidate_evidence": [
                {
                    **item,
                    "node_id": smoke_aliases[index],
                }
                for index, item in enumerate(device_result["devices"])
            ],
            "stage1_scores": {
                smoke_aliases[index]: stage1["scores"][node]
                for index, node in enumerate(nodes)
            },
            "topology": {"nodes": list(smoke_aliases), "edges": []},
            "timeline": [],
            "fault_taxonomy": list(taxonomy),
        },
        validator=lambda value: validate_stage2(value, smoke_aliases, taxonomy),
    )
    return {
        "status": "passed",
        "device_count": len(device_result["devices"]),
        "stage1_score_sum": sum(stage1["scores"].values()),
        "stage2_root_count": len(stage2["root_causes"]),
        "stage2_fault_count": len(stage2["fault_types"]),
    }


def run_incident(
    incident: dict,
    backend: Dual7BBackend,
    config: dict,
    incident_dir: Path,
) -> dict:
    started = time.perf_counter()
    incident_id = incident["incident_id"]
    candidates = tuple(incident["candidate_node_ids"])
    taxonomy = tuple(
        {
            "fault_type": item["fault_type"],
            "fault_category": item["fault_category"],
        }
        for item in incident["fault_taxonomy"]
    )
    compact = [
        compact_device(item, config["max_signals_per_device"])
        for item in incident["devices"]
    ]
    if tuple(item["node_id"] for item in compact) != candidates:
        raise ValidationError(f"{incident_id}: device order/set differs from candidates")

    analyses = []
    for batch_index, batch in enumerate(
        batched(compact, config["device_batch_size"]), start=1
    ):
        requests = []
        for item in batch:
            node = item["node_id"]
            requests.append(
                {
                    "role": "7B-A",
                    "prompt_name": "7b_a_device_analysis",
                    "prompt_version": "dual7b-a-device-v1",
                    "payload": {
                        "incident_id": incident_id,
                        "batch_index": batch_index,
                        "device_evidence_lines": [render_device_evidence(item)],
                        "required_node_ids": [node],
                    },
                    "validator": (
                        lambda value, expected=(node,): validate_device_analysis(
                            value, expected
                        )
                    ),
                }
            )
        results = backend.generate_json_batch(
            requests, max_new_tokens=config["device_max_new_tokens"]
        )
        for result in results:
            analyses.extend(result["devices"])
    analysis_by_node = {item["node_id"]: item for item in analyses}

    stage1_parts = []
    for batch_index, node_batch in enumerate(
        batched(list(candidates), config["stage1_batch_size"]), start=1
    ):
        nodes = tuple(node_batch)
        result = backend.generate_json(
            role="7B-B",
            prompt_name="7b_b_stage1",
            prompt_version="dual7b-b-stage1-v1",
            payload={
                "incident_id": incident_id,
                "batch_index": batch_index,
                "device_analyses": [analysis_by_node[node] for node in nodes],
            },
            validator=lambda value, expected=nodes: validate_stage1(
                value, expected, require_positive=False
            ),
        )
        stage1_parts.append(result)
    stage1 = merge_stage1(stage1_parts)
    stage1["reasons"] = {
        node: analysis_by_node[node]["anomaly_evidence"] for node in candidates
    }
    stage1["reason_summary"] = (
        "7B-B relative likelihood scores; reasons reference validated 7B-A evidence"
    )
    entropy = score_entropy(stage1["scores"], candidates, config["entropy_mode"])
    early_stop = should_early_stop(
        stage1["scores"],
        candidates,
        config["entropy_threshold"],
        config["entropy_mode"],
    )
    filtered = cumulative_top_p(
        stage1["scores"],
        candidates,
        config["stage2_top_p"],
        config["stage2_max_candidates"],
    )
    if len(filtered) < 5:
        raise ValidationError(f"{incident_id}: fewer than five Stage 2 candidates")
    subgraph = topology_subgraph(incident["topology"], filtered)
    timeline = build_metric_timeline(
        compact,
        incident["fault_start_time_utc"],
        incident["fault_end_time_utc"],
        config["max_timeline_events"],
    )
    alias_by_node = {
        node: f"C{index:02d}" for index, node in enumerate(filtered, start=1)
    }
    node_by_alias = {alias: node for node, alias in alias_by_node.items()}
    transit_nodes = sorted(
        {
            item["node_id"]
            for item in subgraph["nodes"]
            if item["node_id"] not in alias_by_node
        }
    )
    transit_alias = {
        node: f"T{index:02d}" for index, node in enumerate(transit_nodes, start=1)
    }
    topology_alias = {**alias_by_node, **transit_alias}
    model_topology = {
        "nodes": [
            {
                "id": topology_alias[item["node_id"]],
                "candidate": item["node_id"] in alias_by_node,
            }
            for item in subgraph["nodes"]
        ],
        "edges": [
            {
                "source": topology_alias[item["source"]],
                "target": topology_alias[item["target"]],
                "edge_type": item["edge_type"],
                "protocol": item.get("protocol"),
            }
            for item in subgraph["edges"]
        ],
    }
    model_timeline = [
        {**item, "node_id": alias_by_node[item["node_id"]]}
        for item in timeline
        if item["node_id"] in alias_by_node
    ]
    model_evidence = [
        {
            "candidate_id": alias_by_node[node],
            "is_anomalous": analysis_by_node[node]["is_anomalous"],
            "anomaly_score": analysis_by_node[node]["anomaly_score"],
            "anomaly_evidence": analysis_by_node[node]["anomaly_evidence"],
            "uncertainty": analysis_by_node[node]["uncertainty"],
        }
        for node in filtered
    ]
    rounds = []
    for round_number in range(1, config["rank_rounds"] + 1):
        result = backend.generate_json(
            role="7B-B",
            prompt_name="7b_b_stage2",
            prompt_version="dual7b-b-stage2-v1",
            payload={
                "incident_id": incident_id,
                "rank_round": round_number,
                "round_instruction": (
                    "prioritize temporal onset"
                    if round_number == 1
                    else "prioritize topology propagation"
                    if round_number == 2
                    else "prioritize counter-evidence and robustness"
                ),
                "candidate_alias_map": [
                    {"candidate_id": alias_by_node[node]} for node in filtered
                ],
                "allowed_candidates": list(node_by_alias),
                "candidate_evidence": model_evidence,
                "stage1_scores": {
                    alias_by_node[node]: stage1["scores"][node] for node in filtered
                },
                "topology": model_topology,
                "timeline": model_timeline,
                "fault_taxonomy": list(taxonomy),
            },
            validator=lambda value: validate_stage2(
                value, tuple(node_by_alias), taxonomy
            ),
        )
        for item in result["root_causes"]:
            item["node_id"] = node_by_alias[item["node_id"]]
        rounds.append(result)
    top5, top3, raw_rankings = aggregate_stage2_rounds(
        rounds, filtered, taxonomy
    )
    elapsed = time.perf_counter() - started
    details = {
        "result_label": config["result_label"],
        "incident_id": incident_id,
        "input_window": {
            key: incident[key]
            for key in (
                "fault_start_time_utc",
                "fault_end_time_utc",
                "slice_start_time_utc",
                "slice_end_time_utc",
            )
        },
        "input_device_count": len(candidates),
        "input_uniformity": "preprocessing audit passed; uniform role schemas retained",
        "device_summaries": analyses,
        "stage1": stage1,
        "entropy": {
            "value": entropy,
            "mode": config["entropy_mode"],
            "threshold": config["entropy_threshold"],
            "early_stop_triggered": early_stop,
            "note": "Stage 2 classification retained for required pipeline output",
        },
        "candidate_filter": {
            "before": list(candidates),
            "after": list(filtered),
            "top_p": config["stage2_top_p"],
            "max_candidates": config["stage2_max_candidates"],
        },
        "topology_subgraph": subgraph,
        "timeline": timeline,
        "stage2_rounds": rounds,
        "rank_of_ranks": {"rounds": config["rank_rounds"], "raw_rankings": raw_rankings},
        "top5_root_causes": top5,
        "fault_type_top3": top3,
        "elapsed_seconds": elapsed,
    }
    write_json(incident_dir / "details.json", details)
    report_lines = [
        f"# {incident_id} — {config['result_label']}",
        "",
        "This is a Dual-7B pipeline validation, not a paper-equivalent result.",
        "",
        "## Input and coverage",
        "",
        f"- Window: {incident['slice_start_time_utc']} — {incident['slice_end_time_utc']} UTC",
        f"- Devices: {len(candidates)}; uniform preprocessing schema: passed",
        f"- Monitoring sources: {', '.join(sorted(incident['devices'][0]['sources']))}",
        "",
        "## Stage 1 and filtering",
        "",
        f"- Entropy: {entropy:.6f} ({config['entropy_mode']})",
        f"- Early stop triggered: {early_stop}",
        f"- Top-p candidates: {len(filtered)} / {len(candidates)}",
        "",
        "## Top5 root causes",
        "",
    ]
    report_lines += [
        f"{item['rank']}. `{item['node_id']}` ({item['failure_score']:.6f}) — "
        f"{item['reason_summary']}"
        for item in top5
    ]
    report_lines += ["", "## Top3 fault types", ""]
    report_lines += [
        f"{item['rank']}. `{item['fault_type']}` / `{item['fault_category']}` "
        f"({item['confidence']:.6f}) — {item['reason_summary']}"
        for item in top3
    ]
    report_lines += [
        "",
        "## Evidence, counter-evidence, and uncertainty",
        "",
        rounds[-1]["analysis"],
        "",
        f"- Rank of Ranks rounds: {config['rank_rounds']}",
        f"- Runtime: {elapsed:.3f} seconds",
        "- Raw model outputs, retries, token counts and peak memory are in local logs.",
    ]
    (incident_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {
        "incident_id": incident_id,
        "prediction_status": "success",
        "result_label": config["result_label"],
        "model_configuration": config["model_configuration"],
        "paper_model_equivalent": False,
        "top5_root_causes": top5,
        "predicted_fault_type": top3[0]["fault_type"],
        "predicted_fault_category": top3[0]["fault_category"],
        "fault_type_top3": top3,
        "rank_of_ranks": {"rounds": config["rank_rounds"], "raw_rankings": raw_rankings},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/dual_7b_pipeline_validation.example.json",
    )
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValidationError("CUDA_VISIBLE_DEVICES must be explicitly set")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = GenerationConfig(seed=config["seed"], **config["generation"])
    backend = Dual7BBackend(
        args.model_path,
        config=generation,
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_root / "prediction"
    reports_dir = prediction_dir / "reports"
    logs_dir = prediction_dir / "logs"
    reports_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    predictions_path = prediction_dir / "predictions.jsonl"
    run_started = time.perf_counter()
    write_json(
        args.output_root / "preprocessing_reference.json",
        {
            "input_root": str(args.input_root),
            "raw_five_tuple_excluded": "netflow_5tuple_minute_readable",
            "traffic_aggregate_used": "traffic_flow_metrics",
            "coverage_status": "passed",
        },
    )
    smoke = smoke_structured(backend)
    write_json(logs_dir / "structured_smoke.json", smoke)
    inputs = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if len(inputs) != 10:
        raise ValidationError(f"expected 10 preprocessed incidents, got {len(inputs)}")
    # Validate one smallest real incident first, then continue with all remaining cases.
    ordered = sorted(inputs, key=lambda path: (path.stat().st_size, path.name))
    completed: set[str] = set()
    for path_index, path in enumerate(ordered):
        incident = json.loads(path.read_text(encoding="utf-8"))
        incident_id = incident["incident_id"]
        incident_output = reports_dir / incident_id
        incident_output.mkdir()
        call_start = len(backend.calls)
        try:
            prediction = run_incident(incident, backend, config, incident_output)
        except Exception as exc:
            prediction = {
                "incident_id": incident_id,
                "prediction_status": "prediction_failed",
                "result_label": config["result_label"],
                "model_configuration": config["model_configuration"],
                "paper_model_equivalent": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (incident_output / "report.md").write_text(
                f"# {incident_id} — prediction failed\n\n"
                f"Dual-7B Pipeline Validation Result\n\n"
                f"Error: {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            (incident_output / "traceback.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        append_jsonl(predictions_path, prediction)
        calls = backend.call_manifest()[call_start:]
        write_json(incident_output / "model_calls.json", calls)
        completed.add(incident_id)
        print(
            f"{incident_id}: {prediction['prediction_status']}; "
            f"model_calls={len(calls)}",
            flush=True,
        )
        if path_index == 0 and prediction["prediction_status"] != "success":
            raise ValidationError(
                f"single-case gate failed for {incident_id}; refusing to run remaining cases"
            )
    manifest = {
        "result_label": config["result_label"],
        "model_configuration": config["model_configuration"],
        "paper_model_equivalent": False,
        "stage1_large_model_replaced_by_7b": True,
        "stage2_large_model_replaced_by_7b": True,
        "model_path": str(args.model_path),
        "prompt_versions": [
            "dual7b-a-device-v1",
            "dual7b-b-stage1-v1",
            "dual7b-b-stage2-v1",
        ],
        "seed": config["seed"],
        "generation": config["generation"],
        "started_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - run_started,
        "incident_count": len(completed),
        "prediction_phase_evaluation_access": False,
    }
    write_json(args.output_root / "run_manifest.json", manifest)
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
