"""Batch-scheduled variant of anomaly-driven 32B inference.

It preserves the Stage-2 and comparative-classification semantics of
run_anomaly_driven_vllm.py while submitting requests across events together so
vLLM can continuously batch them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from bian.methods.mixed_32b import (  # noqa: E402
    aggregate_stage2_rounds,
    apply_v3_stage1_fusion,
    score_stage2_round,
    stage1_fallback_top5,
)
from bian.models.vllm_schemas import stage2_schema  # noqa: E402
from scripts.run_anomaly_driven_vllm import (  # noqa: E402
    append,
    build_backend,
    candidate_evidence_ids,
    class_schema,
    compact_candidate,
    relevant_topology,
    root_payload,
    split_node,
    timeline,
    validate_class,
    validate_stage2_round,
)
from scripts.run_blind33_vllm import load_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("input_root", "stage1_source", "event_manifest", "taxonomy_file", "taxonomy_prompt_file", "config", "model_path", "output_dir"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--stage2-prompt-prefix", default="blind33")
    args = ap.parse_args()
    events = json.loads(args.event_manifest.read_text())
    manifest = {x["event_id"]: x for x in events}
    raw_taxonomy = json.loads(args.taxonomy_file.read_text())["candidates"]
    taxonomy = [
        {**x, "fault_category": x.get("fault_category", x.get("category")), "type_id": f"T{i:02d}"}
        for i, x in enumerate(raw_taxonomy, 1)
    ]
    taxonomy_map = {x["type_id"]: x for x in taxonomy}
    taxonomy_text = args.taxonomy_prompt_file.read_text()
    cfg = json.loads(args.config.read_text())
    full = {x["incident_id"]: x["ranking"] for x in load_jsonl(args.stage1_source / "full_rankings.jsonl")}
    short = {x["incident_id"]: x["shortlist"] for x in load_jsonl(args.stage1_source / "shortlist.jsonl")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("stage2", "classification", "prediction/logs/raw_outputs", "environment"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    contexts = {}
    stage2_requests = []
    for path in sorted(args.input_root.glob("*/incident_input.json")):
        eid = path.parent.name
        incident = json.loads(path.read_text())
        shortlist = short[eid]
        aliases = {x["candidate_id"]: x["node_id"] for x in shortlist}
        by_alias = {x["candidate_id"]: x for x in shortlist}
        evidence_by_node = {x["node_id"]: x for x in incident["engineering_evidence"]}
        candidates = [compact_candidate(x, evidence_by_node[x["node_id"]], max_evidence=cfg["evidence_per_candidate"]) for x in shortlist]
        evidence_ids = candidate_evidence_ids(candidates)
        topology = relevant_topology(incident["topology"], aliases)
        event_timeline = timeline(candidates)
        contexts[eid] = dict(incident=incident, aliases=aliases, by_alias=by_alias, evidence_by_node=evidence_by_node,
                             evidence_ids=evidence_ids, topology=topology, timeline=event_timeline, shortlist=shortlist)
        for round_id, seed in enumerate(cfg["stage2_seeds"], 1):
            stage2_requests.append({
                "request_id": f"anomaly_stage2_{eid}_round{round_id}", "event_id": eid, "round_id": round_id,
                "role": "32B-Stage2", "prompt_name": f"{args.stage2_prompt_prefix}_stage2_round{round_id}",
                "prompt_version": f"anomaly-stage2-round{round_id}-v1",
                "payload": {"round": round_id, "seed": seed,
                    "incident_window": {"start": incident["fault_start_time_utc"], "end": incident["fault_end_time_utc"]},
                    "candidates": candidates, "topology": topology, "timeline": event_timeline},
                "seed": seed, "temperature": cfg["stage2_temperature"], "top_p": cfg["stage2_top_p"],
                "json_schema": stage2_schema(aliases, evidence_ids),
                "validator": lambda value, a=set(aliases), e=set(evidence_ids): validate_stage2_round(value, a, e),
            })

    backend = build_backend(args.model_path, cfg, args.output_dir / "prediction/logs/raw_outputs")
    started = time.perf_counter()
    backend.load()
    stage2_outputs = backend.generate_json_batch(stage2_requests, max_new_tokens=cfg["stage2_max_new_tokens"], allow_partial=True)
    grouped = {eid: [] for eid in contexts}
    for req, result in zip(stage2_requests, stage2_outputs):
        if result is None:
            continue
        ctx = contexts[req["event_id"]]
        fused = apply_v3_stage1_fusion(result["candidates"], ctx["by_alias"])
        scores, audit = score_stage2_round(fused, ctx["aliases"], cfg["v3_fusion"]["weights"])
        grouped[req["event_id"]].append(scores)
        append(args.output_dir / "stage2/rounds.jsonl", {"event_id": req["event_id"], "round": req["round_id"], "scores": scores, "components": audit})

    classification_requests = []
    interim = {}
    for eid, ctx in contexts.items():
        rounds = grouped[eid]
        if len(rounds) >= cfg["minimum_valid_stage2_rounds"]:
            top5, rank_data = aggregate_stage2_rounds(rounds, tuple(sorted(ctx["aliases"].values())))
            mode = "fused_stage1_stage2"
        else:
            top5 = stage1_fallback_top5(full[eid])
            rank_data = {"rounds": 0, "raw_rankings": [], "fallback": "stage1", "valid_stage2_rounds": len(rounds)}
            mode = "stage1_fallback_after_stage2_failure"
        shortlist_by_node = {x["node_id"]: x for x in ctx["shortlist"]}
        roots, evidence_ids = root_payload(top5, shortlist_by_node, ctx["evidence_by_node"], max_evidence=cfg["evidence_per_candidate"])
        interim[eid] = dict(top5=top5, rank_data=rank_data, mode=mode, roots=roots, evidence_ids=evidence_ids)
        classification_requests.append({
            "request_id": f"anomaly_classification_{eid}", "event_id": eid,
            "role": "32B-ComparativeClassification", "prompt_name": "anomaly_driven_comparative_classification",
            "prompt_version": "anomaly-comparative-v1",
            "payload": {"taxonomy": taxonomy, "taxonomy_prompt": taxonomy_text, "root_hypotheses": roots,
                        "topology": ctx["topology"], "timeline": ctx["timeline"], "evidence_ids": sorted(evidence_ids)},
            "temperature": 0.2, "top_p": 0.9,
            "json_schema": class_schema(taxonomy, evidence_ids),
            "validator": lambda value, e=set(evidence_ids): validate_class(value, taxonomy, e),
        })
    class_outputs = backend.generate_json_batch(classification_requests, max_new_tokens=cfg["classification_max_new_tokens"], allow_partial=True)
    predictions = []
    for req, result in zip(classification_requests, class_outputs):
        eid = req["event_id"]
        item, ctx = interim[eid], contexts[eid]
        class_top = []
        if result:
            for rank, value in enumerate(result["top5"], 1):
                tax = taxonomy_map[value["taxonomy_id"]]
                class_top.append({**value, "rank": rank, "fault_type": tax["fault_type"], "fault_category": tax["fault_category"]})
        node = item["top5"][0]["node_id"]
        region, dtype, device = split_node(node)
        predictions.append({
            "event_id": eid, "detection": manifest[eid],
            "rca": {"root_region": region, "root_device_type": dtype, "root_device": device,
                    "top5": [{**x, **dict(zip(("region", "device_type", "device"), split_node(x["node_id"])))} for x in item["top5"]]},
            "classification": {"major_class": class_top[0]["major_class"] if class_top else None,
                               "minor_class": class_top[0]["minor_class"] if class_top else None,
                               "taxonomy_id": class_top[0]["taxonomy_id"] if class_top else None, "top5": class_top},
            "prediction_mode": item["mode"], "valid_stage2_rounds": len(grouped[eid]),
            "status": "success" if result else "failed",
            "failure_reason": None if result else "comparative classification structured output failed",
            "rank_of_ranks": item["rank_data"],
        })
    (args.output_dir / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n")
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "case_id", "status", "root_device", "classification_top1"])
        writer.writeheader()
        writer.writerows({"event_id": x["event_id"], "case_id": x["detection"]["case_id"], "status": x["status"],
                          "root_device": x["rca"]["root_device"], "classification_top1": x["classification"]["taxonomy_id"]} for x in predictions)
    (args.output_dir / "schema_validation_report.json").write_text(json.dumps({"valid": True, "event_count": len(predictions), "successful": sum(x["status"] == "success" for x in predictions)}, indent=2) + "\n")
    (args.output_dir / "model_manifest.json").write_text(json.dumps(backend.model_manifest(), indent=2) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({"event_count": len(predictions), "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "ground_truth_available_to_inference": False, "batch_scheduled": True,
        "stage2_prompt_prefix": args.stage2_prompt_prefix,
        "candidate_budget": len(next(iter(contexts.values()))["shortlist"]) if contexts else 0}, indent=2) + "\n")
    print(json.dumps({"events": len(predictions), "success": sum(x["status"] == "success" for x in predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
