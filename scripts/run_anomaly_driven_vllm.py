"""Event-driven BiAn RCA and closed-set comparative classification."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bian.data.validators import ValidationError
from bian.methods.mixed_32b import apply_v3_stage1_fusion, score_stage2_round, aggregate_stage2_rounds, stage1_fallback_top5
from bian.models.vllm_32b_backend import VLLM32BBackend
from bian.models.dual_7b_backend import GenerationConfig
from bian.models.vllm_schemas import stage2_schema
from scripts.run_blind33_vllm import append, build_backend, candidate_evidence_ids, compact_candidate, relevant_topology, root_payload, timeline, validate_stage2_round, load_jsonl


def class_schema(taxonomy: list[dict[str, Any]], evidence_ids: set[str]) -> dict[str, Any]:
    ids = [x["type_id"] for x in taxonomy]
    evidence = {"type": "array", "items": {"type": "string", "enum": sorted(evidence_ids)}, "maxItems": min(10, len(evidence_ids))}
    item = {"rank": {"type": "integer", "minimum": 1, "maximum": 5}, "major_class": {"type": "string"}, "minor_class": {"type": "string"}, "taxonomy_id": {"type": "string", "enum": ids}, "score": {"type": "number", "minimum": 0, "maximum": 1}, "supporting_evidence": evidence, "counter_evidence": evidence}
    return {"type": "object", "additionalProperties": False, "required": ["major_class", "minor_class", "taxonomy_id", "top5"], "properties": {"major_class": {"type": "string"}, "minor_class": {"type": "string"}, "taxonomy_id": {"type": "string", "enum": ids}, "top5": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "object", "additionalProperties": False, "required": list(item), "properties": item}}}}


def validate_class(value: dict[str, Any], taxonomy: list[dict[str, Any]], evidence_ids: set[str]) -> dict[str, Any]:
    top = value.get("top5")
    valid_ids = {x["type_id"] for x in taxonomy}
    if not isinstance(top, list) or len(top) != 5 or len({x.get("taxonomy_id") for x in top}) != 5:
        raise ValidationError("comparative classification requires five distinct types")
    if any(x.get("taxonomy_id") not in valid_ids for x in top): raise ValidationError("illegal taxonomy id")
    for x in top:
        if not set(x.get("supporting_evidence", [])) <= evidence_ids or not set(x.get("counter_evidence", [])) <= evidence_ids:
            raise ValidationError("classification evidence id is not in input")
    return value


def split_node(node: str) -> tuple[str, str, str]:
    parts = node.split("-")
    region = "-".join(parts[:2]); role = "-".join(parts[2:])
    device_type = "service-vm" if role.startswith("service-") else ("br" if role.startswith("br-") else ("cr" if role.startswith("cr-") else role))
    return region, device_type, node


def run_event(backend: VLLM32BBackend, incident: dict[str, Any], stage1: list[dict[str, Any]], shortlist: list[dict[str, Any]], taxonomy: list[dict[str, Any]], config: dict[str, Any], out: Path, taxonomy_text: str, detection: dict[str, Any]) -> dict[str, Any]:
    eid = incident["incident_id"]; aliases = {x["candidate_id"]: x["node_id"] for x in shortlist}; by_alias = {x["candidate_id"]: x for x in shortlist}; evidence_by_node = {x["node_id"]: x for x in incident["engineering_evidence"]}
    candidates = [compact_candidate(x, evidence_by_node[x["node_id"]], max_evidence=config["evidence_per_candidate"]) for x in shortlist]
    evidence_ids = candidate_evidence_ids(candidates); topology = relevant_topology(incident["topology"], aliases); event_timeline = timeline(candidates)
    requests=[]
    for idx, seed in enumerate(config["stage2_seeds"], 1):
        requests.append({"request_id":f"anomaly_stage2_{eid}_round{idx}","role":"32B-Stage2","prompt_name":f"blind33_stage2_round{idx}","prompt_version":f"anomaly-stage2-round{idx}-v1","payload":{"round":idx,"seed":seed,"incident_window":{"start":incident["fault_start_time_utc"],"end":incident["fault_end_time_utc"]},"candidates":candidates,"topology":topology,"timeline":event_timeline},"seed":seed,"temperature":config["stage2_temperature"],"top_p":config["stage2_top_p"],"json_schema":stage2_schema(aliases,evidence_ids),"validator":lambda v: validate_stage2_round(v,set(aliases),evidence_ids)})
    outputs=backend.generate_json_batch(requests,max_new_tokens=config["stage2_max_new_tokens"],allow_partial=True); rounds=[]
    for idx,(req,result) in enumerate(zip(requests,outputs),1):
        if result is None: continue
        fused=apply_v3_stage1_fusion(result["candidates"],by_alias); scores,audit=score_stage2_round(fused,aliases,config["v3_fusion"]["weights"]); rounds.append(scores); append(out/"stage2/rounds.jsonl",{"event_id":eid,"round":idx,"scores":scores,"components":audit})
    if len(rounds)>=config["minimum_valid_stage2_rounds"]: top5,rank_data=aggregate_stage2_rounds(rounds,tuple(sorted(aliases.values()))); mode="fused_stage1_stage2"
    else: top5=stage1_fallback_top5(stage1); rank_data={"rounds":0,"raw_rankings":[],"fallback":"stage1","valid_stage2_rounds":len(rounds)}; mode="stage1_fallback_after_stage2_failure"
    shortlist_by_node={x["node_id"]:x for x in shortlist}; roots,eids=root_payload(top5,shortlist_by_node,evidence_by_node,max_evidence=config["evidence_per_candidate"])
    cls_req={"request_id":f"anomaly_classification_{eid}","role":"32B-ComparativeClassification","prompt_name":"anomaly_driven_comparative_classification","prompt_version":"anomaly-comparative-v1","payload":{"taxonomy":taxonomy,"taxonomy_prompt":taxonomy_text,"root_hypotheses":roots,"topology":topology,"timeline":event_timeline,"evidence_ids":sorted(eids)},"temperature":0.2,"top_p":0.9,"json_schema":class_schema(taxonomy,eids),"validator":lambda v:validate_class(v,taxonomy,eids)}
    result=backend.generate_json_batch([cls_req],max_new_tokens=config["classification_max_new_tokens"],allow_partial=True)[0]; status="success" if result else "failed"; failure=None
    if not result: failure="comparative classification structured output failed"
    mapped={x["type_id"]:x for x in taxonomy}; class_top=[]
    if result:
        for rank,x in enumerate(result["top5"],1):
            t=mapped[x["taxonomy_id"]]; class_top.append({**x,"rank":rank,"fault_type":t["fault_type"],"fault_category":t["fault_category"]})
    node=top5[0]["node_id"]; region,dtype,device=split_node(node)
    return {"event_id":eid,"detection":detection,"rca":{"root_region":region,"root_device_type":dtype,"root_device":device,"top5":[{**x,**dict(zip(("region","device_type","device"),split_node(x["node_id"]))) } for x in top5]},"classification":{"major_class":class_top[0]["major_class"] if class_top else None,"minor_class":class_top[0]["minor_class"] if class_top else None,"taxonomy_id":class_top[0]["taxonomy_id"] if class_top else None,"top5":class_top},"prediction_mode":mode,"valid_stage2_rounds":len(rounds),"status":status,"failure_reason":failure,"rank_of_ranks":rank_data}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--input-root",type=Path,required=True); ap.add_argument("--stage1-source",type=Path,required=True); ap.add_argument("--event-manifest",type=Path,required=True); ap.add_argument("--taxonomy-file",type=Path,required=True); ap.add_argument("--taxonomy-prompt-file",type=Path,required=True); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--model-path",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    events=json.loads(args.event_manifest.read_text()); taxonomy=json.loads(args.taxonomy_file.read_text())["candidates"]; taxonomy=[{**x,"fault_category":x.get("fault_category",x.get("category")),"type_id":f"T{i:02d}"} for i,x in enumerate(taxonomy,1)]; cfg=json.loads(args.config.read_text()); stage1={x["incident_id"]:x["ranking"] for x in load_jsonl(args.stage1_source/"full_rankings.jsonl")}; short={x["incident_id"]:x["shortlist"] for x in load_jsonl(args.stage1_source/"shortlist.jsonl")}; taxtext=args.taxonomy_prompt_file.read_text(); args.output_dir.mkdir(parents=True,exist_ok=False)
    for d in ("stage2","classification","prediction/logs/raw_outputs","environment"): (args.output_dir/d).mkdir(parents=True,exist_ok=True)
    manifest={x["event_id"]:x for x in events}; backend=build_backend(args.model_path,cfg,args.output_dir/"prediction/logs/raw_outputs"); started=time.perf_counter(); backend.load(); predictions=[]
    for path in sorted(args.input_root.glob("*/incident_input.json")):
        eid=path.parent.name; incident=json.loads(path.read_text()); predictions.append(run_event(backend,incident,stage1[eid],short[eid],taxonomy,cfg,args.output_dir,taxtext,manifest[eid])); print(eid,flush=True)
    (args.output_dir/"predictions.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in predictions)+"\n"); (args.output_dir/"predictions.json").write_text(json.dumps(predictions,ensure_ascii=False,indent=2)+"\n")
    with (args.output_dir/"predictions.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["event_id","case_id","status","root_device","classification_top1"]); w.writeheader(); w.writerows({"event_id":x["event_id"],"case_id":x["detection"]["case_id"],"status":x["status"],"root_device":x["rca"]["root_device"],"classification_top1":x["classification"]["taxonomy_id"]} for x in predictions)
    (args.output_dir/"schema_validation_report.json").write_text(json.dumps({"valid":all(x["status"] in ("success","failed") for x in predictions),"event_count":len(predictions),"successful":sum(x["status"]=="success" for x in predictions)},indent=2)+"\n"); (args.output_dir/"model_manifest.json").write_text(json.dumps(backend.model_manifest(),indent=2)+"\n"); (args.output_dir/"run_manifest.json").write_text(json.dumps({"event_count":len(predictions),"elapsed_seconds":time.perf_counter()-started,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT_ROOT,text=True).strip(),"ground_truth_available_to_inference":False},indent=2)+"\n"); subprocess.run(["nvidia-smi"],stdout=(args.output_dir/"environment/gpu_after.txt").open("w"),check=True); print(json.dumps({"events":len(predictions),"success":sum(x["status"]=="success" for x in predictions)})); return 0

if __name__=="__main__": raise SystemExit(main())
