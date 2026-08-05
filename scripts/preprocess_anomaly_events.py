"""Build BiAn inputs from Trusted detection events, never GT intervals."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.enhanced_features import extract_device_evidence, validate_role_schema
from bian.data.real_preprocessing import preprocess_dataset
from bian.data.reference_topology import build_topology, discover_peer_regions, inventory


def utc(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trusted-events", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--region-map-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.trusted_events.read_text().splitlines() if line.strip()]
    if not events or len({event["detection_event_id"] for event in events}) != len(events):
        raise ValueError("Trusted detection events must have unique event IDs")
    if any("case_id" not in event for event in events):
        raise ValueError("Trusted export must retain case mapping for post-run scoring")
    mapping_config = json.loads(args.region_map_config.read_text())
    region_mapping = {
        next(args.raw_root.glob(prefix + "_*")): region_id
        for prefix, region_id in mapping_config["region_directory_prefixes"].items()
    }
    synthetic = []
    event_manifest = []
    for event in events:
        event_id = event["detection_event_id"]
        start = utc(event["detection_start_utc"])
        end = utc(event["detection_end_utc"])
        synthetic.append({
            "incident_id": event_id,
            "dataset_id": "trusted_detection_event_input",
            "timezone": "UTC",
            "start_time": start.isoformat().replace("+00:00", "Z"),
            "end_time": end.isoformat().replace("+00:00", "Z"),
            "analysis_window_start": (start - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "analysis_window_end": (end + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "candidate_scope": "all_72_candidates",
        })
        event_manifest.append({
            "event_id": event_id,
            "case_id": event["case_id"],
            "detection_start": event["detection_start_utc"],
            "detection_end": event["detection_end_utc"],
            "rule_id": event.get("rule_id"),
            "tier": event.get("tier"),
        })
    args.output_root.mkdir(parents=True, exist_ok=False)
    experiment = args.output_root / "trusted_events_experiment.jsonl"
    experiment.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in synthetic))
    config = json.loads(args.config.read_text())
    links = discover_peer_regions(args.raw_root, region_mapping)
    audit = preprocess_dataset(
        raw_root=args.raw_root,
        experiment_path=experiment,
        taxonomy_path=args.taxonomy_file,
        topology=build_topology(links),
        inventory=inventory(),
        region_mapping=region_mapping,
        output_root=args.output_root / "inputs",
        config=config,
        expected_incident_count=len(events),
    )
    all_devices = []
    for path in sorted((args.output_root / "inputs").glob("*/incident_input.json")):
        incident = json.loads(path.read_text())
        all_devices.extend(incident["devices"])
        incident["engineering_evidence"] = [extract_device_evidence(device) for device in incident["devices"]]
        path.write_text(json.dumps(incident, ensure_ascii=False, separators=(",", ":")) + "\n")
    validate_role_schema(all_devices)
    (args.output_root / "event_manifest.json").write_text(json.dumps(event_manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output_root / "preprocessing_manifest.json").write_text(json.dumps({
        "status": audit["status"], "event_count": len(events), "event_ids": [x["detection_event_id"] for x in events],
        "trusted_events_sha256": hashlib.sha256(args.trusted_events.read_bytes()).hexdigest(),
        "ground_truth_intervals_used": False, "case_id_in_model_input": False,
        "evaluation_or_review_path_received": False, "inter_region_links": sorted([list(x) for x in links]),
    }, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "events": len(events), "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
