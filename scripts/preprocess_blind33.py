"""Prepare 33 blind RCA inputs without accepting any truth/review path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.real_preprocessing import preprocess_dataset
from bian.data.reference_topology import (
    build_topology,
    discover_peer_regions,
    inventory,
)
from bian.data.enhanced_features import extract_device_evidence, validate_role_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--experiment-file", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--taxonomy-ids-file", type=Path, required=True)
    parser.add_argument("--region-map-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.experiment_file, args.taxonomy_file, args.taxonomy_ids_file):
        lowered = str(path).lower()
        if "/evaluation/" in lowered or "/review/" in lowered:
            raise ValueError("preprocessing refuses evaluation/review paths")
    taxonomy = json.loads(args.taxonomy_file.read_text())
    ids = json.loads(args.taxonomy_ids_file.read_text())
    if [x["fault_type"] for x in taxonomy["candidates"]] != ids or len(ids) != 32:
        raise ValueError("taxonomy and candidate IDs must match exactly")
    mapping_config = json.loads(args.region_map_config.read_text())
    region_mapping = {
        next(args.raw_root.glob(prefix + "_*")): region_id
        for prefix, region_id in mapping_config["region_directory_prefixes"].items()
    }
    links = discover_peer_regions(args.raw_root, region_mapping)
    config = json.loads(args.config.read_text())
    audit = preprocess_dataset(
        raw_root=args.raw_root,
        experiment_path=args.experiment_file,
        taxonomy_path=args.taxonomy_file,
        topology=build_topology(links),
        inventory=inventory(),
        region_mapping=region_mapping,
        output_root=args.output_root,
        config=config,
        expected_incident_count=33,
    )
    all_devices = []
    enhanced_incidents = []
    for path in sorted(args.output_root.glob("incident-*/incident_input.json")):
        incident = json.loads(path.read_text())
        all_devices.extend(incident["devices"])
        incident["engineering_evidence"] = [
            extract_device_evidence(device) for device in incident["devices"]
        ]
        path.write_text(
            json.dumps(incident, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        enhanced_incidents.append(
            {
                "incident_id": incident["incident_id"],
                "candidate_count": len(incident["candidate_node_ids"]),
                "evidence_count": sum(
                    len(item["evidence"]) for item in incident["engineering_evidence"]
                ),
                "unavailable_contribution_sum": sum(
                    item["feature_summary"][
                        "unavailable_by_role_anomaly_contribution"
                    ]
                    for item in incident["engineering_evidence"]
                ),
            }
        )
    validate_role_schema(all_devices)
    manifest = {
        "status": audit["status"],
        "incident_count": 33,
        "taxonomy_count": 32,
        "experiment_sha256": hashlib.sha256(
            args.experiment_file.read_bytes()
        ).hexdigest(),
        "taxonomy_sha256": hashlib.sha256(args.taxonomy_file.read_bytes()).hexdigest(),
        "taxonomy_ids_sha256": hashlib.sha256(
            args.taxonomy_ids_file.read_bytes()
        ).hexdigest(),
        "region_mapping_config_sha256": hashlib.sha256(
            args.region_map_config.read_bytes()
        ).hexdigest(),
        "inter_region_links": sorted([list(x) for x in links]),
        "ground_truth_available_to_preprocessing": False,
        "evaluation_or_review_path_received": False,
        "feature_rule_version": "engineering-evidence-v2",
        "incidents": enhanced_incidents,
    }
    (args.output_root / "blind_preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"status": audit["status"], "cases": 33}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
