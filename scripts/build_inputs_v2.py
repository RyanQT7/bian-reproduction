"""Create non-overwriting engineering-enhanced inputs from uniform v1 aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.enhanced_features import extract_device_evidence, validate_role_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    inputs = sorted(args.input_root.glob("incident-*/incident_input.json"))
    if len(inputs) != 10:
        raise ValueError(f"expected 10 inputs, found {len(inputs)}")
    all_devices = []
    audit = {"incident_count": 10, "incidents": [], "feature_rule_version": "engineering-evidence-v2"}
    for path in inputs:
        incident = json.loads(path.read_text(encoding="utf-8"))
        all_devices.extend(incident["devices"])
        incident["engineering_evidence"] = [
            extract_device_evidence(device) for device in incident["devices"]
        ]
        if incident["incident_id"] == "incident-0010":
            partial = [
                item["node_id"]
                for item in incident["devices"]
                if item["sources"]["routing_metrics"]["status"] == "partial"
            ]
            if len(partial) != 2:
                raise ValueError("incident-0010 must retain two partial BR routing sources")
        target = args.output_root / incident["incident_id"]
        target.mkdir()
        (target / "incident_input.json").write_text(
            json.dumps(incident, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        audit["incidents"].append(
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
    fields = validate_role_schema(all_devices)
    registry = args.output_root / "schema_registry"
    registry.mkdir()
    names = {
        "br": "br.json",
        "cr": "cr.json",
        "fw": "fw.json",
        "traffic-vm": "traffic_vm.json",
        "service": "service_vm.json",
    }
    for role, filename in names.items():
        (registry / filename).write_text(
            json.dumps(
                {"device_role": role, "fields": fields[role]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (args.output_root / "preprocessing_v2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated v2 inputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
