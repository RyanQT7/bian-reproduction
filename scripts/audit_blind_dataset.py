"""Audit new blind raw data using only explicit experiment/taxonomy inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.blind_audit import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--experiment-file", type=Path, required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--taxonomy-ids-file", type=Path, required=True)
    parser.add_argument("--region-map-config", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.experiment_file, args.taxonomy_file, args.taxonomy_ids_file):
        lowered = str(path).lower()
        if "/evaluation/" in lowered or "/review/" in lowered:
            raise ValueError("blind audit refuses evaluation/review paths")
    taxonomy = json.loads(args.taxonomy_file.read_text())
    ids = json.loads(args.taxonomy_ids_file.read_text())
    candidates = taxonomy.get("candidates", [])
    if (
        taxonomy.get("candidate_count") != 32
        or len(candidates) != 32
        or len(ids) != 32
        or [x["fault_type"] for x in candidates] != ids
    ):
        raise ValueError("taxonomy files do not define the same ordered 32 types")
    map_config = json.loads(args.region_map_config.read_text())
    region_mapping = {
        next(args.raw_root.glob(prefix + "_*")): region_id
        for prefix, region_id in map_config["region_directory_prefixes"].items()
    }
    if len(region_mapping) != 8:
        raise ValueError("region mapping must resolve exactly eight directories")
    audit = audit_dataset(
        raw_root=args.raw_root,
        experiment_path=args.experiment_file,
        region_mapping=region_mapping,
        config=json.loads(args.preprocessing_config.read_text()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "schema_coverage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    summary = [
        "# Blind dataset schema and coverage audit",
        "",
        f"- Incidents: {audit['incident_count']}",
        f"- Regions: {audit['region_count']}",
        f"- Coverage failures: {audit['coverage_failure_count']}",
        f"- Timezones: {', '.join(audit['timezone_values'])}",
        "- Raw NetFlow five-tuples: excluded uniformly without row scanning",
        "- Aggregated traffic_flow_metrics: retained",
        "",
        "## Schema consistency",
        "",
    ]
    for source, item in audit["schema_consistency"].items():
        summary.append(
            f"- {source}: {'consistent' if item['consistent_across_regions'] else 'DIFFERS'}"
        )
    (args.output_dir / "schema_coverage_audit.md").write_text(
        "\n".join(summary) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "success" if not audit["coverage_failures"] else "failed",
                "incidents": audit["incident_count"],
                "regions": audit["region_count"],
                "coverage_failures": audit["coverage_failure_count"],
                "output": str(args.output_dir),
            }
        )
    )
    return 0 if not audit["coverage_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
