"""Re-test real one-vs-rest classification from a cached successful Top5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bian.models.dual_7b_backend import GenerationConfig
from bian.models.sharded_backend import Sharded32BBackend
from run_mixed_7b32b import root_payload, run_classification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident-input", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--cached-prediction", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    incident = json.loads(args.incident_input.read_text())
    prediction = json.loads(args.cached_prediction.read_text())
    shortlist_records = [
        json.loads(line) for line in args.shortlist.read_text().splitlines() if line
    ]
    shortlist = next(
        item["shortlist"]
        for item in shortlist_records
        if item["incident_id"] == incident["incident_id"]
    )
    evidence_by_node = {
        item["node_id"]: item for item in incident["engineering_evidence"]
    }
    roots, _aliases, evidence_ids = root_payload(
        prediction["top5_root_causes"],
        {item["node_id"]: item for item in shortlist},
        evidence_by_node,
    )
    taxonomy = json.loads(
        (args.bundle_root / "schemas/fault_taxonomy.json").read_text()
    )
    backend = Sharded32BBackend(
        args.model_path,
        config=GenerationConfig(
            max_input_tokens=config["stage2_max_input_tokens"],
            max_new_tokens=config["classification_max_new_tokens"],
            temperature=config["stage2_temperature"],
            top_p=config["stage2_top_p"],
            retries=config["structured_retries"],
            seed=42,
        ),
        prompt_dir=PROJECT_ROOT / "src/bian/prompts",
        device_map={"": 0},
        max_memory={0: "44GiB"},
        precision=config["precision"],
    )
    top3, _ = run_classification(
        backend,
        incident["incident_id"],
        taxonomy,
        roots,
        evidence_ids,
        args.output_dir / "type_scores.jsonl",
    )
    (args.output_dir / "top3.json").write_text(
        json.dumps(top3, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Real classification smoke passed: {[x['fault_type'] for x in top3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
