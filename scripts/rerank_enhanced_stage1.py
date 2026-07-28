"""Re-rank cached Stage 1 model analyses after one generic config correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.methods.enhanced_stage1 import rank_stage1


def load(path: Path) -> dict[str, dict]:
    return {
        item["incident_id"]: item
        for item in (
            json.loads(line) for line in path.read_text().splitlines() if line
        )
    }


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analyses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    analyses = load(args.analyses)
    config = json.loads(args.config.read_text())
    rankings = args.output_dir / "full_rankings.jsonl"
    shortlists = args.output_dir / "shortlist.jsonl"
    for path in sorted(args.input_root.glob("incident-*/incident_input.json")):
        incident = json.loads(path.read_text())
        incident_id = incident["incident_id"]
        ranking, shortlist = rank_stage1(
            incident["engineering_evidence"],
            analyses[incident_id]["device_analyses"],
            config["stage1"],
        )
        append(rankings, {"incident_id": incident_id, "ranking": ranking})
        append(
            shortlists,
            {
                "incident_id": incident_id,
                "shortlist": shortlist,
                "candidate_map": {
                    item["candidate_id"]: item["node_id"] for item in shortlist
                },
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
