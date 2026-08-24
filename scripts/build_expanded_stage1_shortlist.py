#!/usr/bin/env python3
"""Render a larger candidate shortlist from an existing truth-free ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()
    if args.max_candidates < 5:
        raise SystemExit("max-candidates must be at least five")
    rankings = load_jsonl(args.source / "full_rankings.jsonl")
    if not rankings:
        raise SystemExit("empty Stage-1 ranking")
    args.output.mkdir(parents=True, exist_ok=False)
    shortlist_rows = []
    for row in rankings:
        ranking = row["ranking"]
        if len(ranking) < args.max_candidates:
            raise SystemExit(f"{row['incident_id']} has fewer than {args.max_candidates} candidates")
        shortlist = []
        for index, original in enumerate(ranking[: args.max_candidates], 1):
            item = dict(original)
            item["candidate_id"] = f"C{index:02d}"
            shortlist.append(item)
        shortlist_rows.append({
            "incident_id": row["incident_id"],
            "shortlist": shortlist,
            "candidate_map": {item["candidate_id"]: item["node_id"] for item in shortlist},
        })
    write_jsonl(args.output / "full_rankings.jsonl", rankings)
    write_jsonl(args.output / "shortlist.jsonl", shortlist_rows)
    manifest = {
        "source_stage1_dir": str(args.source),
        "source_full_rankings_sha256": hashlib.sha256((args.source / "full_rankings.jsonl").read_bytes()).hexdigest(),
        "candidate_budget": args.max_candidates,
        "ranking_recomputed": False,
        "ground_truth_read": False,
        "selection": "first-N frozen Stage-1 ranking entries",
    }
    (args.output / "expanded_shortlist_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"events": len(shortlist_rows), "candidate_budget": args.max_candidates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
