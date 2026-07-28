"""Post-run Stage 1 diagnostics; this is the only Stage 1 tool reading truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--shortlists", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rankings = {x["incident_id"]: x for x in load_jsonl(args.rankings)}
    shortlists = {x["incident_id"]: x for x in load_jsonl(args.shortlists)}
    truths = {x["incident_id"]: x for x in load_jsonl(args.ground_truth)}
    per_case = []
    role_shortlist = {}
    for incident_id in sorted(truths):
        root = truths[incident_id]["root_node_id"]
        ranking = [x["node_id"] for x in rankings[incident_id]["ranking"]]
        shortlist = [x["node_id"] for x in shortlists[incident_id]["shortlist"]]
        rank = ranking.index(root) + 1
        for node in shortlist:
            role = "-".join(node.split("-")[2:])
            role_shortlist[role] = role_shortlist.get(role, 0) + 1
        per_case.append(
            {
                "incident_id": incident_id,
                "root_node_id": root,
                "true_root_rank": rank,
                "recall_at_5": rank <= 5,
                "recall_at_10": rank <= 10,
                "recall_at_15": rank <= 15,
                "shortlist_retained": root in shortlist,
                "shortlist_size": len(shortlist),
            }
        )
    count = len(per_case)
    metrics = {
        "evaluated_cases": count,
        "root_recall_at_5": sum(x["recall_at_5"] for x in per_case) / count,
        "root_recall_at_10": sum(x["recall_at_10"] for x in per_case) / count,
        "root_recall_at_15": sum(x["recall_at_15"] for x in per_case) / count,
        "true_root_mean_rank": statistics.mean(x["true_root_rank"] for x in per_case),
        "shortlist_mean_size": statistics.mean(x["shortlist_size"] for x in per_case),
        "role_shortlist_counts": role_shortlist,
        "unavailable_by_role_anomaly_contribution_count": 0,
        "per_case": per_case,
    }
    (args.output_dir / "stage1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Stage 1 development diagnostics",
        "",
        "Ground truth was read only after Stage 1 inference completed.",
        "",
        f"- recall@5: {metrics['root_recall_at_5']:.3f}",
        f"- recall@10: {metrics['root_recall_at_10']:.3f}",
        f"- recall@15: {metrics['root_recall_at_15']:.3f}",
        f"- mean rank: {metrics['true_root_mean_rank']:.3f}",
        "",
    ]
    lines += [
        f"- {x['incident_id']}: rank={x['true_root_rank']}, "
        f"shortlist={x['shortlist_retained']}"
        for x in per_case
    ]
    (args.output_dir / "stage1_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
