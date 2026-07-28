"""Leakage-safe orchestration helpers for the Dual-7B real-data run."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bian.data.validators import ValidationError, normalize_scores
from bian.methods.early_stop import score_entropy


PHASES = ("pre_fault", "fault", "post_fault")


def batched(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValidationError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def compact_device(device: dict[str, Any], max_signals: int) -> dict[str, Any]:
    """Apply one deterministic metric-selection rule to every device."""
    signals: list[dict[str, Any]] = []
    source_states = {}
    for source_name, source in sorted(device["sources"].items()):
        source_states[source_name] = {
            "status": source["status"],
            "missing_phases": source["missing_phases"],
        }
        phases = source.get("phases", {})
        pre_metrics = phases.get("pre_fault", {}).get("metrics", {})
        fault_metrics = phases.get("fault", {}).get("metrics", {})
        post_metrics = phases.get("post_fault", {}).get("metrics", {})
        for metric in sorted(set(pre_metrics) | set(fault_metrics) | set(post_metrics)):
            pre = pre_metrics.get(metric, {}).get("mean")
            fault = fault_metrics.get(metric, {}).get("mean")
            post = post_metrics.get(metric, {}).get("mean")
            if not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in (pre, fault)):
                continue
            scale = max(abs(float(pre)), 1e-9)
            change = abs(float(fault) - float(pre)) / scale
            signals.append(
                {
                    "source": source_name,
                    "metric": metric,
                    "pre_mean": round(float(pre), 6),
                    "fault_mean": round(float(fault), 6),
                    "post_mean": (
                        round(float(post), 6)
                        if isinstance(post, (int, float)) and math.isfinite(post)
                        else None
                    ),
                    "relative_change": round(min(change, 1_000_000.0), 6),
                }
            )
    signals.sort(
        key=lambda item: (-item["relative_change"], item["source"], item["metric"])
    )
    return {
        "node_id": device["node_id"],
        "device_family": device["device_family"],
        "source_states": source_states,
        "top_changed_signals": signals[:max_signals],
        "signal_selection": {
            "rule": "largest absolute fault-vs-pre relative mean change",
            "max_signals": max_signals,
        },
    }


def cumulative_top_p(
    scores: dict[str, float],
    candidates: tuple[str, ...],
    top_p: float,
    max_candidates: int,
) -> tuple[str, ...]:
    if not 0 < top_p <= 1:
        raise ValidationError("top_p must be in (0, 1]")
    if max_candidates < 5:
        raise ValidationError("max_candidates must be at least five")
    normalized = normalize_scores(scores)
    ordered = sorted(candidates, key=lambda node: (-normalized[node], node))
    kept: list[str] = []
    cumulative = 0.0
    for node in ordered:
        kept.append(node)
        cumulative += normalized[node]
        if cumulative >= top_p and len(kept) >= 5:
            break
    return tuple(kept[:max_candidates])


def topology_subgraph(
    topology: dict[str, Any], candidates: tuple[str, ...]
) -> dict[str, Any]:
    """Connect selected candidates through deterministic shortest paths."""
    adjacency: dict[str, set[str]] = {
        item["node_id"]: set() for item in topology["nodes"]
    }
    edge_by_pair = {}
    for edge in topology["edges"]:
        left, right = edge["source"], edge["target"]
        adjacency[left].add(right)
        adjacency[right].add(left)
        edge_by_pair[frozenset((left, right))] = edge
    selected_nodes = set(candidates)
    selected_edges: dict[frozenset[str], dict[str, Any]] = {}
    anchor = candidates[0]
    for target in candidates[1:]:
        queue = deque([(anchor, (anchor,))])
        seen = {anchor}
        path = None
        while queue:
            node, current = queue.popleft()
            if node == target:
                path = current
                break
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, current + (neighbor,)))
        if path is None:
            raise ValidationError(f"no topology path between {anchor} and {target}")
        selected_nodes.update(path)
        for left, right in zip(path, path[1:]):
            key = frozenset((left, right))
            selected_edges[key] = edge_by_pair[key]
    node_map = {item["node_id"]: item for item in topology["nodes"]}
    return {
        "directed": False,
        "nodes": [node_map[node] for node in sorted(selected_nodes)],
        "edges": [
            selected_edges[key]
            for key in sorted(selected_edges, key=lambda pair: sorted(pair))
        ],
    }


def build_metric_timeline(
    compact_devices: list[dict[str, Any]],
    fault_start: str,
    fault_end: str,
    max_events: int,
) -> list[dict[str, Any]]:
    events = []
    for device in compact_devices:
        for signal in device["top_changed_signals"][:2]:
            if signal["relative_change"] <= 0:
                continue
            events.append(
                {
                    "timestamp_utc": fault_start,
                    "node_id": device["node_id"],
                    "source": signal["source"],
                    "metric": signal["metric"],
                    "event": "fault_phase_change",
                    "relative_change": signal["relative_change"],
                }
            )
            if signal["post_mean"] is not None:
                events.append(
                    {
                        "timestamp_utc": fault_end,
                        "node_id": device["node_id"],
                        "source": signal["source"],
                        "metric": signal["metric"],
                        "event": "post_fault_observation",
                        "relative_change": signal["relative_change"],
                    }
                )
    events.sort(
        key=lambda item: (
            item["timestamp_utc"],
            -item["relative_change"],
            item["node_id"],
            item["source"],
            item["metric"],
        )
    )
    # Preserve chronological ordering after globally limiting strongest events.
    strongest = sorted(
        events,
        key=lambda item: (
            -item["relative_change"],
            item["timestamp_utc"],
            item["node_id"],
        ),
    )[:max_events]
    return sorted(
        strongest,
        key=lambda item: (
            item["timestamp_utc"], item["node_id"], item["source"], item["metric"]
        ),
    )


def merge_stage1(parts: list[dict[str, Any]]) -> dict[str, Any]:
    raw = {}
    reasons = {}
    summaries = []
    for part in parts:
        raw.update(part["scores"])
        reasons.update(part["reasons"])
        summaries.append(part["reason_summary"])
    return {
        "scores": normalize_scores(raw),
        "reasons": reasons,
        "reason_summary": " | ".join(summaries),
    }


def aggregate_stage2_rounds(
    rounds: list[dict[str, Any]],
    candidates: tuple[str, ...],
    taxonomy: tuple[dict[str, str], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[str]]]:
    rank_sums = {node: 0.0 for node in candidates}
    reason_by_node: dict[str, str] = {}
    raw_rankings = []
    for result in rounds:
        ranked = [item["node_id"] for item in result["root_causes"]]
        ranked += [node for node in candidates if node not in ranked]
        raw_rankings.append(ranked)
        for rank, node in enumerate(ranked, start=1):
            rank_sums[node] += rank
        for item in result["root_causes"]:
            reason_by_node.setdefault(item["node_id"], item["reason"])
    final_nodes = sorted(candidates, key=lambda node: (rank_sums[node], node))[:5]
    inverse = {node: 1.0 / (rank_sums[node] / len(rounds)) for node in final_nodes}
    normalized = normalize_scores(inverse)
    top5 = [
        {
            "rank": rank,
            "node_id": node,
            "failure_score": normalized[node],
            "reason_summary": reason_by_node.get(node, "Rank of Ranks consensus"),
        }
        for rank, node in enumerate(final_nodes, start=1)
    ]
    taxonomy_map = {item["fault_type"]: item["fault_category"] for item in taxonomy}
    fault_rank_sums = {fault: 0.0 for fault in taxonomy_map}
    fault_reasons: dict[str, str] = {}
    fault_confidences: dict[str, list[float]] = {fault: [] for fault in taxonomy_map}
    for result in rounds:
        ranked_faults = [item["fault_type"] for item in result["fault_types"]]
        ranked_faults += [fault for fault in taxonomy_map if fault not in ranked_faults]
        for rank, fault in enumerate(ranked_faults, start=1):
            fault_rank_sums[fault] += rank
        for item in result["fault_types"]:
            fault_reasons.setdefault(item["fault_type"], item["reason"])
            fault_confidences[item["fault_type"]].append(item["confidence"])
    final_faults = sorted(taxonomy_map, key=lambda fault: (fault_rank_sums[fault], fault))[:3]
    confidence_raw = {
        fault: (
            sum(fault_confidences[fault]) / len(fault_confidences[fault])
            if fault_confidences[fault]
            else 1.0 / (fault_rank_sums[fault] / len(rounds))
        )
        for fault in final_faults
    }
    confidences = normalize_scores(confidence_raw)
    top3 = [
        {
            "rank": rank,
            "fault_type": fault,
            "fault_category": taxonomy_map[fault],
            "confidence": confidences[fault],
            "reason_summary": fault_reasons.get(fault, "Rank of Ranks consensus"),
        }
        for rank, fault in enumerate(final_faults, start=1)
    ]
    return top5, top3, raw_rankings


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
