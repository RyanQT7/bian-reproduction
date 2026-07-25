"""End-to-end first-phase mock pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bian import VALIDATION_LABEL
from bian.data.schema import IncidentRecord
from bian.data.validators import validate_incident, validate_scores
from bian.evaluation.hot_device import hot_device_ranking
from bian.methods.candidate_filter import top_p_candidates
from bian.methods.early_stop import score_entropy, should_early_stop
from bian.methods.rank_of_ranks import aggregate_rankings
from bian.methods.timeline import build_timeline
from bian.methods.topology import extract_candidate_subgraph
from bian.models.mock_backend import MockModelBackend


def run_incident(
    incident: IncidentRecord,
    backend: MockModelBackend,
    *,
    rank_rounds: int,
    entropy_threshold: float,
    entropy_mode: str,
    stage2_device_top_p: float,
) -> dict[str, Any]:
    validate_incident(incident)
    if rank_rounds <= 0:
        raise ValueError("rank_rounds must be positive")
    stage1_scores = backend.score(incident, incident.candidate_devices, 0, False)
    validate_scores(stage1_scores, incident.candidate_devices)
    entropy = score_entropy(stage1_scores, incident.candidate_devices, entropy_mode)
    early_stopped = should_early_stop(
        stage1_scores, incident.candidate_devices, entropy_threshold, entropy_mode
    )
    retained = (
        incident.candidate_devices
        if early_stopped
        else top_p_candidates(stage1_scores, incident.candidate_devices, stage2_device_top_p)
    )
    subgraph = extract_candidate_subgraph(incident.topology, retained)
    timeline = build_timeline(incident.alerts)
    if early_stopped:
        runs = [stage1_scores]
    else:
        runs = [
            backend.score(incident, incident.candidate_devices, round_index + 1, True)
            for round_index in range(rank_rounds)
        ]
    ranking, average_ranks = aggregate_rankings(runs, incident.candidate_devices)
    return {
        "result_label": VALIDATION_LABEL,
        "incident_id": incident.incident_id,
        "synthetic": True,
        "candidates": list(incident.candidate_devices),
        "ground_truth": list(incident.ground_truth) if incident.ground_truth else None,
        "stage1_scores": stage1_scores,
        "entropy": entropy,
        "entropy_mode": entropy_mode,
        "early_stopped": early_stopped,
        "retained_candidates": list(retained),
        "subgraph": {
            "nodes": list(subgraph.nodes),
            "edges": [list(edge) for edge in subgraph.edges],
        },
        "timeline": [
            {**asdict(event), "timestamp": event.timestamp.isoformat()} for event in timeline
        ],
        "ranking": list(ranking),
        "average_ranks": average_ranks,
        "hot_device_ranking": list(
            hot_device_ranking(incident.alerts, incident.candidate_devices)
        ),
    }
