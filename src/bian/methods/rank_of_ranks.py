"""Rank-of-Ranks aggregation for repeated scoring runs."""

from __future__ import annotations

from bian.data.validators import ValidationError, validate_scores


def aggregate_rankings(
    runs: list[dict[str, float]], candidates: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[str, float]]:
    if not runs:
        raise ValidationError("at least one scoring run is required")
    rank_sums = {device: 0.0 for device in candidates}
    for scores in runs:
        validate_scores(scores, candidates)
        ranked = sorted(candidates, key=lambda device: (-scores[device], device))
        for rank, device in enumerate(ranked, start=1):
            rank_sums[device] += rank
    averages = {device: total / len(runs) for device, total in rank_sums.items()}
    ranking = tuple(sorted(candidates, key=lambda device: (averages[device], device)))
    return ranking, averages
