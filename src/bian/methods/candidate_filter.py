"""Configurable Stage 2 candidate Top-p filtering."""

from __future__ import annotations

import math

from bian.data.validators import ValidationError, validate_scores


def softmax(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    values = {key: math.exp(value - maximum) for key, value in scores.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def top_p_candidates(scores: dict[str, float], candidates: tuple[str, ...], top_p: float) -> tuple[str, ...]:
    validate_scores(scores, candidates)
    if not 0 < top_p <= 1:
        raise ValidationError("top_p must be in (0, 1]")
    probabilities = softmax(scores)
    ranked = sorted(candidates, key=lambda device: (-probabilities[device], device))
    kept: list[str] = []
    cumulative = 0.0
    for device in ranked:
        kept.append(device)
        cumulative += probabilities[device]
        if cumulative >= top_p:
            break
    return tuple(kept)
