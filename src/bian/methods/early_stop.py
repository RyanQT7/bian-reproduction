"""Entropy-based early stopping."""

from __future__ import annotations

import math

from bian.data.validators import ValidationError, validate_scores


def score_entropy(scores: dict[str, float], candidates: tuple[str, ...], mode: str = "raw") -> float:
    validate_scores(scores, candidates)
    entropy = -sum(value * math.log(value) for value in scores.values() if value > 0)
    if mode == "raw":
        return entropy
    if mode == "normalized":
        return entropy / math.log(len(scores)) if len(scores) > 1 else 0.0
    raise ValidationError("entropy mode must be 'raw' or 'normalized'")


def should_early_stop(
    scores: dict[str, float],
    candidates: tuple[str, ...],
    threshold: float,
    mode: str = "raw",
) -> bool:
    if threshold < 0 or not math.isfinite(threshold):
        raise ValidationError("entropy threshold must be finite and non-negative")
    return score_entropy(scores, candidates, mode) < threshold
