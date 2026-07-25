"""Top-k incident localization metrics."""

from __future__ import annotations

from bian.data.validators import ValidationError


def top_k_hit(ranking: tuple[str, ...], ground_truth: tuple[str, ...] | None, k: int) -> bool:
    if ground_truth is None:
        raise ValidationError("ground truth is required to compute accuracy")
    if not ground_truth:
        raise ValidationError("ground truth must not be empty")
    if k <= 0:
        raise ValidationError("k must be positive")
    return bool(set(ranking[:k]) & set(ground_truth))


def top_k_accuracy(
    predictions: list[tuple[str, ...]],
    ground_truths: list[tuple[str, ...] | None],
    k: int,
) -> float:
    if not predictions or len(predictions) != len(ground_truths):
        raise ValidationError("predictions and ground truths must have equal non-zero length")
    return sum(top_k_hit(ranking, truth, k) for ranking, truth in zip(predictions, ground_truths)) / len(predictions)
