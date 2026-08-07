"""Uncertainty summaries for research evaluation (not clinical risk claims)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from medfm.evaluation.metrics import MetricValue


def _probabilities(scores: Any) -> torch.Tensor:
    values = torch.as_tensor(scores, dtype=torch.float64).detach().reshape(-1)
    if values.numel() and not bool(torch.isfinite(values).all()):
        raise ValueError("uncertainty scores must be finite")
    if values.numel() and not bool(((values >= 0) & (values <= 1)).all()):
        values = torch.sigmoid(values)
    return values


def uncertainty_metrics(
    scores: Any,
    *,
    labels: Iterable[int] | None = None,
    unit: str = "per_patient",
    coverage_levels: Iterable[float] = (0.5, 0.8, 0.9),
) -> dict[str, MetricValue]:
    """Report entropy, confidence, and optional selective-risk curves."""

    probabilities = _probabilities(scores)
    count = int(probabilities.numel())
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-12))
        + (1 - probabilities) * torch.log((1 - probabilities).clamp_min(1e-12))
    )
    result = {
        "mean_entropy": MetricValue("mean_entropy", None if not count else float(entropy.mean()), unit, count),
        "mean_confidence": MetricValue(
            "mean_confidence",
            None if not count else float(torch.maximum(probabilities, 1 - probabilities).mean()),
            unit,
            count,
        ),
    }
    if labels is None:
        return result
    truth = torch.as_tensor(list(labels), dtype=torch.int64).reshape(-1)
    if len(truth) != count or (truth.numel() and not bool(torch.logical_or(truth == 0, truth == 1).all())):
        raise ValueError("labels must align with binary scores")
    correctness = ((probabilities >= 0.5) == truth.bool()).to(torch.float64)
    uncertainty = torch.minimum(probabilities, 1 - probabilities)
    order = torch.argsort(uncertainty, descending=True, stable=True)
    for coverage in coverage_levels:
        coverage_value = float(coverage)
        if not 0.0 < coverage_value <= 1.0:
            raise ValueError("coverage levels must be in (0, 1]")
        keep = max(1, int(round(count * coverage_value))) if count else 0
        risk = None if not keep else float(1 - correctness[order[:keep]].mean())
        result[f"selective_risk@{coverage_value:g}"] = MetricValue(
            f"selective_risk@{coverage_value:g}",
            risk,
            unit,
            keep,
            {"coverage": coverage_value, "selection": "lowest uncertainty"},
        )
    return result


selective_risk = uncertainty_metrics

__all__ = ["selective_risk", "uncertainty_metrics"]
