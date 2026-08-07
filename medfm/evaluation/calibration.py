"""Validation-only threshold selection and calibration utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torchmetrics.classification import BinaryCalibrationError

from medfm.evaluation.schemas import ClinicalUnit, EvaluationSchemaError, EvaluationSplit


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold fitted on validation data and safe to apply to test data."""

    threshold: float
    objective: str
    fit_split: EvaluationSplit = EvaluationSplit.VALIDATION
    target: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fit_split is not EvaluationSplit.VALIDATION:
            raise EvaluationSchemaError("thresholds may only be fitted on validation data")
        if not torch.isfinite(torch.tensor(float(self.threshold))):
            raise ValueError("threshold must be finite")
        if self.target is not None and not 0.0 <= float(self.target) <= 1.0:
            raise ValueError("threshold target must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "objective": self.objective,
            "fit_split": self.fit_split.value,
            "target": self.target,
            "metadata": dict(self.metadata),
        }


def _probabilities(scores: torch.Tensor | Iterable[float]) -> torch.Tensor:
    values = (
        scores.detach().float().reshape(-1)
        if isinstance(scores, torch.Tensor)
        else torch.as_tensor(list(scores), dtype=torch.float32)
    )
    if values.numel() and not bool(torch.isfinite(values).all()):
        raise ValueError("scores must be finite")
    if values.numel() and not bool(((values >= 0) & (values <= 1)).all()):
        values = torch.sigmoid(values)
    return values


def _labels(labels: torch.Tensor | Iterable[int]) -> torch.Tensor:
    values = (
        labels.detach().long().reshape(-1)
        if isinstance(labels, torch.Tensor)
        else torch.as_tensor(list(labels), dtype=torch.long)
    )
    if values.numel() and not bool(torch.logical_or(values == 0, values == 1).all()):
        raise ValueError("binary labels must be 0 or 1")
    return values


def _operating_point(
    labels: torch.Tensor, probabilities: torch.Tensor, threshold: float
) -> tuple[float | None, float | None, float | None, float | None]:
    predicted = probabilities >= float(threshold)
    positive = labels.bool()
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    tn = int((~predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    sensitivity = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    f1 = (
        None
        if precision is None or sensitivity is None or precision + sensitivity == 0
        else 2 * precision * sensitivity / (precision + sensitivity)
    )
    return sensitivity, specificity, precision, f1


def _candidate_thresholds(probabilities: torch.Tensor) -> list[float]:
    if probabilities.numel() == 0:
        return [0.5]
    values = sorted({float(value) for value in probabilities.tolist()})
    return sorted({0.0, 0.5, 1.0, *values})


def fit_threshold(
    labels: torch.Tensor | Iterable[int],
    scores: torch.Tensor | Iterable[float],
    *,
    objective: str = "f1",
    target_specificity: float | None = None,
    target_sensitivity: float | None = None,
    split: EvaluationSplit | str = EvaluationSplit.VALIDATION,
    clinical_unit: ClinicalUnit | str = ClinicalUnit.PATIENT,
) -> ThresholdSelection:
    """Fit a deterministic operating threshold using validation rows only."""

    parsed_split = EvaluationSplit.parse(split)
    if parsed_split is not EvaluationSplit.VALIDATION:
        raise EvaluationSchemaError("fit_threshold refuses non-validation data")
    unit = ClinicalUnit.parse(clinical_unit)
    if unit in {ClinicalUnit.TILE, ClinicalUnit.SAMPLE}:
        # Tile-level fitting is allowed for a tile task, but the caller must
        # explicitly declare that unit rather than silently inheriting it.
        pass
    y = _labels(labels)
    p = _probabilities(scores)
    if y.numel() != p.numel():
        raise ValueError("labels and scores must align")
    objective_name = objective.strip().lower()
    valid_target = target_specificity if target_specificity is not None else target_sensitivity
    if valid_target is not None and not 0.0 <= float(valid_target) <= 1.0:
        raise ValueError("operating target must be in [0, 1]")

    candidates = _candidate_thresholds(p)
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        sensitivity, specificity, precision, f1 = _operating_point(y, p, threshold)
        if target_specificity is not None and (specificity is None or specificity + 1e-12 < target_specificity):
            continue
        if target_sensitivity is not None and (sensitivity is None or sensitivity + 1e-12 < target_sensitivity):
            continue
        if objective_name in {"f1", "f1_score"}:
            value = -1.0 if f1 is None else f1
        elif objective_name in {"sensitivity", "recall"}:
            value = -1.0 if sensitivity is None else sensitivity
        elif objective_name == "specificity":
            value = -1.0 if specificity is None else specificity
        elif objective_name in {"youden", "youden_j"}:
            value = -1.0 if sensitivity is None or specificity is None else sensitivity + specificity - 1.0
        else:
            raise ValueError("objective must be f1, sensitivity, specificity, or youden")
        # Prefer a higher objective, then a lower threshold, then a stable
        # candidate order.  This avoids platform-dependent ties.
        ranking = (value, -float(threshold), -float(threshold))
        if best is None or ranking > best:
            best = ranking
            selected = float(threshold)
    if best is None:
        raise ValueError("no threshold satisfies the requested operating target")
    return ThresholdSelection(
        threshold=selected,
        objective=objective_name,
        target=valid_target,
        metadata={
            "clinical_unit": unit.value,
            "candidate_count": len(candidates),
            "target_specificity": target_specificity,
            "target_sensitivity": target_sensitivity,
        },
    )


def apply_threshold(
    scores: torch.Tensor | Iterable[float],
    selection: ThresholdSelection,
    *,
    split: EvaluationSplit | str,
) -> torch.Tensor:
    """Apply a validation-fitted threshold to any declared evaluation split."""

    parsed_split = EvaluationSplit.parse(split)
    if parsed_split is EvaluationSplit.TRAIN:
        raise EvaluationSchemaError("threshold application to training data is not an evaluation result")
    return _probabilities(scores) >= float(selection.threshold)


@dataclass(frozen=True)
class CalibrationModel:
    """Piecewise-constant calibration map fitted on validation data."""

    bin_edges: tuple[float, ...]
    bin_values: tuple[float, ...]
    fit_split: EvaluationSplit = EvaluationSplit.VALIDATION

    def __post_init__(self) -> None:
        if self.fit_split is not EvaluationSplit.VALIDATION:
            raise EvaluationSchemaError("calibration may only be fitted on validation data")
        if len(self.bin_edges) < 2 or len(self.bin_values) != len(self.bin_edges) - 1:
            raise ValueError("calibration bin geometry is invalid")

    def transform(self, scores: torch.Tensor | Iterable[float]) -> torch.Tensor:
        values = _probabilities(scores)
        indices = torch.bucketize(values, torch.tensor(self.bin_edges[1:-1]), right=False)
        calibrated = torch.tensor(self.bin_values, dtype=torch.float32)[indices]
        return calibrated

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_edges": list(self.bin_edges),
            "bin_values": list(self.bin_values),
            "fit_split": self.fit_split.value,
        }


def fit_calibration(
    labels: torch.Tensor | Iterable[int],
    scores: torch.Tensor | Iterable[float],
    *,
    bins: int = 10,
    split: EvaluationSplit | str = EvaluationSplit.VALIDATION,
) -> CalibrationModel:
    """Fit a histogram calibrator on validation data only."""

    if EvaluationSplit.parse(split) is not EvaluationSplit.VALIDATION:
        raise EvaluationSchemaError("fit_calibration refuses non-validation data")
    if bins < 1:
        raise ValueError("bins must be positive")
    y = _labels(labels)
    p = _probabilities(scores)
    if y.numel() != p.numel():
        raise ValueError("labels and scores must align")
    edges = torch.linspace(0.0, 1.0, bins + 1)
    values: list[float] = []
    for index in range(bins):
        mask = (p >= edges[index]) & ((p < edges[index + 1]) if index < bins - 1 else (p <= edges[index + 1]))
        values.append(
            float(y[mask].float().mean()) if bool(mask.any()) else float((edges[index] + edges[index + 1]) / 2)
        )
    return CalibrationModel(tuple(float(value) for value in edges), tuple(values))


def brier_score(labels: torch.Tensor | Iterable[int], scores: torch.Tensor | Iterable[float]) -> float | None:
    y = _labels(labels)
    p = _probabilities(scores)
    if y.numel() != p.numel():
        raise ValueError("labels and scores must align")
    return None if not y.numel() else float(torch.mean((p - y.float()) ** 2))


def expected_calibration_error(
    labels: torch.Tensor | Iterable[int],
    scores: torch.Tensor | Iterable[float],
    *,
    bins: int = 10,
) -> float | None:
    """Compute top-label ECE with equal-width bins and L1 aggregation."""
    if bins < 1:
        raise ValueError("bins must be positive")
    y = _labels(labels)
    p = _probabilities(scores)
    if y.numel() != p.numel():
        raise ValueError("labels and scores must align")
    if not y.numel():
        return None

    value = BinaryCalibrationError(n_bins=bins, norm="l1")(p, y)
    return None if not bool(torch.isfinite(value)) else float(value)


# Backward-compatible names used in reports and notebooks.
select_threshold = fit_threshold
fit_validation_threshold = fit_threshold
calibrate = fit_calibration

__all__ = [
    "CalibrationModel",
    "ThresholdSelection",
    "apply_threshold",
    "brier_score",
    "calibrate",
    "expected_calibration_error",
    "fit_calibration",
    "fit_threshold",
    "fit_validation_threshold",
    "select_threshold",
]
