"""Clinical-unit-aware evaluation metrics used by Phase 13 recipes.

The functions in this module are dependency-light and deterministic.  They
return structured values instead of bare floats so every report records the
clinical unit and the number of examples contributing to a metric.  This is a
metric interface, not a clinical-validation claim: external-site and human
review evidence remain explicit limitations in the report artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn


@dataclass(frozen=True)
class MetricValue:
    """One metric with its denominator and clinical unit."""

    name: str
    value: float | None
    unit: str
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must be non-empty")
        if not self.unit:
            raise ValueError("metric unit must be non-empty")
        if self.sample_count < 0:
            raise ValueError("metric sample_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "metadata": dict(self.metadata),
        }


def _tensor(value: torch.Tensor | Iterable[float] | Iterable[int], *, dtype: torch.dtype) -> torch.Tensor:
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(list(value))
    return result.detach().float() if dtype.is_floating_point else result.detach().to(dtype=dtype)


def _binary_labels_and_scores(
    labels: torch.Tensor | Iterable[int],
    scores: torch.Tensor | Iterable[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    y = _tensor(labels, dtype=torch.int64).reshape(-1)
    score = _tensor(scores, dtype=torch.float32).reshape(-1)
    if y.numel() != score.numel():
        raise ValueError("labels and scores must contain the same number of examples")
    if y.numel() == 0:
        return y, score
    if not bool(torch.isfinite(score).all()):
        raise ValueError("scores must be finite")
    if not bool(torch.logical_or(y == 0, y == 1).all()):
        raise ValueError("binary labels must be 0 or 1")
    return y, score


def _auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    # ADR-0013: delegated to torchmetrics. Tie ordering on equal scores now
    # follows torchmetrics (sample order); single-class sets stay None.
    positives = int(labels.sum())
    negatives = int(labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return None
    from torchmetrics.classification import BinaryAUROC

    return float(BinaryAUROC()(scores, labels))


def _auprc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    from torchmetrics.classification import BinaryAveragePrecision

    return float(BinaryAveragePrecision()(scores, labels))


def _operating_point(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict[str, float | None]:
    predicted = scores >= threshold
    tp = int((predicted & labels.bool()).sum())
    fp = int((predicted & ~labels.bool()).sum())
    tn = int((~predicted & ~labels.bool()).sum())
    fn = int((~predicted & labels.bool()).sum())
    return {
        "threshold": float(threshold),
        "sensitivity": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": None if tn + fp == 0 else tn / (tn + fp),
        "ppv": None if tp + fp == 0 else tp / (tp + fp),
        "npv": None if tn + fn == 0 else tn / (tn + fn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def _legacy_classification_metrics(
    labels: torch.Tensor | Iterable[int],
    scores: torch.Tensor | Iterable[float],
    *,
    group_ids: Iterable[str] | None = None,
    thresholds: Iterable[float] = (0.2, 0.5, 0.8),
    unit: str = "per_patient",
) -> dict[str, MetricValue]:
    """Compute discrimination, calibration, operating points, and subgroups.

    ``scores`` may be probabilities or logits.  Values outside ``[0, 1]`` are
    interpreted as logits and passed through sigmoid.  A single-class sample
    set reports an undefined (``None``) AUROC/AUPRC instead of fabricating a
    score.
    """

    y, raw_scores = _binary_labels_and_scores(labels, scores)
    probabilities = raw_scores if bool(((raw_scores >= 0) & (raw_scores <= 1)).all()) else torch.sigmoid(raw_scores)
    count = int(y.numel())
    result: dict[str, MetricValue] = {
        "auroc": MetricValue("auroc", _auroc(y, probabilities), unit, count),
        "auprc": MetricValue("auprc", _auprc(y, probabilities), unit, count),
        "brier": MetricValue(
            "brier", float(torch.mean((probabilities - y.float()) ** 2)) if count else None, unit, count
        ),
    }
    if count:
        bins = torch.linspace(0.0, 1.0, 11)
        ece = torch.zeros((), dtype=torch.float64)
        for low, high in zip(bins[:-1], bins[1:], strict=True):
            in_bin = (probabilities >= low) & ((probabilities < high) if high < 1 else (probabilities <= high))
            if bool(in_bin.any()):
                confidence = probabilities[in_bin].mean()
                accuracy = y[in_bin].float().mean()
                ece = ece + in_bin.float().mean().to(torch.float64) * (confidence - accuracy).abs()
        ece_value: float | None = float(ece)
    else:
        ece_value = None
    result["ece"] = MetricValue("ece", ece_value, unit, count, {"bins": 10})
    operating = [_operating_point(y, probabilities, threshold) for threshold in thresholds]
    result["operating_points"] = MetricValue("operating_points", None, unit, count, {"points": operating})
    if group_ids is not None:
        groups = list(group_ids)
        if len(groups) != count:
            raise ValueError("group_ids must align with labels")
        subgroup: dict[str, dict[str, Any]] = {}
        for group in sorted(set(groups)):
            mask = torch.tensor([entry == group for entry in groups], dtype=torch.bool)
            subgroup[group] = {
                metric: value.to_dict()
                for metric, value in classification_metrics(
                    y[mask], probabilities[mask], thresholds=thresholds, unit=unit
                ).items()
                if metric in {"auroc", "auprc", "brier", "ece"}
            }
        result["subgroups"] = MetricValue("subgroups", None, unit, count, {"groups": subgroup})
    return result


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        pool = nn.functional.max_pool2d
    elif mask.ndim == 5:
        pool = nn.functional.max_pool3d
    else:
        raise ValueError("segmentation masks must be [B,1,H,W] or [B,1,D,H,W]")
    foreground = mask.bool()
    kernel = 3
    padded = pool(foreground.float(), kernel, stride=1, padding=1) > 0
    eroded = -pool(-foreground.float(), kernel, stride=1, padding=1) < 0
    return foreground & (~eroded | ~padded)


def _surface_dice(predicted: torch.Tensor, target: torch.Tensor, tolerance: int = 1) -> float:
    pred_boundary = _boundary(predicted)
    target_boundary = _boundary(target)
    if not bool(pred_boundary.any()) and not bool(target_boundary.any()):
        return 1.0
    if not bool(pred_boundary.any()) or not bool(target_boundary.any()):
        return 0.0
    pool = nn.functional.max_pool2d if predicted.ndim == 4 else nn.functional.max_pool3d
    kernel = 2 * tolerance + 1
    target_near_pred = pool(target_boundary.float(), kernel, stride=1, padding=tolerance) > 0
    pred_near_target = pool(pred_boundary.float(), kernel, stride=1, padding=tolerance) > 0
    pred_score = (pred_boundary & target_near_pred).sum() / pred_boundary.sum().clamp_min(1)
    target_score = (target_boundary & pred_near_target).sum() / target_boundary.sum().clamp_min(1)
    return float((pred_score + target_score) / 2)


def _legacy_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    unit: str = "per_image",
) -> dict[str, MetricValue]:
    """Compute per-class Dice, surface Dice, sensitivity, and FP/image."""

    if logits.ndim not in (4, 5) or target.ndim != logits.ndim:
        raise ValueError("segmentation logits and target must have matching rank 4 or 5")
    if tuple(logits.shape) != tuple(target.shape):
        raise ValueError("segmentation logits and target must have identical shapes")
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= threshold
    truth = target > 0.5
    batch = int(logits.shape[0])
    classes = int(logits.shape[1])
    spatial = tuple(range(2, logits.ndim))
    result: dict[str, MetricValue] = {}
    dice_values: list[float] = []
    surface_values: list[float] = []
    sensitivity_values: list[float] = []
    fp_values: list[float] = []
    for cls in range(classes):
        pred_cls = predicted[:, cls : cls + 1]
        truth_cls = truth[:, cls : cls + 1]
        intersection = (pred_cls & truth_cls).sum(dim=spatial)
        denominator = pred_cls.sum(dim=spatial) + truth_cls.sum(dim=spatial)
        dice = ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        dice_values.append(float(dice))
        surface_values.append(
            sum(_surface_dice(pred_cls[i : i + 1], truth_cls[i : i + 1]) for i in range(batch)) / max(1, batch)
        )
        tp = (pred_cls & truth_cls).sum(dim=spatial)
        fn = ((~pred_cls) & truth_cls).sum(dim=spatial)
        sensitivity_values.append(float((tp / (tp + fn).clamp_min(1)).mean()))
        fp_values.append(float((pred_cls & ~truth_cls).sum(dim=spatial).float().mean()))
        result[f"dice/class_{cls}"] = MetricValue(
            f"dice/class_{cls}", dice_values[-1], unit, batch, {"class_index": cls}
        )
        result[f"surface_dice/class_{cls}"] = MetricValue(
            f"surface_dice/class_{cls}", surface_values[-1], unit, batch, {"tolerance_pixels": 1, "class_index": cls}
        )
        result[f"sensitivity/class_{cls}"] = MetricValue(
            f"sensitivity/class_{cls}", sensitivity_values[-1], unit, batch, {"class_index": cls}
        )
        result[f"false_positives_per_image/class_{cls}"] = MetricValue(
            f"false_positives_per_image/class_{cls}", fp_values[-1], unit, batch, {"class_index": cls}
        )
    return result


def serialize_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, Any]:
    """Convert metric values into a stable JSON-compatible mapping."""

    return {name: value.to_dict() for name, value in sorted(metrics.items())}


__all__ = [
    "MetricValue",
    "classification_metrics",
    "segmentation_metrics",
    "serialize_metrics",
]


# Phase 16 extends the compact Phase 13 contracts with complete clinical
# metrics while preserving the original import paths.  Lazy wrappers avoid a
# metrics/advanced import cycle when callers import the package root.
def _advanced_metric(name: str) -> Any:
    from medfm.evaluation import advanced

    return getattr(advanced, name)


def classification_metrics(*args: Any, **kwargs: Any) -> dict[str, MetricValue]:
    return cast(dict[str, MetricValue], _advanced_metric("classification_metrics")(*args, **kwargs))


def segmentation_metrics(*args: Any, **kwargs: Any) -> dict[str, MetricValue]:
    return cast(dict[str, MetricValue], _advanced_metric("segmentation_metrics")(*args, **kwargs))


def baseline_metrics(*args: Any, **kwargs: Any) -> dict[str, dict[str, MetricValue]]:
    return cast(dict[str, dict[str, MetricValue]], _advanced_metric("baseline_metrics")(*args, **kwargs))


def box_iou(*args: Any, **kwargs: Any) -> float:
    return cast(float, _advanced_metric("box_iou")(*args, **kwargs))


def cluster_bootstrap_ci(*args: Any, **kwargs: Any) -> Any:
    return _advanced_metric("cluster_bootstrap_ci")(*args, **kwargs)


def evaluate_holdouts(*args: Any, **kwargs: Any) -> dict[str, Mapping[str, MetricValue]]:
    return cast(dict[str, Mapping[str, MetricValue]], _advanced_metric("evaluate_holdouts")(*args, **kwargs))


def generation_metrics(*args: Any, **kwargs: Any) -> dict[str, MetricValue]:
    return cast(dict[str, MetricValue], _advanced_metric("generation_metrics")(*args, **kwargs))


def localization_metrics(*args: Any, **kwargs: Any) -> dict[str, MetricValue]:
    return cast(dict[str, MetricValue], _advanced_metric("localization_metrics")(*args, **kwargs))


def physical_localization_error(*args: Any, **kwargs: Any) -> float:
    return cast(float, _advanced_metric("physical_localization_error")(*args, **kwargs))


def retrieval_metrics(*args: Any, **kwargs: Any) -> dict[str, MetricValue]:
    return cast(dict[str, MetricValue], _advanced_metric("retrieval_metrics")(*args, **kwargs))


def sensitivity_at_fixed_specificity(*args: Any, **kwargs: Any) -> MetricValue:
    return cast(MetricValue, _advanced_metric("sensitivity_at_fixed_specificity")(*args, **kwargs))


def specificity_at_fixed_sensitivity(*args: Any, **kwargs: Any) -> MetricValue:
    return cast(MetricValue, _advanced_metric("specificity_at_fixed_sensitivity")(*args, **kwargs))


def validate_baseline_set(*args: Any, **kwargs: Any) -> tuple[str, ...]:
    return cast(tuple[str, ...], _advanced_metric("validate_baseline_set")(*args, **kwargs))


def visual_grounding_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _advanced_metric("visual_grounding_gate")(*args, **kwargs))


def __getattr__(name: str) -> Any:
    if name == "BootstrapCI":
        return _advanced_metric(name)
    raise AttributeError(name)


__all__ += [
    "baseline_metrics",
    "box_iou",
    "cluster_bootstrap_ci",
    "evaluate_holdouts",
    "generation_metrics",
    "localization_metrics",
    "physical_localization_error",
    "retrieval_metrics",
    "sensitivity_at_fixed_specificity",
    "specificity_at_fixed_sensitivity",
    "validate_baseline_set",
    "visual_grounding_gate",
]
