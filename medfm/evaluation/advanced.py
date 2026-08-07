"""Advanced Phase 16 metrics.

All functions are deterministic and return :class:`MetricValue` objects.  The
module intentionally avoids optional NLP packages: structured clinical
finding evaluation is exact and auditable, while an approved RadGraph-style
adapter can be supplied explicitly by a caller.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from medfm.evaluation.metrics import MetricValue

# ---------------------------------------------------------------------------
# Shared helpers


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    else:
        result = torch.as_tensor(value)
    return result.to(dtype=dtype) if dtype is not None else result


def _probabilities(scores: Any) -> torch.Tensor:
    result = _tensor(scores, dtype=torch.float64)
    if result.numel() and not bool(torch.isfinite(result).all()):
        raise ValueError("scores must be finite")
    if result.numel() and not bool(((result >= 0) & (result <= 1)).all()):
        result = torch.sigmoid(result)
    return result


def _binary_inputs(labels: Any, scores: Any) -> tuple[torch.Tensor, torch.Tensor]:
    y = _tensor(labels, dtype=torch.int64).reshape(-1)
    p = _probabilities(scores).reshape(-1)
    if y.numel() != p.numel():
        raise ValueError("labels and scores must contain the same number of examples")
    if y.numel() and not bool(torch.logical_or(y == 0, y == 1).all()):
        raise ValueError("binary labels must be 0 or 1")
    return y, p


def _metric(
    name: str, value: float | None, unit: str, count: int, metadata: Mapping[str, Any] | None = None
) -> MetricValue:
    return MetricValue(name, None if value is None else float(value), unit, int(count), dict(metadata or {}))


def _required_float(value: float | None) -> float:
    if value is None:
        raise ValueError("expected a defined metric value")
    return float(value)


def _rank_auc(y: torch.Tensor, p: torch.Tensor) -> float | None:
    positives = int(y.sum())
    negatives = int(y.numel() - positives)
    if positives == 0 or negatives == 0:
        return None
    # Average ranks for ties; this is platform-stable and agrees with the
    # Mann-Whitney definition of AUROC.
    order = torch.argsort(p, stable=True)
    sorted_scores = p[order]
    ranks = torch.arange(1, len(sorted_scores) + 1, dtype=torch.float64)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = ranks[i:j].mean()
        i = j
    positive_rank_sum = ranks[y[order] == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(y: torch.Tensor, p: torch.Tensor) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = torch.argsort(p, descending=True, stable=True)
    ordered = y[order].to(torch.float64)
    cumulative = torch.cumsum(ordered, dim=0)
    positions = torch.arange(1, len(ordered) + 1, dtype=torch.float64)
    return float((cumulative / positions * ordered).sum() / positives)


def _operating(y: torch.Tensor, p: torch.Tensor, threshold: float) -> dict[str, Any]:
    predicted = p >= float(threshold)
    positive = y.bool()
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    tn = int((~predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    sensitivity = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = sensitivity
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    balanced = None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _fixed_operating(
    y: torch.Tensor,
    p: torch.Tensor,
    *,
    target_specificity: float | None = None,
    target_sensitivity: float | None = None,
    threshold: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if threshold is not None:
        point = _operating(y, p, threshold)
        return point["sensitivity"], point["specificity"], float(threshold)
    candidates = sorted({0.0, 0.5, 1.0, *(float(v) for v in p.tolist())})
    selected: tuple[float, float] | None = None
    for candidate in candidates:
        point = _operating(y, p, candidate)
        sensitivity = point["sensitivity"]
        specificity = point["specificity"]
        if target_specificity is not None and (specificity is None or specificity < target_specificity):
            continue
        if target_sensitivity is not None and (sensitivity is None or sensitivity < target_sensitivity):
            continue
        score = sensitivity if target_specificity is not None else specificity
        if score is None:
            continue
        rank = (float(score), -candidate)
        if selected is None or rank > (selected[1], -selected[0]):
            selected = (candidate, float(score))
    if selected is None:
        return None, None, None
    point = _operating(y, p, selected[0])
    return point["sensitivity"], point["specificity"], selected[0]


# ---------------------------------------------------------------------------
# Classification, calibration, and clustered confidence intervals


def _binary_classification_metrics(
    labels: Any,
    scores: Any,
    *,
    unit: str,
    thresholds: Iterable[float],
    threshold: float,
    target_specificity: float | None,
    target_sensitivity: float | None,
) -> dict[str, MetricValue]:
    y, p = _binary_inputs(labels, scores)
    count = int(y.numel())
    point = _operating(y, p, threshold) if count else _operating(y, p, threshold)
    result: dict[str, MetricValue] = {
        "auroc": _metric("auroc", _rank_auc(y, p), unit, count),
        "auprc": _metric("auprc", _average_precision(y, p), unit, count),
        "brier": _metric("brier", None if not count else float(torch.mean((p - y.float()) ** 2)), unit, count),
        "ece": _metric("ece", _ece(y, p, 10), unit, count, {"bins": 10}),
    }
    for key in ("sensitivity", "specificity", "precision", "recall", "f1", "balanced_accuracy"):
        result[key] = _metric(key, point[key], unit, count, {"threshold": float(threshold)})
    result["confusion_matrix"] = _metric(
        "confusion_matrix", None, unit, count, {"matrix": point["confusion_matrix"], "threshold": float(threshold)}
    )
    result["operating_points"] = _metric(
        "operating_points", None, unit, count, {"points": [_operating(y, p, value) for value in thresholds]}
    )
    fixed_sensitivity, fixed_specificity, selected = _fixed_operating(
        y,
        p,
        target_specificity=target_specificity,
        target_sensitivity=target_sensitivity,
    )
    result["sensitivity_at_specificity"] = _metric(
        "sensitivity_at_specificity",
        fixed_sensitivity if target_specificity is not None else None,
        unit,
        count,
        {"target_specificity": target_specificity, "threshold": selected},
    )
    result["specificity_at_sensitivity"] = _metric(
        "specificity_at_sensitivity",
        fixed_specificity if target_sensitivity is not None else None,
        unit,
        count,
        {"target_sensitivity": target_sensitivity, "threshold": selected},
    )
    return result


def _ece(y: torch.Tensor, p: torch.Tensor, bins: int) -> float | None:
    if not y.numel():
        return None
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=p.dtype)
    value = torch.tensor(0.0, dtype=torch.float64)
    for index in range(bins):
        mask = (p >= edges[index]) & ((p < edges[index + 1]) if index < bins - 1 else (p <= edges[index + 1]))
        if bool(mask.any()):
            value += mask.float().mean().to(torch.float64) * (p[mask].mean() - y[mask].float().mean()).abs()
    return float(value)


def classification_metrics(
    labels: Any,
    scores: Any,
    *,
    group_ids: Iterable[str] | None = None,
    thresholds: Iterable[float] = (0.5,),
    threshold: float = 0.5,
    unit: str = "per_patient",
    num_classes: int | None = None,
    target_specificity: float | None = None,
    target_sensitivity: float | None = None,
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, MetricValue]:
    """Compute binary or multiclass metrics at an explicitly named unit."""

    scores_tensor = _tensor(scores)
    labels_tensor = _tensor(labels)
    if scores_tensor.ndim == 2 and scores_tensor.shape[1] > 1:
        classes = int(num_classes or scores_tensor.shape[1])
        if classes != scores_tensor.shape[1]:
            raise ValueError("num_classes must match score columns")
        if labels_tensor.ndim == 1:
            if labels_tensor.numel() and (int(labels_tensor.min()) < 0 or int(labels_tensor.max()) >= classes):
                raise ValueError("multiclass labels are out of range")
            one_hot = torch.nn.functional.one_hot(labels_tensor.long(), classes).to(torch.int64)
        elif tuple(labels_tensor.shape) == tuple(scores_tensor.shape):
            one_hot = labels_tensor.to(torch.int64)
        else:
            raise ValueError("multiclass labels must be class indices or one-hot scores")
        per_class: list[dict[str, MetricValue]] = []
        result: dict[str, MetricValue] = {}
        for index in range(classes):
            current = _binary_classification_metrics(
                one_hot[:, index],
                scores_tensor[:, index],
                unit=unit,
                thresholds=thresholds,
                threshold=threshold,
                target_specificity=target_specificity,
                target_sensitivity=target_sensitivity,
            )
            per_class.append(current)
            for name, value in current.items():
                result[f"{name}/class_{index}"] = MetricValue(
                    f"{name}/class_{index}",
                    value.value,
                    value.unit,
                    value.sample_count,
                    {**value.metadata, "class_index": index},
                )
        for name in (
            "auroc",
            "auprc",
            "brier",
            "ece",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ):
            values = [_required_float(entry[name].value) for entry in per_class if entry[name].value is not None]
            result[f"{name}/macro"] = _metric(
                f"{name}/macro",
                None if not values else sum(values) / len(values),
                unit,
                int(labels_tensor.shape[0]),
                {"class_count": classes},
            )
        micro = _binary_classification_metrics(
            one_hot.reshape(-1),
            scores_tensor.reshape(-1),
            unit=unit,
            thresholds=thresholds,
            threshold=threshold,
            target_specificity=target_specificity,
            target_sensitivity=target_sensitivity,
        )
        for name, value in micro.items():
            result[f"{name}/micro"] = MetricValue(
                f"{name}/micro", value.value, value.unit, value.sample_count, {**value.metadata, "class_count": classes}
            )
        if group_ids is not None:
            groups = list(group_ids)
            if len(groups) != labels_tensor.shape[0]:
                raise ValueError("group_ids must align with multiclass rows")
            result["subgroups"] = _subgroup_metrics(labels_tensor, scores_tensor, groups, unit, num_classes=classes)
        return result

    result = _binary_classification_metrics(
        labels,
        scores,
        unit=unit,
        thresholds=thresholds,
        threshold=threshold,
        target_specificity=target_specificity,
        target_sensitivity=target_sensitivity,
    )
    if group_ids is not None:
        groups = list(group_ids)
        if len(groups) != len(labels_tensor.reshape(-1)):
            raise ValueError("group_ids must align with labels")
        result["subgroups"] = _subgroup_metrics(
            labels_tensor.reshape(-1), _probabilities(scores).reshape(-1), groups, unit
        )
    if bootstrap is not None:
        options = dict(bootstrap)
        cluster_ids = options.pop("cluster_ids", None)
        for name in ("auroc", "auprc", "brier", "ece"):
            result[name] = _with_bootstrap(result[name], labels, scores, name, unit, cluster_ids, options)
    return result


def _subgroup_metrics(
    labels: torch.Tensor, scores: torch.Tensor, groups: list[str], unit: str, *, num_classes: int | None = None
) -> MetricValue:
    payload: dict[str, Any] = {}
    for group in sorted(set(str(value) for value in groups)):
        mask = torch.tensor([str(value) == group for value in groups], dtype=torch.bool)
        current = classification_metrics(labels[mask], scores[mask], unit=unit, num_classes=num_classes)
        payload[group] = {
            name: value.to_dict() for name, value in current.items() if name in {"auroc", "auprc", "brier", "ece", "f1"}
        }
    return _metric("subgroups", None, unit, len(groups), {"groups": payload})


def sensitivity_at_fixed_specificity(
    labels: Any,
    scores: Any,
    *,
    specificity: float = 0.9,
    unit: str = "per_patient",
    threshold: float | None = None,
) -> MetricValue:
    """Report sensitivity at a declared specificity target."""

    if not 0.0 <= specificity <= 1.0:
        raise ValueError("specificity target must be in [0, 1]")
    y, p = _binary_inputs(labels, scores)
    sensitivity, achieved, selected = _fixed_operating(
        y,
        p,
        target_specificity=specificity,
        threshold=threshold,
    )
    return _metric(
        "sensitivity_at_specificity",
        sensitivity,
        unit,
        len(y),
        {
            "target_specificity": specificity,
            "achieved_specificity": achieved,
            "threshold": selected,
        },
    )


def specificity_at_fixed_sensitivity(
    labels: Any,
    scores: Any,
    *,
    sensitivity: float = 0.9,
    unit: str = "per_patient",
    threshold: float | None = None,
) -> MetricValue:
    """Report specificity at a declared sensitivity target."""

    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity target must be in [0, 1]")
    y, p = _binary_inputs(labels, scores)
    achieved_sensitivity, specificity_value, selected = _fixed_operating(
        y,
        p,
        target_sensitivity=sensitivity,
        threshold=threshold,
    )
    return _metric(
        "specificity_at_sensitivity",
        specificity_value,
        unit,
        len(y),
        {
            "target_sensitivity": sensitivity,
            "achieved_sensitivity": achieved_sensitivity,
            "threshold": selected,
        },
    )


@dataclass(frozen=True)
class BootstrapCI:
    """Deterministic cluster-bootstrap interval."""

    metric: str
    point: float | None
    lower: float | None
    upper: float | None
    confidence: float
    unit: str
    cluster_count: int
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "unit": self.unit,
            "cluster_count": self.cluster_count,
            "resamples": self.resamples,
            "seed": self.seed,
        }


def _statistic(labels: torch.Tensor, scores: torch.Tensor, metric: str) -> float | None:
    if metric == "auroc":
        return _rank_auc(labels, scores)
    if metric == "auprc":
        return _average_precision(labels, scores)
    if metric == "brier":
        return None if not labels.numel() else float(torch.mean((scores - labels.float()) ** 2))
    if metric == "ece":
        return _ece(labels, scores, 10)
    if metric in {"f1", "sensitivity", "specificity", "precision", "recall", "balanced_accuracy"}:
        value = _operating(labels, scores, 0.5)[metric]
        return None if value is None else float(value)
    raise ValueError(f"unsupported bootstrap metric {metric!r}")


def cluster_bootstrap_ci(
    labels: Any,
    scores: Any,
    *,
    cluster_ids: Iterable[str] | None,
    metric: str = "auroc",
    unit: str = "per_patient",
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap whole patients/studies/slides, never individual slices."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    y, p = _binary_inputs(labels, scores)
    if cluster_ids is None:
        raise ValueError("cluster_ids are required for clinical-unit bootstrap")
    clusters = [str(value) for value in cluster_ids]
    if len(clusters) != len(y):
        raise ValueError("cluster_ids must align with labels and scores")
    unique = sorted(set(clusters))
    if unit.lower().removeprefix("per_") in {"patient", "study", "slide"} and not unique:
        raise ValueError("clinical-unit bootstrap requires at least one cluster")
    point = _statistic(y, p, metric)
    if len(unique) < 2:
        return BootstrapCI(metric, point, None, None, confidence, unit, len(unique), resamples, seed)
    by_cluster = {cluster: [index for index, value in enumerate(clusters) if value == cluster] for cluster in unique}
    generator = torch.Generator().manual_seed(int(seed))
    values: list[float] = []
    for _ in range(resamples):
        sample_indices = torch.randint(0, len(unique), (len(unique),), generator=generator).tolist()
        indices = [index for sample in sample_indices for index in by_cluster[unique[sample]]]
        value = _statistic(y[indices], p[indices], metric)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        lower = upper = None
    else:
        values.sort()
        alpha = (1.0 - confidence) / 2.0
        lower = float(np.quantile(np.asarray(values), alpha, method="linear"))
        upper = float(np.quantile(np.asarray(values), 1.0 - alpha, method="linear"))
    return BootstrapCI(metric, point, lower, upper, confidence, unit, len(unique), resamples, seed)


def _with_bootstrap(
    metric: MetricValue,
    labels: Any,
    scores: Any,
    name: str,
    unit: str,
    cluster_ids: Iterable[str] | None,
    options: Mapping[str, Any],
) -> MetricValue:
    ci = cluster_bootstrap_ci(labels, scores, cluster_ids=cluster_ids, metric=name, unit=unit, **options)
    return MetricValue(
        metric.name,
        metric.value,
        metric.unit,
        metric.sample_count,
        {**metric.metadata, "confidence_interval": ci.to_dict()},
    )


# ---------------------------------------------------------------------------
# Segmentation, physical-space distances, and boxes


def _mask_from_logits(logits: Any) -> torch.Tensor:
    values = _tensor(logits, dtype=torch.float32)
    return torch.sigmoid(values) >= 0.5


def _surface(mask: NDArray[Any]) -> NDArray[Any]:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.asarray(ndimage.generate_binary_structure(mask.ndim, 1), dtype=bool)
    eroded = np.asarray(ndimage.binary_erosion(mask, structure=structure, border_value=0), dtype=bool)
    return np.asarray(mask & ~eroded, dtype=bool)


def _surface_distances(source: NDArray[Any], target: NDArray[Any], spacing: tuple[float, ...]) -> NDArray[np.float64]:
    source_surface = _surface(source)
    target_surface = _surface(target)
    if not source_surface.any() or not target_surface.any():
        return np.asarray([], dtype=np.float64)
    distance = np.asarray(ndimage.distance_transform_edt(~target_surface, sampling=spacing), dtype=np.float64)
    return np.asarray(distance[source_surface], dtype=np.float64)


def _monai_spatial_summary(
    pred: NDArray[Any], truth: NDArray[Any], spacing: tuple[float, ...], tolerance_mm: float
) -> tuple[float, float, float]:
    """Compute non-empty surface metrics with MONAI's physical-space kernels."""
    from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
    from monai.metrics.surface_dice import SurfaceDiceMetric
    from monai.metrics.surface_distance import SurfaceDistanceMetric

    pred_tensor = torch.as_tensor(pred, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    truth_tensor = torch.as_tensor(truth, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    hd95_raw = HausdorffDistanceMetric(percentile=95.0, directed=False, include_background=False, reduction="mean")(
        pred_tensor, truth_tensor, spacing=spacing
    )
    assd_raw = SurfaceDistanceMetric(
        symmetric=True, include_background=False, distance_metric="euclidean", reduction="mean"
    )(pred_tensor, truth_tensor, spacing=spacing)
    surface_raw = SurfaceDiceMetric(
        class_thresholds=[float(tolerance_mm)],
        include_background=False,
        distance_metric="euclidean",
        reduction="mean",
    )(pred_tensor, truth_tensor, spacing=spacing)

    def _repo_value(raw: Any) -> float:
        value = float(_tensor(raw).reshape(-1)[0])
        # Empty masks are handled before this helper.  MONAI's NaN/inf
        # sentinel therefore maps to the finite repo fallback for non-empty
        # inputs rather than leaking into MetricValue aggregation.
        return value if math.isfinite(value) else 0.0

    return _repo_value(hd95_raw), _repo_value(assd_raw), _repo_value(surface_raw)


def _spatial_summary(
    pred: NDArray[Any], truth: NDArray[Any], spacing: tuple[float, ...], tolerance_mm: float
) -> dict[str, float | None | str]:
    pred_count = int(pred.sum())
    truth_count = int(truth.sum())
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    dice: float
    iou: float
    surface: float
    hd95: float | None
    assd: float | None
    if pred_count == 0 and truth_count == 0:
        dice = iou = surface = 1.0
        hd95 = assd = 0.0
        empty_case = "both_empty"
    elif pred_count == 0 or truth_count == 0:
        dice = iou = surface = 0.0
        hd95 = assd = None
        empty_case = "prediction_empty" if pred_count == 0 else "target_empty"
    else:
        dice = 2.0 * intersection / (pred_count + truth_count)
        iou = intersection / union if union else 1.0
        hd95, assd, surface = _monai_spatial_summary(pred, truth, spacing, tolerance_mm)
        empty_case = "neither_empty"
    return {
        "dice": dice,
        "iou": iou,
        "surface_dice": surface,
        "hd95": hd95,
        "assd": assd,
        "empty_case": empty_case,
    }


def _lesion_counts(pred: NDArray[Any], truth: NDArray[Any]) -> tuple[float, float]:
    structure = ndimage.generate_binary_structure(truth.ndim, 1)
    truth_labels, truth_count = ndimage.label(truth, structure=structure)
    pred_labels, pred_count = ndimage.label(pred, structure=structure)
    detected = 0
    for index in range(1, truth_count + 1):
        component = truth_labels == index
        if bool((pred & component).any()):
            detected += 1
    sensitivity = detected / truth_count if truth_count else (1.0 if pred_count == 0 else 0.0)
    false_positive = float(max(0, pred_count - detected))
    return float(sensitivity), false_positive


def segmentation_metrics(
    logits: Any,
    target: Any,
    *,
    threshold: float = 0.5,
    unit: str = "per_image",
    spacing_mm: Iterable[float] | None = None,
    surface_tolerance_mm: float = 1.0,
) -> dict[str, MetricValue]:
    """Compute dense metrics with explicit empty-case and physical-space rules."""

    values = _tensor(logits, dtype=torch.float32)
    truth_tensor = _tensor(target, dtype=torch.float32)
    if values.ndim not in (4, 5) or values.shape != truth_tensor.shape:
        raise ValueError("segmentation logits and target must have matching rank 4 or 5 and shape")
    probabilities = torch.sigmoid(values)
    predicted = probabilities >= float(threshold)
    truth = truth_tensor > 0.5
    batch, classes = int(values.shape[0]), int(values.shape[1])
    spacing = tuple(float(value) for value in (spacing_mm or (1.0,) * (values.ndim - 2)))
    if len(spacing) != values.ndim - 2 or any(value <= 0 for value in spacing):
        raise ValueError("spacing_mm must match spatial rank and contain positive values")
    result: dict[str, MetricValue] = {}
    macro: dict[str, list[float]] = {
        name: []
        for name in (
            "dice",
            "iou",
            "surface_dice",
            "hd95",
            "assd",
            "sensitivity",
            "lesion_sensitivity",
            "false_positives_per_image",
            "false_positive_lesions",
            "volume_error_mm3",
        )
    }
    voxel_volume = float(math.prod(spacing))
    for cls in range(classes):
        rows: list[dict[str, Any]] = []
        sensitivity_values: list[float] = []
        lesion_values: list[float] = []
        fp_pixel_values: list[float] = []
        fp_lesion_values: list[float] = []
        volume_values: list[float] = []
        for row in range(batch):
            pred_np = predicted[row, cls].numpy().astype(bool)
            truth_np = truth[row, cls].numpy().astype(bool)
            rows.append(_spatial_summary(pred_np, truth_np, spacing, surface_tolerance_mm))
            lesion, false_positive_lesions = _lesion_counts(pred_np, truth_np)
            tp = int((pred_np & truth_np).sum())
            fn = int(((~pred_np) & truth_np).sum())
            sensitivity_values.append(float(tp / (tp + fn)) if tp + fn else (1.0 if not pred_np.any() else 0.0))
            lesion_values.append(lesion)
            fp_pixel_values.append(float((pred_np & ~truth_np).sum()))
            fp_lesion_values.append(false_positive_lesions)
            volume_values.append(abs(float(pred_np.sum() - truth_np.sum())) * voxel_volume)
        for name in ("dice", "iou", "surface_dice", "hd95", "assd"):
            finite = [float(row[name]) for row in rows if row[name] is not None and math.isfinite(float(row[name]))]
            value = None if not finite else sum(finite) / len(finite)
            result[f"{name}/class_{cls}"] = _metric(
                f"{name}/class_{cls}",
                value,
                unit,
                batch,
                {
                    "class_index": cls,
                    "spacing_mm": spacing,
                    "empty_cases": [row["empty_case"] for row in rows],
                },
            )
            if value is not None:
                macro[name].append(value)
        extra_values = (
            ("sensitivity", sensitivity_values, unit),
            ("lesion_sensitivity", lesion_values, unit),
            ("false_positives_per_image", fp_pixel_values, unit),
            ("false_positive_lesions", fp_lesion_values, unit),
            ("volume_error_mm3", volume_values, "mm3_per_scan"),
        )
        for name, values_list, metric_unit in extra_values:
            value = None if not values_list else sum(values_list) / len(values_list)
            result[f"{name}/class_{cls}"] = _metric(
                f"{name}/class_{cls}", value, metric_unit, batch, {"class_index": cls}
            )
            if value is not None:
                macro[name].append(value)
    for name, values_list in macro.items():
        result[f"{name}/macro"] = _metric(
            f"{name}/macro",
            None if not values_list else sum(values_list) / len(values_list),
            "mm3_per_scan" if name == "volume_error_mm3" else unit,
            batch,
            {"class_count": classes},
        )
    return result


def _box_bounds(box: Sequence[float], dimensions: int) -> tuple[float, ...]:
    expected = dimensions * 2
    if len(box) != expected:
        raise ValueError(f"{dimensions}D boxes require {expected} coordinates")
    values = tuple(float(value) for value in box)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    starts, ends = values[:dimensions], values[dimensions:]
    if any(end < start for start, end in zip(starts, ends, strict=True)):
        raise ValueError("box ends must not precede starts")
    return values


def box_iou(predicted: Sequence[float], target: Sequence[float]) -> float:
    """Compute 2D or 3D half-open box IoU."""

    if len(predicted) not in {4, 6} or len(predicted) != len(target):
        raise ValueError("boxes must both be 2D [x1,y1,x2,y2] or 3D [x1,y1,z1,x2,y2,z2]")
    dimensions = len(predicted) // 2
    pred = _box_bounds(predicted, dimensions)
    truth = _box_bounds(target, dimensions)
    intersection = 1.0
    pred_volume = truth_volume = 1.0
    for index in range(dimensions):
        intersection *= max(
            0.0, min(pred[index + dimensions], truth[index + dimensions]) - max(pred[index], truth[index])
        )
        pred_volume *= max(0.0, pred[index + dimensions] - pred[index])
        truth_volume *= max(0.0, truth[index + dimensions] - truth[index])
    union = pred_volume + truth_volume - intersection
    return 1.0 if union == 0 and pred == truth else (0.0 if union == 0 else intersection / union)


def physical_localization_error(
    predicted: Sequence[float],
    target: Sequence[float],
    *,
    spacing_mm: Iterable[float] | None = None,
) -> float:
    """Euclidean center error in physical units for 2D/3D boxes."""

    if len(predicted) not in {4, 6} or len(predicted) != len(target):
        raise ValueError("boxes must both be 2D or both be 3D")
    dimensions = len(predicted) // 2
    pred = _box_bounds(predicted, dimensions)
    truth = _box_bounds(target, dimensions)
    spacing = tuple(float(value) for value in (spacing_mm or (1.0,) * dimensions))
    if len(spacing) != dimensions or any(value <= 0 for value in spacing):
        raise ValueError("spacing_mm must match box dimensions")
    error = 0.0
    for index in range(dimensions):
        pred_center = (pred[index] + pred[index + dimensions]) / 2
        truth_center = (truth[index] + truth[index + dimensions]) / 2
        error += ((pred_center - truth_center) * spacing[index]) ** 2
    return math.sqrt(error)


def localization_metrics(
    predicted_boxes: Iterable[Sequence[float]],
    target_boxes: Iterable[Sequence[float]],
    *,
    unit: str = "per_scan",
    spacing_mm: Iterable[float] | None = None,
) -> dict[str, MetricValue]:
    predicted = list(predicted_boxes)
    target = list(target_boxes)
    if len(predicted) != len(target):
        raise ValueError("predicted_boxes and target_boxes must align")
    if not predicted:
        return {
            "box_iou": _metric("box_iou", None, unit, 0),
            "physical_localization_error_mm": _metric("physical_localization_error_mm", None, unit, 0),
        }
    ious = [box_iou(left, right) for left, right in zip(predicted, target, strict=True)]
    errors = [
        physical_localization_error(left, right, spacing_mm=spacing_mm)
        for left, right in zip(predicted, target, strict=True)
    ]
    return {
        "box_iou": _metric("box_iou", sum(ious) / len(ious), unit, len(ious)),
        "physical_localization_error_mm": _metric(
            "physical_localization_error_mm", sum(errors) / len(errors), unit, len(errors)
        ),
    }


# ---------------------------------------------------------------------------
# Retrieval


def _retrieval_direction(
    similarity: torch.Tensor, query_ids: list[str], candidate_ids: list[str], unit: str, prefix: str
) -> dict[str, MetricValue]:
    if similarity.ndim != 2 or similarity.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("similarity must have shape [len(query_ids), len(candidate_ids)]")
    ranks: list[int] = []
    average_precisions: list[float] = []
    for row, query in enumerate(query_ids):
        order = torch.argsort(similarity[row], descending=True, stable=True).tolist()
        relevant = {index for index, candidate in enumerate(candidate_ids) if candidate == query}
        if not relevant:
            # When IDs are not shared, a query may provide positives through a
            # ``query_id -> candidate_id`` mapping in the caller; absent that,
            # fail instead of silently scoring every row as a miss.
            raise ValueError(f"no positive candidate for query {query!r}")
        rank_positions = [position + 1 for position, index in enumerate(order) if index in relevant]
        ranks.append(min(rank_positions))
        hits = 0
        precision_sum = 0.0
        for position, index in enumerate(order, start=1):
            if index in relevant:
                hits += 1
                precision_sum += hits / position
        average_precisions.append(precision_sum / len(relevant))
    result: dict[str, MetricValue] = {}
    for k in (1, 5, 10):
        result[f"{prefix}/recall@{k}"] = _metric(
            f"{prefix}/recall@{k}", sum(rank <= k for rank in ranks) / len(ranks), unit, len(ranks), {"k": k}
        )
    result[f"{prefix}/median_rank"] = _metric(f"{prefix}/median_rank", float(np.median(ranks)), unit, len(ranks))
    result[f"{prefix}/mean_rank"] = _metric(f"{prefix}/mean_rank", sum(ranks) / len(ranks), unit, len(ranks))
    result[f"{prefix}/mAP"] = _metric(
        f"{prefix}/mAP", sum(average_precisions) / len(average_precisions), unit, len(ranks)
    )
    return result


def retrieval_metrics(
    similarity: Any,
    *,
    query_ids: Iterable[str] | None = None,
    candidate_ids: Iterable[str] | None = None,
    unit: str = "per_query",
) -> dict[str, MetricValue]:
    """Compute image→text and text→image retrieval metrics."""

    matrix = _tensor(similarity, dtype=torch.float64)
    queries = [str(value) for value in (query_ids if query_ids is not None else range(matrix.shape[0]))]
    candidates = [str(value) for value in (candidate_ids if candidate_ids is not None else range(matrix.shape[1]))]
    if matrix.shape != (len(queries), len(candidates)):
        raise ValueError("retrieval IDs must align with similarity matrix")
    result = _retrieval_direction(matrix, queries, candidates, unit, "image_to_text")
    reverse = _retrieval_direction(matrix.T, candidates, queries, unit, "text_to_image")
    result.update(reverse)
    return result


# ---------------------------------------------------------------------------
# Structured generation and clinical finding analysis


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _tokens(value: Any) -> list[str]:
    return re.findall(r"\w+", _normalize_text(value), flags=re.UNICODE)


def _token_f1(predicted: Any, target: Any) -> float:
    left = Counter(_tokens(predicted))
    right = Counter(_tokens(target))
    overlap = sum((left & right).values())
    if not left or not right or overlap == 0:
        return 1.0 if not left and not right else 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def _parse_structured(value: Any) -> tuple[Any, bool]:
    if isinstance(value, Mapping | list | tuple):
        return value, True
    try:
        return json.loads(str(value)), True
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, False


def _finding_set(value: Any) -> set[tuple[str, str, str, str, str]]:
    payload, valid = _parse_structured(value)
    if not valid:
        return set()
    if isinstance(payload, Mapping):
        for key in ("findings", "entities", "observations"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list | tuple):
        return set()
    findings: set[tuple[str, str, str, str, str]] = set()
    for item in payload:
        if isinstance(item, str):
            findings.add((_normalize_text(item), "", "", "", ""))
            continue
        if not isinstance(item, Mapping):
            continue
        finding = _normalize_text(item.get("finding", item.get("entity", item.get("text", ""))))
        if not finding:
            continue
        findings.add(
            (
                finding,
                _normalize_text(item.get("negation", item.get("assertion", ""))),
                _normalize_text(item.get("laterality", "")),
                _normalize_text(item.get("severity", "")),
                _normalize_text(item.get("anatomy", item.get("anatomical_site", ""))),
            )
        )
    return findings


def _ngram_precision(predicted: str, target: str, n: int) -> float:
    left_tokens, right_tokens = _tokens(predicted), _tokens(target)
    left = Counter(tuple(left_tokens[index : index + n]) for index in range(max(0, len(left_tokens) - n + 1)))
    right = Counter(tuple(right_tokens[index : index + n]) for index in range(max(0, len(right_tokens) - n + 1)))
    denominator = sum(left.values())
    return 0.0 if denominator == 0 else sum((left & right).values()) / denominator


def _rouge_l(predicted: str, target: str) -> float:
    left, right = _tokens(predicted), _tokens(target)
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, token in enumerate(left, start=1):
        for j, other in enumerate(right, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if token == other else max(table[i - 1][j], table[i][j - 1])
    return 0.0 if not right else table[-1][-1] / len(right)


def generation_metrics(
    predictions: Sequence[Any],
    references: Sequence[Any],
    *,
    schema: Mapping[str, Any] | None = None,
    approved_entity_relation_evaluator: Callable[[Any, Any], Mapping[str, float]] | None = None,
    unit: str = "per_study",
) -> dict[str, MetricValue]:
    """Score exactness, structured validity, clinical findings, and contradictions."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must align")
    count = len(predictions)
    exact = [
        float(_normalize_text(predicted) == _normalize_text(reference))
        for predicted, reference in zip(predictions, references, strict=True)
    ]
    token = [_token_f1(predicted, reference) for predicted, reference in zip(predictions, references, strict=True)]
    valid_values: list[float] = []
    finding_tp = finding_fp = finding_fn = 0
    attribute_totals = {name: [0, 0] for name in ("negation", "laterality", "severity", "anatomy")}
    contradictions = omissions = hallucinations = 0
    bleu_values: list[float] = []
    rouge_values: list[float] = []
    approved_scores: dict[str, list[float]] = {}
    for predicted, reference in zip(predictions, references, strict=True):
        parsed_predicted, valid = _parse_structured(predicted)
        valid_values.append(float(valid))
        if schema is not None and valid:
            try:
                from jsonschema import Draft202012Validator

                valid_values[-1] = float(Draft202012Validator(schema).is_valid(parsed_predicted))
            except ImportError:
                raise RuntimeError("schema evaluation requires jsonschema") from None
        predicted_findings = _finding_set(predicted)
        reference_findings = _finding_set(reference)
        finding_tp += len(predicted_findings & reference_findings)
        finding_fp += len(predicted_findings - reference_findings)
        finding_fn += len(reference_findings - predicted_findings)
        predicted_by_entity = {item[0]: item for item in predicted_findings}
        reference_by_entity = {item[0]: item for item in reference_findings}
        for entity, reference_item in reference_by_entity.items():
            predicted_item = predicted_by_entity.get(entity)
            if predicted_item is None:
                omissions += 1
                continue
            for index, name in enumerate(("negation", "laterality", "severity", "anatomy"), start=1):
                if reference_item[index]:
                    attribute_totals[name][1] += 1
                    attribute_totals[name][0] += int(predicted_item[index] == reference_item[index])
            if reference_item[1] and predicted_item[1] and reference_item[1] != predicted_item[1]:
                contradictions += 1
        hallucinations += len(predicted_findings - reference_findings)
        bleu_values.append(sum(_ngram_precision(str(predicted), str(reference), n) for n in (1, 2)) / 2)
        rouge_values.append(_rouge_l(str(predicted), str(reference)))
        if approved_entity_relation_evaluator is not None:
            for name, value in approved_entity_relation_evaluator(predicted, reference).items():
                approved_scores.setdefault(str(name), []).append(float(value))
    result: dict[str, MetricValue] = {
        "exact_match": _metric("exact_match", None if not count else sum(exact) / count, unit, count),
        "token_f1": _metric("token_f1", None if not count else sum(token) / count, unit, count),
        "schema_validity": _metric(
            "schema_validity",
            None if not count else sum(valid_values) / count,
            unit,
            count,
            {"schema_supplied": schema is not None},
        ),
        "finding_precision": _metric(
            "finding_precision",
            None if finding_tp + finding_fp == 0 else finding_tp / (finding_tp + finding_fp),
            unit,
            count,
        ),
        "finding_recall": _metric(
            "finding_recall",
            None if finding_tp + finding_fn == 0 else finding_tp / (finding_tp + finding_fn),
            unit,
            count,
        ),
        "contradiction_rate": _metric(
            "contradiction_rate", None if count == 0 else contradictions / count, unit, count
        ),
        "omission_rate": _metric("omission_rate", None if count == 0 else omissions / count, unit, count),
        "hallucinated_finding_rate": _metric(
            "hallucinated_finding_rate", None if count == 0 else hallucinations / count, unit, count
        ),
        "bleu_secondary": _metric(
            "bleu_secondary", None if not count else sum(bleu_values) / count, unit, count, {"headline": False}
        ),
        "rouge_l_secondary": _metric(
            "rouge_l_secondary", None if not count else sum(rouge_values) / count, unit, count, {"headline": False}
        ),
    }
    precision = result["finding_precision"].value
    recall = result["finding_recall"].value
    result["finding_f1"] = _metric(
        "finding_f1",
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall),
        unit,
        count,
    )
    for name, (correct, total) in attribute_totals.items():
        result[f"{name}_correctness"] = _metric(
            f"{name}_correctness", None if total == 0 else correct / total, unit, total, {"labeled_findings": total}
        )
    if approved_entity_relation_evaluator is not None:
        for name, values in sorted(approved_scores.items()):
            result[f"clinical_relation/{name}"] = _metric(
                f"clinical_relation/{name}", sum(values) / len(values), unit, len(values), {"approved_evaluator": True}
            )
    else:
        result["clinical_relation/evaluation_status"] = _metric(
            "clinical_relation/evaluation_status", None, unit, count, {"approved_evaluator": False, "status": "not_run"}
        )
    return result


# ---------------------------------------------------------------------------
# Baselines, ablation gates, and generalization tables


REQUIRED_BASELINES = ("random_majority", "frozen_linear_probe", "lora", "conventional_task")
REQUIRED_SEGMENTATION_BASELINES = ("frozen_encoder_decoder", "nnunet_or_comparable_3d")
VLM_ABLATIONS = ("no_visual", "shuffled_visual", "frozen_bridge", "bridge_type", "token_budget", "coordinates")
HOLDOUT_TYPES = (
    "internal",
    "patient_disjoint",
    "external_site",
    "temporal",
    "vendor_protocol",
    "rare_class",
    "missing_sequence",
    "low_quality",
)


def baseline_metrics(
    labels: Any, *, scores: Any | None = None, seed: int = 0, unit: str = "per_patient"
) -> dict[str, dict[str, MetricValue]]:
    y = _tensor(labels, dtype=torch.int64).reshape(-1)
    if y.numel() and not bool(torch.logical_or(y == 0, y == 1).all()):
        raise ValueError("baseline labels must be binary")
    generator = torch.Generator().manual_seed(int(seed))
    random_scores = torch.rand(y.shape, generator=generator, dtype=torch.float64)
    majority = float(y.float().mean() >= 0.5) if y.numel() else 0.0
    majority_scores = torch.full(y.shape, majority, dtype=torch.float64)
    result = {
        "random_majority": classification_metrics(y, random_scores, unit=unit),
        "majority": classification_metrics(y, majority_scores, unit=unit),
    }
    if scores is not None:
        result["provided_model"] = classification_metrics(y, scores, unit=unit)
    return result


def validate_baseline_set(names: Iterable[str], *, segmentation_3d: bool = False) -> tuple[str, ...]:
    required = set(REQUIRED_BASELINES) | (set(REQUIRED_SEGMENTATION_BASELINES) if segmentation_3d else set())
    available = {str(name) for name in names}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"missing required baselines: {', '.join(missing)}")
    return tuple(sorted(available))


def visual_grounding_gate(
    visual_score: float,
    shuffled_score: float,
    *,
    margin: float,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Fail when shuffled visual inputs are within the predeclared margin."""

    if margin < 0:
        raise ValueError("visual-grounding margin must be non-negative")
    delta = float(visual_score - shuffled_score if higher_is_better else shuffled_score - visual_score)
    passed = delta > float(margin)
    return {
        "visual_score": float(visual_score),
        "shuffled_score": float(shuffled_score),
        "delta": delta,
        "margin": float(margin),
        "higher_is_better": bool(higher_is_better),
        "passed": passed,
        "failure_reason": None if passed else "shuffled visual performance is within the predeclared margin",
    }


def evaluate_holdouts(
    rows: Iterable[Mapping[str, Any]], *, metric_fn: Callable[[list[int], list[float]], Mapping[str, MetricValue]]
) -> dict[str, Mapping[str, MetricValue]]:
    """Evaluate declared generalization holdouts without inventing missing sets."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        holdout = str(row.get("holdout", "internal"))
        if holdout not in HOLDOUT_TYPES:
            raise ValueError(f"unsupported holdout type {holdout!r}")
        grouped.setdefault(holdout, []).append(row)
    result: dict[str, Mapping[str, MetricValue]] = {}
    for holdout, values in sorted(grouped.items()):
        result[holdout] = metric_fn(
            [int(value["target"]) for value in values], [float(value["score"]) for value in values]
        )
    return result


# Public aliases used by task-specific evaluators.
bootstrap_confidence_interval = cluster_bootstrap_ci
patient_level_bootstrap = cluster_bootstrap_ci
compute_retrieval_metrics = retrieval_metrics
compute_generation_metrics = generation_metrics
compute_localization_metrics = localization_metrics


__all__ = [
    "BootstrapCI",
    "HOLDOUT_TYPES",
    "REQUIRED_BASELINES",
    "REQUIRED_SEGMENTATION_BASELINES",
    "VLM_ABLATIONS",
    "baseline_metrics",
    "bootstrap_confidence_interval",
    "box_iou",
    "classification_metrics",
    "cluster_bootstrap_ci",
    "compute_generation_metrics",
    "compute_localization_metrics",
    "compute_retrieval_metrics",
    "evaluate_holdouts",
    "generation_metrics",
    "localization_metrics",
    "patient_level_bootstrap",
    "physical_localization_error",
    "retrieval_metrics",
    "segmentation_metrics",
    "sensitivity_at_fixed_specificity",
    "specificity_at_fixed_sensitivity",
    "validate_baseline_set",
    "visual_grounding_gate",
]
