"""Task-specific Phase 16 tables for 3D and pathology evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from medfm.evaluation.advanced import classification_metrics
from medfm.evaluation.metrics import MetricValue


def _metric(
    name: str, value: float | None, unit: str, count: int, metadata: Mapping[str, Any] | None = None
) -> MetricValue:
    return MetricValue(name, value, unit, count, dict(metadata or {}))


def adjacent_slice_consistency(
    predictions: Any,
    *,
    volume_ids: Iterable[str] | None = None,
    unit: str = "per_scan",
) -> MetricValue:
    """Mean IoU of adjacent slice predictions within each declared volume."""

    values = torch.as_tensor(predictions).detach().float()
    if values.ndim < 3:
        raise ValueError("predictions must include a volume and slice dimension")
    # Accept [V,S,...] or [S,...] for a single volume; never compare slices
    # across independent studies.
    if values.ndim == 3:
        values = values.unsqueeze(0)
    volumes = list(volume_ids or [str(index) for index in range(values.shape[0])])
    if len(volumes) != values.shape[0]:
        raise ValueError("volume_ids must align with the leading volume dimension")
    scores: list[float] = []
    for volume in range(values.shape[0]):
        mask = (
            torch.sigmoid(values[volume]) >= 0.5
            if values[volume].min() < 0 or values[volume].max() > 1
            else values[volume] >= 0.5
        )
        for slice_index in range(max(0, mask.shape[0] - 1)):
            left, right = mask[slice_index], mask[slice_index + 1]
            union = (left | right).sum()
            scores.append(1.0 if union == 0 else float((left & right).sum() / union))
    return _metric(
        "adjacent_slice_consistency",
        None if not scores else sum(scores) / len(scores),
        unit,
        len(scores),
        {"volume_count": len(set(volumes))},
    )


def compare_native_3d_and_slice_sequence(
    native: Mapping[str, MetricValue | float | None],
    slice_sequence: Mapping[str, MetricValue | float | None],
    *,
    unit: str = "per_scan",
) -> dict[str, MetricValue]:
    """Compare the same metric keys without collapsing input strategies."""

    result: dict[str, MetricValue] = {}
    for name in sorted(set(native) | set(slice_sequence)):
        left = native.get(name)
        right = slice_sequence.get(name)
        left_value = left.value if isinstance(left, MetricValue) else left
        right_value = right.value if isinstance(right, MetricValue) else right
        delta = None if left_value is None or right_value is None else float(left_value) - float(right_value)
        result[f"native_3d_minus_slice_sequence/{name}"] = _metric(
            f"native_3d_minus_slice_sequence/{name}",
            delta,
            unit,
            min(
                left.sample_count if isinstance(left, MetricValue) else 0,
                right.sample_count if isinstance(right, MetricValue) else 0,
            ),
            {"native_3d": left_value, "slice_sequence": right_value},
        )
    return result


def small_lesion_sensitivity(
    logits: Any,
    target: Any,
    *,
    max_target_voxels: int = 32,
    unit: str = "per_scan",
) -> MetricValue:
    """Sensitivity restricted to connected lesions at or below a size bound."""

    from scipy import ndimage

    predicted = torch.sigmoid(torch.as_tensor(logits).detach().float()) >= 0.5
    truth = torch.as_tensor(target).detach().float() > 0.5
    if predicted.shape != truth.shape or predicted.ndim not in (4, 5):
        raise ValueError("small-lesion inputs must be matching [B,C,H,W] or [B,C,D,H,W]")
    values: list[float] = []
    for row in range(predicted.shape[0]):
        truth_np = truth[row, 0].numpy()
        pred_np = predicted[row, 0].numpy()
        labels, count = ndimage.label(truth_np)
        small = [index for index in range(1, count + 1) if int((labels == index).sum()) <= max_target_voxels]
        if small:
            values.append(sum(bool((pred_np & (labels == index)).any()) for index in small) / len(small))
    return _metric(
        "small_lesion_sensitivity",
        None if not values else sum(values) / len(values),
        unit,
        len(values),
        {"max_target_voxels": max_target_voxels},
    )


def pathology_evaluation_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    tile_ids: Sequence[str] | None = None,
    slide_ids: Sequence[str] | None = None,
    patient_ids: Sequence[str] | None = None,
    scanner_ids: Sequence[str] | None = None,
    site_ids: Sequence[str] | None = None,
    organ_ids: Sequence[str] | None = None,
) -> dict[str, MetricValue]:
    """Report tile/slide/patient and protected pathology subgroup metrics."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must align")
    result: dict[str, MetricValue] = {}
    for prefix, ids, unit in (
        ("tile", tile_ids, "per_tile"),
        ("slide", slide_ids, "per_slide"),
        ("patient", patient_ids, "per_patient"),
    ):
        if prefix == "tile" or ids is None:
            current_labels, current_scores = list(labels), list(scores)
        else:
            groups: dict[str, list[tuple[int, float]]] = {}
            for label, score, identifier in zip(labels, scores, ids, strict=True):
                groups.setdefault(str(identifier), []).append((int(label), float(score)))
            current_labels, current_scores = [], []
            for identifier in sorted(groups):
                rows = groups[identifier]
                if len({row[0] for row in rows}) != 1:
                    raise ValueError(f"{prefix} {identifier!r} has inconsistent targets")
                current_labels.append(rows[0][0])
                current_scores.append(sum(row[1] for row in rows) / len(rows))
        result.update(
            {
                f"{prefix}/{name}": value
                for name, value in classification_metrics(current_labels, current_scores, unit=unit).items()
            }
        )
    for name, ids in (("scanner", scanner_ids), ("site", site_ids), ("organ", organ_ids)):
        if ids is None:
            continue
        if len(ids) != len(labels):
            raise ValueError(f"{name}_ids must align with labels")
        subgroup = classification_metrics(labels, scores, group_ids=ids, unit=f"per_{name}").get("subgroups")
        if subgroup is not None:
            result[f"{name}/subgroups"] = subgroup
    return result


def sweep_pathology_sampling(
    labels: Sequence[int],
    scores_by_condition: Mapping[Any, Sequence[float]],
    *,
    patient_ids: Sequence[str] | None = None,
    slide_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate sampled tile count and magnification conditions independently."""

    rows: list[dict[str, Any]] = []
    for condition, scores in sorted(scores_by_condition.items(), key=lambda pair: str(pair[0])):
        if len(scores) != len(labels):
            raise ValueError("each tile sampling condition must align with labels")
        if isinstance(condition, tuple | list) and len(condition) == 2:
            tile_count, magnification = condition
        else:
            tile_count, magnification = condition, None
        metrics = pathology_evaluation_metrics(labels, scores, patient_ids=patient_ids, slide_ids=slide_ids)
        rows.append(
            {
                "tile_count": tile_count,
                "magnification": magnification,
                "metrics": {name: value.to_dict() for name, value in metrics.items()},
            }
        )
    return tuple(rows)


__all__ = [
    "adjacent_slice_consistency",
    "compare_native_3d_and_slice_sequence",
    "pathology_evaluation_metrics",
    "small_lesion_sensitivity",
    "sweep_pathology_sampling",
]
