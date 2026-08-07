from __future__ import annotations

import numpy as np
import pytest
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric

from medfm.evaluation.advanced import _spatial_summary, _surface_distances


def _metric_scalar(value: torch.Tensor) -> float:
    return float(value.reshape(-1)[0])


def test_surface_distances_match_monai_physical_spacing() -> None:
    spacing = (1.5, 1.0, 0.5)
    source = np.zeros((9, 10, 11), dtype=bool)
    target = np.zeros_like(source)
    source[2:4, 3:5, 3:5] = True
    target[2:7, 3:8, 3:8] = True
    source_tensor = torch.as_tensor(source)[None, None]
    target_tensor = torch.as_tensor(target)[None, None]

    expected = SurfaceDistanceMetric(
        symmetric=False, include_background=False, distance_metric="euclidean", reduction="mean"
    )(source_tensor, target_tensor, spacing=spacing)

    actual_distances = _surface_distances(source, target, spacing)
    assert actual_distances.size > 0
    assert float(actual_distances.mean()) == pytest.approx(_metric_scalar(expected), abs=1e-6)


def test_non_empty_two_object_summary_matches_monai() -> None:
    spacing = (1.5, 1.0, 0.5)
    tolerance = 1.0
    pred = np.zeros((11, 12, 13), dtype=bool)
    truth = np.zeros_like(pred)
    pred[2:4, 2:4, 2:4] = True
    pred[7:9, 7:10, 8:10] = True
    truth[2:4, 3:5, 2:4] = True
    truth[7:10, 7:10, 8:10] = True
    pred_tensor = torch.as_tensor(pred)[None, None]
    truth_tensor = torch.as_tensor(truth)[None, None]

    summary = _spatial_summary(pred, truth, spacing, tolerance)
    expected_hd95 = HausdorffDistanceMetric(
        percentile=95.0, directed=False, include_background=False, reduction="mean"
    )(pred_tensor, truth_tensor, spacing=spacing)
    expected_assd = SurfaceDistanceMetric(
        symmetric=True, include_background=False, distance_metric="euclidean", reduction="mean"
    )(pred_tensor, truth_tensor, spacing=spacing)
    expected_surface = SurfaceDiceMetric(
        class_thresholds=[tolerance], include_background=False, distance_metric="euclidean", reduction="mean"
    )(pred_tensor, truth_tensor, spacing=spacing)

    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    assert summary["empty_case"] == "neither_empty"
    assert summary["dice"] == pytest.approx(2.0 * intersection / (int(pred.sum()) + int(truth.sum())), abs=1e-6)
    assert summary["iou"] == pytest.approx(intersection / union, abs=1e-6)
    assert summary["hd95"] == pytest.approx(_metric_scalar(expected_hd95), abs=1e-6)
    assert summary["assd"] == pytest.approx(_metric_scalar(expected_assd), abs=1e-6)
    assert summary["surface_dice"] == pytest.approx(_metric_scalar(expected_surface), abs=1e-6)
