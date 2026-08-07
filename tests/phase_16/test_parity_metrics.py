from __future__ import annotations

import warnings

import pytest
import torch
from monai.metrics import SurfaceDiceMetric

from medfm.evaluation import advanced, metrics


def _monai_surface_dice(predicted: torch.Tensor, target: torch.Tensor, tolerance: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = SurfaceDiceMetric(
            class_thresholds=[float(tolerance)],
            include_background=False,
            distance_metric="euclidean",
            reduction="mean",
        )(predicted, target, spacing=(1.0,) * (predicted.ndim - 2))
    return float(raw.reshape(-1)[0])


def test_metrics_classification_facade_forwards_to_advanced(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"forwarded": metrics.MetricValue("forwarded", 1.0, "per_patient", 1)}

    def fake_classification_metrics(*args: object, **kwargs: object) -> dict[str, metrics.MetricValue]:
        return sentinel

    monkeypatch.setattr(advanced, "classification_metrics", fake_classification_metrics)
    assert metrics.classification_metrics([0], [0.5]) is sentinel


@pytest.mark.parametrize(
    ("predicted", "target", "expected"),
    [
        (torch.zeros((1, 1, 8, 8), dtype=torch.bool), torch.zeros((1, 1, 8, 8), dtype=torch.bool), 1.0),
        (torch.zeros((1, 1, 8, 8), dtype=torch.bool), torch.ones((1, 1, 8, 8), dtype=torch.bool), 0.0),
        (torch.ones((1, 1, 8, 8), dtype=torch.bool), torch.zeros((1, 1, 8, 8), dtype=torch.bool), 0.0),
    ],
)
def test_surface_dice_preserves_empty_mask_contract(
    predicted: torch.Tensor, target: torch.Tensor, expected: float
) -> None:
    assert metrics._surface_dice(predicted, target) == expected


def test_surface_dice_non_empty_kernel_drift_requires_keep() -> None:
    predicted = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.bool,
    ).unsqueeze(0)
    target = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.bool,
    ).unsqueeze(0)
    tolerance = 1

    current = metrics._surface_dice(predicted, target, tolerance)
    candidate = _monai_surface_dice(predicted, target, tolerance)
    drift = abs(current - candidate)

    assert drift > 1e-6
    assert current == pytest.approx(0.8782051206, abs=1e-6)
    assert candidate == pytest.approx(0.5600000024, abs=1e-6)
