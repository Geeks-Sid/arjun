from __future__ import annotations

import pytest
import torch
from torchmetrics.classification import BinaryCalibrationError

from medfm.evaluation.calibration import expected_calibration_error


def _reference_ece(labels: list[int], scores: list[float], bins: int) -> float | None:
    probabilities = torch.tensor(scores)
    targets = torch.tensor(labels)
    if not targets.numel():
        return None
    edges = torch.linspace(0.0, 1.0, bins + 1)
    total = torch.tensor(0.0, dtype=torch.float64)
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            (probabilities < edges[index + 1]) if index < bins - 1 else (probabilities <= edges[index + 1])
        )
        if bool(mask.any()):
            total += (
                mask.float().mean().to(torch.float64)
                * (probabilities[mask].mean() - targets[mask].float().mean()).abs()
            )
    return float(total)


@pytest.mark.parametrize(
    ("labels", "scores", "bins"),
    [
        pytest.param([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 2, id="balanced"),
        pytest.param([0, 1], [0.0, 1.0], 4, id="boundary-values"),
        pytest.param([0, 1, 0], [0.1, 0.3, 0.9], 10, id="empty-bins"),
        pytest.param([0, 1, 0], [0.0, 0.0, 0.0], 4, id="all-zero-predictions"),
        pytest.param([0, 1, 0], [1.0, 1.0, 1.0], 4, id="all-one-predictions"),
    ],
)
def test_expected_calibration_error_matches_torchmetrics(labels: list[int], scores: list[float], bins: int) -> None:
    expected = expected_calibration_error(labels, scores, bins=bins)
    reference = _reference_ece(labels, scores, bins)
    library_value = BinaryCalibrationError(n_bins=bins, norm="l1")(torch.tensor(scores), torch.tensor(labels))

    assert expected is not None
    assert reference is not None
    assert float(torch.abs(library_value - reference)) <= 1e-6
    assert float(torch.abs(library_value - expected)) <= 1e-6


def test_expected_calibration_error_preserves_empty_input_contract() -> None:
    assert expected_calibration_error([], [], bins=4) is None
    assert torch.isnan(BinaryCalibrationError(n_bins=4, norm="l1")(torch.tensor([]), torch.tensor([])))
