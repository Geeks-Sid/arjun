from __future__ import annotations

import numpy as np
import pytest
import torch
from monai.metrics import DiceMetric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryCalibrationError

from medfm.evaluation.advanced import _average_precision, _ece, _rank_auc, _spatial_summary

CLASSIFICATION_FIXTURES = (
    ("balanced", [0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2]),
    ("imbalanced", [0, 0, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5]),
    ("ties", [0, 1, 0, 1, 0, 1], [0.5, 0.5, 0.2, 0.5, 0.2, 0.9]),
    ("degenerate_positive", [1, 1], [0.1, 0.2]),
    ("degenerate_negative", [0, 0], [0.1, 0.2]),
)


@pytest.mark.parametrize(("name", "labels", "scores"), CLASSIFICATION_FIXTURES)
def test_rank_auc_torchmetrics_parity_and_empty_contract(name: str, labels: list[int], scores: list[float]) -> None:
    del name
    target = torch.tensor(labels)
    prediction = torch.tensor(scores)
    actual = _rank_auc(target, prediction)
    expected = BinaryAUROC()(prediction, target)
    if len(set(labels)) < 2:
        assert actual is None
        assert float(expected) == 0.0
    else:
        assert actual == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize(("name", "labels", "scores"), CLASSIFICATION_FIXTURES)
def test_average_precision_matches_torchmetrics(name: str, labels: list[int], scores: list[float]) -> None:
    del name
    # ADR-0013: _average_precision delegates to torchmetrics (tie ordering
    # follows torchmetrics); only the undefined single-class case stays None.
    target = torch.tensor(labels)
    prediction = torch.tensor(scores)
    actual = _average_precision(target, prediction)
    expected = BinaryAveragePrecision()(prediction, target)
    if not any(labels):
        assert actual is None
    else:
        assert actual == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize(("name", "labels", "scores"), CLASSIFICATION_FIXTURES)
def test_ece_torchmetrics_l1_bin_parity(name: str, labels: list[int], scores: list[float]) -> None:
    del name
    target = torch.tensor(labels)
    prediction = torch.tensor(scores)
    actual = _ece(target, prediction, 3)
    expected = BinaryCalibrationError(n_bins=3, norm="l1")(prediction, target)
    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_ece_empty_contract_remains_none() -> None:
    assert _ece(torch.empty(0, dtype=torch.int64), torch.empty(0), 10) is None


def test_non_empty_dice_iou_use_monai_metric_with_repo_values() -> None:
    predicted = np.zeros((8, 9), dtype=bool)
    truth = np.zeros_like(predicted)
    predicted[2:5, 2:6] = True
    truth[3:6, 3:7] = True
    summary = _spatial_summary(predicted, truth, (1.0, 1.0), 1.0)

    predicted_tensor = torch.as_tensor(predicted)[None, None]
    truth_tensor = torch.as_tensor(truth)[None, None]
    expected_dice = DiceMetric(
        include_background=False,
        reduction="mean",
        ignore_empty=False,
    )(predicted_tensor, truth_tensor)
    intersection = int((predicted & truth).sum())
    union = int((predicted | truth).sum())
    assert summary["empty_case"] == "neither_empty"
    assert summary["dice"] == pytest.approx(float(expected_dice), abs=1e-6)
    assert summary["iou"] == pytest.approx(intersection / union, abs=1e-6)
