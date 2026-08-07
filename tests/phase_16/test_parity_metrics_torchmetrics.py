from __future__ import annotations

import warnings

import pytest
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from medfm.evaluation import metrics

_PARITY_FIXTURES = [
    # Unique scores exercise the ordinary ROC/AP path.
    (torch.tensor([0, 1, 0, 1]), torch.tensor([0.1, 0.9, 0.2, 0.8])),
    (torch.tensor([0, 1, 0, 1]), torch.tensor([0.9, 0.1, 0.8, 0.2])),
    (torch.tensor([1, 0, 1, 0, 1]), torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1])),
    # Ties now follow torchmetrics sample order (ADR-0013).
    (torch.tensor([0, 1]), torch.tensor([0.5, 0.5])),
    (torch.tensor([1, 0]), torch.tensor([0.5, 0.5])),
    (torch.tensor([1, 0, 1, 0]), torch.tensor([0.5, 0.5, 0.5, 0.5])),
]


def _torchmetrics_value(
    metric_type: type[BinaryAUROC] | type[BinaryAveragePrecision],
    labels: torch.Tensor,
    scores: torch.Tensor,
    *,
    update: bool,
) -> float:
    metric = metric_type()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if update:
            metric.update(scores, labels)
            value = metric.compute()
        else:
            value = metric(scores, labels)
    return float(value)


@pytest.mark.parametrize(("labels", "scores"), _PARITY_FIXTURES)
def test_metrics_auroc_auprc_match_torchmetrics(labels: torch.Tensor, scores: torch.Tensor) -> None:
    # ADR-0013: _auroc/_auprc now delegate to torchmetrics; values must match
    # exactly (tie ordering follows torchmetrics).
    candidates = (
        (metrics._auroc, BinaryAUROC),
        (metrics._auprc, BinaryAveragePrecision),
    )
    for legacy, metric_type in candidates:
        current = legacy(labels, scores)
        forward = _torchmetrics_value(metric_type, labels, scores, update=False)
        updated = _torchmetrics_value(metric_type, labels, scores, update=True)
        assert forward == pytest.approx(updated, abs=1e-7)
        if current is not None:
            assert current == pytest.approx(forward, abs=1e-6)


@pytest.mark.parametrize(
    ("labels", "scores", "legacy_auroc", "legacy_auprc", "candidate_auprc"),
    [
        (torch.tensor([1, 1, 1]), torch.tensor([0.1, 0.5, 0.9]), None, 1.0, 1.0),
        (torch.tensor([0, 0, 0]), torch.tensor([0.1, 0.5, 0.9]), None, None, 0.0),
    ],
)
def test_torchmetrics_degenerate_outputs_do_not_replace_none_contract(
    labels: torch.Tensor,
    scores: torch.Tensor,
    legacy_auroc: float | None,
    legacy_auprc: float | None,
    candidate_auprc: float,
) -> None:
    assert metrics._auroc(labels, scores) == legacy_auroc
    assert metrics._auprc(labels, scores) == legacy_auprc
    assert _torchmetrics_value(BinaryAUROC, labels, scores, update=False) == 0.0
    assert _torchmetrics_value(BinaryAveragePrecision, labels, scores, update=False) == candidate_auprc
