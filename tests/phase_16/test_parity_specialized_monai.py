from __future__ import annotations

import pytest
import torch
from monai.metrics import DiceMetric

from medfm.evaluation.specialized import adjacent_slice_consistency


def _monai_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    dice = float(
        DiceMetric(
            include_background=False,
            reduction="mean",
            ignore_empty=False,
        )(left[None, None], right[None, None])
    )
    return dice / (2.0 - dice)


def test_adjacent_slice_consistency_monai_iou_parity_preserves_volume_grouping() -> None:
    predictions = torch.zeros(2, 3, 4, 4)
    predictions[0, 0, :2, :2] = 1
    predictions[0, 1, 1:3, :2] = 1
    predictions[1, 2, 2:, 2:] = 1

    expected_scores = [
        _monai_iou(predictions[0, 0].bool(), predictions[0, 1].bool()),
        0.0,
        1.0,
        0.0,
    ]
    result = adjacent_slice_consistency(predictions, volume_ids=["study-a", "study-b"])

    assert result.value == pytest.approx(sum(expected_scores) / len(expected_scores), abs=1e-6)
    assert result.sample_count == len(expected_scores)
    assert result.metadata["volume_count"] == 2
