from __future__ import annotations

import pytest
import torch

from medfm.evaluation import (
    adjacent_slice_consistency,
    compare_native_3d_and_slice_sequence,
    pathology_evaluation_metrics,
    selective_risk,
    small_lesion_sensitivity,
    sweep_pathology_sampling,
    uncertainty_metrics,
)
from medfm.evaluation.metrics import MetricValue


def test_3d_and_slice_sequence_helpers_keep_strategies_separate() -> None:
    native = {"auroc": MetricValue("auroc", 0.9, "per_scan", 2)}
    sequence = {"auroc": MetricValue("auroc", 0.8, "per_scan", 2)}
    comparison = compare_native_3d_and_slice_sequence(native, sequence)
    assert comparison["native_3d_minus_slice_sequence/auroc"].value == pytest.approx(0.1)
    masks = torch.zeros(1, 3, 4, 4)
    masks[:, :, 1:3, 1:3] = 10
    assert adjacent_slice_consistency(masks).value == 1.0


def test_small_lesion_and_pathology_sweeps_report_declared_units() -> None:
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., 2:3, 2:3, 2:3] = 1
    logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    assert small_lesion_sensitivity(logits, target, max_target_voxels=2).value == 1.0
    metrics = pathology_evaluation_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        tile_ids=["t0", "t1", "t2", "t3"],
        slide_ids=["s0", "s0", "s1", "s1"],
        patient_ids=["p0", "p0", "p1", "p1"],
        site_ids=["a", "a", "b", "b"],
    )
    assert metrics["tile/auroc"].unit == "per_tile"
    assert metrics["slide/auroc"].unit == "per_slide"
    assert metrics["patient/auroc"].unit == "per_patient"
    rows = sweep_pathology_sampling([0, 1], {(4, "10x"): [0.1, 0.9]})
    assert rows[0]["tile_count"] == 4


def test_uncertainty_selective_risk_is_explicit() -> None:
    metrics = uncertainty_metrics([0.1, 0.9], labels=[0, 1], unit="per_patient")
    assert metrics["mean_entropy"].unit == "per_patient"
    assert selective_risk([0.1, 0.9], labels=[0, 1])["selective_risk@0.5"].value == 0.0
