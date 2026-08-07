from __future__ import annotations

import torch

from medfm.recipes.phase14 import (
    benchmark_slice_token_budgets,
    language_conditioned_segmentation_metrics,
    native_3d_classification_metrics,
    native_3d_segmentation_metrics,
    native_vlm_grounding_metrics,
    sliding_window_predict,
)


def test_native_3d_classification_reports_patient_and_study_units() -> None:
    metrics = native_3d_classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        patient_ids=["p0", "p0", "p1", "p1"],
        study_ids=["s0", "s1", "s2", "s3"],
    )
    assert metrics["patient/auroc"].unit == "per_patient"
    assert metrics["study/auroc"].unit == "per_study"
    assert metrics["patient/auroc"].sample_count == 2


def test_native_3d_segmentation_reports_surface_lesion_and_volume_metrics() -> None:
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:5, 2:5, 2:5] = 1
    logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    metrics = native_3d_segmentation_metrics(logits, target, spacing_mm=(2.0, 1.0, 1.0))
    assert metrics["dice/class_0"].value > 0.99
    assert metrics["hd95/class_0"].unit == "per_scan"
    assert metrics["lesion_recall/class_0"].value == 1.0
    assert metrics["false_positives_per_scan/class_0"].value == 0.0
    assert metrics["volume_error_mm3/class_0"].unit == "mm3_per_scan"


def test_sliding_window_gaussian_blending_covers_every_voxel() -> None:
    volume = torch.ones(1, 1, 6, 6, 6)

    def predictor(crop: torch.Tensor) -> torch.Tensor:
        return crop * 3.0

    output = sliding_window_predict(
        volume,
        predictor,
        window_shape=(4, 4, 4),
        overlap=0.5,
        blend_mode="gaussian",
    )
    assert output.shape == volume.shape
    assert torch.allclose(output, torch.full_like(output, 3.0), atol=1e-5)


def test_language_segmentation_separates_mask_and_query_grounding() -> None:
    target = torch.zeros(2, 1, 4, 4, 4)
    target[0, :, 1:3, 1:3, 1:3] = 1
    logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    metrics = language_conditioned_segmentation_metrics(
        logits,
        target,
        query_mask=torch.tensor([True, False]),
        query_grounding=torch.tensor([1.0, 0.0]),
    )
    assert metrics["mask_accuracy"].unit == "per_query"
    assert metrics["mask_accuracy"].sample_count == 1
    assert metrics["query_grounding"].value == 1.0
    query_target = target.unsqueeze(1).repeat(1, 2, 1, 1, 1, 1)
    query_logits = torch.where(query_target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    query_metrics = language_conditioned_segmentation_metrics(
        query_logits,
        query_target,
        query_mask=torch.tensor([[True, False], [True, True]]),
        query_grounding=torch.tensor([[1.0, 0.0], [1.0, 0.5]]),
    )
    assert query_metrics["mask_accuracy"].sample_count == 3


def test_vlm_ablation_and_slice_budget_contracts() -> None:
    metrics = native_vlm_grounding_metrics(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[0.5, 1.0]]),
        torch.tensor([[0.8, 1.5]]),
    )
    assert metrics["image_dependence"].unit == "mean_logit_delta"
    rows = benchmark_slice_token_budgets(slice_buckets=(4, 8), visual_token_buckets=(32,), text_token_buckets=(8,))
    assert len(rows) == 2
    assert all(row["selector_on_host"] for row in rows)
    assert all(row["within_48gb_cap"] for row in rows)
