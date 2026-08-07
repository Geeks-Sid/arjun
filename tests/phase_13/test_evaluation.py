from __future__ import annotations

import torch

from medfm.evaluation import classification_metrics, run_visual_dependence_ablation, segmentation_metrics
from medfm.recipes.phase13 import build_phase13_recipe, make_phase13_artifact


def test_classification_metrics_report_clinical_units_and_subgroups() -> None:
    metrics = classification_metrics(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        group_ids=["site_a", "site_a", "site_b", "site_b"],
        unit="per_patient",
    )

    assert metrics["auroc"].value == 1.0
    assert metrics["auprc"].value == 1.0
    assert metrics["auroc"].unit == "per_patient"
    assert metrics["auroc"].sample_count == 4
    assert set(metrics["subgroups"].metadata["groups"]) == {"site_a", "site_b"}


def test_segmentation_metrics_report_dice_surface_and_false_positives() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[..., 1:3, 1:3] = 1
    logits = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
    metrics = segmentation_metrics(logits, target)

    assert metrics["dice/class_0"].value > 0.99
    assert metrics["surface_dice/class_0"].value > 0.99
    assert metrics["sensitivity/class_0"].value > 0.99
    assert metrics["false_positives_per_image/class_0"].value == 0.0


def test_visual_ablation_exposes_masking_and_shuffle_modes(recipe_config) -> None:
    config = recipe_config("external_vlm_smoke.yaml")
    built = build_phase13_recipe(config)
    result = run_visual_dependence_ablation(built.model, built.task, built.train_data[0])

    assert result.criteria["requires_no_image_degradation"] in {True, False}
    assert result.criteria["requires_shuffled_degradation"] in {True, False}
    assert result.to_dict()["criteria"] == result.criteria
    output = built.model.forward_mode(built.train_data[0], mode="none")
    assert output.mode == "none"
    assert not bool(output.visual_tokens.token_mask.any())


def test_phase13_artifact_requires_provenance_and_limitations(recipe_config) -> None:
    config = recipe_config("classification_smoke.yaml")
    metrics = classification_metrics([0, 1], [0.2, 0.8])
    artifact = make_phase13_artifact(config, metrics)

    payload = artifact.to_dict()
    assert payload["recipe_id"] == "classification-smoke"
    assert payload["reproducibility"]["config_hash"] == config.config_hash()
    assert payload["clinical_units"]["auroc"] == "per_patient"
    assert payload["limitations"]
