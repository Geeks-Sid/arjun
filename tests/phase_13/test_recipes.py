from __future__ import annotations

import pytest
import torch

from medfm.core.enums import TaskType
from medfm.recipes.phase13 import build_phase13_recipe, phase13_builders, restore_mask_to_original
from medfm.training.pipeline import TrainingPipeline


@pytest.mark.parametrize(
    "recipe_name, expected_family",
    [
        ("classification_smoke.yaml", "classification"),
        ("classification_multilabel_smoke.yaml", "classification"),
        ("segmentation_smoke.yaml", "segmentation"),
        ("segmentation_promptable_smoke.yaml", "promptable_segmentation"),
        ("native_vlm_smoke.yaml", "native_vlm"),
        ("native_structured_findings_smoke.yaml", "native_vlm"),
        ("external_vlm_smoke.yaml", "external_vlm"),
        ("external_vlm_linear_64.yaml", "external_vlm"),
        ("external_vlm_perceiver_32.yaml", "external_vlm"),
        ("external_vlm_perceiver_128.yaml", "external_vlm"),
    ],
)
def test_offline_recipe_builds_are_pinned(recipe_config, recipe_name: str, expected_family: str) -> None:
    config = recipe_config(recipe_name)
    built = build_phase13_recipe(config)

    assert built.metadata.family == expected_family
    assert built.metadata.recipe_id
    assert built.metadata.dataset_id
    assert built.metadata.dataset_revision
    assert built.metadata.preprocessing_revision
    assert built.train_data
    assert all(batch.sample_ids for batch in built.train_data)


def test_multilabel_and_promptable_task_contracts(recipe_config) -> None:
    multilabel = build_phase13_recipe(recipe_config("classification_multilabel_smoke.yaml"))
    promptable = build_phase13_recipe(recipe_config("segmentation_promptable_smoke.yaml"))

    assert multilabel.task.task_type == TaskType.MULTILABEL_CLASSIFICATION
    assert multilabel.train_data[0].task_targets["classification"].shape[1] == 2
    assert promptable.task.task_type == TaskType.PROMPTABLE_SEGMENTATION
    assert "prompt_map" in promptable.train_data[0].task_targets


def test_staged_lora_recipe_exposes_lora_groups(recipe_config) -> None:
    config = recipe_config("classification_development_lora.yaml")
    pipeline = TrainingPipeline(config, builders=phase13_builders())
    built = pipeline.build()
    groups = {entry["name"] for entry in built.optimizer.group_summary()}

    assert "task_head" in groups
    assert "vision_lora" in groups
    assert all(parameter.requires_grad for parameter in built.task.parameters())


def test_recipe_training_smokes_end_to_end(recipe_config) -> None:
    names = (
        "classification_smoke.yaml",
        "segmentation_smoke.yaml",
        "native_vlm_smoke.yaml",
        "external_vlm_smoke.yaml",
    )
    for name in names:
        config = recipe_config(name)
        built = TrainingPipeline(config, builders=phase13_builders()).build()
        result = built.trainer.train()
        assert result.success, name
        assert result.optimizer_steps == 1
        assert result.metadata["effective_batch_size"] == config.global_batch_size
        assert result.metadata["trainable_parameters"] > 0


def test_external_token_buckets_remain_static(recipe_config) -> None:
    small = build_phase13_recipe(recipe_config("external_vlm_perceiver_32.yaml"))
    large = build_phase13_recipe(recipe_config("external_vlm_perceiver_128.yaml"))

    assert small.metadata.visual_token_count == 32
    assert large.metadata.visual_token_count == 128
    assert small.metadata.bridge_type == "perceiver"
    assert large.metadata.bridge_type == "perceiver"


def test_segmentation_restores_original_coordinates() -> None:
    mask = torch.ones(1, 1, 2, 3)
    restored = restore_mask_to_original(mask, original_size=(8, 10), crop_box=(2, 1, 8, 7))

    assert restored.shape == (1, 1, 8, 10)
    assert torch.all(restored[..., :1, :] == 0)
    assert torch.all(restored[..., 1:7, 2:8] == 1)
    assert torch.all(restored[..., 7:, :] == 0)
