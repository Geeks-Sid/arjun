from __future__ import annotations

from pathlib import Path

import pytest
import torch

from medfm.core.enums import Modality, TaskType
from medfm.recipes.phase14 import (
    build_phase14_recipe,
    phase14_builders,
    restore_volume_mask_to_original,
    select_volume_input_policy,
)
from medfm.training.pipeline import TrainingPipeline


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("classification_ct_fm_cuda.yaml", "classification"),
        ("classification_triad_mri.yaml", "classification"),
        ("segmentation_ct_fm_baseline.yaml", "segmentation"),
        ("native_vlm_structured_findings.yaml", "native_3d_vlm"),
        ("native_vlm_cached_tpu_bf16.yaml", "native_3d_vlm"),
        ("slice_sequence_vlm_uniform.yaml", "slice_sequence_vlm"),
        ("language_conditioned_segmentation.yaml", "language_conditioned_segmentation"),
    ],
)
def test_phase14_recipe_families_are_pinned(phase14_config, name: str, family: str) -> None:
    built = build_phase14_recipe(phase14_config(name))
    assert built.metadata.family == family
    assert built.metadata.recipe_id
    assert built.metadata.dataset_revision
    assert built.metadata.preprocessing_revision
    assert built.metadata.shape_buckets == ((16, 16, 16),)
    assert built.metadata.global_batch_size == 2 if "tpu" not in name else 1
    assert built.metadata.limitations


def test_native_3d_classification_keeps_volume_semantics(phase14_config) -> None:
    built = build_phase14_recipe(phase14_config("classification_triad_mri.yaml"))
    batch = built.train_data[0]
    output = built.model(batch)
    assert batch.modality is Modality.MRI_3D
    assert batch.pixel_values is not None and batch.pixel_values.ndim == 5
    assert output.pooled_embedding is not None and output.pooled_embedding.shape == (2, 32)
    assert output.spatial_tokens is not None and output.spatial_tokens.ndim == 3
    assert built.task.task_type is TaskType.BINARY_CLASSIFICATION


def test_segmentation_records_positive_sampling_and_restores_space(phase14_config) -> None:
    built = build_phase14_recipe(phase14_config("segmentation_ct_fm_baseline.yaml"))
    batch = built.train_data[0]
    assert batch.modality is Modality.CT_3D
    assert 0.0 <= float(batch.task_targets["positive_patch_rate"]) <= 1.0
    assert batch.task_targets["voxel_mask"].shape == (2, 16, 16, 16)
    patch = torch.ones(1, 1, 2, 3, 4)
    restored = restore_volume_mask_to_original(patch, original_size=(6, 7, 8), crop_origin=(2, 1, 3))
    assert restored.shape == (1, 1, 6, 7, 8)
    assert torch.all(restored[..., 2:4, 1:4, 3:7] == 1)
    assert torch.all(restored[..., :2] == 0)


def test_native_vlm_exposes_coordinates_cache_and_ablations(phase14_config) -> None:
    built = build_phase14_recipe(phase14_config("native_vlm_cached_tpu_bf16.yaml"))
    batch = built.train_data[0]
    result = built.model.forward_mode(batch, mode="image")
    assert result.visual_tokens is not None
    assert result.visual_tokens.tokens.shape == (1, 32, 32)
    assert result.source_coordinates is not None and result.source_coordinates.shape[-1] == 3
    assert result.language.auxiliary["cached_spatial_tokens"] is True
    no_image = built.model.forward_mode(batch, mode="none")
    shuffled = built.model.forward_mode(batch, mode="shuffle")
    assert no_image.language.logits.shape == result.language.logits.shape
    assert shuffled.language.logits.shape == result.language.logits.shape
    generated = built.model.generate(batch)
    assert len(generated.texts) == 1
    assert generated.token_ids is not None and generated.token_ids.ndim == 2


def test_slice_sequence_is_distinct_and_preserves_selector_metadata(phase14_config) -> None:
    built = build_phase14_recipe(phase14_config("slice_sequence_vlm_uniform.yaml"))
    batch = built.train_data[0]
    assert batch.modality is Modality.MULTI_IMAGE_2D
    assert batch.pixel_values is not None and batch.pixel_values.shape == (2, 4, 1, 32, 32)
    records = batch.task_targets["slice_metadata"]
    assert len(records) == 2 and len(records[0]) == 4
    assert {"index", "normalized_z", "physical_z_mm", "series_order", "window"} <= set(records[0][0])
    assert built.metadata.selector_revision
    assert built.metadata.slice_count == 4


def test_language_conditioned_segmentation_has_fixed_query_contract(phase14_config) -> None:
    built = build_phase14_recipe(phase14_config("language_conditioned_segmentation.yaml"))
    batch = built.train_data[0]
    output = built.model(batch)
    loss = built.task.compute_loss(output, batch)
    assert built.task.task_type is TaskType.LANGUAGE_CONDITIONED_SEGMENTATION
    assert output["text_embeddings"].shape == (2, 8, 32)
    assert output["query_mask"].shape == (2,)
    assert torch.isfinite(loss.total)


def test_phase14_pipeline_one_batch_smoke_and_checkpoint_resume(phase14_config) -> None:
    config = phase14_config("classification_triad_mri.yaml")
    built = TrainingPipeline(config, builders=phase14_builders()).build()
    result = built.trainer.train()
    assert result.success
    assert result.optimizer_steps == 1
    assert result.metadata["effective_batch_size"] == config.global_batch_size
    checkpoint = Path(result.checkpoint) if result.checkpoint is not None else None
    assert checkpoint is not None and checkpoint.exists()
    resumed = TrainingPipeline(config, builders=phase14_builders()).build()
    state = resumed.trainer.resume(checkpoint)
    assert state.global_step == 1


def test_volume_policy_is_fingerprint_driven_but_static() -> None:
    policy = select_volume_input_policy(
        {
            "input_strategy": "global_local",
            "recommended_shape_buckets": [
                {"kind": "3d_patch", "shape": [96, 96, 96]},
                {"kind": "images", "shape": [16, 16]},
            ],
            "global_shape": [128, 128, 128],
            "local_shape": [96, 96, 96],
        }
    )
    assert policy.strategy == "global_local"
    assert policy.shape_buckets == ((96, 96, 96),)
    assert policy.global_shape == (128, 128, 128)
    assert policy.local_shape == (96, 96, 96)
