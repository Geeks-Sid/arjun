from __future__ import annotations

from pathlib import Path

import pytest
import torch

from medfm.core.batch import BucketKind
from medfm.core.enums import Modality, TaskType
from medfm.recipes.phase15 import (
    build_phase15_recipe,
    contrastive_alignment_loss,
    evaluate_wsi_tile_counts_and_magnification,
    pathology_classification_metrics,
    patient_disjoint_split,
    select_wsi_visual_tokens,
)
from medfm.training.pipeline import TrainingPipeline
from medfm.recipes.phase15 import phase15_builders


ALL_RECIPE_NAMES = tuple(sorted(path.name for path in Path(__file__).parents[2].joinpath("configs/recipes/pathology").glob("*.yaml")))


@pytest.mark.parametrize("name", ALL_RECIPE_NAMES)
def test_pathology_recipes_are_pinned_and_bounded(phase15_config, name: str) -> None:
    built = build_phase15_recipe(phase15_config(name))
    metadata = built.metadata
    assert built.train_data
    assert metadata.dataset_revision
    assert metadata.preprocessing_revision
    assert metadata.model_revision
    assert metadata.max_tiles_per_slide > 0
    assert all(count <= metadata.max_tiles_per_slide for count in metadata.sampled_tile_counts)
    assert metadata.shard_unit == "slide"
    assert metadata.actual_tile_count_logging
    assert metadata.backend_observability["padded_tiles_excluded_from_loss_and_metrics"]
    serialized = metadata.to_dict()
    assert serialized["slide_reader_revision"] == "phase03-slide-reader-v1"
    assert serialized["tile_index_revision"] == "phase04-tile-index-v1"
    assert serialized["failure_rates"]["corrupt_tiles"] == 0.0
    assert metadata.backend_observability["tile_encoder_status"] == "offline_contract"


def test_tile_stage_matrix_and_contrastive_alignment(phase15_config) -> None:
    for name, stage in (
        ("tile_classification_hoptimus_linear.yaml", "1"),
        ("tile_classification_mlp.yaml", "2"),
        ("tile_classification_vision_lora.yaml", "3"),
        ("tile_classification_contrastive.yaml", "4"),
    ):
        built = build_phase15_recipe(phase15_config(name))
        assert built.metadata.stage == stage
        assert built.train_data[0].modality is Modality.PATHOLOGY_TILE
        if stage == "4":
            assert "text_embeddings" in built.train_data[0].task_targets
            loss = contrastive_alignment_loss(
                torch.eye(3),
                torch.eye(3),
                valid_mask=torch.tensor([True, True, False]),
            )
            assert torch.isfinite(loss)


def test_wsi_aggregators_keep_coordinates_and_masks(phase15_config) -> None:
    for name, aggregator in (
        ("wsi_classification_smoke.yaml", "mean"),
        ("wsi_classification_attention_mil.yaml", "attention_mil"),
        ("wsi_classification_gated_attention_mil.yaml", "gated_attention_mil"),
        ("wsi_classification_transformer.yaml", "transformer"),
    ):
        built = build_phase15_recipe(phase15_config(name))
        batch = built.train_data[0]
        assert built.metadata.aggregator == aggregator
        assert batch.tile_coordinates is not None
        assert batch.bucket.kind is BucketKind.VISUAL_TOKENS
        mask = batch.task_targets["tile_mask"]
        counts = batch.task_targets["actual_tile_count"]
        assert torch.equal(mask.sum(dim=1).to(dtype=counts.dtype), counts)
        assert built.metadata.eval_selector == "grid"


def test_wsi_vlm_fixed_tokens_coordinates_and_ablations(phase15_config) -> None:
    built = build_phase15_recipe(phase15_config("wsi_vlm_cached_smoke.yaml"))
    batch = built.train_data[0]
    model = built.model
    image = model.forward_mode(batch, mode="image")
    no_slide = model.forward_mode(batch, mode="none")
    shuffled_tiles = model.forward_mode(batch, mode="shuffle_tiles")
    shuffled_coordinates = model.forward_mode(batch, mode="shuffle_coordinates")
    assert image.visual_tokens.tokens.shape[1] == 32
    assert image.visual_tokens.token_mask.all()
    assert not no_slide.visual_tokens.token_mask.any()
    assert not torch.equal(image.language.logits, no_slide.language.logits)
    assert not torch.equal(image.language.logits, shuffled_tiles.language.logits)
    assert not torch.equal(image.source_coordinates, shuffled_coordinates.source_coordinates)
    assert image.language.auxiliary["selector_revision"] == "phase15-selector-v1"
    assert all(len(rows) <= 4 for rows in image.evidence_tiles)

    payload = model.evidence_json(slide_id="slide-0")
    from medfm.recipes.pathology_stitching import validate_evidence_json

    assert validate_evidence_json(payload, slide_shape=(64, 64)) == []


def test_patient_split_and_tile_count_magnification_evaluation() -> None:
    records = [
        {"slide_id": "s0", "patient_id": "p0"},
        {"slide_id": "s1", "patient_id": "p0"},
        {"slide_id": "s2", "patient_id": "p1"},
        {"slide_id": "s3", "patient_id": "p2"},
        {"slide_id": "s4", "patient_id": "p3"},
    ]
    split = patient_disjoint_split(records, seed=7)
    groups = {split.patient_by_slide[slide] for slide in split.train}
    assert groups.isdisjoint({split.patient_by_slide[slide] for slide in split.validation})
    assert groups.isdisjoint({split.patient_by_slide[slide] for slide in split.test})
    rows = evaluate_wsi_tile_counts_and_magnification(
        [0, 1, 0, 1],
        {(4, "10x"): [0.1, 0.8, 0.2, 0.7], (8, "20x"): [0.2, 0.9, 0.3, 0.8]},
        patient_ids=["p0", "p1", "p2", "p3"],
        slide_ids=["s0", "s1", "s2", "s3"],
    )
    assert [(row["tile_count"], row["magnification"]) for row in rows] == [(4, "10x"), (8, "20x")]
    assert all("slide/auroc" in row["metrics"] for row in rows)


def test_pathology_metrics_exclude_padded_entries() -> None:
    metrics = pathology_classification_metrics(
        [0, 1, 1],
        [0.1, 0.9, 0.2],
        patient_ids=["p0", "p1", "p2"],
        slide_ids=["s0", "s1", "s2"],
        scanner_ids=["a", "a", "b"],
        valid_mask=[True, True, False],
    )
    assert metrics["tile/auroc"].sample_count == 2
    assert metrics["slide/auroc"].unit == "per_slide"
    assert metrics["patient/auroc"].unit == "per_patient"
    assert "scanner/subgroups" in metrics


@pytest.mark.parametrize(
    "name",
    [
        "tile_classification_hoptimus_linear.yaml",
        "wsi_classification_smoke.yaml",
        "wsi_vlm_cached_smoke.yaml",
        "segmentation_smoke.yaml",
    ],
)
def test_one_step_pipeline_for_each_recipe_family(phase15_config, name: str) -> None:
    config = phase15_config(name)
    built = TrainingPipeline(config, builders=phase15_builders()).build()
    result = built.trainer.train()
    assert result.success
    assert result.optimizer_steps == 1

def test_phase15_training_checkpoint_resume(phase15_config) -> None:
    config = phase15_config("wsi_classification_smoke.yaml")
    built = TrainingPipeline(config, builders=phase15_builders()).build()
    result = built.trainer.train()
    checkpoint = Path(result.checkpoint) if result.checkpoint is not None else None
    assert result.success and checkpoint is not None and checkpoint.exists()
    resumed = TrainingPipeline(config, builders=phase15_builders()).build()
    state = resumed.trainer.resume(checkpoint)
    assert state.global_step == 1


def test_task_types_for_language_variants(phase15_config) -> None:
    expected = {
        "wsi_vlm_report.yaml": TaskType.REPORT_GENERATION,
        "wsi_vlm_vqa.yaml": TaskType.VISUAL_QUESTION_ANSWERING,
        "wsi_vlm_retrieval.yaml": TaskType.IMAGE_TEXT_RETRIEVAL,
    }
    for name, task_type in expected.items():
        assert build_phase15_recipe(phase15_config(name)).task.task_type is task_type


def test_selector_budget_is_fixed_and_padding_is_masked() -> None:
    embeddings = [torch.arange(15, dtype=torch.float32).reshape(5, 3), torch.ones(2, 3)]
    records = [
        [
            {"tile_id": f"a{i}", "x": i, "y": 0, "width": 4, "height": 4}
            for i in range(5)
        ],
        [{"tile_id": f"b{i}", "x": i, "y": 1, "width": 4, "height": 4} for i in range(2)],
    ]
    selected = select_wsi_visual_tokens(embeddings, records, budget=None, selector="grid")
    assert selected.tokens.shape[1] == 64
    assert selected.mask[0].sum() == 5
    assert selected.mask[1].sum() == 2
    assert selected.actual_counts == (5, 2)
