"""Phase 04 collator tests: contracts, buckets, static shapes, final batches."""

import warnings

import pytest
import torch

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.core.enums import Modality
from medfm.core.sample import SpatialMetadata
from medfm.data.collators import (
    BucketPlan,
    ClassificationCollator,
    ContrastiveCollator,
    FinalBatchPolicy,
    MultiImageVLCollator,
    MultitaskCollator,
    Segmentation2DCollator,
    Segmentation3DCollator,
    VolumeVLCollator,
    WSIVLCollator,
)
from medfm.data.errors import CollatorError

# BucketPlan warns the first time each bucket shape is exercised; the dedicated
# warning test asserts that explicitly, everywhere else it is noise.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _plan_2d() -> BucketPlan:
    return BucketPlan(
        buckets={
            BucketKind.IMAGE_2D: (
                BucketId(BucketKind.IMAGE_2D, (64, 64)),
                BucketId(BucketKind.IMAGE_2D, (128, 128)),
            ),
            BucketKind.TEXT_TOKENS: (BucketId(BucketKind.TEXT_TOKENS, (16,)),),
        },
        mode="static",
    )


def _xray_example(sample_id: str, hw: tuple[int, int] = (32, 32), label: int = 1) -> dict:
    return {
        "sample_id": sample_id,
        "modality": Modality.XRAY_2D,
        "image": torch.rand(1, *hw),
        "label": torch.tensor(label),
    }


def _text(sample: dict, length: int = 10) -> dict:
    sample = dict(sample)
    sample["input_ids"] = torch.arange(length, dtype=torch.int64)
    labels = torch.full((length,), -100, dtype=torch.int64)
    labels[length // 2 :] = torch.arange(length // 2, dtype=torch.int64)  # assistant span
    sample["lm_labels"] = labels
    return sample


def _ct_example(sample_id: str, dhw: tuple[int, int, int] = (4, 16, 16)) -> dict:
    shape = dhw
    affine = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    return {
        "sample_id": sample_id,
        "modality": Modality.CT_3D,
        "image": torch.rand(1, *dhw),
        "spatial": SpatialMetadata(
            original_shape=shape,
            current_shape=shape,
            affine=affine,
            spacing_mm=(1.0, 1.0, 1.0),
            orientation="RAS",
        ),
    }


# ---------------------------------------------------------------------------
# BucketPlan policy
# ---------------------------------------------------------------------------


def test_bucket_plan_assigns_smallest_covering_bucket():
    plan = _plan_2d()
    assert plan.assign(BucketKind.IMAGE_2D, (32, 48)).shape == (64, 64)
    assert plan.assign(BucketKind.IMAGE_2D, (100, 65)).shape == (128, 128)


def test_bucket_plan_out_of_bucket_error_names_unplanned_compilation():
    plan = _plan_2d()
    with pytest.raises(CollatorError, match="unplanned TPU compilation"):
        plan.assign(BucketKind.IMAGE_2D, (256, 256))


def test_bucket_plan_pad_to_max_falls_back_to_largest():
    plan = BucketPlan(
        buckets={BucketKind.IMAGE_2D: (BucketId(BucketKind.IMAGE_2D, (64, 64)),)},
        mode="static",
        out_of_bucket_policy="pad_to_max",
    )
    assert plan.assign(BucketKind.IMAGE_2D, (256, 32)).shape == (64, 64)


def test_bucket_plan_bounds_bucket_count():
    with pytest.raises(CollatorError, match="max_buckets"):
        BucketPlan(
            buckets={
                BucketKind.IMAGE_2D: tuple(BucketId(BucketKind.IMAGE_2D, (16 * i, 16)) for i in range(1, 6)),
            },
            max_buckets=3,
        )


def test_bucket_plan_first_exercise_warns():
    plan = BucketPlan(
        buckets={BucketKind.IMAGE_2D: (BucketId(BucketKind.IMAGE_2D, (64, 64)),)},
        mode="static",
    )
    with pytest.warns(UserWarning, match="exercised for the first time"):
        plan.assign(BucketKind.IMAGE_2D, (32, 32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan.assign(BucketKind.IMAGE_2D, (48, 48))  # same bucket: no second warning


def test_bucket_plan_config_roundtrip_and_independent_train_val_hashes():
    plan = _plan_2d()
    clone = BucketPlan.from_config(plan.to_config())
    assert clone.to_config() == plan.to_config()
    assert clone.config_hash() == plan.config_hash()
    val_config = plan.to_config()
    val_config["buckets"]["IMAGE_2D"] = [[128, 128]]  # validation: separately configured
    val_plan = BucketPlan.from_config(val_config)
    assert val_plan.config_hash() != plan.config_hash()


# ---------------------------------------------------------------------------
# Classification collator
# ---------------------------------------------------------------------------


def test_classification_happy_path_dynamic():
    collator = ClassificationCollator(Modality.XRAY_2D)
    batch = collator([_xray_example("b", (32, 48), 0), _xray_example("a", (40, 32), 1)])
    assert isinstance(batch, MedicalBatch)
    assert batch.pixel_values.shape == (2, 1, 40, 48)  # per-batch max
    assert batch.sample_ids == ["b", "a"]  # input order preserved
    assert batch.labels.tolist() == [0, 1]
    assert batch.image_mask.tolist() == [True, True]
    assert batch.bucket is None


def test_classification_static_repeated_batches_identical_shapes():
    collator = ClassificationCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    shapes = []
    for sizes in [((32, 48), (50, 60)), ((10, 10), (64, 64)), ((33, 33), (20, 55))]:
        batch = collator([_xray_example(f"s{i}", size) for i, size in enumerate(sizes)])
        shapes.append(tuple(batch.pixel_values.shape))
        assert batch.bucket is not None and batch.bucket.kind is BucketKind.IMAGE_2D
        assert batch.image_mask.all()
    assert shapes == [(2, 1, 64, 64)] * 3  # every batch lands on the same declared bucket
    bigger = collator([_xray_example("big", (100, 70))])
    assert tuple(bigger.pixel_values.shape) == (1, 1, 128, 128)
    assert bigger.bucket.shape == (128, 128)


def test_classification_out_of_bucket_sample_rejected_in_static_mode():
    collator = ClassificationCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    with pytest.raises(CollatorError, match="unplanned TPU compilation"):
        collator([_xray_example("huge", (512, 512))])


def test_classification_3d_volume_bucket():
    plan = BucketPlan(
        buckets={BucketKind.VOLUME_3D: (BucketId(BucketKind.VOLUME_3D, (8, 32, 32)),)},
        mode="static",
    )
    collator = ClassificationCollator(Modality.CT_3D, plan, static=True)
    batch = collator([_ct_example("v1", (4, 16, 16)), _ct_example("v2", (6, 20, 24))])
    assert batch.pixel_values.shape == (2, 1, 8, 32, 32)
    assert batch.bucket.kind is BucketKind.VOLUME_3D
    assert batch.spatial_metadata[0].spacing_mm == (1.0, 1.0, 1.0)  # metadata preserved


def test_classification_rejects_mixed_modalities():
    collator = ClassificationCollator(Modality.XRAY_2D)
    other = {"sample_id": "ct", "modality": Modality.CT_2D_SLICE, "image": torch.rand(1, 32, 32)}
    with pytest.raises(CollatorError, match="mixed incompatible modalities"):
        collator([_xray_example("x"), other])


# ---------------------------------------------------------------------------
# Segmentation collators
# ---------------------------------------------------------------------------


def test_segmentation_2d_stacks_masks_with_matching_spatial_dims():
    collator = Segmentation2DCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    examples = []
    for index, size in enumerate([(32, 32), (48, 40)]):
        examples.append(
            {
                "sample_id": f"s{index}",
                "modality": Modality.XRAY_2D,
                "image": torch.rand(1, *size),
                "mask": torch.randint(0, 2, (2, *size)).float(),
            }
        )
    batch = collator(examples)
    assert batch.pixel_values.shape == (2, 1, 64, 64)
    segmentation = batch.task_targets["segmentation"]
    assert segmentation.shape == (2, 2, 64, 64)  # [B, K, H, W] matching pixel spatial dims
    # Padded regions are zeroed in both image and mask.
    assert float(segmentation[0, :, 32:, :].abs().sum()) == 0.0
    assert float(segmentation[0, :, :, 32:].abs().sum()) == 0.0


def test_segmentation_3d_volume():
    plan = BucketPlan(
        buckets={BucketKind.VOLUME_3D: (BucketId(BucketKind.VOLUME_3D, (8, 32, 32)),)},
        mode="static",
    )
    collator = Segmentation3DCollator(Modality.MRI_3D, plan, static=True)
    example = {
        "sample_id": "m1",
        "modality": Modality.MRI_3D,
        "image": torch.rand(1, 4, 16, 16),
        "mask": torch.randint(0, 2, (1, 4, 16, 16)).float(),
    }
    batch = collator([example])
    assert batch.task_targets["segmentation"].shape == (1, 1, 8, 32, 32)


def test_segmentation_rejects_mask_shape_mismatch():
    collator = Segmentation2DCollator(Modality.XRAY_2D)
    example = {
        "sample_id": "bad",
        "modality": Modality.XRAY_2D,
        "image": torch.rand(1, 32, 32),
        "mask": torch.rand(1, 16, 16),
    }
    with pytest.raises(CollatorError, match="spatial dims"):
        collator([example])


# ---------------------------------------------------------------------------
# Contrastive collator
# ---------------------------------------------------------------------------


def test_contrastive_pairs_images_with_padded_text():
    collator = ContrastiveCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    examples = [_text(_xray_example("a", (32, 32)), 10), _text(_xray_example("b", (40, 40)), 6)]
    batch = collator(examples)
    assert batch.pixel_values.shape == (2, 1, 64, 64)
    assert batch.input_ids.shape == (2, 16)  # TEXT_TOKENS bucket
    assert batch.attention_mask[0].tolist() == [True] * 10 + [False] * 6
    assert batch.attention_mask[1].tolist() == [True] * 6 + [False] * 10
    assert batch.bucket.kind is BucketKind.TEXT_TOKENS
    # LM labels padded with -100: padding is never supervised.
    assert (batch.task_targets["lm_labels"][:, 10:] == -100).all()


def test_contrastive_rejects_mixed_modalities():
    collator = ContrastiveCollator(Modality.XRAY_2D)
    other = _text({"sample_id": "m", "modality": Modality.MRI_2D_SLICE, "image": torch.rand(1, 32, 32)})
    with pytest.raises(CollatorError, match="mixed incompatible modalities"):
        collator([_text(_xray_example("x")), other])


# ---------------------------------------------------------------------------
# Multi-image VL collator
# ---------------------------------------------------------------------------


def _multi_image_plan() -> BucketPlan:
    return BucketPlan(
        buckets={
            BucketKind.IMAGE_2D: (BucketId(BucketKind.IMAGE_2D, (64, 64)),),
            BucketKind.MULTI_IMAGE: (BucketId(BucketKind.MULTI_IMAGE, (4,)),),
            BucketKind.TEXT_TOKENS: (BucketId(BucketKind.TEXT_TOKENS, (16,)),),
            BucketKind.VISUAL_TOKENS: (BucketId(BucketKind.VISUAL_TOKENS, (8,)),),
        },
        mode="static",
    )


def _mi_example(sample_id: str, count: int, hw: tuple[int, int] = (32, 32)) -> dict:
    return _text(
        {
            "sample_id": sample_id,
            "modality": Modality.MULTI_IMAGE_2D,
            "images": [torch.rand(1, *hw) for _ in range(count)],
        }
    )


def test_multi_image_vl_pads_image_count_with_mask():
    collator = MultiImageVLCollator(Modality.MULTI_IMAGE_2D, _multi_image_plan(), static=True)
    batch = collator([_mi_example("a", 2), _mi_example("b", 4), _mi_example("c", 1)])
    assert batch.pixel_values.shape == (3, 4, 1, 64, 64)  # MULTI_IMAGE bucket count
    assert batch.image_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ]
    assert batch.bucket.kind is BucketKind.MULTI_IMAGE
    assert batch.input_ids.shape == (3, 16)


def test_multi_image_vl_visual_token_limit_validated():
    collator = MultiImageVLCollator(Modality.MULTI_IMAGE_2D, _multi_image_plan(), static=True)
    ok = _mi_example("ok", 2)
    ok["visual_tokens"] = torch.rand(5, 8)
    batch = collator([ok])
    assert batch.task_targets["visual_tokens"].shape == (1, 8, 8)
    assert batch.task_targets["visual_token_mask"].tolist() == [[True] * 5 + [False] * 3]
    too_many = _mi_example("overflow", 2)
    too_many["visual_tokens"] = torch.rand(9, 8)  # exceeds the (8,) bucket
    with pytest.raises(CollatorError, match="unplanned TPU compilation"):
        collator([too_many])


def test_text_token_limit_validated():
    collator = ContrastiveCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    example = _text(_xray_example("long"), length=40)  # exceeds the (16,) text bucket
    with pytest.raises(CollatorError, match="unplanned TPU compilation"):
        collator([example])


# ---------------------------------------------------------------------------
# Volume VL collator
# ---------------------------------------------------------------------------


def test_volume_vl_single_volume_static():
    plan = BucketPlan(
        buckets={
            BucketKind.VOLUME_3D: (BucketId(BucketKind.VOLUME_3D, (8, 32, 32)),),
            BucketKind.TEXT_TOKENS: (BucketId(BucketKind.TEXT_TOKENS, (16,)),),
        },
        mode="static",
    )
    collator = VolumeVLCollator(Modality.CT_3D, plan, static=True)
    batch = collator([_text(_ct_example("v1", (4, 16, 16))), _text(_ct_example("v2", (8, 30, 30)))])
    assert batch.pixel_values.shape == (2, 1, 8, 32, 32)
    assert batch.bucket.kind is BucketKind.VOLUME_3D
    assert batch.input_ids.shape == (2, 16)
    assert batch.spatial_metadata[1].current_shape == (8, 30, 30)


def test_volume_vl_multi_series_pads_series_count():
    collator = VolumeVLCollator(Modality.MULTI_SERIES_3D)
    examples = [
        _text(
            {
                "sample_id": "ms1",
                "modality": Modality.MULTI_SERIES_3D,
                "volumes": [torch.rand(1, 4, 16, 16), torch.rand(1, 4, 16, 16)],
            }
        ),
        _text(
            {
                "sample_id": "ms2",
                "modality": Modality.MULTI_SERIES_3D,
                "volumes": [torch.rand(1, 4, 16, 16)],
            }
        ),
    ]
    batch = collator(examples)
    assert batch.pixel_values.shape == (2, 2, 1, 4, 16, 16)  # [B, S, C, D, H, W]
    assert batch.image_mask.tolist() == [[True, True], [True, False]]


# ---------------------------------------------------------------------------
# WSI VL collator
# ---------------------------------------------------------------------------


def _wsi_plan() -> BucketPlan:
    return BucketPlan(
        buckets={
            BucketKind.IMAGE_2D: (BucketId(BucketKind.IMAGE_2D, (32, 32)),),
            BucketKind.WSI_TILES: (BucketId(BucketKind.WSI_TILES, (4,)),),
            BucketKind.TEXT_TOKENS: (BucketId(BucketKind.TEXT_TOKENS, (16,)),),
            BucketKind.VISUAL_TOKENS: (BucketId(BucketKind.VISUAL_TOKENS, (8,)),),
        },
        mode="static",
    )


def _wsi_example(sample_id: str, tiles: int, hw: tuple[int, int] = (16, 16)) -> dict:
    return _text(
        {
            "sample_id": sample_id,
            "modality": Modality.PATHOLOGY_WSI,
            "tiles": [torch.rand(3, *hw) for _ in range(tiles)],
            "tile_coordinates": torch.stack([torch.tensor([index * 16, index * 32]) for index in range(tiles)]).to(
                torch.int64
            ),
        }
    )


def test_wsi_vl_pads_tiles_coordinates_and_mask():
    collator = WSIVLCollator(Modality.PATHOLOGY_WSI, _wsi_plan(), static=True)
    batch = collator([_wsi_example("w1", 3), _wsi_example("w2", 1)])
    assert batch.pixel_values.shape == (2, 4, 3, 32, 32)  # WSI_TILES bucket
    assert batch.image_mask.tolist() == [[True, True, True, False], [True, False, False, False]]
    assert batch.tile_coordinates.shape == (2, 4, 2)
    assert batch.tile_coordinates[0, 2].tolist() == [32, 64]  # real coords preserved
    assert batch.tile_coordinates[0, 3].tolist() == [0, 0]  # padded coords zeroed
    assert batch.bucket.kind is BucketKind.WSI_TILES


def test_wsi_vl_precomputed_embeddings_use_visual_token_bucket():
    collator = WSIVLCollator(Modality.PATHOLOGY_WSI, _wsi_plan(), static=True)
    example = _text({"sample_id": "e1", "modality": Modality.PATHOLOGY_WSI})
    example["visual_tokens"] = torch.rand(6, 4)
    batch = collator([example])
    assert batch.pixel_values is None
    assert batch.task_targets["visual_tokens"].shape == (1, 8, 4)
    assert batch.task_targets["visual_token_mask"].tolist() == [[True] * 6 + [False] * 2]
    assert batch.bucket.kind is BucketKind.VISUAL_TOKENS


def test_wsi_vl_rejects_coordinate_tile_mismatch():
    collator = WSIVLCollator(Modality.PATHOLOGY_WSI)
    example = _wsi_example("bad", 2)
    example["tile_coordinates"] = torch.zeros(3, 2, dtype=torch.int64)
    with pytest.raises(CollatorError, match="tile_coordinates"):
        collator([example])


# ---------------------------------------------------------------------------
# Multitask collator
# ---------------------------------------------------------------------------


def test_multitask_dispatches_declared_modalities_only():
    multitask = MultitaskCollator(
        {
            Modality.XRAY_2D: ClassificationCollator(Modality.XRAY_2D),
            Modality.CT_3D: ClassificationCollator(Modality.CT_3D),
        }
    )
    examples = [_ct_example("ct-1"), _xray_example("x-1"), _xray_example("x-2", (16, 16))]
    result = multitask(examples)
    assert set(result.batches) == {Modality.XRAY_2D, Modality.CT_3D}
    assert result.modality_index == ("CT_3D", "XRAY_2D", "XRAY_2D")  # input order
    assert result.sample_ids == ("ct-1", "x-1", "x-2")
    assert result.batches[Modality.XRAY_2D].sample_ids == ["x-1", "x-2"]


def test_multitask_rejects_undeclared_modality():
    multitask = MultitaskCollator({Modality.XRAY_2D: ClassificationCollator(Modality.XRAY_2D)})
    with pytest.raises(CollatorError, match="undeclared modality"):
        multitask([_xray_example("x"), _ct_example("ct")])


def test_multitask_eval_never_drops():
    multitask = MultitaskCollator(
        {
            Modality.XRAY_2D: ClassificationCollator(Modality.XRAY_2D, final_batch_policy=FinalBatchPolicy.DROP),
            Modality.CT_3D: ClassificationCollator(Modality.CT_3D, final_batch_policy=FinalBatchPolicy.DROP),
        }
    )
    examples = [_xray_example("x-1"), _ct_example("ct-1")]
    result = multitask(examples, training=False, target_batch_size=8)
    covered = sorted(sid for batch in result.batches.values() for sid in batch.sample_ids)
    assert covered == ["ct-1", "x-1"]  # evaluation samples are sacred


# ---------------------------------------------------------------------------
# Final-batch policy
# ---------------------------------------------------------------------------


def test_final_batch_drop_policy_training_only():
    collator = ClassificationCollator(Modality.XRAY_2D, final_batch_policy=FinalBatchPolicy.DROP)
    short = [_xray_example(f"s{i}") for i in range(3)]
    assert collator(short, training=True, target_batch_size=4) is None
    # Evaluation never drops, even when short of a target size.
    batch = collator(short, training=False, target_batch_size=4)
    assert batch is not None and batch.sample_ids == ["s0", "s1", "s2"]


def test_final_batch_pad_policy_masks_replicas():
    collator = ClassificationCollator(Modality.XRAY_2D, final_batch_policy=FinalBatchPolicy.PAD)
    batch = collator([_xray_example(f"s{i}") for i in range(3)], training=True, target_batch_size=5)
    assert batch.pixel_values.shape[0] == 5
    assert batch.sample_ids[3:] == ["s2::pad0", "s2::pad1"]
    assert batch.image_mask.tolist() == [True, True, True, False, False]


def test_final_batch_oversized_batch_rejected():
    collator = ClassificationCollator(Modality.XRAY_2D)
    with pytest.raises(CollatorError, match="exceeds target_batch_size"):
        collator([_xray_example(f"s{i}") for i in range(5)], training=True, target_batch_size=4)


# ---------------------------------------------------------------------------
# Padding invariance for losses and metrics
# ---------------------------------------------------------------------------


def test_text_loss_is_padding_invariant():
    dynamic = ContrastiveCollator(Modality.XRAY_2D)
    static = ContrastiveCollator(Modality.XRAY_2D, _plan_2d(), static=True)
    examples = [_text(_xray_example("a", (32, 32)), 10), _text(_xray_example("b", (40, 40)), 6)]
    losses = []
    for batch in (dynamic(examples), static(examples)):
        labels = batch.task_targets["lm_labels"]
        supervised = labels != -100
        # Toy "loss": mean of supervised label ids — identical regardless of padding.
        losses.append(labels[supervised].float().mean().item())
        # Toy "metric": total attention — computed over masks, padding cannot leak in.
        assert float(batch.attention_mask.sum()) == 16.0
    assert losses[0] == pytest.approx(losses[1])


def test_visual_token_loss_is_padding_invariant():
    collator = WSIVLCollator(Modality.PATHOLOGY_WSI, _wsi_plan(), static=True)
    example = _text({"sample_id": "e1", "modality": Modality.PATHOLOGY_WSI})
    tokens = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10.0
    example["visual_tokens"] = tokens
    batch = collator([example])
    padded_tokens = batch.task_targets["visual_tokens"][0]
    mask = batch.task_targets["visual_token_mask"][0]
    masked_mean = padded_tokens[mask].mean()
    assert masked_mean.item() == pytest.approx(float(tokens.mean()))
