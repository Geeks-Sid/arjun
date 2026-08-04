"""MedicalBatch shape contract, rank/modality mismatches, buckets, transfers."""

import pytest
import torch
from contract_fixtures import make_batch, make_spatial

from medfm.core import (
    BucketError,
    BucketId,
    BucketKind,
    MedicalBatch,
    Modality,
    ShapeContractError,
)

ALL_MODALITIES = tuple(Modality)


@pytest.mark.parametrize("modality", ALL_MODALITIES, ids=lambda m: m.value)
def test_synthetic_batch_validates_for_every_modality(modality):
    batch = make_batch(modality)
    assert batch.modality is modality
    assert len(batch.sample_ids) == 2


def test_modality_is_authoritative_over_rank():
    # The same rank-5 tensor is legal for CT_3D but illegal for XRAY_2D.
    volume = torch.randn(2, 1, 8, 16, 16)
    MedicalBatch(modality=Modality.CT_3D, pixel_values=volume, sample_ids=["a", "b"])
    with pytest.raises(ShapeContractError, match="rank 4"):
        MedicalBatch(modality=Modality.XRAY_2D, pixel_values=volume, sample_ids=["a", "b"])


def test_rank_mismatch_error_is_actionable():
    with pytest.raises(ShapeContractError) as excinfo:
        MedicalBatch(
            modality=Modality.CT_3D,
            pixel_values=torch.randn(2, 3, 16, 16),
            sample_ids=["a", "b"],
        )
    message = str(excinfo.value)
    assert "CT_3D" in message and "[B, C, D, H, W]" in message and "got rank 4" in message


def test_text_only_rejects_pixel_values():
    with pytest.raises(ShapeContractError, match="TEXT"):
        MedicalBatch(
            modality=Modality.TEXT_ONLY,
            pixel_values=torch.randn(2, 1, 8, 8),
            sample_ids=["a", "b"],
        )


def test_text_only_requires_input_ids():
    with pytest.raises(ShapeContractError, match="input_ids"):
        MedicalBatch(modality=Modality.TEXT_ONLY, sample_ids=["a", "b"])


def test_sample_ids_must_match_batch_dim():
    with pytest.raises(ShapeContractError, match="sample_ids"):
        MedicalBatch(modality=Modality.XRAY_2D, pixel_values=torch.randn(2, 1, 8, 8), sample_ids=["a"])


def test_attention_mask_shape_must_match_input_ids():
    with pytest.raises(ShapeContractError, match="attention_mask"):
        MedicalBatch(
            modality=Modality.TEXT_ONLY,
            sample_ids=["a", "b"],
            input_ids=torch.zeros(2, 8, dtype=torch.int64),
            attention_mask=torch.ones(2, 7, dtype=torch.bool),
        )


def test_image_mask_shapes_per_modality():
    # WSI needs [B, T]; [B] is a contract violation.
    with pytest.raises(ShapeContractError, match="image_mask"):
        MedicalBatch(
            modality=Modality.PATHOLOGY_WSI,
            pixel_values=torch.randn(2, 4, 3, 8, 8),
            image_mask=torch.ones(2, dtype=torch.bool),
            sample_ids=["a", "b"],
        )


def test_segmentation_target_shape_validated_against_pixels():
    with pytest.raises(ShapeContractError, match="segmentation"):
        MedicalBatch(
            modality=Modality.CT_3D,
            pixel_values=torch.randn(2, 1, 8, 16, 16),
            task_targets={"segmentation": torch.zeros(2, 3, 16, 16)},  # 2D mask on 3D batch
            sample_ids=["a", "b"],
        )
    with pytest.raises(ShapeContractError, match="spatial dims"):
        MedicalBatch(
            modality=Modality.XRAY_2D,
            pixel_values=torch.randn(2, 1, 16, 16),
            task_targets={"segmentation": torch.zeros(2, 3, 32, 32)},
            sample_ids=["a", "b"],
        )
    # Segmentation is undefined for aggregation/text modalities.
    with pytest.raises(ShapeContractError, match="not defined"):
        MedicalBatch(
            modality=Modality.MULTI_IMAGE_2D,
            pixel_values=torch.randn(2, 3, 3, 16, 16),
            image_mask=torch.ones(2, 3, dtype=torch.bool),
            task_targets={"segmentation": torch.zeros(2, 1, 16, 16)},
            sample_ids=["a", "b"],
        )


def test_valid_segmentation_shapes_pass():
    MedicalBatch(
        modality=Modality.XRAY_2D,
        pixel_values=torch.randn(2, 1, 16, 16),
        task_targets={"segmentation": torch.zeros(2, 3, 16, 16)},
        sample_ids=["a", "b"],
    )
    MedicalBatch(
        modality=Modality.CT_3D,
        pixel_values=torch.randn(2, 1, 8, 16, 16),
        task_targets={"segmentation": torch.zeros(2, 3, 8, 16, 16)},
        sample_ids=["a", "b"],
    )


def test_visual_token_batch_shape_and_mask():
    batch = MedicalBatch(
        modality=Modality.PATHOLOGY_WSI,
        sample_ids=["a", "b"],
        task_targets={
            "visual_tokens": torch.randn(2, 5, 7),
            "visual_token_mask": torch.ones(2, 5, dtype=torch.bool),
        },
    )
    assert batch.device is not None
    with pytest.raises(ShapeContractError, match="visual_token_mask"):
        MedicalBatch(
            modality=Modality.PATHOLOGY_WSI,
            sample_ids=["a", "b"],
            task_targets={
                "visual_tokens": torch.randn(2, 5, 7),
                "visual_token_mask": torch.ones(2, 4, dtype=torch.bool),
            },
        )


def test_bucket_requires_masks_for_padded_content():
    bucket = BucketId(kind=BucketKind.TEXT_TOKENS, shape=(8,))
    with pytest.raises(BucketError, match="attention_mask"):
        MedicalBatch(
            modality=Modality.TEXT_ONLY,
            sample_ids=["a", "b"],
            input_ids=torch.zeros(2, 8, dtype=torch.int64),
            bucket=bucket,
        )
    with pytest.raises(BucketError, match="image_mask"):
        MedicalBatch(
            modality=Modality.PATHOLOGY_WSI,
            pixel_values=torch.randn(2, 4, 3, 8, 8),
            sample_ids=["a", "b"],
            bucket=BucketId(kind=BucketKind.WSI_TILES, shape=(4,)),
        )


def test_bucket_shape_must_match_padded_dims():
    with pytest.raises(BucketError, match="bucket shape"):
        MedicalBatch(
            modality=Modality.XRAY_2D,
            pixel_values=torch.randn(2, 1, 16, 16),
            image_mask=torch.ones(2, dtype=torch.bool),
            sample_ids=["a", "b"],
            bucket=BucketId(kind=BucketKind.IMAGE_2D, shape=(32, 32)),
        )
    with pytest.raises(BucketError, match="CT_3D or MRI_3D"):
        MedicalBatch(
            modality=Modality.XRAY_2D,
            pixel_values=torch.randn(2, 1, 16, 16),
            image_mask=torch.ones(2, dtype=torch.bool),
            sample_ids=["a", "b"],
            bucket=BucketId(kind=BucketKind.VOLUME_3D, shape=(8, 16, 16)),
        )


def test_bucket_id_validation():
    with pytest.raises(BucketError):
        BucketId(kind=BucketKind.IMAGE_2D, shape=(448,))  # needs (H, W)
    assert str(BucketId(kind=BucketKind.WSI_TILES, shape=(512,))) == "WSI_TILES:512"
    bucket = BucketId(kind=BucketKind.VOLUME_3D, shape=(96, 128, 128))
    assert BucketId.from_dict(bucket.to_dict()) == bucket


def test_static_bucket_padding_masks_preserve_unpadded_content():
    # Two real images padded to a bucket of four; mask must recover originals.
    real = torch.randn(2, 2, 3, 8, 8)
    padded = torch.zeros(2, 4, 3, 8, 8)
    padded[:, :2] = real
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    batch = MedicalBatch(
        modality=Modality.MULTI_IMAGE_2D,
        pixel_values=padded,
        image_mask=mask,
        sample_ids=["a", "b"],
        bucket=BucketId(kind=BucketKind.MULTI_IMAGE, shape=(4,)),
    )
    recovered = batch.pixel_values[batch.image_mask].reshape(2, 2, 3, 8, 8)
    assert torch.equal(recovered, real)


def test_device_transfer_preserves_non_tensor_metadata():
    batch = MedicalBatch(
        modality=Modality.CT_3D,
        pixel_values=torch.randn(2, 1, 8, 16, 16),
        image_mask=torch.ones(2, dtype=torch.bool),
        spatial_metadata=[make_spatial((8, 16, 16)), None],
        labels=torch.tensor([0, 1]),
        task_targets={"segmentation": torch.zeros(2, 2, 8, 16, 16)},
        sample_ids=["a", "b"],
        bucket=BucketId(kind=BucketKind.VOLUME_3D, shape=(8, 16, 16)),
    )
    moved = batch.to("cpu").to(torch.device("cpu"))
    assert moved.modality is batch.modality
    assert moved.sample_ids == batch.sample_ids
    assert moved.bucket == batch.bucket
    assert moved.spatial_metadata[0] is not None
    assert moved.spatial_metadata[0].spacing_mm == batch.spatial_metadata[0].spacing_mm
    assert moved.spatial_metadata[1] is None
    assert torch.equal(moved.labels, batch.labels)


def test_pin_memory_marks_batch_and_keeps_metadata():
    batch = make_batch(Modality.XRAY_2D)
    pinned = batch.pin_memory()
    assert pinned.pinned is True
    assert pinned.sample_ids == batch.sample_ids
    assert torch.equal(pinned.pixel_values, batch.pixel_values)
    # Staying on the CPU keeps the pinned flag.
    assert pinned.to("cpu").pinned is True


def test_batch_metadata_dict_has_no_payloads_or_devices():
    batch = make_batch(Modality.PATHOLOGY_WSI)
    meta = batch.to_metadata_dict()
    assert meta["tensors"]["pixel_values"]["shape"] == [2, 4, 3, 16, 16]
    assert meta["tensors"]["pixel_values"]["dtype"] == "float32"
    assert "device" not in str(meta)
