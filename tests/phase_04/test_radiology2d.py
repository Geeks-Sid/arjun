"""Tests for medfm.data.transforms.radiology2d (2D radiology transforms)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from medfm.data.errors import TransformError
from medfm.data.transforms.base import InversionMode, Transform, TransformContext, TransformData, invert_history
from medfm.data.transforms.pipeline import TransformPipeline
from medfm.data.transforms.radiology2d import (
    BodyRegionCrop,
    DecodeGrayscale,
    LetterboxResize,
    NormalizeImage,
    RandomFlip2D,
    RandomGaussianNoise,
    RandomIntensityShift,
    RandomRotate2D,
    RandomScale2D,
    RandomTranslate2D,
    RescaleIntensity,
    ToChannels,
)
from medfm.data.transforms.specs import NormalizationSpec, PreprocessSpec


def _data(image: torch.Tensor, metadata: dict[str, object] | None = None) -> TransformData:
    return TransformData(image=image, metadata=metadata or {})


def _ctx(seed: int = 1234) -> TransformContext:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return TransformContext(rng=rng, seed=seed)


# ---------------------------------------------------------------------------
# DecodeGrayscale
# ---------------------------------------------------------------------------


def test_decode_monochrome1_applies_inversion_correction() -> None:
    image = torch.tensor([[[0.0, 2.0], [4.0, 10.0]]])
    data = DecodeGrayscale("MONOCHROME1")(_data(image), None)
    expected = 10.0 - image
    assert torch.equal(data.image, expected)
    record = data.history[-1]
    assert record.name == "decode_grayscale"
    assert record.stage == "deterministic"
    assert record.params["inversion_applied"] is True


def test_decode_monochrome2_passthrough() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    data = DecodeGrayscale("MONOCHROME2")(_data(image), None)
    assert torch.equal(data.image, image)
    assert data.history[-1].params["inversion_applied"] is False


def test_decode_rejects_multi_channel() -> None:
    image = torch.zeros(3, 8, 8)
    with pytest.raises(TransformError):
        DecodeGrayscale("MONOCHROME2")(_data(image), None)


def test_decode_rejects_unknown_photometric_interpretation() -> None:
    with pytest.raises(TransformError):
        DecodeGrayscale("RGB")


# ---------------------------------------------------------------------------
# LetterboxResize
# ---------------------------------------------------------------------------


def test_letterbox_preserves_aspect_and_pads_symmetrically() -> None:
    image = torch.rand(1, 100, 200, generator=torch.Generator().manual_seed(0))
    data = LetterboxResize((224, 224))(_data(image), None)
    assert tuple(data.image.shape) == (1, 224, 224)
    record = data.history[-1]
    assert record.spatial is True
    # Aspect preserved: content is the largest 2:1 rectangle fitting 224x224.
    assert record.params["scale"] == pytest.approx(224 / 200)
    assert record.params["content_size"] == [112, 224]
    # Symmetric padding, odd remainder absorbed on the bottom/right.
    assert record.params["pad_top"] == 56
    assert record.params["pad_bottom"] == 56
    assert record.params["pad_left"] == 0
    assert record.params["pad_right"] == 0


def test_letterbox_inversion_roundtrips_to_original_shape_image_and_label() -> None:
    image = torch.rand(1, 50, 80, generator=torch.Generator().manual_seed(1))
    data = LetterboxResize((64, 64))(_data(image), None)
    assert tuple(data.image.shape) == (1, 64, 64)
    modes: list[InversionMode] = ["image", "label"]
    for mode in modes:
        restored = invert_history(data.history, data.image, mode=mode)
        assert tuple(restored.shape) == (1, 50, 80)


def test_letterbox_inversion_recovers_smooth_image_approximately() -> None:
    rows = torch.linspace(0.0, 1.0, 96).view(1, 96, 1)
    cols = torch.linspace(0.0, 1.0, 192).view(1, 1, 192)
    image = (rows + cols) / 2.0  # smooth ramp, tolerant to bilinear roundtrip
    data = LetterboxResize((64, 64))(_data(image.clone()), None)
    restored = invert_history(data.history, data.image, mode="image")
    assert torch.allclose(restored, image, atol=0.05)


# ---------------------------------------------------------------------------
# BodyRegionCrop
# ---------------------------------------------------------------------------


def _body_image() -> torch.Tensor:
    image = torch.zeros(1, 64, 64)
    image[:, 20:40, 24:44] = 1.0
    return image


def test_body_crop_box_and_margin() -> None:
    data = BodyRegionCrop(margin=4)(_data(_body_image()), None)
    record = data.history[-1]
    assert record.spatial is True
    assert record.params["crop_box"] == [16, 20, 44, 48]
    assert tuple(data.image.shape) == (1, 28, 28)


def test_body_crop_inversion_reembeds_exactly() -> None:
    image = _body_image()
    data = BodyRegionCrop(margin=8)(_data(image.clone()), None)
    restored = invert_history(data.history, data.image, mode="image")
    assert tuple(restored.shape) == tuple(image.shape)
    assert torch.equal(restored, image)


def test_body_crop_empty_foreground_is_recorded_noop() -> None:
    image = torch.zeros(1, 32, 32)
    data = BodyRegionCrop()(_data(image), None)
    assert tuple(data.image.shape) == (1, 32, 32)
    assert data.history[-1].params["crop_box"] == [0, 0, 32, 32]


# ---------------------------------------------------------------------------
# ToChannels / NormalizeImage / RescaleIntensity
# ---------------------------------------------------------------------------


def test_to_channels_repeats_single_channel() -> None:
    image = torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(2))
    data = ToChannels(3)(_data(image), None)
    assert tuple(data.image.shape) == (3, 8, 8)
    for channel in range(3):
        assert torch.equal(data.image[channel], image[0])


def test_to_channels_passthrough_and_validation() -> None:
    image = torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(3))
    data = ToChannels(1)(_data(image), None)
    assert torch.equal(data.image, image)
    with pytest.raises(TransformError):
        ToChannels(2)
    with pytest.raises(TransformError):
        ToChannels(1)(_data(torch.zeros(3, 8, 8)), None)


def test_normalize_image_math() -> None:
    image = torch.tensor([[[1.0, 3.0], [5.0, 7.0]], [[10.0, 20.0], [30.0, 40.0]]])
    spec = NormalizationSpec(mean=(4.0, 25.0), std=(2.0, 5.0))
    data = NormalizeImage(spec)(_data(image), None)
    expected = torch.tensor([[[-1.5, -0.5], [0.5, 1.5]], [[-3.0, -1.0], [1.0, 3.0]]])
    assert torch.allclose(data.image, expected)
    with pytest.raises(TransformError):
        NormalizeImage(NormalizationSpec(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))(_data(image), None)


def test_rescale_intensity_min_max() -> None:
    image = torch.tensor([[[2.0, 4.0], [6.0, 10.0]]])
    data = RescaleIntensity((0.0, 1.0))(_data(image), None)
    assert torch.allclose(data.image, torch.tensor([[[0.0, 0.25], [0.5, 1.0]]]))
    constant = RescaleIntensity((0.0, 1.0))(_data(torch.full((1, 4, 4), 7.0)), None)
    assert torch.equal(constant.image, torch.zeros(1, 4, 4))


def test_rescale_intensity_percentiles_clamps() -> None:
    image = torch.zeros(1, 10, 10)
    image[:, :5, :] = 1.0
    image[0, 0, 0] = 100.0  # outlier clamped away by the percentile anchors
    data = RescaleIntensity((0.0, 1.0), percentiles=(10.0, 90.0))(_data(image), None)
    assert float(data.image.min()) >= 0.0
    assert float(data.image.max()) <= 1.0


# ---------------------------------------------------------------------------
# Stochastic augmentation reproducibility
# ---------------------------------------------------------------------------


def _augmented_pair(transform: Transform, seed_a: int, seed_b: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.rand(1, 32, 32, generator=torch.Generator().manual_seed(9))
    out_a = transform(_data(image.clone()), _ctx(seed_a)).image
    out_b = transform(_data(image.clone()), _ctx(seed_b)).image
    return out_a, out_b


@pytest.mark.parametrize(
    "transform_factory",
    [
        lambda: RandomRotate2D(10.0),
        lambda: RandomTranslate2D(0.1),
        lambda: RandomScale2D((0.9, 1.1)),
        lambda: RandomIntensityShift((-0.1, 0.1), (0.9, 1.1)),
        lambda: RandomGaussianNoise((0.01, 0.05)),
        lambda: RandomFlip2D(allow_horizontal=True),
    ],
)
def test_augmentation_same_seed_identical_different_seed_differs(transform_factory: Callable[[], Transform]) -> None:
    transform = transform_factory()
    first, second = _augmented_pair(transform, seed_a=42, seed_b=42)
    assert torch.equal(first, second)
    _, third = _augmented_pair(transform, seed_a=42, seed_b=43)
    assert not torch.equal(first, third)


def test_stochastic_transform_requires_context() -> None:
    with pytest.raises(TransformError):
        RandomGaussianNoise((0.1, 0.1))(_data(torch.zeros(1, 8, 8)), None)


# ---------------------------------------------------------------------------
# RandomFlip2D gating
# ---------------------------------------------------------------------------


def _asymmetric_image() -> torch.Tensor:
    image = torch.zeros(1, 8, 8)
    image[0, 0, 0] = 1.0  # single corner: h/v flips land in distinct corners
    return image


def test_flip_disabled_everywhere_is_noop_but_records() -> None:
    image = _asymmetric_image()
    transform = RandomFlip2D(allow_horizontal=False, allow_vertical=False)
    for seed in range(50):
        data = transform(_data(image.clone()), _ctx(seed))
        assert torch.equal(data.image, image)
        record = data.history[-1]
        assert record.name == "random_flip_2d"
        assert record.params["flipped_horizontal"] is False
        assert record.params["flipped_vertical"] is False


def test_flip_horizontal_only_never_flips_vertical() -> None:
    image = _asymmetric_image()
    h_flipped = torch.flip(image, dims=(-1,))
    transform = RandomFlip2D(allow_horizontal=True, p=0.5)
    seen_flip = False
    for seed in range(100):
        data = transform(_data(image.clone()), _ctx(seed))
        assert torch.equal(data.image, image) or torch.equal(data.image, h_flipped)
        assert data.history[-1].params["flipped_vertical"] is False
        seen_flip = seen_flip or torch.equal(data.image, h_flipped)
    assert seen_flip  # flips actually happen when allowed


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_metadata_passes_through_untouched() -> None:
    metadata = {"view_position": "PA", "view_order": 2, "longitudinal_index": 0}
    image = torch.rand(1, 40, 60, generator=torch.Generator().manual_seed(5))
    data = _data(image, metadata=dict(metadata))
    data = DecodeGrayscale("MONOCHROME2")(data, None)
    data = LetterboxResize((64, 64))(data, None)
    data = BodyRegionCrop(margin=2)(data, None)
    data = ToChannels(3)(data, None)
    data = RescaleIntensity((0.0, 1.0))(data, None)
    data = RandomFlip2D(allow_horizontal=True)(data, _ctx(7))
    assert data.metadata == metadata


# ---------------------------------------------------------------------------
# Full pipeline + spec validation
# ---------------------------------------------------------------------------


def _dummy_spec() -> PreprocessSpec:
    return PreprocessSpec(
        model_id="dummy-xray",
        spatial_shape=(224, 224),
        channels=3,
        value_range=(-8.0, 8.0),
        normalization=NormalizationSpec(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    )


def _xray_pipeline() -> TransformPipeline:
    return TransformPipeline(
        deterministic=[
            DecodeGrayscale("MONOCHROME1"),
            BodyRegionCrop(margin=8),
            LetterboxResize((224, 224)),
            ToChannels(3),
            RescaleIntensity((0.0, 1.0)),
            NormalizeImage(NormalizationSpec(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))),
        ],
        stochastic=[RandomRotate2D(5.0), RandomFlip2D(allow_horizontal=True)],
        spec=_dummy_spec(),
        name="dummy-xray-pipeline",
    )


def test_full_pipeline_output_passes_spec_validation() -> None:
    image = torch.rand(1, 100, 140, generator=torch.Generator().manual_seed(11)) * 0.8 + 0.1
    pipeline = _xray_pipeline()
    result = pipeline(_data(image), _ctx(123))
    assert tuple(result.image.shape) == (3, 224, 224)
    assert pipeline.spec is not None
    pipeline.spec.validate(result.image)
    names = [record.name for record in result.history]
    assert names == [
        "decode_grayscale",
        "body_region_crop",
        "letterbox_resize",
        "to_channels",
        "rescale_intensity",
        "normalize_image",
        "random_rotate_2d",
        "random_flip_2d",
    ]


def test_pipeline_reproducible_under_fixed_seed() -> None:
    image = torch.rand(1, 100, 140, generator=torch.Generator().manual_seed(12))
    pipeline = _xray_pipeline()
    out_a = pipeline(_data(image.clone()), _ctx(99)).image
    out_b = pipeline(_data(image.clone()), _ctx(99)).image
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Worker/epoch/sample seeding
# ---------------------------------------------------------------------------


def test_for_sample_streams_reproducible_but_distinct_across_workers_and_epochs() -> None:
    def first_draw(ctx: TransformContext) -> torch.Tensor:
        return torch.rand(8, generator=ctx.rng)

    base = TransformContext.for_sample(base_seed=42, epoch=0, worker_id=0, sample_key="sample-1")
    same = TransformContext.for_sample(base_seed=42, epoch=0, worker_id=0, sample_key="sample-1")
    other_worker = TransformContext.for_sample(base_seed=42, epoch=0, worker_id=1, sample_key="sample-1")
    other_epoch = TransformContext.for_sample(base_seed=42, epoch=1, worker_id=0, sample_key="sample-1")
    other_sample = TransformContext.for_sample(base_seed=42, epoch=0, worker_id=0, sample_key="sample-2")

    assert torch.equal(first_draw(base), first_draw(same))
    assert not torch.equal(first_draw(base), first_draw(other_worker))
    assert not torch.equal(first_draw(base), first_draw(other_epoch))
    assert not torch.equal(first_draw(base), first_draw(other_sample))
