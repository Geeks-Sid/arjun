"""Phase 04 CT transform tests: HU calibration, windowing, spatial inversion."""

import pytest
import torch

from medfm.core.sample import SpatialMetadata
from medfm.data.errors import TransformError
from medfm.data.transforms.base import TransformData, invert_history
from medfm.data.transforms.ct import ClipHU, ToHounsfieldUnits, WindowChannels
from medfm.data.transforms.pipeline import TransformPipeline
from medfm.data.transforms.spatial3d import (
    CanonicalizeOrientation,
    ForegroundCrop3D,
    ResampleToSpacing,
)
from medfm.data.transforms.specs import PreprocessSpec


def _spatial(
    shape: tuple[int, int, int], spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), orientation: str = "RAS"
) -> SpatialMetadata:
    affine = torch.diag(torch.as_tensor([*spacing, 1.0], dtype=torch.float64))
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        original_affine=affine.clone(),
        spacing_mm=spacing,
        orientation=orientation,
        anatomical_axes=tuple(orientation),
    )


def _volume(
    shape: tuple[int, int, int] = (4, 5, 6),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "RAS",
) -> TransformData:
    image = torch.arange(float(shape[0] * shape[1] * shape[2])).reshape(1, *shape) / 10.0
    return TransformData(image=image, spatial=_spatial(shape, spacing, orientation))


# ---------------------------------------------------------------------------
# HU calibration
# ---------------------------------------------------------------------------


def test_hu_conversion_applies_slope_intercept():
    data = _volume()
    raw = data.image.clone()
    out = ToHounsfieldUnits(slope=2.0, intercept=-1024.0)(data, None)
    assert torch.allclose(out.image, raw * 2.0 - 1024.0)
    assert out.history[-1].params["applied"] is True


def test_hu_already_calibrated_is_recorded_noop():
    data = _volume()
    raw = data.image.clone()
    out = ToHounsfieldUnits(slope=2.0, intercept=-1024.0, units="HU")(data, None)
    assert torch.equal(out.image, raw)
    assert out.history[-1].params["applied"] is False


def test_hu_unknown_units_rejected():
    with pytest.raises(TransformError, match="units"):
        ToHounsfieldUnits(slope=1.0, intercept=0.0, units="normalized")


def test_clip_hu_bounds():
    data = _volume()
    out = ClipHU(min_hu=-100.0, max_hu=300.0)(data, None)
    out.image -= 1000.0  # sanity: post-clip values are bounded by the window
    assert float(ClipHU(-100.0, 300.0)(out, None).image.min()) >= -100.0
    assert float(ClipHU(-100.0, 300.0)(out, None).image.max()) <= 300.0


def test_clip_hu_rejects_inverted_range():
    with pytest.raises(TransformError):
        ClipHU(min_hu=100.0, max_hu=-100.0)


# ---------------------------------------------------------------------------
# Window channels
# ---------------------------------------------------------------------------


def test_window_math_center_width_convention():
    image = torch.as_tensor([40.0, 80.0, 0.0, -500.0]).reshape(1, 4, 1, 1)  # [C=1, D=4, H=1, W=1]
    data = TransformData(image=image, spatial=_spatial((4, 1, 1)))
    out = WindowChannels(windows=((40.0, 80.0),))(data, None)
    # v maps to clip((v - (center - width/2)) / width, 0, 1)
    assert out.image.shape == (1, 4, 1, 1)
    values = out.image[0, :, 0, 0].tolist()
    assert values == pytest.approx([0.5, 1.0, 0.0, 0.0])


def test_multi_window_channel_count_and_order():
    data = TransformData(image=torch.zeros(1, 3, 4, 5), spatial=_spatial((3, 4, 5)))
    out = WindowChannels(windows=((40.0, 80.0), (-600.0, 1500.0), (300.0, 500.0)))(data, None)
    assert out.image.shape == (3, 3, 4, 5)
    assert out.history[-1].params["windows"][1] == [-600.0, 1500.0]


def test_window_channels_requires_single_channel():
    data = TransformData(image=torch.zeros(2, 3, 4, 5), spatial=_spatial((3, 4, 5)))
    with pytest.raises(TransformError, match="single-channel"):
        WindowChannels(windows=((40.0, 80.0),))(data, None)


# ---------------------------------------------------------------------------
# Spatial transforms: orientation, resample, crop — with inversion
# ---------------------------------------------------------------------------


def test_orientation_canonicalization_and_exact_inversion():
    data = _volume(shape=(2, 3, 4), orientation="LPS")
    original = data.image.clone()
    out = CanonicalizeOrientation(target="RAS")(data, None)
    assert out.spatial is not None and out.spatial.orientation == "RAS"
    # LPS -> RAS flips all three axes; shape preserved, content permuted.
    assert out.spatial_shape == (2, 3, 4)
    restored = invert_history(out.history, out.image)
    assert torch.equal(restored, original)


def test_orientation_updates_affine_and_rejects_bad_codes():
    data = _volume(orientation="LPS")
    assert data.spatial is not None
    prior_affine = data.spatial.affine.clone()
    out = CanonicalizeOrientation(target="RAS")(data, None)
    assert out.spatial is not None
    assert not torch.equal(out.spatial.affine, prior_affine)
    with pytest.raises(TransformError):
        CanonicalizeOrientation(target="XYZ")


def test_resample_to_spacing_updates_geometry_and_restores_shape():
    data = _volume(shape=(4, 5, 6), spacing=(1.0, 1.0, 1.0))
    mask = (data.image[0] > 3.0).to(torch.int64)[None]
    data.targets["mask"] = mask
    out = ResampleToSpacing(spacing_mm=(2.0, 2.0, 2.0))(data, None)
    assert out.spatial is not None and out.spatial.spacing_mm == (2.0, 2.0, 2.0)
    assert out.spatial_shape == (2, 2, 3)  # round(n * 0.5) per axis
    # Label inversion: nearest back to the original lattice, original shape.
    restored_mask = invert_history(out.history, out.targets["mask"], mode="label")
    assert tuple(restored_mask.shape) == tuple(mask.shape)
    assert set(torch.unique(restored_mask).tolist()) <= {0, 1}
    # Image inversion restores the original shape (values approximate).
    restored_image = invert_history(out.history, out.image)
    assert tuple(restored_image.shape) == (1, 4, 5, 6)


def test_image_and_label_interpolation_differ():
    shape = (8, 1, 1)
    image = torch.zeros(1, *shape)
    image[0, 4:, 0, 0] = 1.0  # step edge
    data = TransformData(image=image, targets={"mask": image.clone()}, spatial=_spatial(shape))
    out = ResampleToSpacing(spacing_mm=(0.5, 1.0, 1.0))(data, None)
    zoomed_image = out.image
    zoomed_mask = out.targets["mask"]
    interior = (zoomed_image > 1e-6) & (zoomed_image < 1.0 - 1e-6)
    assert bool(interior.any()), "image zoom (order=3) must produce smooth intermediate values"
    assert set(torch.unique(zoomed_mask).tolist()) <= {0.0, 1.0}, "label zoom (order=0) must stay discrete"


def test_foreground_crop_records_invertible_coordinates():
    shape = (10, 12, 14)
    image = torch.zeros(1, *shape)
    image[0, 3:6, 4:8, 5:10] = 100.0
    mask = torch.zeros(1, *shape, dtype=torch.int64)
    mask[0, 3:6, 4:8, 5:10] = 1
    data = TransformData(image=image, targets={"mask": mask}, spatial=_spatial(shape))
    out = ForegroundCrop3D(margin=0, threshold=50.0)(data, None)
    assert out.spatial_shape == (3, 4, 5)
    # Inversion re-embeds in the original lattice: exact for image and mask.
    restored_image = invert_history(out.history, out.image)
    restored_mask = invert_history(out.history, out.targets["mask"], mode="label")
    assert torch.equal(restored_image, image)
    assert torch.equal(restored_mask, mask)


def test_crop_orientation_chain_mask_reconstruction_in_original_coordinates():
    """Exit-criteria test: invert a CT spatial chain and compare masks exactly."""
    shape = (8, 9, 10)
    image = torch.zeros(1, *shape)
    image[0, 2:6, 3:7, 4:8] = 500.0
    mask = torch.zeros(1, *shape, dtype=torch.int64)
    mask[0, 3:5, 4:6, 5:7] = 2
    data = TransformData(
        image=image,
        targets={"mask": mask},
        spatial=_spatial(shape, spacing=(1.0, 1.0, 1.0), orientation="LPS"),
    )
    pipeline = TransformPipeline(
        deterministic=[CanonicalizeOrientation(target="RAS"), ForegroundCrop3D(margin=1, threshold=100.0)]
    )
    out = pipeline.run_deterministic(data)
    restored_mask = invert_history(out.history, out.targets["mask"], mode="label")
    assert torch.equal(restored_mask, mask)


def test_spatial_transforms_require_spatial_metadata():
    data = TransformData(image=torch.zeros(1, 4, 5, 6))
    with pytest.raises(TransformError, match="SpatialMetadata"):
        ResampleToSpacing(spacing_mm=(1.0, 1.0, 1.0))(data, None)


# ---------------------------------------------------------------------------
# End-to-end CT pipeline against a dummy PreprocessSpec
# ---------------------------------------------------------------------------


def test_ct_pipeline_conforms_to_dummy_preprocess_spec():
    shape = (4, 6, 8)
    image = torch.full((1, *shape), 100.0)  # raw stored values
    data = TransformData(image=image, spatial=_spatial(shape))
    spec = PreprocessSpec(
        model_id="dummy-ct-v1",
        spatial_shape=shape,
        channels=2,
        value_range=(0.0, 1.0),
    )
    pipeline = TransformPipeline(
        deterministic=[
            ToHounsfieldUnits(slope=1.0, intercept=-1024.0),
            ClipHU(min_hu=-1024.0, max_hu=3071.0),
            WindowChannels(windows=((40.0, 400.0), (-600.0, 1500.0))),
        ],
        spec=spec,
    )
    out = pipeline(data)
    assert out.image.shape == spec.expected_tensor_shape()
    spec.validate(out.image)  # exact shape/range conformance


def test_ct_pipeline_rejects_nonconforming_shape():
    data = TransformData(image=torch.zeros(1, 4, 6, 8), spatial=_spatial((4, 6, 8)))
    spec = PreprocessSpec(model_id="dummy-ct-v1", spatial_shape=(8, 8, 8), channels=1)
    pipeline = TransformPipeline(deterministic=[ClipHU(-1024.0, 3071.0)], spec=spec)
    with pytest.raises(Exception, match="shape"):
        pipeline(data)
