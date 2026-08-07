"""Phase 04 MRI transform tests: sequences, normalization, stacking, bias field."""

import pytest
import torch

from medfm.core.sample import SpatialMetadata
from medfm.data.errors import TransformError
from medfm.data.transforms.base import TransformData, invert_history
from medfm.data.transforms.mri import (
    SEQUENCE_METADATA_KEY,
    ForegroundZScoreNormalize,
    RobustPercentileNormalize,
    SequenceResolver,
    apply_n4_bias_field_correction,
    select_sequences,
    stack_sequences,
)
from medfm.data.transforms.pipeline import TransformPipeline
from medfm.data.transforms.spatial3d import CanonicalizeOrientation, ResampleToSpacing
from medfm.data.transforms.specs import PreprocessSpec


def _resolver() -> SequenceResolver:
    return SequenceResolver(
        {
            "T1": ("T1W", "T1-WEIGHTED"),
            "T1CE": ("T1C", "T1GD", "T1-CE"),
            "T2": ("T2W",),
            "FLAIR": ("T2-FLAIR", "FLAIR-T2"),
        }
    )


def _spatial(shape: tuple[int, int, int]) -> SpatialMetadata:
    affine = torch.diag(torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        spacing_mm=(1.0, 1.0, 1.0),
        orientation="RAS",
        anatomical_axes=("R", "A", "S"),
    )


def _sequence_volume(sequence: str, shape: tuple[int, int, int] = (3, 4, 5), fill: float = 1.0) -> TransformData:
    image = torch.full((1, *shape), fill)
    return TransformData(
        image=image,
        spatial=_spatial(shape),
        metadata={SEQUENCE_METADATA_KEY: sequence},
    )


# ---------------------------------------------------------------------------
# Sequence identification
# ---------------------------------------------------------------------------


def test_sequence_resolver_aliases_case_insensitive():
    resolver = _resolver()
    assert resolver.resolve("t1c") == "T1CE"
    assert resolver.resolve("T1Gd") == "T1CE"
    assert resolver.resolve(" flair-t2 ") == "FLAIR"
    assert resolver.resolve("T1") == "T1"
    assert resolver.is_known("t2w") is True
    assert resolver.is_known("DWI") is False


def test_sequence_resolver_rejects_unknown_with_legal_values():
    resolver = _resolver()
    with pytest.raises(TransformError, match="unknown MRI sequence"):
        resolver.resolve("DWI")


def test_sequence_resolver_rejects_ambiguous_alias_config():
    with pytest.raises(TransformError, match="unambiguous"):
        SequenceResolver({"T1": ("SHARED",), "T1CE": ("shared",)})


def test_select_sequences_rejects_missing_without_substitution():
    resolver = _resolver()
    available = {"t1c": _sequence_volume("t1c"), "T2W": _sequence_volume("T2W")}
    selected = select_sequences(available, ("T1CE", "T2"), resolver)
    assert set(selected) == {"T1CE", "T2"}
    with pytest.raises(TransformError, match="not available"):
        select_sequences(available, ("T1CE", "FLAIR"), resolver)


def test_select_sequences_rejects_duplicate_canonical():
    resolver = _resolver()
    available = {"t1c": _sequence_volume("t1c"), "T1GD": _sequence_volume("T1GD")}
    with pytest.raises(TransformError, match="unambiguous"):
        select_sequences(available, ("T1CE",), resolver)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_foreground_zscore_normalizes_foreground_only():
    image = torch.zeros(1, 2, 3, 4)
    image[0, :, :, :2] = torch.as_tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    data = TransformData(image=image.clone(), spatial=_spatial((2, 3, 4)))
    out = ForegroundZScoreNormalize()(data, None)
    foreground = out.image[out.image != 0]
    assert float(foreground.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(foreground.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)
    # Background stays exactly zero.
    assert float(out.image[0, :, :, 2:].abs().sum()) == 0.0


def test_foreground_zscore_constant_channel_is_noop():
    data = TransformData(image=torch.full((1, 2, 2, 2), 7.0), spatial=_spatial((2, 2, 2)))
    out = ForegroundZScoreNormalize()(data, None)
    assert torch.equal(out.image, torch.full((1, 2, 2, 2), 7.0))


def test_robust_percentile_normalize_range_and_background():
    image = torch.zeros(1, 4, 4, 4)
    image[0, :2] = torch.arange(32.0).reshape(2, 4, 4)
    data = TransformData(image=image.clone(), spatial=_spatial((4, 4, 4)))
    out = RobustPercentileNormalize(lower=1.0, upper=99.0)(data, None)
    foreground = out.image[out.image != 0]
    assert float(foreground.min()) > 0.0 and float(foreground.max()) <= 1.0
    assert float(out.image[0, 2:].abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Multi-sequence stacking
# ---------------------------------------------------------------------------


def test_stack_sequences_channel_layout_and_mask():
    resolver = _resolver()
    datas = [_sequence_volume("t1c", fill=1.0), _sequence_volume("T2W", fill=2.0)]
    out = stack_sequences(datas, ("T1CE", "T2"), resolver)
    assert out.image.shape == (2, 3, 4, 5)
    assert float(out.image[0].mean()) == 1.0 and float(out.image[1].mean()) == 2.0
    assert out.targets["sequence_mask"].tolist() == [True, True]
    assert out.metadata["sequences"] == ["T1CE", "T2"]


def test_stack_sequences_missing_required_rejected():
    resolver = _resolver()
    datas = [_sequence_volume("t1c")]
    with pytest.raises(TransformError, match="never substituted"):
        stack_sequences(datas, ("T1CE", "FLAIR"), resolver)


def test_stack_sequences_allowed_missing_zero_channel_and_mask():
    resolver = _resolver()
    datas = [_sequence_volume("t1c", fill=3.0)]
    out = stack_sequences(datas, ("T1CE", "FLAIR"), resolver, allowed_missing=("FLAIR",))
    assert out.image.shape == (2, 3, 4, 5)
    assert float(out.image[0].mean()) == 3.0
    assert float(out.image[1].abs().sum()) == 0.0
    assert out.targets["sequence_mask"].tolist() == [True, False]


def test_stack_sequences_requires_sequence_metadata():
    resolver = _resolver()
    data = TransformData(image=torch.ones(1, 3, 4, 5), spatial=_spatial((3, 4, 5)))
    with pytest.raises(TransformError, match="sequence"):
        stack_sequences([data], ("T1",), resolver)


def test_stack_sequences_rejects_mismatched_shapes():
    resolver = _resolver()
    datas = [_sequence_volume("t1c", shape=(3, 4, 5)), _sequence_volume("T2W", shape=(3, 4, 6))]
    with pytest.raises(TransformError, match="mismatched"):
        stack_sequences(datas, ("T1CE", "T2"), resolver)


# ---------------------------------------------------------------------------
# Orientation / spacing canonicalization (shared 3D machinery on MRI)
# ---------------------------------------------------------------------------


def test_mri_orientation_and_spacing_canonicalization():
    shape = (4, 5, 6)
    image = torch.arange(float(4 * 5 * 6)).reshape(1, *shape)
    affine = torch.diag(torch.as_tensor([-1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    spatial = SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        spacing_mm=(1.0, 1.0, 1.0),
        orientation="LPS",
        anatomical_axes=("L", "P", "S"),
    )
    data = TransformData(image=image.clone(), spatial=spatial)
    out = CanonicalizeOrientation(target="RAS")(data, None)
    assert out.spatial is not None and out.spatial.orientation == "RAS"
    restored = invert_history(out.history, out.image)
    assert torch.equal(restored, image)

    resampled = ResampleToSpacing(spacing_mm=(1.0, 1.0, 1.0))(data, None)
    assert resampled.spatial_shape == shape  # unity spacing is a recorded no-op grid


# ---------------------------------------------------------------------------
# Bias-field correction: explicit, deterministic, off-pipeline
# ---------------------------------------------------------------------------


def test_bias_field_correction_is_explicit_and_deterministic():
    image = torch.full((1, 8, 8, 8), 100.0)
    ramp = torch.linspace(1.0, 1.5, 8).reshape(1, 8, 1, 1).expand(1, 8, 8, 8)
    data = TransformData(image=(image * ramp).contiguous(), spatial=_spatial((8, 8, 8)))
    first = apply_n4_bias_field_correction(data)
    second = apply_n4_bias_field_correction(
        TransformData(image=(image * ramp).contiguous(), spatial=_spatial((8, 8, 8)))
    )
    assert torch.equal(first.image, second.image), "bias-field correction must be deterministic"
    assert first.history[-1].name == "n4_bias_field_correction"
    # It is a standalone function, not a Transform — it cannot be placed in a
    # TransformPipeline (type-level opt-in-only guarantee).
    assert not isinstance(apply_n4_bias_field_correction, type)


def test_bias_field_correction_rejects_nonpositive_sigma():
    data = TransformData(image=torch.ones(1, 4, 4, 4), spatial=_spatial((4, 4, 4)))
    with pytest.raises(TransformError, match="smoothing_sigma_voxels"):
        apply_n4_bias_field_correction(data, smoothing_sigma_voxels=0.0)


# ---------------------------------------------------------------------------
# Pipeline conformance against a dummy PreprocessSpec
# ---------------------------------------------------------------------------


def test_mri_pipeline_conforms_to_dummy_preprocess_spec():
    resolver = _resolver()
    shape = (3, 4, 5)
    datas = [_sequence_volume("t1c", shape, fill=5.0), _sequence_volume("T2W", shape, fill=9.0)]
    stacked = stack_sequences(datas, ("T1CE", "T2"), resolver)
    spec = PreprocessSpec(model_id="dummy-mri-v1", spatial_shape=shape, channels=2)
    pipeline = TransformPipeline(deterministic=[ForegroundZScoreNormalize()], spec=spec)
    out = pipeline(stacked)
    assert out.image.shape == spec.expected_tensor_shape()
    spec.validate(out.image)
