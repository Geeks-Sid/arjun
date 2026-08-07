"""Parity checks for MRI intensity transfer kernels."""

import torch
from monai.transforms import NormalizeIntensity, ScaleIntensityRangePercentiles

from medfm.core.sample import SpatialMetadata
from medfm.data.transforms.base import TransformData
from medfm.data.transforms.mri import ForegroundZScoreNormalize, RobustPercentileNormalize


def _spatial(shape: tuple[int, int, int]) -> SpatialMetadata:
    affine = torch.diag(torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        original_affine=affine.clone(),
        spacing_mm=(1.0, 1.0, 1.0),
        orientation="RAS",
        anatomical_axes=("R", "A", "S"),
    )


def test_foreground_zscore_matches_monai_nonzero_kernel() -> None:
    image = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [0.0, 4.0, 8.0]]],
            [[[0.0, -3.0, 1.0], [0.0, 5.0, 9.0]]],
        ],
        dtype=torch.float32,
    )
    expected = NormalizeIntensity(nonzero=True, channel_wise=True, dtype=torch.float32)(image)
    actual = ForegroundZScoreNormalize()(TransformData(image=image, spatial=_spatial((1, 2, 3))), None).image

    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(actual[:, :, :, 0], torch.zeros(2, 1, 2))


def test_foreground_zscore_preserves_epsilon_floor_noop() -> None:
    image = torch.tensor([[[[0.0, 1e-9, 2e-9]]]], dtype=torch.float32)
    actual = ForegroundZScoreNormalize()(TransformData(image=image, spatial=_spatial((1, 1, 3))), None)

    assert torch.equal(actual.image, image)
    assert actual.history[-1].params["stds"][0] <= 1e-8


def test_robust_percentile_matches_monai_on_foreground_values() -> None:
    image = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]]],
        dtype=torch.float32,
    )
    lower, upper = 10.0, 90.0
    foreground = image[0][image[0] != 0]
    expected_values = ScaleIntensityRangePercentiles(
        lower,
        upper,
        b_min=0.0,
        b_max=1.0,
        clip=True,
        relative=False,
        dtype=torch.float32,
    )(foreground)
    expected = image.clone()
    expected[0][image[0] != 0] = expected_values
    actual = RobustPercentileNormalize(lower, upper)(
        TransformData(image=image, spatial=_spatial((1, 1, 10))), None
    ).image

    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual[0, 0, 0, 0] == 0.0
