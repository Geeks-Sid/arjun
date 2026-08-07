"""Parity checks for CT intensity transfer kernels."""

import torch
from monai.transforms import ScaleIntensityRange

from medfm.core.sample import SpatialMetadata
from medfm.data.transforms.base import TransformData
from medfm.data.transforms.ct import WindowChannels


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


def test_window_channels_matches_monai_scale_intensity_range() -> None:
    image = torch.tensor(
        [[[-1200.0, -500.0, 0.0], [40.0, 80.0, 400.0]]],
        dtype=torch.float32,
    ).reshape(1, 2, 1, 3)
    windows = ((40.0, 80.0), (0.0, 400.0), (100.0, 200.0))
    hu = image[0]
    expected = torch.stack([((hu - (center - width / 2.0)) / width).clamp(0.0, 1.0) for center, width in windows])
    monai = torch.stack(
        [
            ScaleIntensityRange(
                a_min=center - width / 2.0,
                a_max=center + width / 2.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                dtype=hu.dtype,
            )(hu)
            for center, width in windows
        ]
    )
    actual = WindowChannels(windows)(TransformData(image=image, spatial=_spatial((2, 1, 3))), None).image

    assert actual.dtype == torch.float32
    assert torch.equal(monai, expected)
    assert torch.equal(actual, expected)
