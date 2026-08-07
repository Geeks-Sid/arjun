"""Parity checks for the optional MONAI kernels used by spatial3d transforms."""

import torch
from monai.data import MetaTensor
from monai.transforms import CropForeground, Orientation, Spacing

from medfm.core.sample import SpatialMetadata
from medfm.data.transforms.base import TransformData
from medfm.data.transforms.spatial3d import CanonicalizeOrientation, ForegroundCrop3D, ResampleToSpacing, _zoom_tensor


def _spatial(
    shape: tuple[int, int, int],
    affine: torch.Tensor,
    spacing: tuple[float, float, float],
    orientation: str,
) -> SpatialMetadata:
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=affine,
        spacing_mm=spacing,
        orientation=orientation,
        anatomical_axes=tuple(orientation),
    )


def test_orientation_matches_monai_for_permuted_and_flipped_fixture() -> None:
    shape = (2, 3, 4)
    image = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, *shape)
    label = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(1, *shape)
    affine = torch.tensor(
        [[0.0, 0.0, 1.0, 11.0], [-2.0, 0.0, 0.0, 22.0], [0.0, 3.0, 0.0, 33.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    metadata = _spatial(shape, affine, (2.0, 3.0, 1.0), "PSR")
    out = CanonicalizeOrientation("RAS")(TransformData(image.clone(), {"mask": label.clone()}, metadata), None)

    expected_image_meta = Orientation(axcodes="RAS")(MetaTensor(image, affine=affine))
    expected_label_meta = Orientation(axcodes="RAS")(MetaTensor(label, affine=affine))
    expected_image = expected_image_meta.as_tensor()
    expected_label = expected_label_meta.as_tensor()
    assert torch.equal(out.image, expected_image)
    assert torch.equal(out.targets["mask"], expected_label)
    assert out.image.dtype == image.dtype and out.targets["mask"].dtype == label.dtype
    assert out.spatial is not None and torch.equal(out.spatial.affine, expected_image_meta.affine)


def test_spacing_matches_monai_output_shape_and_dtype() -> None:
    shape = (5, 9, 13)
    image = torch.arange(2 * 5 * 9 * 13, dtype=torch.float32).reshape(2, *shape)
    affine = torch.diag(torch.tensor([1.5, 3.0, 6.0, 1.0], dtype=torch.float64))
    out = ResampleToSpacing((2.0, 4.0, 8.0))(
        TransformData(image.clone(), spatial=_spatial(shape, affine, (1.5, 3.0, 6.0), "RAS")), None
    )
    expected = Spacing(pixdim=(2.0, 4.0, 8.0), mode="bilinear")(MetaTensor(image, affine=affine))
    scipy_kernel = _zoom_tensor(image, (0.75, 0.75, 0.75), 3)
    assert tuple(out.image.shape) == tuple(expected.shape) == tuple(scipy_kernel.shape)
    assert out.image.dtype == expected.dtype == scipy_kernel.dtype


def test_foreground_bounds_match_monai_and_empty_keeps_whole_volume() -> None:
    shape = (7, 8, 9)
    image = torch.zeros(1, *shape)
    image[0, 2:5, 3:6, 4:7] = 5.0
    affine = torch.eye(4, dtype=torch.float64)
    threshold = 1.0
    margin = 1
    out = ForegroundCrop3D(margin=margin, threshold=threshold)(
        TransformData(image.clone(), spatial=_spatial(shape, affine, (1.0, 1.0, 1.0), "RAS")), None
    )
    expected, starts, ends = CropForeground(
        select_fn=lambda values: values > threshold,
        margin=margin,
        allow_smaller=True,
        return_coords=True,
    )(image)
    assert torch.equal(out.image, expected)
    assert out.history[-1].params["origin"] == [int(value) for value in starts]
    assert out.spatial_shape == tuple(int(value) for value in expected.shape[-3:])
    assert out.history[-1].params["origin"] == [1, 2, 3]
    assert [int(value) for value in ends] == [6, 7, 8]
    empty = ForegroundCrop3D(margin=margin, threshold=threshold)(
        TransformData(torch.zeros(1, *shape), spatial=_spatial(shape, affine, (1.0, 1.0, 1.0), "RAS")), None
    )
    assert empty.spatial_shape == shape
    assert empty.history[-1].params["origin"] == [0, 0, 0]
