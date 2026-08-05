"""NIfTI / MHA / NumPy / PNG-JPEG readers: geometry preserved, dtypes canonical."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from synthetic import write_mha, write_nifti, write_numpy_volume, write_png

from medfm.data.errors import ReaderError, UnsupportedFormatError
from medfm.data.readers.base import reader_for_path, resolve_local_path
from medfm.data.readers.radiology import MHAReader, NiftiReader, NumpyVolumeReader, PngJpegReader


def test_nifti_preserves_affine_spacing_dtype_orientation(tmp_path: Path) -> None:
    affine = np.array([[-1.5, 0.0, 0.0, 90.0], [0.0, 1.5, 0.0, -120.0], [0.0, 0.0, 2.0, -30.0], [0.0, 0.0, 0.0, 1.0]])
    array = write_nifti(tmp_path / "v.nii.gz", shape=(20, 18, 10), affine=affine, seed=1)
    read = NiftiReader().read(tmp_path / "v.nii.gz")
    assert np.array_equal(read.image.numpy(), array)
    assert read.image.dtype == torch.int16  # on-disk dtype preserved, not upcast
    assert np.allclose(read.spatial.affine.numpy(), affine)
    assert np.allclose(read.spatial.original_affine.numpy(), affine)
    assert read.spatial.spacing_mm == pytest.approx((1.5, 1.5, 2.0))
    assert read.spatial.orientation == "LAS"  # from the affine, never dropped
    assert read.spatial.original_shape == read.spatial.current_shape == array.shape


def test_mha_transposes_to_framework_axis_order(tmp_path: Path) -> None:
    array_zyx = write_mha(tmp_path / "v.mha", shape=(8, 12, 16), seed=2)
    read = MHAReader().read(tmp_path / "v.mha")
    assert tuple(read.image.shape) == (16, 12, 8)  # (i, j, k)
    assert np.array_equal(read.image.numpy(), array_zyx.transpose(2, 1, 0))
    assert read.spatial.spacing_mm == pytest.approx((1.0, 1.25, 2.5))
    assert np.allclose(read.spatial.affine.numpy()[:3, 3], (10.0, -5.0, 3.0))


def test_numpy_volume_with_and_without_sidecar(tmp_path: Path) -> None:
    array = write_numpy_volume(tmp_path / "v.npy", shape=(6, 5, 4), seed=3, with_meta=True)
    read = NumpyVolumeReader().read(tmp_path / "v.npy")
    assert np.array_equal(read.image.numpy(), array)
    assert read.spatial.spacing_mm == (1.0, 1.0, 1.0)

    bare = tmp_path / "bare.npy"
    write_numpy_volume(bare, shape=(4, 4, 4), seed=4, with_meta=False)
    read_bare = NumpyVolumeReader().read(bare)
    assert read_bare.spatial.affine is None
    assert read_bare.spatial.spacing_mm is None
    assert read_bare.spatial.current_shape == (4, 4, 4)


def test_numpy_volume_rejects_bad_sidecar_affine(tmp_path: Path) -> None:
    write_numpy_volume(tmp_path / "v.npy", shape=(4, 4, 4), seed=5, with_meta=False)
    (tmp_path / "v.meta.json").write_text('{"affine": [[1, 0], [0, 1]]}', encoding="utf-8")
    with pytest.raises(ReaderError, match="4x4"):
        NumpyVolumeReader().read(tmp_path / "v.npy")


def test_png_and_jpeg_rgb_and_grayscale(tmp_path: Path) -> None:
    rgb = write_png(tmp_path / "rgb.png", size=(32, 24), mode="RGB", seed=6)
    read_rgb = PngJpegReader().read(tmp_path / "rgb.png")
    assert np.array_equal(read_rgb.image.numpy(), rgb)
    assert read_rgb.image.dtype == torch.uint8

    gray = write_png(tmp_path / "gray.png", size=(16, 16), mode="L", seed=7)
    read_gray = PngJpegReader().read(tmp_path / "gray.png")
    assert tuple(read_gray.image.shape) == (16, 16, 1)  # (H, W, 1)
    assert np.array_equal(read_gray.image.numpy()[..., 0], gray)


def test_reader_for_path_dispatches_by_suffix(tmp_path: Path) -> None:
    assert isinstance(reader_for_path(tmp_path / "a.nii.gz"), NiftiReader)
    assert isinstance(reader_for_path(tmp_path / "a.nii"), NiftiReader)
    assert isinstance(reader_for_path(tmp_path / "a.mha"), MHAReader)
    assert isinstance(reader_for_path(tmp_path / "a.npy"), NumpyVolumeReader)
    assert isinstance(reader_for_path(tmp_path / "a.png"), PngJpegReader)
    with pytest.raises(ReaderError, match="no reader claims"):
        reader_for_path(tmp_path / "a.xyz")


def test_resolve_local_path_rejects_remote_and_missing(tmp_path: Path) -> None:
    target = tmp_path / "img.nii.gz"
    target.write_bytes(b"x")
    assert resolve_local_path(str(target)) == target
    assert resolve_local_path(f"file://{target}") == target
    assert resolve_local_path("img.nii.gz", base_dir=tmp_path) == target
    with pytest.raises(ReaderError, match="s3"):
        resolve_local_path("s3://bucket/img.nii.gz")
    with pytest.raises(ReaderError, match="base_dir"):
        resolve_local_path("img.nii.gz")
    with pytest.raises(ReaderError, match="not found"):
        resolve_local_path(str(tmp_path / "nope.nii.gz"))


def test_unsupported_dtype_is_rejected(tmp_path: Path) -> None:
    import nibabel as nib

    array = np.zeros((4, 4, 4), dtype=np.complex64)
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(tmp_path / "c.nii.gz"))
    with pytest.raises(UnsupportedFormatError, match="complex"):
        NiftiReader().read(tmp_path / "c.nii.gz")
