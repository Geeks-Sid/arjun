"""DICOM series discovery, physical sorting, calibration, and rejection paths."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from medfm.data.errors import ReaderError, UnsupportedFormatError
from medfm.data.readers.dicom import DICOMSeriesReader, discover_dicom_series
from phase_03.synthetic import write_dicom_series


@pytest.fixture()
def series_dir(tmp_path: Path) -> Path:
    return tmp_path / "series"


def test_discovers_single_series_by_hashed_uid(series_dir: Path) -> None:
    uid, _ = write_dicom_series(series_dir, num_slices=4, series_seed="one")
    groups = discover_dicom_series(series_dir)
    assert len(groups) == 1
    (key,) = groups
    assert key != uid  # hashed, never the raw UID
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert len(groups[key]) == 4


def test_sorts_slices_by_physical_position_not_filename(series_dir: Path) -> None:
    _, raw = write_dicom_series(series_dir, num_slices=6, shuffle_files=True, value_seed=3)
    read = DICOMSeriesReader().read(series_dir)
    # Volume is (i, j, k); k indexes ascending position along the slice normal.
    calibrated = raw.astype(np.float64) * 2.0 - 1000.0
    volume = read.image.numpy()
    assert volume.shape == (16, 16, 6)
    assert np.allclose(volume.transpose(2, 1, 0), calibrated)


def test_affine_maps_voxel_origin_to_first_slice_position(series_dir: Path) -> None:
    origin = (12.5, -30.0, 4.0)
    write_dicom_series(series_dir, num_slices=4, origin=origin, slice_spacing=2.5)
    read = DICOMSeriesReader().read(series_dir)
    affine = read.spatial.affine.numpy()
    assert np.allclose(affine[:3, 3], origin)
    # k-th slice position advances along the normal (here +z) by slice_spacing.
    positions = read.spatial.slice_positions_mm.numpy()
    assert np.allclose(np.diff(positions), 2.5)
    assert read.spatial.spacing_mm == pytest.approx((1.0, 1.0, 2.5))


def test_applies_rescale_slope_and_intercept(series_dir: Path) -> None:
    _, raw = write_dicom_series(series_dir, num_slices=2, slope=3.0, intercept=-500.0, value_seed=5)
    read = DICOMSeriesReader().read(series_dir)
    expected = raw.astype(np.float64) * 3.0 - 500.0
    assert np.allclose(read.image.numpy().transpose(2, 1, 0), expected)
    assert read.source_metadata["rescale_applied"] is True


def test_monochrome1_is_inverted(series_dir: Path) -> None:
    _, raw = write_dicom_series(series_dir, num_slices=2, photometric="MONOCHROME1", value_seed=7)
    read = DICOMSeriesReader().read(series_dir)
    inverted = raw.max(axis=(1, 2), keepdims=True).astype(np.float64) - raw.astype(np.float64)
    expected = inverted * 2.0 - 1000.0
    assert np.allclose(read.image.numpy().transpose(2, 1, 0), expected)


def test_hashes_uids_and_never_exposes_raw_identifiers(series_dir: Path) -> None:
    patient_id = "MRN-SECRET-99"
    uid, _ = write_dicom_series(series_dir, num_slices=2, patient_id=patient_id, series_seed="priv")
    read = DICOMSeriesReader().read(series_dir)
    meta_text = repr(read.source_metadata)
    assert patient_id not in meta_text
    assert uid not in meta_text
    assert read.source_metadata["patient_id_hash"]
    assert read.source_metadata["series_id_hash"] != uid


def test_rejects_inconsistent_orientation(series_dir: Path) -> None:
    write_dicom_series(
        series_dir,
        num_slices=4,
        value_seed=8,
        per_slice_overrides={2: {"ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]}},
    )
    with pytest.raises(ReaderError, match="orientation changes"):
        DICOMSeriesReader().read(series_dir)


def test_rejects_inconsistent_pixel_spacing(series_dir: Path) -> None:
    write_dicom_series(series_dir, num_slices=4, value_seed=9, per_slice_overrides={1: {"PixelSpacing": [3.0, 3.0]}})
    with pytest.raises(ReaderError, match="pixel spacing changes"):
        DICOMSeriesReader().read(series_dir)


def test_rejects_multiframe_objects(series_dir: Path) -> None:
    write_dicom_series(series_dir, num_slices=2, value_seed=10, per_slice_overrides={0: {"NumberOfFrames": 2}})
    with pytest.raises(UnsupportedFormatError, match="MULTIFRAME"):
        DICOMSeriesReader().read(series_dir)


def test_rejects_scout_localizer_series(series_dir: Path) -> None:
    write_dicom_series(series_dir, num_slices=2, value_seed=11, image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))
    with pytest.raises(UnsupportedFormatError, match="scout/localizer"):
        DICOMSeriesReader().read(series_dir)


def test_rejects_mixed_series_unless_selected(series_dir: Path) -> None:
    write_dicom_series(series_dir, num_slices=2, series_seed="A", file_prefix="a", value_seed=12)
    write_dicom_series(series_dir, num_slices=2, series_seed="B", file_prefix="b", value_seed=13)
    groups = discover_dicom_series(series_dir)
    assert len(groups) == 2
    with pytest.raises(ReaderError, match="series"):
        DICOMSeriesReader().read(series_dir)
    chosen = next(iter(groups))
    read = DICOMSeriesReader().read(series_dir, series_id_hash=chosen)
    assert read.source_metadata["series_id_hash"] == chosen


def test_empty_directory_is_actionable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ReaderError, match="no DICOM files"):
        DICOMSeriesReader().read(empty)


def test_summaries_report_counts_without_pixels(series_dir: Path) -> None:
    write_dicom_series(series_dir, num_slices=3, series_seed="sum", patient_id="MRN-X")
    summaries = DICOMSeriesReader().summaries(series_dir)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.file_count == 3
    assert summary.modality == "CT"
    assert "MRN-X" not in repr(summary)
