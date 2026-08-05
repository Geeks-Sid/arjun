"""DICOM series discovery and reading (pydicom).

Contract highlights:

- Slices are sorted by PHYSICAL POSITION (``ImagePositionPatient`` projected
  on the slice normal), never by filename.
- Orientation (``ImageOrientationPatient``) and in-plane spacing
  (``PixelSpacing``) must be consistent across the series within tolerance;
  inconsistent series are rejected with actionable errors.
- CT calibration: ``RescaleSlope``/``RescaleIntercept`` are applied so
  output values are in the stored calibrated units (HU for CT).
  ``MONOCHROME1`` inputs are inverted slice-wise so higher values are always
  brighter (MONOCHROME2 semantics) downstream.
- Unsupported variants fail explicitly: multiframe objects, scout/localizer
  series, mixed series in one directory (unless configured), and pixel data
  this reader cannot decode.
- Privacy: UIDs are hashed at the boundary; ``source_metadata`` carries only
  SHA-256 digests plus acquisition facts (never raw identifiers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medfm.core.sample import SpatialMetadata
from medfm.data.errors import ReaderError, UnsupportedFormatError
from medfm.data.readers.base import PayloadRead, Reader, hash_identifier

#: Orientation cosine deviation (per component) tolerated within one series.
DEFAULT_ORIENTATION_TOLERANCE = 1e-3
#: Relative in-plane spacing deviation tolerated within one series.
DEFAULT_SPACING_TOLERANCE = 1e-3
#: Relative deviation of an inter-slice gap from the median slice spacing that
#: is still considered a uniform volume (gaps beyond this are rejected).
DEFAULT_SLICE_GAP_TOLERANCE = 0.25

_LOCALIZER_MARKERS = ("LOCALIZER", "SCOUT")
_SUPPORTED_PHOTOMETRIC = ("MONOCHROME1", "MONOCHROME2")


def _require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:
        raise UnsupportedFormatError(
            "reading DICOM requires pydicom; install medfm with the 'medical' extra (uv sync --extra medical)"
        ) from exc
    return pydicom


def _float_list(value: Any, name: str, length: int) -> tuple[float, ...]:
    try:
        items = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ReaderError(f"DICOM attribute {name} is not numeric: {value!r}") from exc
    if len(items) != length:
        raise ReaderError(f"DICOM attribute {name} must have {length} values; got {items!r}")
    return items


@dataclass(frozen=True)
class DicomSeriesSummary:
    """Hashed, privacy-safe summary of one discovered DICOM series."""

    series_id_hash: str
    study_id_hash: str
    patient_id_hash: str
    frame_of_reference_hash: str | None
    file_count: int
    modality: str | None


@dataclass
class _Slice:
    position: tuple[float, ...]
    position_on_normal: float
    file_path: Path
    dataset: Any = field(repr=False, default=None)


def discover_dicom_series(directory: Path) -> dict[str, list[Path]]:
    """Group DICOM files under ``directory`` (non-recursive) by hashed series UID.

    Returns ``{series_id_hash: [file paths...]}``; files without a readable
    DICOM preamble are skipped silently (directories often mix sidecars).
    """
    pydicom = _require_pydicom()
    if not directory.is_dir():
        raise ReaderError(f"DICOM series path {directory} is not a directory")
    series: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None or not str(uid):
            continue
        series.setdefault(hash_identifier(str(uid)), []).append(path)
    return series


class DICOMSeriesReader(Reader):
    """Reads one single-frame DICOM series directory into a calibrated volume.

    Parameters
    ----------
    allow_mixed_series:
        When ``False`` (default) a directory containing more than one series
        raises :class:`ReaderError`; set ``True`` and pass ``series_id_hash``
        to :meth:`read` to pick one.
    orientation_tolerance / spacing_tolerance / slice_gap_tolerance:
        Consistency validation thresholds.
    """

    reader_id = "dicom_series"
    reader_version = "1.0.0"

    def __init__(
        self,
        *,
        allow_mixed_series: bool = False,
        orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE,
        spacing_tolerance: float = DEFAULT_SPACING_TOLERANCE,
        slice_gap_tolerance: float = DEFAULT_SLICE_GAP_TOLERANCE,
    ) -> None:
        self._allow_mixed_series = allow_mixed_series
        self._orientation_tolerance = orientation_tolerance
        self._spacing_tolerance = spacing_tolerance
        self._slice_gap_tolerance = slice_gap_tolerance

    def supports(self, path: Path) -> bool:
        return path.is_dir() and bool(discover_dicom_series(path))

    def summaries(self, directory: Path) -> list[DicomSeriesSummary]:
        """Hashed summaries of every series in ``directory`` (no pixel reads)."""
        pydicom = _require_pydicom()
        grouped = discover_dicom_series(directory)
        summaries: list[DicomSeriesSummary] = []
        for series_hash, paths in sorted(grouped.items()):
            ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            patient_id = getattr(ds, "PatientID", None)
            frame_uid = getattr(ds, "FrameOfReferenceUID", None)
            summaries.append(
                DicomSeriesSummary(
                    series_id_hash=series_hash,
                    study_id_hash=hash_identifier(str(study_uid)) if study_uid else "",
                    patient_id_hash=hash_identifier(str(patient_id)) if patient_id else "",
                    frame_of_reference_hash=hash_identifier(str(frame_uid)) if frame_uid else None,
                    file_count=len(paths),
                    modality=str(getattr(ds, "Modality", "")) or None,
                )
            )
        return summaries

    def read(self, path: Path, *, series_id_hash: str | None = None) -> PayloadRead:
        pydicom = _require_pydicom()
        grouped = discover_dicom_series(path)
        if not grouped:
            raise ReaderError(f"no DICOM files found in {path}; check the series directory")
        if series_id_hash is not None:
            if series_id_hash not in grouped:
                raise ReaderError(
                    f"series hash {series_id_hash} not found in {path}; available: {sorted(grouped)} (hashes)"
                )
            selected = series_id_hash
        elif len(grouped) == 1:
            selected = next(iter(grouped))
        else:
            if not self._allow_mixed_series:
                raise ReaderError(
                    f"{path} contains {len(grouped)} DICOM series {sorted(grouped)} (hashes); mixed-series "
                    "directories are rejected — pass series_id_hash to select one or set allow_mixed_series"
                )
            raise ReaderError(
                f"{path} contains {len(grouped)} series and allow_mixed_series is set but no series_id_hash "
                "was passed; choose one explicitly"
            )

        slices: list[_Slice] = []
        for file_path in grouped[selected]:
            ds = pydicom.dcmread(str(file_path), force=True)
            self._reject_unsupported(ds, file_path)
            position = _float_list(ds.ImagePositionPatient, "ImagePositionPatient", 3)
            slices.append(_Slice(position=position, position_on_normal=0.0, file_path=file_path, dataset=ds))

        first = slices[0].dataset
        row_cos, col_cos = self._validate_orientation(slices)
        normal = np.cross(row_cos, col_cos)
        for item in slices:
            item.position_on_normal = float(np.dot(np.asarray(item.position), normal))
        slices.sort(key=lambda item: item.position_on_normal)

        spacing_xy = self._validate_in_plane_spacing(slices)
        slice_spacing, slice_positions = self._validate_slice_positions(slices)
        arrays = [self._decode_slice(item.dataset, item.file_path) for item in slices]
        self._validate_slice_shapes(arrays, slices)
        volume_kji = np.stack(arrays, axis=0)  # (slices, rows, cols) = (k, j, i)
        volume = np.ascontiguousarray(volume_kji.transpose(2, 1, 0))  # -> (i, j, k) framework contract

        origin = np.asarray(slices[0].position, dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        # Voxel axis i = row cosines (column index), j = column cosines (row index), k = slice normal.
        affine[:3, 0] = row_cos * spacing_xy[1]
        affine[:3, 1] = col_cos * spacing_xy[0]
        affine[:3, 2] = normal * slice_spacing
        affine[:3, 3] = origin

        spacing_mm = (float(spacing_xy[1]), float(spacing_xy[0]), float(slice_spacing))
        frame_hash = getattr(first, "FrameOfReferenceUID", None)
        spatial = SpatialMetadata(
            original_shape=tuple(int(d) for d in volume.shape),
            current_shape=tuple(int(d) for d in volume.shape),
            affine=torch.as_tensor(affine, dtype=torch.float64),
            original_affine=torch.as_tensor(affine, dtype=torch.float64),
            spacing_mm=spacing_mm,
            # nibabel-style axcodes assume RAS; the DICOM affine is LPS, so no
            # RAS axcode is derived here (Phase 04 owns canonical orientation).
            orientation=None,
            slice_positions_mm=torch.as_tensor(slice_positions, dtype=torch.float64),
            frame_of_reference_hash=hash_identifier(str(frame_hash)) if frame_hash else None,
        )
        study_uid = getattr(first, "StudyInstanceUID", None)
        patient_id = getattr(first, "PatientID", None)
        source_metadata = {
            "reader": self.reader_id,
            "reader_version": self.reader_version,
            "series_id_hash": selected,
            "study_id_hash": hash_identifier(str(study_uid)) if study_uid else None,
            "patient_id_hash": hash_identifier(str(patient_id)) if patient_id else None,
            "modality": str(getattr(first, "Modality", "")) or None,
            "slice_count": len(slices),
            "rescale_applied": True,
        }
        return PayloadRead(
            tensors={"image": torch.from_numpy(volume)}, spatial=spatial, source_metadata=source_metadata
        )

    # -- validation / decoding -------------------------------------------------

    def _reject_unsupported(self, ds: Any, file_path: Path) -> None:
        where = file_path.name
        frames = getattr(ds, "NumberOfFrames", 1)
        if int(frames) > 1:
            raise UnsupportedFormatError(
                f"{where} is a MULTIFRAME DICOM object ({frames} frames); only single-frame series are "
                "supported — export single frames or use a multiframe-capable tool"
            )
        image_type = [str(t).upper() for t in getattr(ds, "ImageType", [])]
        if any(marker in value for value in image_type for marker in _LOCALIZER_MARKERS):
            raise UnsupportedFormatError(
                f"{where} is a scout/localizer acquisition (ImageType {image_type}); localizers are never "
                "ingested as training volumes"
            )
        if not hasattr(ds, "PixelData"):
            raise UnsupportedFormatError(f"{where} has no PixelData attribute; nothing to decode")
        photometric = str(getattr(ds, "PhotometricInterpretation", ""))
        if photometric not in _SUPPORTED_PHOTOMETRIC:
            raise UnsupportedFormatError(
                f"{where} has PhotometricInterpretation {photometric!r}; supported: {list(_SUPPORTED_PHOTOMETRIC)} "
                "(RGB/palette DICOM is not supported by this reader)"
            )
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise UnsupportedFormatError(f"{where} has SamplesPerPixel != 1; only grayscale DICOM is supported")
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
            raise ReaderError(
                f"{where} lacks ImagePositionPatient/ImageOrientationPatient; physical sorting is impossible — "
                "rejecting the series fail-closed"
            )

    def _validate_orientation(self, slices: list[_Slice]) -> tuple[np.ndarray, np.ndarray]:
        reference = _float_list(slices[0].dataset.ImageOrientationPatient, "ImageOrientationPatient", 6)
        row_cos = np.asarray(reference[:3], dtype=np.float64)
        col_cos = np.asarray(reference[3:], dtype=np.float64)
        for item in slices[1:]:
            candidate = np.asarray(
                _float_list(item.dataset.ImageOrientationPatient, "ImageOrientationPatient", 6), dtype=np.float64
            )
            if np.abs(candidate[:3] - row_cos).max() > self._orientation_tolerance or (
                np.abs(candidate[3:] - col_cos).max() > self._orientation_tolerance
            ):
                raise ReaderError(
                    f"orientation changes within the series: {item.file_path.name} has ImageOrientationPatient "
                    f"{candidate.tolist()} vs {reference} at the reference slice; mixed orientations cannot form "
                    "one volume — split the series or reject the acquisition"
                )
        return row_cos, col_cos

    def _validate_in_plane_spacing(self, slices: list[_Slice]) -> tuple[float, float]:
        reference = _float_list(slices[0].dataset.PixelSpacing, "PixelSpacing", 2)
        for item in slices[1:]:
            candidate = _float_list(item.dataset.PixelSpacing, "PixelSpacing", 2)
            if (
                abs(candidate[0] - reference[0]) > self._spacing_tolerance * reference[0]
                or abs(candidate[1] - reference[1]) > self._spacing_tolerance * reference[1]
            ):
                raise ReaderError(
                    f"pixel spacing changes within the series: {item.file_path.name} has PixelSpacing "
                    f"{candidate} vs {reference} at the reference slice; resample before ingestion"
                )
        return (reference[0], reference[1])  # (row spacing, column spacing)

    def _validate_slice_positions(self, slices: list[_Slice]) -> tuple[float, np.ndarray]:
        positions = np.asarray([item.position_on_normal for item in slices], dtype=np.float64)
        if len(slices) == 1:
            # Single slice: fall back to the slice thickness when present.
            thickness = getattr(slices[0].dataset, "SliceThickness", None)
            spacing = float(thickness) if thickness is not None and float(thickness) > 0 else 1.0
            return spacing, positions
        gaps = np.abs(np.diff(positions))
        if gaps.min() <= 1e-6:
            duplicates = [item.file_path.name for item, gap in zip(slices[1:], gaps, strict=True) if gap <= 1e-6]
            raise ReaderError(
                f"duplicate slice positions detected ({duplicates}); the series would stack identical planes — "
                "check the acquisition/export"
            )
        median = float(np.median(gaps))
        if np.abs(gaps - median).max() > self._slice_gap_tolerance * median:
            raise ReaderError(
                f"slice spacing is irregular: gaps range {gaps.min():.4f}-{gaps.max():.4f} mm around median "
                f"{median:.4f} mm; volumes with irregular spacing are rejected (re-sort or re-acquire)"
            )
        return median, positions

    def _decode_slice(self, ds: Any, file_path: Path) -> np.ndarray:
        try:
            raw = ds.pixel_array
        except Exception as exc:
            raise UnsupportedFormatError(
                f"cannot decode pixel data in {file_path.name}: {exc}; the transfer syntax or compression may be "
                "unsupported by this reader"
            ) from exc
        if raw.ndim != 2:
            raise UnsupportedFormatError(f"{file_path.name} decoded to rank-{raw.ndim} pixel data; expected 2D")
        values = raw.astype(np.float64)
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            # Lower values are brighter: invert slice-wise so downstream always
            # sees MONOCHROME2 semantics (higher = brighter). Linear rescale
            # afterwards preserves ordering.
            values = float(values.max()) - values
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        calibrated: np.ndarray = values * slope + intercept
        return calibrated

    def _validate_slice_shapes(self, arrays: list[np.ndarray], slices: list[_Slice]) -> None:
        reference = arrays[0].shape
        for array, item in zip(arrays[1:], slices[1:], strict=True):
            if array.shape != reference:
                raise ReaderError(
                    f"slice dimensions change within the series: {item.file_path.name} is {array.shape} vs "
                    f"{reference} at the reference slice; mixed dimensions cannot form one volume"
                )
