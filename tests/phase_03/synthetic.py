"""Synthetic, legally-safe fixture builders for Phase 03 tests.

Every payload here is generated mathematically (seeded RNGs, no real patient
data). Deterministic given a seed, so fixtures and assertions are stable.
Shared by the unit tests and the committed fingerprint fixture
(``tests/fixtures/manifests/mixed_synthetic.parquet``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def hid(seed: str) -> str:
    """Deterministic identifier hash (sha256 hex)."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def manifest_row(**overrides: Any) -> dict[str, Any]:
    """One valid manifest row (defaults to a CT_3D sample)."""
    base: dict[str, Any] = {
        "sample_id": "unset",
        "patient_id_hash": hid("patient-a"),
        "study_id_hash": hid("study-a1"),
        "series_id_hash": hid("series-a1"),
        "modality": "CT_3D",
        "image_uri": "images/ct/0001.nii.gz",
        "secondary_image_uris": None,
        "mask_uri": None,
        "annotation_uri": None,
        "report_uri": None,
        "label_json": None,
        "split": "TRAIN",
        "site_id": "site-01",
        "scanner_vendor": "SIEMENS",
        "acquisition_date_bucket": "2019-Q3",
        "dataset_name": "synthetic-mixed",
        "dataset_version": "1.0.0",
        "license": "CC-BY-4.0",
        "provenance_uri": "s3://bucket/provenance/record.json",
    }
    base.update(overrides)
    return base


def build_mixed_manifest(
    *,
    patients: int = 12,
    seed: int = 7,
    splits: tuple[str, float] = (("TRAIN", 0.6), ("VAL", 0.2), ("TEST", 0.2)),
) -> pd.DataFrame:
    """Deterministic mixed-modality manifest with a patient-disjoint split.

    Patients are assigned to splits by hashing ``"<seed>:<patient>"`` into the
    cumulative ratios (same construction medfm.data.splits uses), so the
    committed fixture is reproducible without importing medfm internals.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    modalities = ("CT_3D", "MRI_3D", "XRAY_2D", "PATHOLOGY_WSI", "PATHOLOGY_TILE", "TEXT_ONLY")

    cumulative: list[tuple[str, float]] = []
    total = 0.0
    for name, weight in splits:
        total += weight
        cumulative.append((name, total))

    def split_for(patient_key: str) -> str:
        bucket = int(hashlib.sha256(f"{seed}:{patient_key}".encode()).hexdigest()[:8], 16) / float(16**8)
        for name, edge in cumulative:
            if bucket < edge:
                return name
        return cumulative[-1][0]

    sample_counter = 0
    for patient in range(patients):
        patient_key = f"patient-{patient:03d}"
        patient_hash = hid(patient_key)
        split = split_for(patient_key)
        modality = modalities[patient % len(modalities)]
        sample_counter += 1
        sample_id = f"{modality.lower()}-{sample_counter:04d}"
        study_hash = hid(f"study-{patient:03d}-0")
        series_hash = hid(f"series-{patient:03d}-0")
        site = f"site-{patient % 3 + 1:02d}"
        vendor = ("SIEMENS", "GE", "PHILIPS")[patient % 3]
        date_bucket = ("2018-Q2", "2019-Q1", "2019-Q3", "2020-Q1")[patient % 4]

        row = manifest_row(
            sample_id=sample_id,
            patient_id_hash=patient_hash,
            study_id_hash=study_hash,
            series_id_hash=series_hash,
            modality=modality,
            split=split,
            site_id=site,
            scanner_vendor=vendor,
            acquisition_date_bucket=date_bucket,
            image_sha256=hid(f"payload-{sample_id}"),
        )
        if modality == "TEXT_ONLY":
            row["image_uri"] = None
            row["report_uri"] = f"s3://restricted-store/reports/{sample_id}.json"
            row["report_chars"] = int(rng.integers(200, 3000))
        else:
            row["image_uri"] = f"images/{modality.lower()}/{sample_id}"
        if modality in ("CT_3D", "MRI_3D"):
            shape = [int(rng.integers(48, 96)), int(rng.integers(128, 256)), int(rng.integers(128, 256))]
            row["shape"] = shape
            row["spacing_mm"] = [round(float(u), 3) for u in rng.uniform(0.7, 3.0, size=3)]
            row["num_slices"] = shape[0]
            row["mask_uri"] = f"masks/{modality.lower()}/{sample_id}.nii.gz"
            row["label_json"] = json.dumps({"task": "BINARY_CLASSIFICATION", "values": [int(rng.integers(0, 2))]})
            row["intensity_stats_json"] = json.dumps(
                {
                    "p01": float(rng.uniform(-1000, -500)),
                    "p50": float(rng.uniform(-100, 100)),
                    "p99": float(rng.uniform(200, 1000)),
                }
            )
            if rng.random() < 0.5:
                row["seg_class_volumes_json"] = json.dumps({"liver_mm3": float(rng.uniform(1000, 20000))})
        elif modality == "XRAY_2D":
            row["shape"] = [1, int(rng.integers(512, 1024)), int(rng.integers(512, 1024))]
            row["label_json"] = json.dumps({"task": "MULTICLASS_CLASSIFICATION", "values": [int(rng.integers(0, 3))]})
        elif modality == "PATHOLOGY_WSI":
            row["num_tiles"] = int(rng.integers(128, 2048))
            row["microns_per_pixel"] = round(float(rng.choice([0.25, 0.5])), 3)
            row["magnification"] = float(rng.choice([20.0, 40.0]))
        elif modality == "PATHOLOGY_TILE":
            row["shape"] = [1, 256, 256]
        rows.append(row)
    return pd.DataFrame(rows)


# --- payload builders ----------------------------------------------------------


def write_nifti(
    path: Path, *, shape: tuple[int, int, int] = (32, 32, 16), affine: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """Write a deterministic int16 NIfTI volume; returns the array."""
    import nibabel as nib

    rng = np.random.default_rng(seed)
    array = rng.integers(-1000, 400, size=shape, dtype=np.int16)
    if affine is None:
        affine = np.array(
            [[-1.5, 0.0, 0.0, 90.0], [0.0, 1.5, 0.0, -120.0], [0.0, 0.0, 2.0, -30.0], [0.0, 0.0, 0.0, 1.0]]
        )
    image = nib.Nifti1Image(array, affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return array


def write_mha(path: Path, *, shape: tuple[int, int, int] = (16, 24, 32), seed: int = 0) -> np.ndarray:
    """Write a deterministic int16 MHA volume (SimpleITK); returns the zyx array."""
    import SimpleITK as sitk

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 500, size=shape, dtype=np.int16)  # (k, j, i) = z, y, x
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.25, 2.5))  # (x, y, z)
    image.SetOrigin((10.0, -5.0, 3.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return array


def write_numpy_volume(
    path: Path, *, shape: tuple[int, int, int] = (24, 20, 18), seed: int = 0, with_meta: bool = True
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = rng.random(shape).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    if with_meta:
        meta = {
            "affine": np.eye(4).tolist(),
            "spacing_mm": [1.0, 1.0, 1.0],
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return array


def write_png(path: Path, *, size: tuple[int, int] = (48, 64), mode: str = "RGB", seed: int = 0) -> np.ndarray:
    from PIL import Image

    rng = np.random.default_rng(seed)
    if mode == "L":
        array = rng.integers(0, 255, size=size[::-1], dtype=np.uint8)
        image = Image.fromarray(array, mode="L")
    else:
        array = rng.integers(0, 255, size=(*size[::-1], 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return array


def _dicom_uid(seed_text: str) -> str:
    """Deterministic DICOM-UID-shaped string (not registered; synthetic only)."""
    digest = hashlib.sha256(seed_text.encode()).hexdigest()[:24]
    components = [str(int(digest[i : i + 6], 16) % 100000) for i in range(0, 24, 6)]
    return "1.2.826.0.1.3680043.9." + ".".join(components)


def write_dicom_series(
    directory: Path,
    *,
    num_slices: int = 8,
    rows: int = 16,
    cols: int = 16,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_spacing: float = 2.0,
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    slope: float = 2.0,
    intercept: float = -1000.0,
    photometric: str = "MONOCHROME2",
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY", "AXIAL"),
    patient_id: str = "12345678",
    series_seed: str = "series-1",
    value_seed: int = 0,
    shuffle_files: bool = False,
    file_prefix: str = "slice",
    per_slice_overrides: dict[int, dict[str, Any]] | None = None,
) -> tuple[str, np.ndarray]:
    """Write a synthetic single-frame CT series; returns (series_uid, raw stack).

    ``raw stack`` is the stored (pre-rescale) int16 pixel data in SLICE ORDER
    (index k = k-th position along the slice normal). ``per_slice_overrides``
    lets tests corrupt individual slices (orientation/spacing/frames/type).
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    rng = np.random.default_rng(value_seed)
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = _dicom_uid(series_seed)
    study_uid = _dicom_uid(f"study-{series_seed}")
    row_cos = np.asarray(orientation[:3], dtype=np.float64)
    col_cos = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row_cos, col_cos)

    raw_stack = np.zeros((num_slices, rows, cols), dtype=np.int16)
    file_names: list[str] = []
    for k in range(num_slices):
        array = rng.integers(0, 2000, size=(rows, cols)).astype(np.int16)
        raw_stack[k] = array
        position = np.asarray(origin, dtype=np.float64) + normal * (k * slice_spacing)

        meta = FileMetaDataset()
        instance_uid = _dicom_uid(f"{series_seed}-instance-{k}")
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = instance_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = instance_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.ImageType = list(image_type)
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = photometric
        ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        ds.ImageOrientationPatient = [float(v) for v in orientation]
        ds.ImagePositionPatient = [float(v) for v in position]
        ds.SliceThickness = float(slice_spacing)
        ds.RescaleSlope = float(slope)
        ds.RescaleIntercept = float(intercept)
        ds.PixelData = array.tobytes()

        overrides = (per_slice_overrides or {}).get(k, {})
        for attr, value in overrides.items():
            setattr(ds, attr, value)

        name = f"{file_prefix}_{k:04d}.dcm"
        file_names.append(name)
        pydicom.dcmwrite(str(directory / name), ds, enforce_file_format=True)

    if shuffle_files:
        # Rename into a scrambled order to prove sorting is physical, not lexical.
        order = rng.permutation(num_slices)
        for original_index, target in enumerate(order):
            (directory / file_names[original_index]).rename(directory / f"{file_prefix}_scrambled_{target:02d}.dcm")
    return series_uid, raw_stack


def write_pyramid_tiff(
    path: Path, *, size: tuple[int, int] = (512, 512), levels: int = 3, mpp: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Write a synthetic OME-TIFF pyramid readable by TiffSlide/OpenSlide.

    The resolution tag encodes MPP (pixels-per-centimetre); returns the
    level-0 RGB array.
    """
    import tifffile

    rng = np.random.default_rng(seed)
    height, width = size
    base = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    pyramid = [base]
    current = base
    for _ in range(levels - 1):
        current = current[::2, ::2]
        pyramid.append(current)
    pixels_per_cm = 10000.0 / mpp
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(path), ome=True) as tiff:
        tiff.write(
            pyramid[0],
            resolution=(pixels_per_cm, pixels_per_cm),
            resolutionunit=3,
            tile=(128, 128),
            subifds=len(pyramid) - 1,
        )
        for level in pyramid[1:]:
            tiff.write(level, subfiletype=1, resolutionunit=3)
    return base


def write_tile_store(
    directory: Path, *, tile_count: int = 6, channels: int = 3, tile_size: int = 16, seed: int = 0
) -> tuple[Any, Any]:
    """Write a pre-extracted tile store (tiles + level-0 coords) as safetensors."""
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(seed)
    tiles = torch.from_numpy(rng.integers(0, 255, size=(tile_count, channels, tile_size, tile_size), dtype=np.uint8))
    coords = torch.from_numpy((np.arange(tile_count) * tile_size).reshape(-1, 1).repeat(2, axis=1).astype(np.int64))
    directory.mkdir(parents=True, exist_ok=True)
    save_file({"tiles": tiles, "coords": coords}, str(directory / "tiles.safetensors"))
    return tiles, coords


def write_embedding_store(root: Path, slide_key: str, *, embeddings: int = 5, dim: int = 8, seed: int = 0) -> None:
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(seed)
    values = torch.from_numpy(rng.standard_normal((embeddings, dim)).astype(np.float32))
    coords = torch.from_numpy(np.arange(embeddings).reshape(-1, 1).repeat(2, axis=1).astype(np.int64) * 256)
    root.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": values, "coords": coords}, str(root / f"{slide_key}.safetensors"))
