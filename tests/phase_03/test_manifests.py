"""Tests for the canonical manifest schema and IO (Phase 03 foundation slice)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from medfm.core.serialization import config_hash
from medfm.data.errors import ManifestError, ManifestSecurityError, ManifestVersionError
from medfm.data.manifests import (
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION_METADATA_KEY,
    inspect_manifest,
    manifest_content_hash,
    manifest_schema_dict,
    read_manifest,
    read_manifest_with_version,
    validate_manifest,
    validate_uri,
    write_manifest,
)


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _valid_frame() -> pd.DataFrame:
    """Small valid mixed-modality manifest (CT_3D, XRAY_2D, PATHOLOGY_WSI, TEXT_ONLY)."""

    def row(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "sample_id": "unset",
            "patient_id_hash": _hash("patient-a"),
            "study_id_hash": _hash("study-a1"),
            "series_id_hash": _hash("series-a1"),
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

    rows = [
        row(
            sample_id="ct-0001",
            modality="CT_3D",
            image_uri="images/ct/0001.nii.gz",
            mask_uri="masks/ct/0001.nii.gz",
            label_json=json.dumps({"task": "BINARY_CLASSIFICATION", "values": [1]}),
            shape=[1, 64, 256, 256],
            spacing_mm=[3.0, 0.7, 0.7],
            num_slices=64,
            image_sha256=_hash("payload-ct-0001"),
        ),
        row(
            sample_id="xray-0002",
            patient_id_hash=_hash("patient-b"),
            study_id_hash=_hash("study-b1"),
            series_id_hash=_hash("series-b1"),
            modality="XRAY_2D",
            image_uri="https://data.example.org/xray/0002.png",
            secondary_image_uris=["images/xray/0002_lat.png"],
            split="VAL",
            scanner_vendor="GE",
        ),
        row(
            sample_id="wsi-0003",
            patient_id_hash=_hash("patient-c"),
            study_id_hash=None,
            series_id_hash=None,
            modality="PATHOLOGY_WSI",
            image_uri="gs://bucket/wsi/0003.svs",
            split="TEST",
            num_tiles=256,
            microns_per_pixel=0.25,
            magnification=40.0,
            shape_bucket_kind="tile_count",
            shape_bucket_shape=[256, 256, 3],
        ),
        row(
            sample_id="text-0004",
            patient_id_hash=_hash("patient-d"),
            study_id_hash=None,
            series_id_hash=None,
            modality="TEXT_ONLY",
            image_uri=None,
            report_uri="s3://restricted-store/reports/0004.json",
            report_chars=1523,
            split=None,
        ),
    ]
    return pd.DataFrame(rows)


def _cell_norm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return [_cell_norm(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_cell_norm(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _assert_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert sorted(str(c) for c in left.columns) == sorted(str(c) for c in right.columns)
    left = left.sort_values("sample_id").reset_index(drop=True)
    right = right.sort_values("sample_id").reset_index(drop=True)
    assert len(left) == len(right)
    for column in left.columns:
        for lval, rval in zip(left[column], right[column], strict=True):
            assert _cell_norm(lval) == _cell_norm(rval), f"column {column}: {lval!r} != {rval!r}"


def _write_raw_parquet(df: pd.DataFrame, path: Path, version: bytes | None) -> None:
    """Write a parquet manifest, stamping (or omitting) the version metadata key directly."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    if version is not None:
        metadata = dict(table.schema.metadata or {})
        metadata[SCHEMA_VERSION_METADATA_KEY] = version
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


# --- valid frame + round trips -------------------------------------------------


def test_valid_frame_passes_validation() -> None:
    validate_manifest(_valid_frame())  # must not raise


def test_parquet_round_trip_preserves_values(tmp_path: Path) -> None:
    df = _valid_frame()
    path = tmp_path / "manifest.parquet"
    write_manifest(df, path, base_dir=tmp_path)
    _assert_frames_equal(df, read_manifest(path))


def test_jsonl_round_trip_preserves_values_and_warns(tmp_path: Path) -> None:
    df = _valid_frame()
    path = tmp_path / "manifest.jsonl"
    with pytest.warns(UserWarning, match="DEBUGGING INTERCHANGE"):
        write_manifest(df, path, base_dir=tmp_path)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line) == {"manifest_schema_version": MANIFEST_SCHEMA_VERSION}
    _assert_frames_equal(df, read_manifest(path))


def test_read_manifest_with_version_reports_stamped_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    write_manifest(_valid_frame(), path, base_dir=tmp_path)
    df, version = read_manifest_with_version(path)
    assert version == MANIFEST_SCHEMA_VERSION
    assert len(df) == 4


def test_write_manifest_rejects_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="unsupported manifest suffix"):
        write_manifest(_valid_frame(), tmp_path / "manifest.csv", base_dir=tmp_path)


def test_write_manifest_validates_before_writing(tmp_path: Path) -> None:
    df = _valid_frame().drop(columns=["license"])
    path = tmp_path / "bad.parquet"
    with pytest.raises(ManifestError):
        write_manifest(df, path)
    assert not path.exists()


# --- column rules ----------------------------------------------------------------


def test_missing_required_column_rejected() -> None:
    df = _valid_frame().drop(columns=["license"])
    with pytest.raises(ManifestError, match="missing required column 'license'"):
        validate_manifest(df)


@pytest.mark.parametrize("column", ["report", "report_text", "report_body"])
def test_embedded_report_text_column_rejected(column: str) -> None:
    df = _valid_frame()
    df[column] = "Patient presents with..."
    with pytest.raises(ManifestError, match="report_uri references only"):
        validate_manifest(df)


def test_report_chars_and_report_uri_columns_allowed() -> None:
    df = _valid_frame()
    df["report_chars"] = [None, None, None, 1523]
    validate_manifest(df)  # must not raise


# --- identifier hygiene ------------------------------------------------------------


@pytest.mark.parametrize("raw_id", ["12345678", "1.2.840.113619"])
def test_raw_patient_identifiers_rejected(raw_id: str) -> None:
    df = _valid_frame()
    df.loc[0, "patient_id_hash"] = raw_id
    with pytest.raises(ManifestError, match="patient_id_hash"):
        validate_manifest(df)


def test_null_patient_id_hash_rejected() -> None:
    df = _valid_frame()
    df.loc[0, "patient_id_hash"] = None
    with pytest.raises(ManifestError, match="patient_id_hash is required"):
        validate_manifest(df)


def test_nullable_study_series_hashes_accepted_when_null() -> None:
    df = _valid_frame()
    df["study_id_hash"] = None
    df["series_id_hash"] = None
    validate_manifest(df)  # must not raise


def test_duplicate_sample_id_rejected() -> None:
    df = _valid_frame()
    df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
    with pytest.raises(ManifestError, match="duplicate sample_id"):
        validate_manifest(df)


# --- enum values -----------------------------------------------------------------


def test_bad_modality_rejected_with_actionable_message() -> None:
    df = _valid_frame()
    df.loc[0, "modality"] = "CT4D"
    with pytest.raises(ManifestError, match=r"unknown Modality value 'CT4D'.*legal values"):
        validate_manifest(df)


def test_bad_split_rejected_with_actionable_message() -> None:
    df = _valid_frame()
    df.loc[0, "split"] = "train"
    with pytest.raises(ManifestError, match=r"unknown SplitName value 'train'.*legal values"):
        validate_manifest(df)


def test_null_split_accepted() -> None:
    df = _valid_frame()
    df["split"] = None
    validate_manifest(df)  # must not raise


# --- URI validation ----------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["ftp", "gopher"])
def test_disallowed_uri_schemes_rejected(scheme: str) -> None:
    with pytest.raises(ManifestSecurityError, match=f"scheme '{scheme}'"):
        validate_uri(f"{scheme}://host/payload.nii.gz", base_dir=None)


def test_disallowed_scheme_in_manifest_collected() -> None:
    df = _valid_frame()
    df.loc[0, "image_uri"] = "ftp://host/payload.nii.gz"
    with pytest.raises(ManifestError, match="ftp"):
        validate_manifest(df)


def test_path_traversal_outside_base_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestSecurityError, match="outside base_dir"):
        validate_uri("../../etc/passwd", base_dir=tmp_path)


def test_dotdot_relative_uri_without_base_dir_rejected() -> None:
    with pytest.raises(ManifestSecurityError, match="no base_dir"):
        validate_uri("../sibling/img.nii.gz", base_dir=None)


def test_relative_path_inside_base_dir_allowed(tmp_path: Path) -> None:
    validate_uri("subdir/img.nii.gz", base_dir=tmp_path)
    validate_uri("subdir/../img.nii.gz", base_dir=tmp_path)  # resolves back inside


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/images/0001.nii.gz",
        "gs://bucket/wsi/0003.svs",
        "https://data.example.org/xray/0002.png",
        "file:///data/images/0001.nii.gz",
        "/absolute/environment/specific/path.nii.gz",
    ],
)
def test_allowed_uri_forms_accepted(uri: str, tmp_path: Path) -> None:
    validate_uri(uri, base_dir=tmp_path)


def test_empty_and_control_character_uris_rejected() -> None:
    with pytest.raises(ManifestSecurityError):
        validate_uri("", base_dir=None)
    with pytest.raises(ManifestSecurityError, match="control characters"):
        validate_uri("images/\t0001.nii.gz", base_dir=None)


# --- modality-conditional image_uri -------------------------------------------------


def test_text_only_row_with_null_image_uri_accepted() -> None:
    df = _valid_frame()
    assert df.loc[3, "modality"] == "TEXT_ONLY"
    assert df.loc[3, "image_uri"] is None
    validate_manifest(df)  # must not raise


def test_ct3d_row_with_null_image_uri_rejected() -> None:
    df = _valid_frame()
    df.loc[0, "image_uri"] = None
    with pytest.raises(ManifestError, match="image_uri is required for modality 'CT_3D'"):
        validate_manifest(df)


# --- governance audit columns --------------------------------------------------------


@pytest.mark.parametrize("column", ["dataset_name", "dataset_version", "license"])
def test_empty_provenance_columns_rejected(column: str) -> None:
    df = _valid_frame()
    df.loc[1, column] = ""
    with pytest.raises(ManifestError, match=rf"{column} is required non-empty"):
        validate_manifest(df)


# --- optional columns -----------------------------------------------------------------


@pytest.mark.parametrize("bad_shape", [[0, 256], [-1, 256], [], ["a", "b"]])
def test_bad_shape_values_rejected(bad_shape: list[Any]) -> None:
    df = _valid_frame()
    df.at[1, "shape"] = bad_shape
    with pytest.raises(ManifestError, match="shape must be a non-empty list of positive ints"):
        validate_manifest(df)


def test_bad_spacing_rejected() -> None:
    df = _valid_frame()
    df.at[0, "spacing_mm"] = [3.0, -0.7]
    with pytest.raises(ManifestError, match="spacing_mm"):
        validate_manifest(df)


def test_bad_image_sha256_rejected() -> None:
    df = _valid_frame()
    df.loc[0, "image_sha256"] = "not-a-sha"
    with pytest.raises(ManifestError, match="image_sha256 must be a 64-char lowercase hex digest"):
        validate_manifest(df)


def test_bad_group_id_hash_rejected() -> None:
    df = _valid_frame()
    df["group_id_hash"] = None
    df.loc[0, "group_id_hash"] = "12345678"
    with pytest.raises(ManifestError, match="group_id_hash"):
        validate_manifest(df)


def test_negative_report_chars_rejected() -> None:
    df = _valid_frame()
    df.loc[3, "report_chars"] = -5
    with pytest.raises(ManifestError, match="report_chars must be a non-negative integer"):
        validate_manifest(df)


def test_shape_bucket_hints_are_optional() -> None:
    df = _valid_frame().drop(columns=["shape_bucket_kind", "shape_bucket_shape"])
    validate_manifest(df)  # hints may be absent entirely


# --- aggregation of problems -----------------------------------------------------------


def test_multiple_problems_collected_into_one_error() -> None:
    df = _valid_frame()
    df.loc[0, "patient_id_hash"] = "12345678"
    df.loc[1, "modality"] = "CT4D"
    with pytest.raises(ManifestError) as excinfo:
        validate_manifest(df)
    message = str(excinfo.value)
    assert "patient_id_hash" in message
    assert "CT4D" in message
    assert "2 problem(s)" in message


# --- inspect + content hash --------------------------------------------------------------


def test_inspect_manifest_is_deterministic(tmp_path: Path) -> None:
    df = _valid_frame()
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    write_manifest(df, path_a, base_dir=tmp_path)
    write_manifest(df, path_b, base_dir=tmp_path)
    first = inspect_manifest(path_a)
    second = inspect_manifest(path_b)
    assert first == second
    assert config_hash(first) == config_hash(second)
    assert first["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert first["format"] == "parquet"
    assert first["row_count"] == 4
    assert first["columns_missing"] == []
    assert first["modality_counts"] == {"CT_3D": 1, "PATHOLOGY_WSI": 1, "TEXT_ONLY": 1, "XRAY_2D": 1}
    assert first["split_counts"] == {"TEST": 1, "TRAIN": 1, "VAL": 1}
    assert first["datasets"] == [{"dataset_name": "synthetic-mixed", "dataset_version": "1.0.0"}]
    assert first["licenses"] == ["CC-BY-4.0"]


def test_manifest_content_hash_deterministic_across_formats(tmp_path: Path) -> None:
    df = _valid_frame()
    direct = manifest_content_hash(df)

    parquet_path = tmp_path / "m.parquet"
    write_manifest(df, parquet_path, base_dir=tmp_path)
    from_parquet = manifest_content_hash(read_manifest(parquet_path))

    jsonl_path = tmp_path / "m.jsonl"
    with pytest.warns(UserWarning):
        write_manifest(df, jsonl_path, base_dir=tmp_path)
    from_jsonl = manifest_content_hash(read_manifest(jsonl_path))

    assert direct == from_parquet == from_jsonl


def test_identical_frames_written_twice_hash_identically(tmp_path: Path) -> None:
    df = _valid_frame()
    hashes = []
    for name in ("first.parquet", "second.parquet"):
        path = tmp_path / name
        write_manifest(df, path, base_dir=tmp_path)
        hashes.append(manifest_content_hash(read_manifest(path)))
    assert hashes[0] == hashes[1]
    assert len(hashes[0]) == 64


# --- versioning / migration ---------------------------------------------------------------


def test_newer_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.parquet"
    _write_raw_parquet(_valid_frame(), path, b"99")
    with pytest.raises(ManifestVersionError, match="newer than supported"):
        read_manifest(path)
    with pytest.raises(ManifestVersionError):
        read_manifest_with_version(path)


def test_missing_version_metadata_assumed_v1(tmp_path: Path) -> None:
    path = tmp_path / "unstamped.parquet"
    _write_raw_parquet(_valid_frame(), path, None)
    df, version = read_manifest_with_version(path)
    assert version == 1
    _assert_frames_equal(_valid_frame(), df)


# --- schema introspection ------------------------------------------------------------------


def test_manifest_schema_dict_covers_required_columns() -> None:
    schema = manifest_schema_dict()
    assert schema["schema_version"] == MANIFEST_SCHEMA_VERSION
    by_name = {column["name"]: column for column in schema["columns"]}
    for name in REQUIRED_COLUMNS:
        assert name in by_name
        assert by_name[name]["required_column"] is True
    assert by_name["report_chars"]["nullable"] is True
    assert schema["allowed_uri_schemes"] == ["file", "gs", "https", "s3"]
    assert config_hash(schema) == config_hash(manifest_schema_dict())  # deterministic
