"""Governance: exceptions/logs never expose raw identifiers or report text."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from medfm.data.errors import ManifestError, SplitLeakageError, UnsupportedFormatError
from medfm.data.manifests.schema import validate_manifest
from medfm.data.readers.dicom import DICOMSeriesReader, discover_dicom_series
from medfm.data.splits import SplitPolicy, assert_no_split_leakage, generate_split_assignment
from phase_03.synthetic import hid, manifest_row, write_dicom_series

RAW_MRN = "MRN-0009182736"
RAW_UID = "1.2.840.113619.2.55.3.604688"
REPORT_TEXT = "Patient presents with a 2.3 cm lesion in the right upper lobe."


def test_manifest_rejects_raw_identifiers_not_just_hashes() -> None:
    df = pd.DataFrame([manifest_row(sample_id="x", patient_id_hash=RAW_MRN)])
    with pytest.raises(ManifestError, match="patient_id_hash") as excinfo:
        validate_manifest(df)
    # The raw MRN must be rejected AND never echoed back into the message.
    assert RAW_MRN not in str(excinfo.value)


def test_manifest_rejects_embedded_report_text_columns() -> None:
    df = pd.DataFrame([manifest_row(sample_id="x")])
    df["report_body"] = REPORT_TEXT
    with pytest.raises(ManifestError, match="report_uri references only") as excinfo:
        validate_manifest(df)
    assert REPORT_TEXT not in str(excinfo.value), "report text must not be echoed in the error"


def test_dicom_errors_do_not_leak_raw_uids_or_mrn(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    uid, _ = write_dicom_series(
        series_dir,
        num_slices=1,
        patient_id=RAW_MRN,
        series_seed="leak-check",
        per_slice_overrides={0: {"NumberOfFrames": 3}},  # multiframe -> actionable failure
    )
    (series_hash,) = discover_dicom_series(series_dir)
    with pytest.raises(UnsupportedFormatError) as excinfo:
        DICOMSeriesReader().read(series_dir, series_id_hash=series_hash)
    message = str(excinfo.value)
    assert RAW_MRN not in message
    assert RAW_UID not in message
    assert uid not in message  # raw series UID never appears; only file names / hashes may


def test_split_leakage_error_reports_hashes_only() -> None:
    rows = []
    for i in range(6):
        rows.append(manifest_row(sample_id=f"s{i}", patient_id_hash=hid(f"p{i}"), image_uri=f"img/{i}.nii.gz"))
    df = generate_split_assignment(pd.DataFrame(rows), policy=SplitPolicy.PATIENT, seed=1)
    leaked = df.copy()
    leaked.loc[leaked.index[-1], "patient_id_hash"] = leaked["patient_id_hash"].iloc[0]
    leaked.loc[leaked.index[-1], "split"] = "TEST" if leaked.loc[leaked.index[-1], "split"] != "TEST" else "VAL"
    with pytest.raises(SplitLeakageError) as excinfo:
        assert_no_split_leakage(leaked)
    message = str(excinfo.value)
    # Hashes may appear; raw patient labels must not.
    assert "p0" not in message.replace(hid("p0"), "")
    assert hid("p0") in message or "patient_id_hash" in message


def test_data_layer_logging_carries_no_phi(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        df = pd.DataFrame([manifest_row(sample_id="x")])
        df["report_text"] = REPORT_TEXT
        with pytest.raises(ManifestError):
            validate_manifest(df)
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert REPORT_TEXT not in combined
    assert RAW_MRN not in combined
