"""Manifest-row -> MedicalSample bridge (the reader contract handed to Phase 04)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from medfm.core.enums import Modality, SplitName, TaskType
from medfm.data.readers.base import sample_from_manifest_row
from medfm.data.readers.pathology import TiffSlideReader
from medfm.data.readers.radiology import NiftiReader
from phase_03.synthetic import manifest_row, write_nifti, write_pyramid_tiff


def test_volumetric_row_builds_contract_sample(tmp_path: Path) -> None:
    nii = tmp_path / "ct.nii.gz"
    write_nifti(nii, shape=(16, 16, 8), seed=1)
    read = NiftiReader().read(nii)
    row = pd.Series(
        manifest_row(
            sample_id="ct-1",
            modality="CT_3D",
            image_uri=str(nii),
            label_json=json.dumps({"task": "BINARY_CLASSIFICATION", "values": [1]}),
        )
    )
    sample = sample_from_manifest_row(row, read)
    assert sample.modality is Modality.CT_3D
    assert sample.spatial is not None
    assert sample.spatial.affine.shape == (4, 4)
    assert sample.labels is not None and sample.labels.task is TaskType.BINARY_CLASSIFICATION
    assert sample.provenance.dataset_name == "synthetic-mixed"
    assert sample.provenance.split is SplitName.TRAIN
    assert sample.image_references and sample.image_references[0].uri == str(nii)
    sample.validate_for_task(TaskType.BINARY_CLASSIFICATION)


def test_wsi_row_builds_contract_sample(tmp_path: Path) -> None:
    slide = tmp_path / "slide.tif"
    write_pyramid_tiff(slide, size=(256, 256), levels=2, mpp=0.5, seed=2)
    reader = TiffSlideReader(slide)
    from medfm.data.readers.base import PayloadRead

    read = PayloadRead(tensors={"image": reader.thumbnail((64, 64))}, pathology=reader.pathology_metadata())
    row = pd.Series(manifest_row(sample_id="wsi-1", modality="PATHOLOGY_WSI", image_uri=str(slide)))
    sample = sample_from_manifest_row(row, read)
    assert sample.modality is Modality.PATHOLOGY_WSI
    assert sample.pathology is not None
    assert sample.pathology.microns_per_pixel == pytest.approx(0.5)
    assert sample.spatial is None
    reader.close()


def test_text_only_row_needs_no_payload() -> None:
    row = pd.Series(
        manifest_row(
            sample_id="text-1",
            modality="TEXT_ONLY",
            image_uri=None,
            report_uri="s3://store/report.json",
            split=None,
        )
    )
    # TEXT_ONLY with no payload must still fail the contract (needs text content);
    # the bridge constructs references/metadata only, so this must raise.
    from medfm.core.errors import SchemaValidationError

    with pytest.raises(SchemaValidationError):
        sample_from_manifest_row(row, None)
