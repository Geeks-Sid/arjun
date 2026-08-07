from __future__ import annotations

import numpy as np
import pytest
import torch

from medfm.core.sample import SpatialMetadata
from medfm.inference import DICOMExportPolicy, validate_derived_geometry
from medfm.inference.errors import RequestValidationError


def test_derived_geometry_requires_shape_and_affine_match() -> None:
    affine = np.eye(4)
    validate_derived_geometry((4, 5, 6), (4, 5, 6), source_affine=affine, derived_affine=affine.copy())
    with pytest.raises(RequestValidationError):
        validate_derived_geometry((4, 5, 6), (4, 5, 5))
    shifted = affine.copy()
    shifted[0, 3] = 1.0
    with pytest.raises(RequestValidationError):
        validate_derived_geometry((4, 5, 6), (4, 5, 6), source_affine=affine, derived_affine=shifted)


def test_dicom_export_requires_explicit_reviewed_workflow() -> None:
    with pytest.raises(RequestValidationError):
        DICOMExportPolicy().require("SEG")
    DICOMExportPolicy(approved=True).require("SEG")


def test_nifti_export_roundtrip_when_medical_extra_is_available(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    from medfm.inference import export_nifti, reopen_and_validate_nifti

    metadata = SpatialMetadata(
        original_shape=(4, 5, 6),
        current_shape=(4, 5, 6),
        original_affine=torch.eye(4),
        affine=torch.eye(4),
        spacing_mm=(1.0, 1.0, 1.0),
        orientation="RAS",
    )
    output = export_nifti(torch.ones(4, 5, 6), tmp_path / "mask.nii.gz", metadata=metadata)
    assert nib.load(str(output)).shape == (4, 5, 6)
    reopen_and_validate_nifti(output, metadata)
