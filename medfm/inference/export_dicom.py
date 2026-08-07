"""Reviewed highdicom output gates with privacy-safe source references."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medfm.core.sample import SpatialMetadata
from medfm.inference.errors import OptionalDependencyError, RequestValidationError


@dataclass(frozen=True)
class DICOMExportPolicy:
    """Explicit approval required before creating a derived DICOM object."""

    approved: bool = False
    workflow_version: str = "highdicom-reviewed-v1"
    allow_seg: bool = True
    allow_sr: bool = False
    allow_parametric_map: bool = False

    def require(self, kind: str) -> None:
        if not self.approved:
            raise RequestValidationError(
                details={"format": kind, "reason": "reviewed DICOM workflow approval required"}
            )
        if kind == "SEG" and not self.allow_seg:
            raise RequestValidationError(details={"format": kind, "reason": "DICOM SEG disabled by policy"})
        if kind == "SR" and not self.allow_sr:
            raise RequestValidationError(details={"format": kind, "reason": "DICOM SR disabled by policy"})
        if kind == "PARAMETRIC_MAP" and not self.allow_parametric_map:
            raise RequestValidationError(details={"format": kind, "reason": "parametric maps disabled by policy"})


@dataclass(frozen=True)
class DICOMSourceReference:
    """Hashed source references suitable for general operational logs."""

    study_hash: str
    series_hash: str
    frame_of_reference_hash: str | None = None


def hash_dicom_identifier(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def source_reference(dataset: Any) -> DICOMSourceReference:
    return DICOMSourceReference(
        study_hash=hash_dicom_identifier(getattr(dataset, "StudyInstanceUID", "")),
        series_hash=hash_dicom_identifier(getattr(dataset, "SeriesInstanceUID", "")),
        frame_of_reference_hash=(
            hash_dicom_identifier(dataset.FrameOfReferenceUID)
            if getattr(dataset, "FrameOfReferenceUID", None)
            else None
        ),
    )


def validate_derived_geometry(
    source_shape: Sequence[int],
    derived_shape: Sequence[int],
    *,
    source_affine: np.ndarray | torch.Tensor | None = None,
    derived_affine: np.ndarray | torch.Tensor | None = None,
    tolerance: float = 1e-5,
) -> None:
    if tuple(int(value) for value in source_shape) != tuple(int(value) for value in derived_shape):
        raise RequestValidationError(details={"reason": "derived DICOM geometry shape differs from source"})
    if source_affine is not None and derived_affine is not None:
        source = np.asarray(
            source_affine.detach().to("cpu") if isinstance(source_affine, torch.Tensor) else source_affine
        )
        derived = np.asarray(
            derived_affine.detach().to("cpu") if isinstance(derived_affine, torch.Tensor) else derived_affine
        )
        if source.shape != derived.shape or not np.allclose(source, derived, atol=tolerance):
            raise RequestValidationError(details={"reason": "derived DICOM geometry affine differs from source"})


def _require_dicom() -> tuple[Any, Any]:
    try:
        import highdicom
        import pydicom
    except ImportError as exc:
        raise OptionalDependencyError(details={"extra": "medical", "format": "highdicom"}) from exc
    return highdicom, pydicom


def export_dicom_seg(
    mask: torch.Tensor,
    source_datasets: Sequence[Any],
    path: str | Path,
    *,
    metadata: SpatialMetadata | None = None,
    policy: DICOMExportPolicy | None = None,
    label: str = "prediction",
) -> Path:
    """Create and reopen a DICOM SEG only through an approved highdicom path.

    The exact highdicom constructor differs across minor releases, so this
    function accepts only the reviewed constructor surface and fails closed
    rather than falling back to hand-written DICOM tags.
    """

    export_policy = policy or DICOMExportPolicy()
    export_policy.require("SEG")
    highdicom, pydicom = _require_dicom()
    if not source_datasets:
        raise RequestValidationError(details={"field": "source_datasets"})
    if mask.ndim < 3:
        raise RequestValidationError(details={"field": "mask", "reason": "DICOM SEG mask must be 3D"})
    source_shape = tuple(int(value) for value in source_datasets[0].pixel_array.shape)
    derived = mask.detach().to("cpu").numpy()
    derived_shape = tuple(int(value) for value in derived.shape[-2:])
    if len(source_shape) == 2 and derived_shape != source_shape:
        raise RequestValidationError(details={"reason": "mask in-plane geometry differs from source"})
    if metadata is not None:
        validate_derived_geometry(metadata.original_shape[-2:], derived_shape)
    try:
        # highdicom owns UID generation and source-reference semantics.  Do not
        # expose source UID values in exceptions or audit logs.
        segment_description = highdicom.seg.SegmentDescription(
            segment_number=1,
            segment_label=str(label)[:64],
            segmented_property_category=highdicom.sr.CodedConcept("M-01000", "SRT", "Anatomical Structure"),
            segmented_property_type=highdicom.sr.CodedConcept("T-D0050", "SRT", "Tissue"),
            algorithm_type=highdicom.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_name="medfm",
        )
        # The source-image helper keeps geometry/reference tags coherent.
        segmentation = highdicom.seg.Segmentation(
            source_images=list(source_datasets),
            pixel_array=(derived > 0).astype(np.uint8),
            segmentation_type=highdicom.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[segment_description],
            series_instance_uid=pydicom.uid.generate_uid(),
            series_number=1,
            sop_instance_uid=pydicom.uid.generate_uid(),
            instance_number=1,
            manufacturer="medfm",
            manufacturer_model_name="medfm-inference",
            software_versions=export_policy.workflow_version,
            device_serial_number="medfm",
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        segmentation.save_as(str(output))
        reopened = pydicom.dcmread(str(output), stop_before_pixels=False)
        if not hasattr(reopened, "SegmentSequence") or not hasattr(reopened, "PixelData"):
            raise RequestValidationError(details={"reason": "derived DICOM SEG could not be reopened"})
        return output
    except RequestValidationError:
        raise
    except Exception as exc:
        raise RequestValidationError(details={"reason": "reviewed highdicom SEG workflow failed"}) from exc


__all__ = [
    "DICOMExportPolicy",
    "DICOMSourceReference",
    "export_dicom_seg",
    "hash_dicom_identifier",
    "source_reference",
    "validate_derived_geometry",
]
