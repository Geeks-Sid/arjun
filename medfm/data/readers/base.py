"""Shared reader contract for radiology and pathology payloads.

A reader decodes ONE payload location into a :class:`PayloadRead`: CPU
tensors with canonical dtypes plus geometry metadata (:class:`SpatialMetadata`
for radiology, :class:`PathologyMetadata` for pathology) and *hashed*
source metadata. Readers never return raw identifiers (UIDs are hashed at
this boundary; ``medfm/data/errors.py`` privacy rule) and never discard
affine/spacing/orientation or MPP/pyramid geometry.

Volumetric axis contract: every volumetric reader returns its image array in
voxel order ``(i, j, k)`` such that ``affine @ [i, j, k, 1]`` yields patient
coordinates in millimetres — the nibabel convention, applied uniformly to
NIfTI, MHA, and DICOM series. Callers needing original-space output mapping
(mask overlays, DICOM re-export) use the preserved affine +
``frame_of_reference_hash``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import pandas as pd
import torch

from medfm.core.enums import Modality
from medfm.core.errors import SchemaValidationError
from medfm.core.sample import ImageReference, LabelTarget, MedicalSample, ProvenanceMetadata
from medfm.core.sample import patient_id_hash as _patient_id_hash
from medfm.core.sample import series_id_hash as _series_id_hash
from medfm.core.sample import study_id_hash as _study_id_hash
from medfm.data.errors import ReaderError

#: Bump when the reader contract changes meaning (cache keys carry the
#: per-reader version separately; this is the contract generation).
READER_CONTRACT_VERSION = 1

#: Suffixes each radiology reader claims (used by :func:`reader_for_path`).
_VOLUME_SUFFIXES = {".nii", ".gz", ".mha", ".mhd", ".npy"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def hash_identifier(value: str) -> str:
    """SHA-256 of an identifier (UID, MRN, accession) — the ONLY form of a
    raw identifier allowed past a reader boundary."""
    if not value:
        raise ReaderError("cannot hash an empty identifier")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_local_path(uri: str, *, base_dir: Path | None = None) -> Path:
    """Resolve a manifest URI to a local filesystem path for reading.

    Only scheme-less paths and ``file://`` URIs are readable locally;
    object-store URIs need a storage-backed loader (out of Phase 03 scope).
    Relative paths are anchored at ``base_dir`` (required in that case).
    """
    if not uri:
        raise ReaderError("cannot read an empty URI; the manifest row has no payload reference")
    path_str = uri
    if uri.startswith("file://"):
        parts = urlsplit(uri)
        if parts.netloc not in ("", "localhost"):
            raise ReaderError(f"file:// URI with remote host {parts.netloc!r} is not readable locally")
        path_str = parts.path
    elif "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ReaderError(
            f"URI {uri!r} uses scheme {scheme!r}; only local paths and file:// URIs are readable by readers — "
            "materialize the payload locally or use a storage-backed loader"
        )
    path = Path(path_str)
    if not path.is_absolute():
        if base_dir is None:
            raise ReaderError(f"relative URI {uri!r} needs base_dir to be resolved; pass the dataset root as base_dir")
        path = base_dir / path
    if not path.exists():
        raise ReaderError(f"payload not found at {path}; check the manifest URI and dataset root")
    return path


@dataclass(frozen=True)
class PayloadRead:
    """One decoded payload: tensors + geometry + hashed source metadata.

    ``tensors`` always contains ``"image"``; ``"mask"`` is present when a
    segmentation payload was read alongside. All tensors are CPU with
    canonical dtypes (accelerator-neutral, cache-safe).
    """

    tensors: dict[str, torch.Tensor]
    spatial: Any | None = None  # SpatialMetadata | None (typed in radiology readers)
    pathology: Any | None = None  # PathologyMetadata | None (typed in pathology readers)
    source_metadata: dict[str, Any] = field(default_factory=dict)  # hashed IDs + acquisition facts only

    @property
    def image(self) -> torch.Tensor:
        return self.tensors["image"]


@runtime_checkable
class Reader(Protocol):
    """The contract every payload reader implements."""

    #: Stable machine-readable reader identity (used in cache keys).
    reader_id: str
    #: Reader behavior version; any decode-behavior change must bump it so
    #: caches invalidate (``medfm.data.caching.keys.CacheKey.reader_version``).
    reader_version: str

    def supports(self, path: Path) -> bool:
        """Whether this reader claims ``path`` (suffix/sniff based)."""
        ...

    def read(self, path: Path) -> PayloadRead:
        """Decode ``path``; raise a :class:`ReaderError` subclass on failure."""
        ...


def _row_value(row: pd.Series, column: str) -> Any | None:
    if column not in row.index:
        return None
    value = row[column]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _label_from_json(label_json: str | None) -> LabelTarget | None:
    if label_json is None:
        return None
    import json

    payload = json.loads(label_json)
    if not isinstance(payload, dict):
        raise ReaderError(f"label_json must decode to an object; got {type(payload).__name__}")
    from medfm.core.enums import TaskType

    task = TaskType.from_value(str(payload["task"]))
    values = tuple(float(v) for v in payload["values"])
    class_names = tuple(str(n) for n in payload["class_names"]) if payload.get("class_names") else None
    return LabelTarget(task=task, values=values, class_names=class_names)


def sample_from_manifest_row(row: pd.Series, read: PayloadRead | None = None) -> MedicalSample:
    """Build a :class:`MedicalSample` from a manifest row (+ decoded payload).

    The row supplies identity/provenance (never recomputed); ``read`` supplies
    geometry metadata decoded from the payload. Manifest columns map 1:1 onto
    the sample contract (Phase 02 handoff).
    """
    modality = Modality.from_value(str(row["modality"]))
    provenance = ProvenanceMetadata(
        dataset_name=str(row["dataset_name"]),
        dataset_version=str(row["dataset_version"]),
        split=None if _row_value(row, "split") is None else _split_from_row(str(row["split"])),
        site_id=_row_value(row, "site_id"),
        license=_row_value(row, "license"),
        source_uri=_row_value(row, "provenance_uri"),
        source_sha256=_row_value(row, "image_sha256"),
        acquisition_date_bucket=_row_value(row, "acquisition_date_bucket"),
    )

    references: list[ImageReference] = []
    image_uri = _row_value(row, "image_uri")
    if image_uri is not None:
        references.append(ImageReference(uri=str(image_uri), role="primary", sha256=_row_value(row, "image_sha256")))
    secondary = _row_value(row, "secondary_image_uris")
    if secondary is not None:
        from medfm.data.manifests.schema import coerce_list_cell

        for item in coerce_list_cell(secondary) or []:
            references.append(ImageReference(uri=str(item), role="secondary"))
    mask_uri = _row_value(row, "mask_uri")
    if mask_uri is not None:
        references.append(ImageReference(uri=str(mask_uri), role="mask"))

    study_hash = _row_value(row, "study_id_hash")
    series_hash = _row_value(row, "series_id_hash")
    return MedicalSample(
        sample_id=str(row["sample_id"]),
        patient_id_hash=_patient_id_hash(str(row["patient_id_hash"])),
        study_id_hash=_study_id_hash(str(study_hash)) if study_hash is not None else None,
        series_id_hash=_series_id_hash(str(series_hash)) if series_hash is not None else None,
        modality=modality,
        provenance=provenance,
        image_references=tuple(references),
        labels=_label_from_json(_row_value(row, "label_json")),
        spatial=read.spatial if read is not None else None,
        pathology=read.pathology if read is not None else None,
    )


def _split_from_row(value: str) -> Any:
    from medfm.core.enums import SplitName

    return SplitName.from_value(value)


def reader_for_path(path: Path) -> Reader:
    """Pick the radiology reader for ``path`` by suffix (no payload decoding).

    DICOM is directory-based and handled by :class:`DICOMSeriesReader`
    directly; WSI formats are handled by the slide readers.
    """
    from medfm.data.readers.radiology import MHAReader, NiftiReader, NumpyVolumeReader, PngJpegReader

    suffix = path.suffix.lower()
    if suffix == ".gz" and path.name.endswith(".nii.gz"):
        return NiftiReader()
    if suffix in (".nii",):
        return NiftiReader()
    if suffix in (".mha", ".mhd"):
        return MHAReader()
    if suffix == ".npy":
        return NumpyVolumeReader()
    if suffix in _IMAGE_SUFFIXES:
        return PngJpegReader()
    raise ReaderError(
        f"no reader claims {path.name!r} (suffix {suffix!r}); supported radiology suffixes: "
        f"{sorted(_VOLUME_SUFFIXES | _IMAGE_SUFFIXES)} — for DICOM pass the series directory to DICOMSeriesReader"
    )


def validate_canonical_tensor(name: str, tensor: torch.Tensor) -> None:
    """Boundary check: reader outputs must be CPU tensors with canonical dtypes."""
    from medfm.core.serialization import canonical_dtype_name

    if tensor.device.type != "cpu":
        raise SchemaValidationError(f"reader tensor {name!r} must be on CPU; got device {tensor.device}")
    canonical_dtype_name(tensor.dtype)  # raises SerializationError (non-canonical) — let it propagate typed
