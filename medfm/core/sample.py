"""Core sample schema: MedicalSample and its nested metadata types.

A ``MedicalSample`` is the unit exchanged between readers (Phase 03),
preprocessing (Phase 04), and collators. It carries references and metadata,
never raw payloads. Structural invariants are enforced at construction time
(``__post_init__``); task-specific requirements are checked with
:meth:`MedicalSample.validate_for_task`.

All ``*_id_hash`` fields accept only hexadecimal digests (32–128 chars). Raw
identifiers (MRNs, accession numbers, DICOM UIDs) fail validation — clinical
data is de-identified before entering this environment, and these fields are
the enforcement point.

Serialization note: small metadata tensors (affines, coordinates) serialize
inline as nested lists; this is metadata, not payload. Round-trips are
lossless and always land on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, NewType, cast

import torch

from medfm.core.enums import CoordinateSystem, Modality, SplitName, TaskType
from medfm.core.errors import IdentifierError, SchemaValidationError
from medfm.core.serialization import tensor_from_data, tensor_to_data
from medfm.core.versioning import SCHEMA_VERSION, migrate_payload

#: Typed de-identified identifiers. The NewTypes make accidental mixing of
#: patient/study/series hashes a type error, and the factories reject values
#: that look like raw identifiers.
SampleId = NewType("SampleId", str)
PatientIdHash = NewType("PatientIdHash", str)
StudyIdHash = NewType("StudyIdHash", str)
SeriesIdHash = NewType("SeriesIdHash", str)

_HASH_RE = re.compile(r"[0-9a-f]{32,128}")
#: Patterns that indicate a raw (unhashed) clinical identifier.
_DICOM_UID_RE = re.compile(r"[0-9]+(\.[0-9]+)+")
_MRN_RE = re.compile(r"[0-9]{4,}")


def _hash_id(value: str, field_name: str) -> str:
    if _DICOM_UID_RE.fullmatch(value):
        raise IdentifierError(f"{field_name} looks like a raw DICOM UID; store only a salted hash of the identifier")
    if _MRN_RE.fullmatch(value):
        raise IdentifierError(f"{field_name} looks like a raw numeric identifier (MRN); store only a hash")
    if not _HASH_RE.fullmatch(value):
        raise IdentifierError(
            f"{field_name} must be a lowercase hex digest (32–128 chars); got {value!r}. "
            "Raw identifiers are not accepted anywhere in the contract layer."
        )
    return value


def patient_id_hash(value: str) -> PatientIdHash:
    return PatientIdHash(_hash_id(value, "patient_id_hash"))


def study_id_hash(value: str) -> StudyIdHash:
    return StudyIdHash(_hash_id(value, "study_id_hash"))


def series_id_hash(value: str) -> SeriesIdHash:
    return SeriesIdHash(_hash_id(value, "series_id_hash"))


def _optional_str_tuple(value: Any, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(str(v) for v in value)
    if not items or any(not v for v in items):
        raise SchemaValidationError(f"{field_name} must be a non-empty tuple of non-empty strings")
    return items


def _shape_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    shape = tuple(int(d) for d in value)
    if not shape or any(d <= 0 for d in shape):
        raise SchemaValidationError(f"{field_name} must be a non-empty tuple of positive ints; got {value!r}")
    return shape


def _optional_matrix(value: Any, field_name: str) -> torch.Tensor | None:
    """Validate an optional square affine matrix stored as a tensor."""
    if value is None:
        return None
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] not in (3, 4):
        raise SchemaValidationError(f"{field_name} must be a 3x3 or 4x4 matrix tensor; got shape {tuple(value.shape)}")
    if not value.dtype.is_floating_point:
        raise SchemaValidationError(f"{field_name} must have a floating dtype; got {value.dtype}")
    return cast(torch.Tensor, value)


def _dim2(value: Any, field_name: str) -> tuple[int, int]:
    """Validate a (width, height)-style pair of positive ints."""
    pair = tuple(int(d) for d in value)
    if len(pair) != 2 or any(d <= 0 for d in pair):
        raise SchemaValidationError(f"{field_name} must be a (width, height) pair of positive ints")
    return (pair[0], pair[1])


@dataclass(frozen=True)
class ImageReference:
    """A reference to one image/series payload, never the payload itself."""

    uri: str
    role: str = "primary"  # primary | secondary | localizer | mask | thumbnail
    frame_index: int | None = None  # frame within a multi-frame object
    sha256: str | None = None  # payload content hash, when known

    def __post_init__(self) -> None:
        if not self.uri:
            raise SchemaValidationError("ImageReference.uri must be non-empty")
        if self.frame_index is not None and self.frame_index < 0:
            raise SchemaValidationError("ImageReference.frame_index must be >= 0")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise SchemaValidationError("ImageReference.sha256 must be a 64-char lowercase hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "role": self.role,
            "frame_index": self.frame_index,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageReference:
        return cls(
            uri=str(data["uri"]),
            role=str(data.get("role", "primary")),
            frame_index=data.get("frame_index"),
            sha256=data.get("sha256"),
        )


@dataclass(frozen=True)
class LabelTarget:
    """Classification-style labels.

    ``values`` semantics by ``task``:
    - BINARY: one value in {0, 1}
    - MULTICLASS / ORDINAL: one class index in [0, num_classes)
    - MULTILABEL: one 0/1 entry per class (length == num_classes)
    """

    task: TaskType
    values: tuple[float, ...]
    class_names: tuple[str, ...] | None = None

    _LABEL_TASKS = (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.ORDINAL_CLASSIFICATION,
    )

    def __post_init__(self) -> None:
        if self.task not in self._LABEL_TASKS:
            raise SchemaValidationError(f"LabelTarget.task must be a classification task; got {self.task}")
        if not self.values:
            raise SchemaValidationError("LabelTarget.values must be non-empty")
        if self.task is TaskType.BINARY_CLASSIFICATION:
            if len(self.values) != 1 or self.values[0] not in (0.0, 1.0):
                raise SchemaValidationError("BINARY labels must be a single 0/1 value")
        elif self.task is TaskType.MULTILABEL_CLASSIFICATION:
            if any(v not in (0.0, 1.0) for v in self.values):
                raise SchemaValidationError("MULTILABEL labels must be 0/1 per class")
            if self.class_names is not None and len(self.class_names) != len(self.values):
                raise SchemaValidationError("MULTILABEL class_names must match values length")
        else:  # MULTICLASS / ORDINAL
            if len(self.values) != 1 or int(self.values[0]) != self.values[0] or self.values[0] < 0:
                raise SchemaValidationError(f"{self.task} label must be a single non-negative class index")
            if self.class_names is not None and int(self.values[0]) >= len(self.class_names):
                raise SchemaValidationError("class index out of range for class_names")

    @property
    def num_classes(self) -> int | None:
        if self.task is TaskType.MULTILABEL_CLASSIFICATION:
            return len(self.values)
        if self.class_names is not None:
            return len(self.class_names)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "values": list(self.values),
            "class_names": list(self.class_names) if self.class_names is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, schema_version: int | None = None) -> LabelTarget:
        names = data.get("class_names")
        return cls(
            task=TaskType.from_value(str(data["task"]), schema_version=schema_version),
            values=tuple(float(v) for v in data["values"]),
            class_names=tuple(str(n) for n in names) if names is not None else None,
        )


@dataclass(frozen=True)
class SegmentationTarget:
    """Reference to a segmentation mask plus its class map."""

    mask_uri: str
    class_index_to_name: dict[int, str]
    instance_aware: bool = False

    def __post_init__(self) -> None:
        if not self.mask_uri:
            raise SchemaValidationError("SegmentationTarget.mask_uri must be non-empty")
        if not self.class_index_to_name:
            raise SchemaValidationError("SegmentationTarget.class_index_to_name must be non-empty")
        if any(k < 0 for k in self.class_index_to_name):
            raise SchemaValidationError("class indices must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_uri": self.mask_uri,
            "class_index_to_name": {str(k): v for k, v in sorted(self.class_index_to_name.items())},
            "instance_aware": self.instance_aware,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationTarget:
        return cls(
            mask_uri=str(data["mask_uri"]),
            class_index_to_name={int(k): str(v) for k, v in data["class_index_to_name"].items()},
            instance_aware=bool(data.get("instance_aware", False)),
        )


@dataclass(frozen=True)
class BoxTarget:
    """Bounding boxes with an explicit coordinate system.

    Each box has 4 (2D: x0, y0, x1, y1) or 6 (3D: x0, y0, z0, x1, y1, z1)
    coordinates interpreted in ``coordinate_system``.
    """

    boxes: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]
    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None:
        if len(self.boxes) != len(self.labels):
            raise SchemaValidationError(f"BoxTarget has {len(self.boxes)} boxes but {len(self.labels)} labels")
        for box in self.boxes:
            if len(box) not in (4, 6):
                raise SchemaValidationError(f"boxes must have 4 (2D) or 6 (3D) coordinates; got {len(box)}")
            half = len(box) // 2
            if any(box[i + half] < box[i] for i in range(half)):
                raise SchemaValidationError(f"box max coordinates must be >= min coordinates; got {box}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": [list(b) for b in self.boxes],
            "labels": list(self.labels),
            "coordinate_system": self.coordinate_system.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, schema_version: int | None = None) -> BoxTarget:
        return cls(
            boxes=tuple(tuple(float(c) for c in b) for b in data["boxes"]),
            labels=tuple(str(lb) for lb in data["labels"]),
            coordinate_system=CoordinateSystem.from_value(
                str(data["coordinate_system"]), schema_version=schema_version
            ),
        )


@dataclass(frozen=True)
class ConversationTurn:
    """One instruction-tuning turn."""

    role: str  # system | user | assistant
    content: str

    _ROLES = ("system", "user", "assistant")

    def __post_init__(self) -> None:
        if self.role not in self._ROLES:
            raise SchemaValidationError(f"ConversationTurn.role must be one of {self._ROLES}; got {self.role!r}")
        if not self.content:
            raise SchemaValidationError("ConversationTurn.content must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(role=str(data["role"]), content=str(data["content"]))


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Where a sample came from. Required on every sample."""

    dataset_name: str
    dataset_version: str
    split: SplitName | None = None
    site_id: str | None = None  # site identifier, itself de-identified
    license: str | None = None  # dataset license identifier (e.g. "CC-BY-4.0")
    source_uri: str | None = None
    source_sha256: str | None = None
    acquisition_date_bucket: str | None = None  # e.g. "2019-Q3"; never an exact date
    deidentification_method: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise SchemaValidationError("ProvenanceMetadata.dataset_name must be non-empty")
        if not self.dataset_version:
            raise SchemaValidationError("ProvenanceMetadata.dataset_version must be non-empty")
        if self.source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise SchemaValidationError("ProvenanceMetadata.source_sha256 must be a 64-char hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "split": self.split.value if self.split is not None else None,
            "site_id": self.site_id,
            "license": self.license,
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "acquisition_date_bucket": self.acquisition_date_bucket,
            "deidentification_method": self.deidentification_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, schema_version: int | None = None) -> ProvenanceMetadata:
        split = data.get("split")
        return cls(
            dataset_name=str(data["dataset_name"]),
            dataset_version=str(data["dataset_version"]),
            split=SplitName.from_value(str(split), schema_version=schema_version) if split is not None else None,
            site_id=data.get("site_id"),
            license=data.get("license"),
            source_uri=data.get("source_uri"),
            source_sha256=data.get("source_sha256"),
            acquisition_date_bucket=data.get("acquisition_date_bucket"),
            deidentification_method=data.get("deidentification_method"),
        )


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class SpatialMetadata:
    """Geometry of a radiology sample. Never discard during preprocessing.

    ``affine``/``original_affine`` map voxel indices to patient-space
    millimetres (current / pre-resampling geometry). ``slice_positions_mm``
    holds per-slice patient positions when slices are not uniformly spaced.
    """

    original_shape: tuple[int, ...]
    current_shape: tuple[int, ...]
    affine: torch.Tensor | None = None  # 4x4 (or 3x3 for 2D), float
    original_affine: torch.Tensor | None = None
    spacing_mm: tuple[float, ...] | None = None
    orientation: str | None = None  # e.g. "RAS", "LPS"
    anatomical_axes: tuple[str, ...] | None = None  # per-axis labels, e.g. ("R", "A", "S")
    slice_positions_mm: torch.Tensor | None = None  # [D] float
    frame_of_reference_hash: str | None = None

    _AXIS_LABELS = ("R", "L", "A", "P", "S", "I")

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_shape", _shape_tuple(self.original_shape, "original_shape"))
        object.__setattr__(self, "current_shape", _shape_tuple(self.current_shape, "current_shape"))
        if len(self.original_shape) != len(self.current_shape):
            raise SchemaValidationError("original_shape and current_shape must have the same rank")
        _optional_matrix(self.affine, "affine")
        _optional_matrix(self.original_affine, "original_affine")
        if self.spacing_mm is not None:
            spacing = tuple(float(s) for s in self.spacing_mm)
            if len(spacing) != len(self.current_shape) or any(s <= 0 for s in spacing):
                raise SchemaValidationError(
                    "spacing_mm must be positive and match the spatial rank; "
                    f"got {self.spacing_mm!r} for shape {self.current_shape}"
                )
            object.__setattr__(self, "spacing_mm", spacing)
        if self.anatomical_axes is not None:
            axes = _optional_str_tuple(self.anatomical_axes, "anatomical_axes")
            assert axes is not None
            if len(axes) != len(self.current_shape) or any(a not in self._AXIS_LABELS for a in axes):
                raise SchemaValidationError(
                    f"anatomical_axes must be one of {self._AXIS_LABELS} per spatial dim; got {axes}"
                )
        if self.slice_positions_mm is not None:
            if self.slice_positions_mm.ndim != 1 or not self.slice_positions_mm.dtype.is_floating_point:
                raise SchemaValidationError("slice_positions_mm must be a 1D float tensor")
        if self.frame_of_reference_hash is not None:
            _hash_id(self.frame_of_reference_hash, "frame_of_reference_hash")

    @property
    def spatial_rank(self) -> int:
        return len(self.current_shape)

    def to(self, device: torch.device | str) -> SpatialMetadata:
        """Return a copy with tensor fields moved to ``device`` (metadata preserved)."""
        return SpatialMetadata(
            original_shape=self.original_shape,
            current_shape=self.current_shape,
            affine=self.affine.to(device) if self.affine is not None else None,
            original_affine=self.original_affine.to(device) if self.original_affine is not None else None,
            spacing_mm=self.spacing_mm,
            orientation=self.orientation,
            anatomical_axes=self.anatomical_axes,
            slice_positions_mm=(self.slice_positions_mm.to(device) if self.slice_positions_mm is not None else None),
            frame_of_reference_hash=self.frame_of_reference_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "original_shape": list(self.original_shape),
            "current_shape": list(self.current_shape),
            "affine": tensor_to_data(self.affine) if self.affine is not None else None,
            "affine_dtype": str(self.affine.dtype).removeprefix("torch.") if self.affine is not None else None,
            "original_affine": (tensor_to_data(self.original_affine) if self.original_affine is not None else None),
            "original_affine_dtype": (
                str(self.original_affine.dtype).removeprefix("torch.") if self.original_affine is not None else None
            ),
            "spacing_mm": list(self.spacing_mm) if self.spacing_mm is not None else None,
            "orientation": self.orientation,
            "anatomical_axes": list(self.anatomical_axes) if self.anatomical_axes is not None else None,
            "slice_positions_mm": (
                tensor_to_data(self.slice_positions_mm) if self.slice_positions_mm is not None else None
            ),
            "slice_positions_mm_dtype": (
                str(self.slice_positions_mm.dtype).removeprefix("torch.")
                if self.slice_positions_mm is not None
                else None
            ),
            "frame_of_reference_hash": self.frame_of_reference_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialMetadata:
        data = migrate_payload("SpatialMetadata", data)

        def _tensor(payload: Any, dtype_name: Any) -> torch.Tensor | None:
            if payload is None:
                return None
            dtype = getattr(torch, str(dtype_name)) if dtype_name else torch.float32
            if not isinstance(dtype, torch.dtype):
                raise SchemaValidationError(f"unknown tensor dtype name {dtype_name!r}")
            return tensor_from_data(payload, dtype)

        return cls(
            original_shape=tuple(int(d) for d in data["original_shape"]),
            current_shape=tuple(int(d) for d in data["current_shape"]),
            affine=_tensor(data.get("affine"), data.get("affine_dtype")),
            original_affine=_tensor(data.get("original_affine"), data.get("original_affine_dtype")),
            spacing_mm=tuple(float(s) for s in data["spacing_mm"]) if data.get("spacing_mm") else None,
            orientation=data.get("orientation"),
            anatomical_axes=(tuple(str(a) for a in data["anatomical_axes"]) if data.get("anatomical_axes") else None),
            slice_positions_mm=_tensor(data.get("slice_positions_mm"), data.get("slice_positions_mm_dtype")),
            frame_of_reference_hash=data.get("frame_of_reference_hash"),
        )


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class PathologyMetadata:
    """Whole-slide / tile geometry. MPP and pyramid data are never discarded.

    ``tile_coordinates`` is a ``[T, 2]`` or ``[T, 4]`` integer tensor of
    level-0 slide pixel coordinates: ``(x, y)`` or ``(x, y, w, h)`` per tile.
    """

    microns_per_pixel: float | None = None
    magnification: float | None = None
    slide_dimensions: tuple[int, int] | None = None  # level-0 (width, height)
    level_dimensions: tuple[tuple[int, int], ...] = ()
    stain: str | None = None  # e.g. "H&E", "IHC"
    scanner_vendor: str | None = None
    tile_coordinates: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.microns_per_pixel is not None and self.microns_per_pixel <= 0:
            raise SchemaValidationError("microns_per_pixel must be positive")
        if self.magnification is not None and self.magnification <= 0:
            raise SchemaValidationError("magnification must be positive")
        if self.slide_dimensions is not None:
            object.__setattr__(self, "slide_dimensions", _dim2(self.slide_dimensions, "slide_dimensions"))
        object.__setattr__(
            self,
            "level_dimensions",
            tuple(_dim2(level, "level_dimensions") for level in self.level_dimensions),
        )
        if self.tile_coordinates is not None:
            tc = self.tile_coordinates
            if tc.ndim != 2 or tc.shape[1] not in (2, 4):
                raise SchemaValidationError(f"tile_coordinates must have shape [T, 2] or [T, 4]; got {tuple(tc.shape)}")
            if tc.dtype not in (torch.int32, torch.int64):
                raise SchemaValidationError(f"tile_coordinates must be int32/int64; got {tc.dtype}")
            if bool((tc < 0).any()):
                raise SchemaValidationError("tile_coordinates must be non-negative")

    def to(self, device: torch.device | str) -> PathologyMetadata:
        """Return a copy with tensor fields moved to ``device`` (metadata preserved)."""
        return PathologyMetadata(
            microns_per_pixel=self.microns_per_pixel,
            magnification=self.magnification,
            slide_dimensions=self.slide_dimensions,
            level_dimensions=self.level_dimensions,
            stain=self.stain,
            scanner_vendor=self.scanner_vendor,
            tile_coordinates=self.tile_coordinates.to(device) if self.tile_coordinates is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "microns_per_pixel": self.microns_per_pixel,
            "magnification": self.magnification,
            "slide_dimensions": list(self.slide_dimensions) if self.slide_dimensions is not None else None,
            "level_dimensions": [list(level) for level in self.level_dimensions],
            "stain": self.stain,
            "scanner_vendor": self.scanner_vendor,
            "tile_coordinates": (tensor_to_data(self.tile_coordinates) if self.tile_coordinates is not None else None),
            "tile_coordinates_dtype": (
                str(self.tile_coordinates.dtype).removeprefix("torch.") if self.tile_coordinates is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathologyMetadata:
        data = migrate_payload("PathologyMetadata", data)
        tc = data.get("tile_coordinates")
        dtype_name = data.get("tile_coordinates_dtype")
        return cls(
            microns_per_pixel=data.get("microns_per_pixel"),
            magnification=data.get("magnification"),
            slide_dimensions=_dim2(data["slide_dimensions"], "slide_dimensions")
            if data.get("slide_dimensions")
            else None,
            level_dimensions=tuple(_dim2(level, "level_dimensions") for level in data.get("level_dimensions", [])),
            stain=data.get("stain"),
            scanner_vendor=data.get("scanner_vendor"),
            tile_coordinates=(
                tensor_from_data(tc, getattr(torch, str(dtype_name), torch.int64)) if tc is not None else None
            ),
        )


@dataclass(frozen=True, eq=False)  # contains tensor-bearing metadata; compare via to_dict()
class MedicalSample:
    """The canonical per-sample contract exchanged across the data stack.

    Carries references and metadata only. Structural rules (modality-specific
    required metadata, identifier hygiene) are enforced at construction;
    task-specific target requirements via :meth:`validate_for_task`.
    """

    sample_id: str
    patient_id_hash: str
    modality: Modality
    provenance: ProvenanceMetadata
    study_id_hash: str | None = None
    series_id_hash: str | None = None
    image_references: tuple[ImageReference, ...] = ()
    labels: LabelTarget | None = None
    segmentation: SegmentationTarget | None = None
    boxes: BoxTarget | None = None
    report: str | None = None
    question: str | None = None
    answer: str | None = None
    conversations: tuple[ConversationTurn, ...] = ()
    spatial: SpatialMetadata | None = None
    pathology: PathologyMetadata | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise SchemaValidationError("MedicalSample.sample_id must be non-empty")
        _hash_id(self.patient_id_hash, "patient_id_hash")
        if self.study_id_hash is not None:
            _hash_id(self.study_id_hash, "study_id_hash")
        if self.series_id_hash is not None:
            _hash_id(self.series_id_hash, "series_id_hash")
        if self.schema_version > SCHEMA_VERSION:
            raise SchemaValidationError(
                f"sample schema_version {self.schema_version} is newer than supported {SCHEMA_VERSION}"
            )
        self._validate_modality_requirements()

    def _validate_modality_requirements(self) -> None:
        modality = self.modality
        if modality.is_text_only:
            if self.image_references:
                raise SchemaValidationError("TEXT_ONLY samples must not carry image_references")
            if self.spatial is not None or self.pathology is not None:
                raise SchemaValidationError("TEXT_ONLY samples must not carry spatial/pathology metadata")
            if not (self.report or self.question or self.conversations):
                raise SchemaValidationError("TEXT_ONLY samples need text content (report, question, or conversations)")
            return

        if not self.image_references:
            raise SchemaValidationError(f"{modality} samples require at least one image reference")

        if modality.is_pathology:
            if self.pathology is None:
                raise SchemaValidationError(f"{modality} samples require PathologyMetadata")
            if self.spatial is not None:
                raise SchemaValidationError(f"{modality} samples use PathologyMetadata, not SpatialMetadata")
        elif modality.is_volumetric:
            if self.spatial is None:
                raise SchemaValidationError(f"{modality} samples require SpatialMetadata")
            if self.spatial.affine is None and self.spatial.spacing_mm is None:
                raise SchemaValidationError(
                    f"{modality} SpatialMetadata must carry an affine or spacing_mm (geometry is never lost)"
                )
            if self.pathology is not None:
                raise SchemaValidationError(f"{modality} samples use SpatialMetadata, not PathologyMetadata")
        else:
            # 2D radiology / multi-image: spatial metadata optional, pathology forbidden.
            if self.pathology is not None:
                raise SchemaValidationError(f"{modality} samples must not carry PathologyMetadata")

    #: Task -> required sample fields ("text"/"answer_or_conversations" are
    #: virtual fields checked by MedicalSample._has_field); see
    #: ``_TASK_FIELD_REQUIREMENTS`` below.

    def validate_for_task(self, task: TaskType) -> None:
        """Raise :class:`SchemaValidationError` if required targets are missing."""
        requirements = _TASK_FIELD_REQUIREMENTS.get(task)
        if requirements is None:
            return
        missing = [name for name in requirements if not self._has_field(name)]
        if missing:
            raise SchemaValidationError(
                f"sample {self.sample_id!r} ({self.modality}) is missing required fields for {task.value}: {missing}"
            )

    def _has_field(self, name: str) -> bool:
        if name == "text":
            return bool(self.report or self.question or self.conversations)
        if name == "answer_or_conversations":
            return bool(self.answer or self.conversations)
        return getattr(self, name) is not None

    def to_dict(self) -> dict[str, Any]:
        """Canonical non-tensor representation (metadata tensors inline)."""
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "patient_id_hash": self.patient_id_hash,
            "study_id_hash": self.study_id_hash,
            "series_id_hash": self.series_id_hash,
            "modality": self.modality.value,
            "image_references": [ref.to_dict() for ref in self.image_references],
            "labels": self.labels.to_dict() if self.labels is not None else None,
            "segmentation": self.segmentation.to_dict() if self.segmentation is not None else None,
            "boxes": self.boxes.to_dict() if self.boxes is not None else None,
            "report": self.report,
            "question": self.question,
            "answer": self.answer,
            "conversations": [turn.to_dict() for turn in self.conversations],
            "spatial": self.spatial.to_dict() if self.spatial is not None else None,
            "pathology": self.pathology.to_dict() if self.pathology is not None else None,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MedicalSample:
        # Enums are parsed against the payload's ORIGINAL version so enum
        # migrations (keyed by the version the payload was written with) apply
        # before/while the schema migration chain runs.
        payload_version = int(data.get("schema_version", SCHEMA_VERSION))
        data = migrate_payload("MedicalSample", data)
        spatial = data.get("spatial")
        pathology = data.get("pathology")
        labels = data.get("labels")
        segmentation = data.get("segmentation")
        boxes = data.get("boxes")
        return cls(
            sample_id=str(data["sample_id"]),
            patient_id_hash=str(data["patient_id_hash"]),
            study_id_hash=data.get("study_id_hash"),
            series_id_hash=data.get("series_id_hash"),
            modality=Modality.from_value(str(data["modality"]), schema_version=payload_version),
            image_references=tuple(ImageReference.from_dict(r) for r in data.get("image_references", [])),
            labels=LabelTarget.from_dict(labels, schema_version=payload_version) if labels is not None else None,
            segmentation=SegmentationTarget.from_dict(segmentation) if segmentation is not None else None,
            boxes=BoxTarget.from_dict(boxes, schema_version=payload_version) if boxes is not None else None,
            report=data.get("report"),
            question=data.get("question"),
            answer=data.get("answer"),
            conversations=tuple(ConversationTurn.from_dict(t) for t in data.get("conversations", [])),
            spatial=SpatialMetadata.from_dict(spatial) if spatial is not None else None,
            pathology=PathologyMetadata.from_dict(pathology) if pathology is not None else None,
            provenance=ProvenanceMetadata.from_dict(data["provenance"], schema_version=payload_version),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


#: Task -> required sample fields ("text"/"answer_or_conversations" are
#: virtual fields checked by MedicalSample._has_field).
_TASK_FIELD_REQUIREMENTS: dict[TaskType, tuple[str, ...]] = {
    TaskType.BINARY_CLASSIFICATION: ("labels",),
    TaskType.MULTICLASS_CLASSIFICATION: ("labels",),
    TaskType.MULTILABEL_CLASSIFICATION: ("labels",),
    TaskType.ORDINAL_CLASSIFICATION: ("labels",),
    TaskType.IMAGE_TEXT_RETRIEVAL: ("text",),
    TaskType.TEXT_IMAGE_RETRIEVAL: ("text",),
    TaskType.CONTRASTIVE_ALIGNMENT: ("text",),
    TaskType.SEMANTIC_SEGMENTATION: ("segmentation",),
    TaskType.INSTANCE_SEGMENTATION: ("segmentation",),
    TaskType.PROMPTABLE_SEGMENTATION: ("segmentation",),
    TaskType.LANGUAGE_CONDITIONED_SEGMENTATION: ("segmentation", "text"),
    TaskType.BOUNDING_BOX_LOCALIZATION: ("boxes",),
    TaskType.VISUAL_QUESTION_ANSWERING: ("question", "answer_or_conversations"),
    TaskType.REPORT_GENERATION: ("report",),
    TaskType.STRUCTURED_FINDING_GENERATION: ("report",),
}
