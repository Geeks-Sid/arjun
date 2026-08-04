"""Canonical enums and identifier types for the medfm contract layer.

The value sets are frozen by Phase 00 governance docs
(``docs/supported_modalities.md``, ``docs/supported_tasks.md``). Parsing is
strict by default: unknown values raise :class:`UnknownEnumValueError`.
Explicit, versioned migrations may be registered through
:func:`register_enum_migration` so old payloads can be upgraded deliberately
instead of silently accepted.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from medfm.core.errors import UnknownEnumValueError

E = TypeVar("E", bound="StrictStrEnum")

# (enum class name, schema version the payload was written with, old value)
# -> value valid at the current schema version. Migrations are explicit and
# append-only; they are the ONLY way a retired value stays readable.
EnumMigrationKey = tuple[str, int, str]
_ENUM_MIGRATIONS: dict[EnumMigrationKey, str] = {}

#: Schema version the current enum sets correspond to.
CURRENT_SCHEMA_VERSION = 1


def register_enum_migration(enum_name: str, from_version: int, old_value: str, new_value: str) -> None:
    """Register a versioned migration from a retired enum value to a current one."""
    key = (enum_name, from_version, old_value)
    if key in _ENUM_MIGRATIONS and _ENUM_MIGRATIONS[key] != new_value:
        raise UnknownEnumValueError(f"conflicting migration registered for {key}")
    _ENUM_MIGRATIONS[key] = new_value


class StrictStrEnum(StrEnum):
    """String enum that rejects unknown values unless a migration applies."""

    @classmethod
    def from_value(cls: type[E], value: str, *, schema_version: int | None = None) -> E:
        """Parse ``value`` strictly; apply a registered migration if one exists.

        Raises :class:`UnknownEnumValueError` listing the legal values when the
        value is unknown and no migration covers ``(cls, schema_version, value)``.
        """
        try:
            return cls(value)
        except ValueError:
            pass
        if schema_version is not None and schema_version != CURRENT_SCHEMA_VERSION:
            migrated = _ENUM_MIGRATIONS.get((cls.__name__, schema_version, value))
            if migrated is not None:
                try:
                    return cls(migrated)
                except ValueError:
                    raise UnknownEnumValueError(
                        f"migration for {cls.__name__} value {value!r} (schema v{schema_version}) "
                        f"targets unknown value {migrated!r}; legal values: {[m.value for m in cls]}"
                    ) from None
        legal = [m.value for m in cls]
        hint = (
            f" (no migration registered from schema v{schema_version})"
            if schema_version is not None and schema_version != CURRENT_SCHEMA_VERSION
            else ""
        )
        raise UnknownEnumValueError(f"unknown {cls.__name__} value {value!r}{hint}; legal values: {legal}")

    def __str__(self) -> str:
        return str(self.value)


class Modality(StrictStrEnum):
    """Canonical modalities; the only legal values of ``modality`` fields.

    Source of truth: ``docs/supported_modalities.md``. Adding, renaming, or
    removing a value requires an ADR plus a contract version bump.
    """

    XRAY_2D = "XRAY_2D"
    CT_2D_SLICE = "CT_2D_SLICE"
    CT_3D = "CT_3D"
    MRI_2D_SLICE = "MRI_2D_SLICE"
    MRI_3D = "MRI_3D"
    PATHOLOGY_TILE = "PATHOLOGY_TILE"
    PATHOLOGY_WSI = "PATHOLOGY_WSI"
    MULTI_IMAGE_2D = "MULTI_IMAGE_2D"
    MULTI_SERIES_3D = "MULTI_SERIES_3D"
    TEXT_ONLY = "TEXT_ONLY"

    @property
    def is_text_only(self) -> bool:
        return self is Modality.TEXT_ONLY

    @property
    def is_pathology(self) -> bool:
        return self in (Modality.PATHOLOGY_TILE, Modality.PATHOLOGY_WSI)

    @property
    def is_volumetric(self) -> bool:
        """Native 3D or multi-series 3D input (affine/spacing semantics apply)."""
        return self in (Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D)

    @property
    def expected_pixel_rank(self) -> int | None:
        """Rank of ``pixel_values`` in a :class:`MedicalBatch` for this modality.

        ``None`` means no pixel tensor is expected (``TEXT_ONLY``). Rank alone
        never determines modality — it is checked *against* the authoritative
        ``modality`` field.
        """
        return _MODALITY_PIXEL_RANK[self]


_MODALITY_PIXEL_RANK: dict[Modality, int | None] = {
    Modality.XRAY_2D: 4,  # [B, C, H, W]
    Modality.CT_2D_SLICE: 4,  # [B, C, H, W]
    Modality.CT_3D: 5,  # [B, C, D, H, W]
    Modality.MRI_2D_SLICE: 4,  # [B, C, H, W]
    Modality.MRI_3D: 5,  # [B, C, D, H, W]
    Modality.PATHOLOGY_TILE: 4,  # [B, C, H, W]
    Modality.PATHOLOGY_WSI: 5,  # [B, T, C, H, W]
    Modality.MULTI_IMAGE_2D: 5,  # [B, I, C, H, W]
    Modality.MULTI_SERIES_3D: 6,  # [B, S, C, D, H, W]
    Modality.TEXT_ONLY: None,  # text tokens only
}

#: Pixel-tensor axis layouts per modality (documentation-grade, used in errors).
MODALITY_PIXEL_AXES: dict[Modality, str | None] = {
    Modality.XRAY_2D: "BCHW",
    Modality.CT_2D_SLICE: "BCHW",
    Modality.CT_3D: "BCDHW",
    Modality.MRI_2D_SLICE: "BCHW",
    Modality.MRI_3D: "BCDHW",
    Modality.PATHOLOGY_TILE: "BCHW",
    Modality.PATHOLOGY_WSI: "BTCHW",
    Modality.MULTI_IMAGE_2D: "BICHW",
    Modality.MULTI_SERIES_3D: "BSCDHW",
    Modality.TEXT_ONLY: None,
}


class TaskType(StrictStrEnum):
    """Canonical tasks; the only legal task identifiers.

    Source of truth: ``docs/supported_tasks.md``.
    """

    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTICLASS_CLASSIFICATION = "MULTICLASS_CLASSIFICATION"
    MULTILABEL_CLASSIFICATION = "MULTILABEL_CLASSIFICATION"
    ORDINAL_CLASSIFICATION = "ORDINAL_CLASSIFICATION"
    IMAGE_TEXT_RETRIEVAL = "IMAGE_TEXT_RETRIEVAL"
    TEXT_IMAGE_RETRIEVAL = "TEXT_IMAGE_RETRIEVAL"
    SEMANTIC_SEGMENTATION = "SEMANTIC_SEGMENTATION"
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
    PROMPTABLE_SEGMENTATION = "PROMPTABLE_SEGMENTATION"
    LANGUAGE_CONDITIONED_SEGMENTATION = "LANGUAGE_CONDITIONED_SEGMENTATION"
    BOUNDING_BOX_LOCALIZATION = "BOUNDING_BOX_LOCALIZATION"
    VISUAL_QUESTION_ANSWERING = "VISUAL_QUESTION_ANSWERING"
    REPORT_GENERATION = "REPORT_GENERATION"
    STRUCTURED_FINDING_GENERATION = "STRUCTURED_FINDING_GENERATION"
    CONTRASTIVE_ALIGNMENT = "CONTRASTIVE_ALIGNMENT"
    MULTITASK = "MULTITASK"


class LoadingMode(StrictStrEnum):
    """How a model's weights are loaded/adapted for training.

    ``QLORA_NF4`` is CUDA-only (bitsandbytes); it must never be selected for a
    TPU configuration (see ``implementation_plan/accelerator_training_strategy.md``).
    """

    FULL = "FULL"
    FROZEN = "FROZEN"
    LORA = "LORA"
    QLORA_NF4 = "QLORA_NF4"


class CoordinateSystem(StrictStrEnum):
    """Coordinate systems for boxes and token coordinates.

    - ``NORMALIZED_IMAGE``: x/y in [0, 1] relative to image width/height.
    - ``MILLIMETERS``: patient-space millimetres (radiology; affine-defined).
    - ``MICRONS``: slide microns (pathology; MPP-defined).
    - ``SLIDE_PIXELS``: level-0 slide pixel coordinates (pathology).
    """

    NORMALIZED_IMAGE = "NORMALIZED_IMAGE"
    MILLIMETERS = "MILLIMETERS"
    MICRONS = "MICRONS"
    SLIDE_PIXELS = "SLIDE_PIXELS"


class PrecisionMode(StrictStrEnum):
    """Training/inference precision policy (distinct from quantization)."""

    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"


class SplitName(StrictStrEnum):
    """Dataset split names. Splits are assigned patient-first, then site,
    then time (``docs/architecture/adr_0004_patient_level_splitting.md``)."""

    TRAIN = "TRAIN"
    VAL = "VAL"
    TEST = "TEST"
    EXTERNAL_VAL = "EXTERNAL_VAL"
    TEMPORAL_VAL = "TEMPORAL_VAL"
