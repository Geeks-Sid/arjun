"""Versioned, clinical-unit-aware schemas for evaluation data.

The schemas in this module are deliberately dependency-light.  They are the
boundary between inference and evaluation: prediction files can be copied,
reviewed, and scored again without loading a model or rerunning inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from medfm.core.serialization import canonical_json, config_hash

EVALUATION_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1


class EvaluationSchemaError(ValueError):
    """Raised when an evaluation record violates a schema invariant."""


class ClinicalUnit(str, Enum):
    """Allowed aggregation units for clinical evaluation."""

    PATIENT = "patient"
    STUDY = "study"
    SLIDE = "slide"
    SCAN = "scan"
    TILE = "tile"
    QUERY = "query"
    SAMPLE = "sample"

    @classmethod
    def parse(cls, value: str | ClinicalUnit) -> ClinicalUnit:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().removeprefix("per_")
        try:
            return cls(normalized)
        except ValueError as exc:
            raise EvaluationSchemaError(f"unsupported clinical unit {value!r}") from exc

    @property
    def metric_name(self) -> str:
        return f"per_{self.value}"


class EvaluationSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXTERNAL = "external"
    TEMPORAL = "temporal"
    HOLDOUT = "holdout"

    @classmethod
    def parse(cls, value: str | EvaluationSplit) -> EvaluationSplit:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise EvaluationSchemaError(f"unsupported evaluation split {value!r}") from exc


@dataclass(frozen=True)
class ClinicalIdentity:
    """Identifiers needed to aggregate a row at a declared clinical unit."""

    patient_id: str | None = None
    study_id: str | None = None
    slide_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if value is not None and not str(value).strip():
                raise EvaluationSchemaError(f"{name} must be non-empty when provided")

    def require(self, unit: ClinicalUnit | str) -> str:
        parsed = ClinicalUnit.parse(unit)
        field_name = {
            ClinicalUnit.PATIENT: "patient_id",
            ClinicalUnit.STUDY: "study_id",
            ClinicalUnit.SLIDE: "slide_id",
            ClinicalUnit.SCAN: "study_id",
            ClinicalUnit.TILE: "slide_id",
            ClinicalUnit.QUERY: "study_id",
            ClinicalUnit.SAMPLE: "patient_id",
        }[parsed]
        value = getattr(self, field_name)
        if value is None:
            raise EvaluationSchemaError(f"{parsed.value} clinical-unit evaluation requires {field_name}")
        if not isinstance(value, str):
            raise EvaluationSchemaError(f"{field_name} must be a string")
        return value

    def key(self, unit: ClinicalUnit | str) -> str:
        return self.require(unit)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "slide_id": self.slide_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ClinicalIdentity:
        raw = dict(data or {})
        return cls(
            patient_id=None if raw.get("patient_id") is None else str(raw["patient_id"]),
            study_id=None if raw.get("study_id") is None else str(raw["study_id"]),
            slide_id=None if raw.get("slide_id") is None else str(raw["slide_id"]),
        )


@dataclass(frozen=True)
class TargetRecord:
    """Versioned target payload associated with one prediction row."""

    sample_id: str
    value: Any
    clinical_unit: ClinicalUnit
    identity: ClinicalIdentity
    split: EvaluationSplit = EvaluationSplit.TEST
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise EvaluationSchemaError("target sample_id must be non-empty")
        object.__setattr__(self, "clinical_unit", ClinicalUnit.parse(self.clinical_unit))
        object.__setattr__(self, "split", EvaluationSplit.parse(self.split))
        self.identity.require(self.clinical_unit)
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise EvaluationSchemaError("unsupported target schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "value": self.value,
            "clinical_unit": self.clinical_unit.value,
            "identity": self.identity.to_dict(),
            "split": self.split.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetRecord:
        return cls(
            sample_id=str(data["sample_id"]),
            value=data.get("value"),
            clinical_unit=ClinicalUnit.parse(str(data["clinical_unit"])),
            identity=ClinicalIdentity.from_dict(data.get("identity")),
            split=EvaluationSplit.parse(str(data.get("split", EvaluationSplit.TEST.value))),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", PREDICTION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class GroupRecord:
    """Protected subgroup label linked to a clinical identity."""

    name: str
    value: str
    clinical_unit: ClinicalUnit
    identity: ClinicalIdentity
    schema_version: int = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.value).strip():
            raise EvaluationSchemaError("group name and value must be non-empty")
        object.__setattr__(self, "clinical_unit", ClinicalUnit.parse(self.clinical_unit))
        self.identity.require(self.clinical_unit)
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise EvaluationSchemaError("unsupported group schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "value": self.value,
            "clinical_unit": self.clinical_unit.value,
            "identity": self.identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroupRecord:
        return cls(
            name=str(data["name"]),
            value=str(data["value"]),
            clinical_unit=ClinicalUnit.parse(str(data["clinical_unit"])),
            identity=ClinicalIdentity.from_dict(data.get("identity")),
            schema_version=int(data.get("schema_version", PREDICTION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class PredictionRecord:
    """A model output and optional target that can be scored offline."""

    sample_id: str
    prediction: Any
    clinical_unit: ClinicalUnit
    identity: ClinicalIdentity
    target: Any | None = None
    split: EvaluationSplit = EvaluationSplit.TEST
    groups: Mapping[str, str] = field(default_factory=dict)
    is_padding: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise EvaluationSchemaError("prediction sample_id must be non-empty")
        object.__setattr__(self, "clinical_unit", ClinicalUnit.parse(self.clinical_unit))
        object.__setattr__(self, "split", EvaluationSplit.parse(self.split))
        self.identity.require(self.clinical_unit)
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise EvaluationSchemaError("unsupported prediction schema version")
        if any(not str(name).strip() or not str(value).strip() for name, value in self.groups.items()):
            raise EvaluationSchemaError("group names and values must be non-empty")

    def target_record(self) -> TargetRecord:
        if self.target is None:
            raise EvaluationSchemaError(f"prediction {self.sample_id!r} has no target")
        return TargetRecord(
            sample_id=self.sample_id,
            value=self.target,
            clinical_unit=self.clinical_unit,
            identity=self.identity,
            split=self.split,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "target": self.target,
            "clinical_unit": self.clinical_unit.value,
            "identity": self.identity.to_dict(),
            "split": self.split.value,
            "groups": dict(sorted(self.groups.items())),
            "is_padding": bool(self.is_padding),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredictionRecord:
        identity_data = data.get("identity")
        if identity_data is None:
            identity_data = {
                key: data.get(key) for key in ("patient_id", "study_id", "slide_id") if data.get(key) is not None
            }
        return cls(
            sample_id=str(data["sample_id"]),
            prediction=data.get("prediction"),
            target=data.get("target"),
            clinical_unit=ClinicalUnit.parse(str(data["clinical_unit"])),
            identity=ClinicalIdentity.from_dict(identity_data),
            split=EvaluationSplit.parse(str(data.get("split", EvaluationSplit.TEST.value))),
            groups=dict(data.get("groups", {})),
            is_padding=bool(data.get("is_padding", False)),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", PREDICTION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class MetricResult:
    """Stable schema for a reported metric result."""

    name: str
    value: float | None
    clinical_unit: ClinicalUnit
    sample_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise EvaluationSchemaError("metric name must be non-empty")
        object.__setattr__(self, "clinical_unit", ClinicalUnit.parse(self.clinical_unit))
        if self.sample_count < 0:
            raise EvaluationSchemaError("metric sample_count must be >= 0")
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise EvaluationSchemaError("unsupported metric-result schema version")

    @property
    def unit(self) -> str:
        return self.clinical_unit.metric_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "value": self.value,
            "clinical_unit": self.clinical_unit.value,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeProvenance:
    """Execution metadata stored with every saved prediction artifact."""

    model_hash: str
    data_hash: str
    preprocess_hash: str
    backend: str
    precision: str
    topology: str
    bucket: str
    checkpoint_format: str
    model_revision: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_hash",
            "data_hash",
            "preprocess_hash",
            "backend",
            "precision",
            "topology",
            "bucket",
            "checkpoint_format",
        ):
            if not str(getattr(self, name)).strip():
                raise EvaluationSchemaError(f"runtime provenance field {name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "preprocess_hash": self.preprocess_hash,
            "backend": self.backend,
            "precision": self.precision,
            "topology": self.topology,
            "bucket": self.bucket,
            "checkpoint_format": self.checkpoint_format,
            "model_revision": self.model_revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeProvenance:
        # Older callers used the more verbose hash names.  Accept both on
        # input while writing one canonical schema.
        return cls(
            model_hash=str(data.get("model_hash", data.get("model_revision", "unknown-model"))),
            data_hash=str(data.get("data_hash", data.get("dataset_hash", "unknown-data"))),
            preprocess_hash=str(data.get("preprocess_hash", data.get("preprocessing_hash", "unknown-preprocess"))),
            backend=str(data.get("backend", "cpu")),
            precision=str(data.get("precision", "fp32")),
            topology=str(data.get("topology", data.get("distribution", "single"))),
            bucket=str(data.get("bucket", "unbucketed")),
            checkpoint_format=str(data.get("checkpoint_format", data.get("checkpoint", "unknown"))),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
        )


@dataclass(frozen=True)
class PredictionArtifact:
    """Serializable prediction set and its complete provenance."""

    artifact_id: str
    task: str
    clinical_unit: ClinicalUnit
    split: EvaluationSplit
    predictions: tuple[PredictionRecord, ...]
    provenance: RuntimeProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.artifact_id).strip() or not str(self.task).strip():
            raise EvaluationSchemaError("artifact_id and task must be non-empty")
        object.__setattr__(self, "clinical_unit", ClinicalUnit.parse(self.clinical_unit))
        object.__setattr__(self, "split", EvaluationSplit.parse(self.split))
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise EvaluationSchemaError("unsupported prediction artifact schema version")
        seen: set[str] = set()
        seen_valid: set[str] = set()
        for row in self.predictions:
            if not isinstance(row, PredictionRecord):
                raise TypeError("predictions must contain PredictionRecord values")
            if not row.is_padding and (row.sample_id in seen_valid or row.sample_id in seen):
                raise EvaluationSchemaError(f"duplicate non-padding prediction sample_id {row.sample_id!r}")
            seen.add(row.sample_id)
            if not row.is_padding:
                seen_valid.add(row.sample_id)
            if row.clinical_unit != self.clinical_unit:
                raise EvaluationSchemaError("all prediction rows must use artifact clinical_unit")
            if row.split != self.split:
                raise EvaluationSchemaError("all prediction rows must use artifact split")

    @property
    def valid_predictions(self) -> tuple[PredictionRecord, ...]:
        """Return non-padding rows; padded distributed rows never score."""

        return tuple(row for row in self.predictions if not row.is_padding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "task": self.task,
            "clinical_unit": self.clinical_unit.value,
            "split": self.split.value,
            "predictions": [row.to_dict() for row in self.predictions],
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def content_hash(self) -> str:
        return config_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredictionArtifact:
        return cls(
            artifact_id=str(data["artifact_id"]),
            task=str(data["task"]),
            clinical_unit=ClinicalUnit.parse(str(data["clinical_unit"])),
            split=EvaluationSplit.parse(str(data["split"])),
            predictions=tuple(PredictionRecord.from_dict(row) for row in data.get("predictions", [])),
            provenance=RuntimeProvenance.from_dict(data.get("provenance", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", PREDICTION_SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, path: str | Path) -> PredictionArtifact:
        source = Path(path)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


# Explicit schema aliases make the public contract discoverable to callers
# that prefer ``PredictionSchema``/``TargetSchema`` terminology.
PredictionSchema = PredictionRecord
TargetSchema = TargetRecord
GroupSchema = GroupRecord
MetricResultSchema = MetricResult


__all__ = [
    "ClinicalIdentity",
    "ClinicalUnit",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationSchemaError",
    "EvaluationSplit",
    "GroupRecord",
    "GroupSchema",
    "MetricResult",
    "MetricResultSchema",
    "PredictionArtifact",
    "PredictionRecord",
    "PredictionSchema",
    "PREDICTION_SCHEMA_VERSION",
    "RuntimeProvenance",
    "TargetRecord",
    "TargetSchema",
]
