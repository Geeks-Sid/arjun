"""
Versioned evaluation report artifacts for recipe acceptance and release review.

Reports contain research metrics and explicit limitations.  They never label a
model clinically validated; external validation and qualified human review are
separate evidence gates.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medfm.core.serialization import canonical_json, config_hash
from medfm.evaluation.artifacts import load_prediction_artifact, recompute_metrics
from medfm.evaluation.metrics import MetricValue, serialize_metrics
from medfm.evaluation.schemas import PredictionArtifact

SAFE_CLINICAL_STATEMENT = (
    "Generated outputs require task-specific clinical validation and qualified human review before any clinical use."
)
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationArtifact:
    """A machine-readable evaluation record with explicit limitations."""

    recipe_id: str
    metrics: Mapping[str, MetricValue]
    reproducibility: Mapping[str, Any]
    clinical_units: Mapping[str, str]
    memory: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ValueError("recipe_id must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported evaluation artifact schema version")
        if not self.limitations:
            raise ValueError("evaluation artifacts must state limitations")
        for name, metric in self.metrics.items():
            if not isinstance(metric, MetricValue):
                raise TypeError(f"metric {name!r} is not a MetricValue")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "metrics": serialize_metrics(self.metrics),
            "reproducibility": dict(self.reproducibility),
            "clinical_units": dict(self.clinical_units),
            "memory": dict(self.memory),
            "limitations": list(self.limitations),
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return destination


def make_artifact(
    recipe_id: str,
    metrics: Mapping[str, MetricValue],
    *,
    config_hash: str,
    seed: int,
    dataset_hash: str | None,
    preprocessing_hash: str | None,
    model_revision: str | None,
    memory: Mapping[str, Any] | None = None,
    limitations: tuple[str, ...] = (
        "Synthetic/offline contract evidence is not clinical validation.",
        "External-site and human-review evidence must be collected before clinical use.",
    ),
) -> EvaluationArtifact:
    """Build an artifact with the run provenance fields required by Phase 16."""

    units = {name: metric.unit for name, metric in metrics.items()}
    reproducibility = {
        "config_hash": config_hash,
        "seed": int(seed),
        "dataset_manifest_sha256": dataset_hash,
        "preprocessing_hash": preprocessing_hash,
        "model_revision": model_revision,
    }
    return EvaluationArtifact(
        recipe_id=recipe_id,
        metrics=metrics,
        reproducibility=reproducibility,
        clinical_units=units,
        memory=dict(memory or {}),
        limitations=limitations,
    )


def _metric_payload(metrics: Mapping[str, MetricValue | Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in sorted(metrics.items()):
        result[name] = value.to_dict() if isinstance(value, MetricValue) else dict(value)
    return result


def _assert_no_unsupported_claims(payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False).casefold()
    # A negative limitation is allowed; an affirmative unsupported label is
    # not.  Keep the check structural as well as textual for machine reports.
    if payload.get("claims", {}).get("clinically_validated") is True:
        raise ValueError("reports must not label a model clinically validated")
    forbidden = (
        r"(?<!not )clinically validated",
        r"(?<!not )clinical performance claim",
        r"(?<!not )diagnostic performance claim",
        r"autonomous diagnosis",
    )
    for pattern in forbidden:
        if re.search(pattern, text):
            raise ValueError("report contains an unsupported clinical claim")


@dataclass(frozen=True)
class EvaluationReport:
    """Release-facing report with scientific and accelerator evidence separated."""

    report_id: str
    task: str
    prediction_artifact_id: str
    metrics: Mapping[str, MetricValue | Mapping[str, Any]]
    provenance: Mapping[str, Any]
    clinical_unit: str
    scientific_metrics: Mapping[str, MetricValue | Mapping[str, Any]] = field(default_factory=dict)
    accelerator_metrics: Mapping[str, MetricValue | Mapping[str, Any]] = field(default_factory=dict)
    subgroup_results: Mapping[str, Any] = field(default_factory=dict)
    baselines: Mapping[str, Any] = field(default_factory=dict)
    ablations: Mapping[str, Any] = field(default_factory=dict)
    human_review: Mapping[str, Any] = field(default_factory=dict)
    examples: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[str, ...] = (SAFE_CLINICAL_STATEMENT,)
    claims: Mapping[str, Any] = field(default_factory=lambda: {"clinically_validated": False})
    schema_version: int = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.report_id or not self.task or not self.prediction_artifact_id:
            raise ValueError("report_id, task, and prediction_artifact_id must be non-empty")
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation report schema version")
        if not self.limitations:
            raise ValueError("evaluation reports must state known limitations")
        if not any("clinical validation" in value.casefold() for value in self.limitations):
            raise ValueError("evaluation reports must state clinical-validation limitations")
        for example in self.examples:
            if "text" in example or "image" in example or "pixels" in example:
                raise ValueError("reports may reference reviewed content but must not embed protected text/images")
            if not example.get("artifact_uri"):
                raise ValueError("report examples require access-controlled artifact_uri references")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "task": self.task,
            "prediction_artifact_id": self.prediction_artifact_id,
            "clinical_unit": self.clinical_unit,
            "metrics": _metric_payload(self.metrics),
            "scientific_metrics": _metric_payload(self.scientific_metrics),
            "accelerator_metrics": _metric_payload(self.accelerator_metrics),
            "provenance": dict(self.provenance),
            "subgroup_results": dict(self.subgroup_results),
            "baselines": dict(self.baselines),
            "ablations": dict(self.ablations),
            "human_review": dict(self.human_review),
            "examples": [dict(example) for example in self.examples],
            "limitations": list(self.limitations),
            "claims": dict(self.claims),
        }
        _assert_no_unsupported_claims(payload)
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def checksum(self) -> str:
        return config_hash(self.to_dict())

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> EvaluationReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            report_id=str(data["report_id"]),
            task=str(data["task"]),
            prediction_artifact_id=str(data["prediction_artifact_id"]),
            clinical_unit=str(data["clinical_unit"]),
            metrics=dict(data.get("metrics", {})),
            scientific_metrics=dict(data.get("scientific_metrics", {})),
            accelerator_metrics=dict(data.get("accelerator_metrics", {})),
            provenance=dict(data.get("provenance", {})),
            subgroup_results=dict(data.get("subgroup_results", {})),
            baselines=dict(data.get("baselines", {})),
            ablations=dict(data.get("ablations", {})),
            human_review=dict(data.get("human_review", {})),
            examples=tuple(data.get("examples", ())),
            limitations=tuple(data.get("limitations", ())),
            claims=dict(data.get("claims", {"clinically_validated": False})),
            schema_version=int(data.get("schema_version", REPORT_SCHEMA_VERSION)),
        )


def build_evaluation_report(
    artifact: PredictionArtifact,
    metrics: Mapping[str, MetricValue | Mapping[str, Any]],
    *,
    report_id: str | None = None,
    subgroup_results: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
    ablations: Mapping[str, Any] | None = None,
    human_review: Mapping[str, Any] | None = None,
    examples: tuple[Mapping[str, Any], ...] = (),
    accelerator_metrics: Mapping[str, Any] | None = None,
    limitations: tuple[str, ...] = (
        SAFE_CLINICAL_STATEMENT,
        "External-site and human-review evidence remain release limitations until collected under approved governance.",
    ),
) -> EvaluationReport:
    """Create a report from a saved prediction artifact and computed metrics."""

    report_id = report_id or f"{artifact.artifact_id}-report"
    scientific = dict(metrics)
    provenance = artifact.provenance.to_dict()
    provenance["prediction_artifact_hash"] = artifact.content_hash()
    provenance["evaluation_split"] = artifact.split.value
    return EvaluationReport(
        report_id=report_id,
        task=artifact.task,
        prediction_artifact_id=artifact.artifact_id,
        metrics=scientific,
        scientific_metrics=scientific,
        accelerator_metrics=dict(accelerator_metrics or {}),
        provenance=provenance,
        clinical_unit=artifact.clinical_unit.metric_name,
        subgroup_results=dict(subgroup_results or {}),
        baselines=dict(baselines or {}),
        ablations=dict(ablations or {}),
        human_review=dict(human_review or {}),
        examples=examples,
        limitations=limitations,
    )


def recompute_report_from_predictions(
    prediction_path: str | Path,
    *,
    metric_options: Mapping[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> EvaluationReport:
    """Recompute and optionally write a report without model inference."""

    artifact = load_prediction_artifact(prediction_path)
    metrics = recompute_metrics(artifact, metric_options=metric_options)
    report = build_evaluation_report(artifact, metrics)
    if report_path is not None:
        report.write(report_path)
    return report


# Public names used by release tooling.
VersionedEvaluationReport = EvaluationReport
write_versioned_report = EvaluationReport.write
assert_no_unsupported_clinical_claim = _assert_no_unsupported_claims


__all__ = [
    "EvaluationArtifact",
    "EvaluationReport",
    "REPORT_SCHEMA_VERSION",
    "SAFE_CLINICAL_STATEMENT",
    "VersionedEvaluationReport",
    "assert_no_unsupported_clinical_claim",
    "build_evaluation_report",
    "make_artifact",
    "recompute_report_from_predictions",
    "write_versioned_report",
]
