"""Prediction artifact persistence and deterministic metric recomputation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from medfm.core.serialization import canonical_json
from medfm.evaluation.metrics import MetricValue, serialize_metrics
from medfm.evaluation.schemas import (
    ClinicalUnit,
    EvaluationSchemaError,
    PredictionArtifact,
    PredictionRecord,
)


@dataclass(frozen=True)
class AggregatedPredictions:
    """Scores and labels after aggregation at the configured clinical unit."""

    keys: tuple[str, ...]
    labels: tuple[Any, ...]
    scores: tuple[Any, ...]
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_counts: tuple[int, ...] = ()
    unit: ClinicalUnit = ClinicalUnit.PATIENT

    @property
    def sample_count(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit.value,
            "keys": list(self.keys),
            "labels": list(self.labels),
            "scores": list(self.scores),
            "groups": {name: list(values) for name, values in sorted(self.groups.items())},
            "source_counts": list(self.source_counts),
        }


def _same_value(left: Any, right: Any) -> bool:
    """Compare JSON-like values without tensor identity or float noise."""

    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().tolist()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().tolist()
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return bool(left == right)


def _mean_values(values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack([value.detach().float() for value in values]).mean(dim=0)
    if isinstance(first, int | float) and not isinstance(first, bool):
        return sum(float(value) for value in values) / len(values)
    if isinstance(first, list) and all(isinstance(value, list) for value in values):
        tensors = [torch.as_tensor(value, dtype=torch.float32) for value in values]
        return torch.stack(tensors).mean(dim=0).tolist()
    if all(_same_value(first, value) for value in values[1:]):
        return first
    raise EvaluationSchemaError("cannot mean non-numeric predictions at a clinical unit")


def _target_value(rows: list[PredictionRecord]) -> Any:
    targets = [row.target for row in rows]
    if any(target is None for target in targets):
        raise EvaluationSchemaError("all rows in a scored clinical unit must include a target")
    first = targets[0]
    if not all(_same_value(first, target) for target in targets[1:]):
        raise EvaluationSchemaError("a clinical unit contains inconsistent target values")
    return first


def aggregate_prediction_records(
    records: Iterable[PredictionRecord],
    *,
    unit: ClinicalUnit | str,
    score_reducer: str = "mean",
) -> AggregatedPredictions:
    """Aggregate rows without allowing padded distributed duplicates to score."""

    parsed_unit = ClinicalUnit.parse(unit)
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in records:
        if row.is_padding:
            continue
        grouped[row.identity.key(parsed_unit)].append(row)
    if score_reducer not in {"mean", "first"}:
        raise ValueError("score_reducer must be 'mean' or 'first'")

    keys = tuple(sorted(grouped))
    labels: list[Any] = []
    scores: list[Any] = []
    source_counts: list[int] = []
    group_values: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        rows = grouped[key]
        labels.append(_target_value(rows))
        predictions = [row.prediction for row in rows]
        scores.append(predictions[0] if score_reducer == "first" else _mean_values(predictions))
        source_counts.append(len(rows))
        for name in sorted({name for row in rows for name in row.groups}):
            values = {str(row.groups[name]) for row in rows if name in row.groups}
            if len(values) > 1:
                raise EvaluationSchemaError(f"group {name!r} is inconsistent within clinical unit {key!r}")
            if values:
                group_values[name].append(next(iter(values)))
    return AggregatedPredictions(
        keys=keys,
        labels=tuple(labels),
        scores=tuple(scores),
        groups={name: tuple(values) for name, values in sorted(group_values.items())},
        source_counts=tuple(source_counts),
        unit=parsed_unit,
    )


def save_prediction_artifact(artifact: PredictionArtifact, path: str | Path) -> Path:
    """Write a deterministic JSON prediction artifact."""

    return artifact.save(path)


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    """Load and validate a saved prediction artifact."""

    return PredictionArtifact.load(path)


def recompute_metrics(
    artifact: PredictionArtifact,
    *,
    metric_options: Mapping[str, Any] | None = None,
) -> dict[str, MetricValue]:
    """Recompute metrics from predictions only; inference is never invoked."""

    options = dict(metric_options or {})
    task = artifact.task.strip().lower().replace("-", "_")
    if task in {"classification", "binary_classification", "multiclass_classification"}:
        from medfm.evaluation.metrics import classification_metrics

        aggregated = aggregate_prediction_records(artifact.valid_predictions, unit=artifact.clinical_unit)
        return classification_metrics(
            aggregated.labels,
            aggregated.scores,
            unit=artifact.clinical_unit.metric_name,
            group_ids=aggregated.groups.get(str(options.get("group", ""))) if options.get("group") else None,
            **{
                key: value
                for key, value in options.items()
                if key
                in {"thresholds", "threshold", "num_classes", "bootstrap", "target_specificity", "target_sensitivity"}
            },
        )
    if task in {"segmentation", "binary_segmentation", "3d_segmentation"}:
        from medfm.evaluation.metrics import segmentation_metrics

        rows = artifact.valid_predictions
        if not rows:
            raise EvaluationSchemaError("cannot recompute segmentation metrics from an empty artifact")
        logits = torch.stack([torch.as_tensor(row.prediction) for row in rows])
        segmentation_targets = torch.stack([torch.as_tensor(row.target) for row in rows])
        return segmentation_metrics(
            logits,
            segmentation_targets,
            unit=artifact.clinical_unit.metric_name,
            spacing_mm=tuple(options.get("spacing_mm", (1.0,) * (logits.ndim - 2))),
        )
    if task in {"retrieval", "image_text_retrieval", "vqa", "generation", "report_generation"}:
        from medfm.evaluation.advanced import generation_metrics, retrieval_metrics

        if task == "retrieval" or task == "image_text_retrieval":
            matrix = torch.as_tensor([row.prediction for row in artifact.valid_predictions])
            ids = [row.sample_id for row in artifact.valid_predictions]
            return retrieval_metrics(matrix, query_ids=ids, candidate_ids=ids)
        predictions = [row.prediction for row in artifact.valid_predictions]
        generation_targets: list[Any] = [row.target for row in artifact.valid_predictions]
        return generation_metrics(predictions, generation_targets, **options)
    raise EvaluationSchemaError(f"unsupported artifact task {artifact.task!r}")


def recompute_metrics_from_path(
    path: str | Path,
    *,
    metric_options: Mapping[str, Any] | None = None,
) -> dict[str, MetricValue]:
    """Convenience wrapper used by the CLI and golden-report tests."""

    return recompute_metrics(load_prediction_artifact(path), metric_options=metric_options)


def serialize_recomputed_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, Any]:
    """Stable JSON payload for a recomputed metric set."""

    return serialize_metrics(metrics)


# Names used by callers that prefer a shorter API.
aggregate_at_clinical_unit = aggregate_prediction_records
recompute_from_predictions = recompute_metrics


__all__ = [
    "AggregatedPredictions",
    "aggregate_at_clinical_unit",
    "aggregate_prediction_records",
    "load_prediction_artifact",
    "recompute_from_predictions",
    "recompute_metrics",
    "recompute_metrics_from_path",
    "save_prediction_artifact",
    "serialize_recomputed_metrics",
]
