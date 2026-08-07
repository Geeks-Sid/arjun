"""Offline evaluation CLI: ``python -m medfm.cli.evaluate --config ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import yaml

from medfm.core.serialization import config_hash
from medfm.evaluation.artifacts import recompute_metrics, save_prediction_artifact
from medfm.evaluation.report import build_evaluation_report
from medfm.evaluation.schemas import (
    ClinicalIdentity,
    ClinicalUnit,
    EvaluationSplit,
    PredictionArtifact,
    PredictionRecord,
    RuntimeProvenance,
)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("evaluation config must contain a mapping")
    return dict(data)


def _hash_value(value: Any, fallback: str) -> str:
    return str(value) if value is not None and str(value).strip() else fallback


def _records_from_config(config: dict[str, Any], *, config_path: Path) -> PredictionArtifact:
    raw = dict(config.get("evaluation", config))
    artifact_path = raw.get("prediction_artifact", raw.get("predictions_path"))
    if artifact_path:
        source = Path(str(artifact_path))
        if not source.is_absolute():
            source = config_path.parent / source
        return PredictionArtifact.load(source)
    unit = ClinicalUnit.parse(raw.get("clinical_unit", "patient"))
    split = EvaluationSplit.parse(raw.get("split", "test"))
    task = str(raw.get("task", raw.get("task_type", "classification"))).strip().lower().replace("-", "_")
    rows = raw.get("predictions")
    if rows is None:
        labels = list(raw.get("labels", ()))
        scores = list(cast(Iterable[Any], raw.get("scores", raw.get("predicted", ()))))
        if len(labels) != len(scores):
            raise ValueError("labels and scores must have the same length")
        patient_ids = list(raw.get("patient_ids", ()))
        study_ids = list(raw.get("study_ids", ()))
        slide_ids = list(raw.get("slide_ids", ()))
        groups = list(raw.get("groups", ()))
        rows = []
        for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "prediction": score,
                    "target": label,
                    "patient_id": patient_ids[index] if index < len(patient_ids) else f"patient-{index}",
                    "study_id": study_ids[index] if index < len(study_ids) else f"study-{index}",
                    "slide_id": slide_ids[index] if index < len(slide_ids) else f"slide-{index}",
                    "groups": groups[index] if index < len(groups) and isinstance(groups[index], dict) else {},
                }
            )
    if not isinstance(rows, list):
        raise ValueError("predictions must be a list of records")
    records: list[PredictionRecord] = []
    for index, value in enumerate(rows):
        if not isinstance(value, dict):
            raise ValueError("each prediction must be a mapping")
        identity = value.get("identity") or {
            "patient_id": value.get("patient_id", f"patient-{index}"),
            "study_id": value.get("study_id", f"study-{index}"),
            "slide_id": value.get("slide_id", f"slide-{index}"),
        }
        records.append(
            PredictionRecord(
                sample_id=str(value.get("sample_id", f"sample-{index}")),
                prediction=value.get("prediction", value.get("score")),
                target=value.get("target", value.get("label")),
                clinical_unit=unit,
                identity=ClinicalIdentity.from_dict(identity),
                split=split,
                groups={str(name): str(group) for name, group in dict(value.get("groups", {})).items()},
                is_padding=bool(value.get("is_padding", False)),
                metadata=dict(value.get("metadata", {})),
            )
        )
    provenance_raw = dict(raw.get("provenance", {}))
    config_hash_value = config_hash(config)
    provenance = RuntimeProvenance(
        model_hash=_hash_value(provenance_raw.get("model_hash", raw.get("model_hash")), config_hash_value[:16]),
        data_hash=_hash_value(provenance_raw.get("data_hash", raw.get("dataset_hash")), config_hash_value[16:32]),
        preprocess_hash=_hash_value(
            provenance_raw.get("preprocess_hash", raw.get("preprocessing_hash")), config_hash_value[32:]
        ),
        backend=str(provenance_raw.get("backend", raw.get("backend", "cpu"))),
        precision=str(provenance_raw.get("precision", raw.get("precision", "fp32"))),
        topology=str(provenance_raw.get("topology", raw.get("topology", "single"))),
        bucket=str(provenance_raw.get("bucket", raw.get("bucket", "unbucketed"))),
        checkpoint_format=str(provenance_raw.get("checkpoint_format", raw.get("checkpoint_format", "safetensors"))),
        model_revision=None if provenance_raw.get("model_revision") is None else str(provenance_raw["model_revision"]),
    )
    return PredictionArtifact(
        artifact_id=str(raw.get("artifact_id", raw.get("id", "phase16-evaluation"))),
        task=task,
        clinical_unit=unit,
        split=split,
        predictions=tuple(records),
        provenance=provenance,
        metadata={"config_hash": config_hash_value, **dict(raw.get("metadata", {}))},
    )


def run_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Run offline evaluation and return a JSON-compatible result."""

    path = Path(config_path)
    config = _load_config(path)
    raw = dict(config.get("evaluation", config))
    artifact = _records_from_config(config, config_path=path)
    output_dir = Path(str(raw.get("output_dir", "artifacts/evaluation/phase16")))
    if not output_dir.is_absolute():
        output_dir = (
            path.parent.parent.parent / output_dir
            if str(output_dir).startswith("artifacts/")
            else path.parent / output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{artifact.artifact_id}.predictions.json"
    save_prediction_artifact(artifact, prediction_path)
    metric_options = dict(raw.get("metric_options", {}))
    metrics = recompute_metrics(artifact, metric_options=metric_options)
    subgroup_results: dict[str, Any] = {}
    for group_name in sorted({name for row in artifact.valid_predictions for name in row.groups}):
        subgroup_rows = [row for row in artifact.valid_predictions if group_name in row.groups]
        if subgroup_rows:
            subgroup_artifact = PredictionArtifact(
                artifact_id=f"{artifact.artifact_id}-{group_name}",
                task=artifact.task,
                clinical_unit=artifact.clinical_unit,
                split=artifact.split,
                predictions=tuple(subgroup_rows),
                provenance=artifact.provenance,
            )
            subgroup_results[group_name] = {
                name: value.to_dict()
                for name, value in recompute_metrics(subgroup_artifact, metric_options=metric_options).items()
            }
    report = build_evaluation_report(
        artifact,
        metrics,
        report_id=str(raw.get("report_id", f"{artifact.artifact_id}-report")),
        subgroup_results=subgroup_results,
        baselines=dict(raw.get("baselines", {})),
        ablations=dict(raw.get("ablations", {})),
        human_review=dict(raw.get("human_review", {})),
        examples=tuple(raw.get("examples", ())),
        accelerator_metrics=dict(raw.get("accelerator_metrics", {})),
    )
    report_path = output_dir / "evaluation_report.json"
    report.write(report_path)
    return {
        "artifact_path": str(prediction_path),
        "report_path": str(report_path),
        "report_checksum": report.checksum(),
        "metrics": {name: value.to_dict() for name, value in sorted(metrics.items())},
        "clinical_unit": artifact.clinical_unit.metric_name,
        "split": artifact.split.value,
        "limitations": list(report.limitations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation from saved or inline predictions.")
    parser.add_argument("--config", required=True, help="YAML/JSON evaluation configuration")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args.config)
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Evaluation report: {result['report_path']}")
        print(f"Prediction artifact: {result['artifact_path']}")
        print(f"Clinical unit: {result['clinical_unit']} ({result['split']})")
        for name, metric in result["metrics"].items():
            if metric["value"] is not None:
                print(f"  {name}: {metric['value']:.6f}")
        print(
            "Clinical safety: Generated outputs require task-specific clinical validation "
            "and qualified human review before any clinical use."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
