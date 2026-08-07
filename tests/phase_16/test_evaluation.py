from __future__ import annotations

import math

import pytest
import torch

from medfm.evaluation import (
    ClinicalIdentity,
    EvaluationReport,
    EvaluationSchemaError,
    PredictionArtifact,
    PredictionRecord,
    aggregate_prediction_records,
    apply_threshold,
    box_iou,
    build_evaluation_report,
    classification_metrics,
    cluster_bootstrap_ci,
    compare_backend_predictions,
    expected_calibration_error,
    fit_calibration,
    fit_threshold,
    generation_metrics,
    inter_rater_agreement,
    localization_metrics,
    recompute_metrics,
    retrieval_metrics,
    save_prediction_artifact,
    segmentation_metrics,
    summarize_reviews,
    visual_grounding_gate,
)
from medfm.evaluation.human_review import ReviewRecord, default_review_protocol, sample_review_items


def test_prediction_artifact_roundtrip_aggregates_at_patient_and_drops_padding(prediction_artifact, tmp_path) -> None:
    aggregated = aggregate_prediction_records(prediction_artifact.predictions, unit="patient")
    assert aggregated.keys == ("p0", "p1")
    assert aggregated.labels == (0, 1)
    assert aggregated.source_counts == (2, 1)
    assert len(prediction_artifact.valid_predictions) == 3
    destination = save_prediction_artifact(prediction_artifact, tmp_path / "predictions.json")
    loaded = PredictionArtifact.load(destination)
    assert loaded.canonical_json() == prediction_artifact.canonical_json()
    assert recompute_metrics(loaded)["auroc"].value == 1.0


def test_clinical_unit_declarations_are_required() -> None:
    with pytest.raises(EvaluationSchemaError, match="patient_id"):
        PredictionRecord(
            sample_id="missing-patient",
            prediction=0.5,
            target=1,
            clinical_unit="patient",
            identity=ClinicalIdentity(study_id="study"),
        )


def test_classification_hand_fixture_reports_required_metrics() -> None:
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], unit="per_patient")
    assert metrics["auroc"].value == 1.0
    assert metrics["auprc"].value == 1.0
    assert metrics["sensitivity"].value == 1.0
    assert metrics["specificity"].value == 1.0
    assert metrics["precision"].value == 1.0
    assert metrics["recall"].value == 1.0
    assert metrics["f1"].value == 1.0
    assert metrics["balanced_accuracy"].value == 1.0
    assert metrics["confusion_matrix"].metadata["matrix"] == [[2, 0], [0, 2]]
    assert metrics["brier"].value == pytest.approx(0.025)
    assert metrics["ece"].value is not None


def test_calibration_and_thresholds_refuse_test_fit() -> None:
    selection = fit_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], split="validation")
    assert apply_threshold([0.2, 0.8], selection, split="test").tolist() == [False, True]
    assert fit_calibration([0, 1], [0.1, 0.9], split="validation").transform([0.1]).shape == (1,)
    with pytest.raises(EvaluationSchemaError):
        fit_threshold([0, 1], [0.1, 0.9], split="test")
    with pytest.raises(EvaluationSchemaError):
        fit_calibration([0, 1], [0.1, 0.9], split="test")


def test_patient_cluster_bootstrap_is_deterministic_and_not_slice_level() -> None:
    first = cluster_bootstrap_ci(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        cluster_ids=["p0", "p0", "p1", "p1"],
        metric="auroc",
        unit="per_patient",
        resamples=50,
        seed=17,
    )
    second = cluster_bootstrap_ci(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        cluster_ids=["p0", "p0", "p1", "p1"],
        metric="auroc",
        unit="per_patient",
        resamples=50,
        seed=17,
    )
    assert first.to_dict() == second.to_dict()
    with pytest.raises(ValueError, match="cluster_ids"):
        cluster_bootstrap_ci([0, 1], [0.1, 0.9], cluster_ids=None, unit="per_patient")


def test_empty_segmentation_and_physical_localization_metrics() -> None:
    empty = segmentation_metrics(torch.full((1, 1, 4, 4), -10.0), torch.zeros(1, 1, 4, 4), spacing_mm=(2.0, 1.0))
    assert empty["dice/class_0"].value == 1.0
    assert empty["iou/class_0"].value == 1.0
    assert empty["hd95/class_0"].value == 0.0
    target = torch.zeros(1, 1, 8, 8, 8)
    target[..., 2:5, 2:5, 2:5] = 1
    perfect = segmentation_metrics(
        torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0)),
        target,
        spacing_mm=(2.0, 1.0, 1.0),
        unit="per_scan",
    )
    assert perfect["assd/class_0"].value == pytest.approx(0.0)
    assert box_iou((0, 0, 4, 4), (0, 0, 4, 4)) == 1.0
    localization = localization_metrics([(0, 0, 0, 2, 2, 2)], [(1, 0, 0, 3, 2, 2)], spacing_mm=(2, 1, 1))
    assert localization["physical_localization_error_mm"].value == pytest.approx(2.0)


def test_retrieval_and_generation_metrics_cover_both_directions_and_clinical_errors() -> None:
    retrieval = retrieval_metrics([[1.0, 0.0], [0.0, 1.0]], query_ids=["a", "b"], candidate_ids=["a", "b"])
    assert retrieval["image_to_text/recall@1"].value == 1.0
    assert retrieval["text_to_image/mAP"].value == 1.0
    generation = generation_metrics(
        ['{"findings": [{"finding": "mass", "negation": "present", "anatomy": "left lung"}]}'],
        ['{"findings": [{"finding": "mass", "negation": "absent", "anatomy": "left lung"}]}'],
        schema={"type": "object", "required": ["findings"]},
    )
    assert generation["schema_validity"].value == 1.0
    assert generation["contradiction_rate"].value == 1.0
    assert generation["anatomy_correctness"].value == 1.0
    assert generation["bleu_secondary"].metadata["headline"] is False


def test_visual_grounding_gate_fails_shuffled_inputs_within_margin() -> None:
    assert visual_grounding_gate(0.80, 0.78, margin=0.05)["passed"] is False
    assert visual_grounding_gate(0.80, 0.60, margin=0.05)["passed"] is True


def test_backend_parity_retains_divergence_and_report_is_recomputable(prediction_artifact, tmp_path) -> None:
    parity = compare_backend_predictions([0.0, 1.0], [0.0, 1.2], backend="cuda")
    assert parity.within_tolerance is False
    metrics = recompute_metrics(prediction_artifact)
    report = build_evaluation_report(prediction_artifact, metrics)
    assert report.checksum()
    report_path = report.write(tmp_path / "report.json")
    loaded = EvaluationReport.load(report_path)
    assert loaded.checksum() == report.checksum()


def test_review_protocol_and_agreement_are_access_controlled() -> None:
    protocol = default_review_protocol(sample_size=2, seed=3)
    assert sample_review_items(["b", "a", "c"], sample_size=2, seed=3) == sample_review_items(
        ["b", "a", "c"], sample_size=2, seed=3
    )
    records = [
        ReviewRecord("i0", "r0", "correct", "protected://review/i0"),
        ReviewRecord("i0", "r1", "major_error", "protected://review/i0"),
    ]
    agreement = inter_rater_agreement(records)
    assert agreement["pair_count"] == 1
    summary = summarize_reviews(records)
    assert summary["major_or_harmful_count"] == 1
    assert protocol.access_control["classification"] == "restricted"


def test_expected_calibration_error_empty_is_explicit() -> None:
    assert expected_calibration_error([], [], bins=10) is None
    assert math.isfinite(expected_calibration_error([0, 1], [0.1, 0.9]) or 0.0)
