from __future__ import annotations

from pathlib import Path

import pytest

from medfm.evaluation import (
    ClinicalIdentity,
    ClinicalUnit,
    EvaluationSplit,
    PredictionArtifact,
    PredictionRecord,
    RuntimeProvenance,
)


@pytest.fixture
def prediction_artifact(tmp_path: Path) -> PredictionArtifact:
    rows = (
        PredictionRecord(
            sample_id="p0-s0",
            prediction=0.1,
            target=0,
            clinical_unit=ClinicalUnit.PATIENT,
            identity=ClinicalIdentity(patient_id="p0", study_id="s0"),
            split=EvaluationSplit.TEST,
            groups={"site": "a"},
        ),
        PredictionRecord(
            sample_id="p0-s1",
            prediction=0.2,
            target=0,
            clinical_unit=ClinicalUnit.PATIENT,
            identity=ClinicalIdentity(patient_id="p0", study_id="s1"),
            split=EvaluationSplit.TEST,
            groups={"site": "a"},
        ),
        PredictionRecord(
            sample_id="p1-s2",
            prediction=0.8,
            target=1,
            clinical_unit=ClinicalUnit.PATIENT,
            identity=ClinicalIdentity(patient_id="p1", study_id="s2"),
            split=EvaluationSplit.TEST,
            groups={"site": "b"},
        ),
        PredictionRecord(
            sample_id="p1-s2",
            prediction=0.8,
            target=1,
            clinical_unit=ClinicalUnit.PATIENT,
            identity=ClinicalIdentity(patient_id="p1", study_id="s2"),
            split=EvaluationSplit.TEST,
            is_padding=True,
        ),
    )
    return PredictionArtifact(
        artifact_id="phase16-fixture",
        task="classification",
        clinical_unit=ClinicalUnit.PATIENT,
        split=EvaluationSplit.TEST,
        predictions=rows,
        provenance=RuntimeProvenance(
            model_hash="model-v1",
            data_hash="data-v1",
            preprocess_hash="pre-v1",
            backend="cpu",
            precision="fp32",
            topology="single",
            bucket="tiny",
            checkpoint_format="safetensors",
        ),
    )
