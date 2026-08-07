from __future__ import annotations

import torch

from medfm.evaluation.report import EvaluationArtifact
from medfm.recipes.phase15 import make_phase15_artifact, pathology_segmentation_metrics
from medfm.training.config import RunConfig


def test_pathology_segmentation_reports_tile_and_slide_units() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    predicted = torch.full_like(target, -8.0)
    predicted[..., 2:6, 2:6] = 8
    metrics = pathology_segmentation_metrics(predicted, target, slide_predicted=predicted, slide_target=target)
    assert metrics["tile/dice/class_0"].unit == "per_image"
    assert metrics["slide/dice/class_0"].unit == "per_image"
    assert metrics["slide/dice/class_0"].value > 0.9


def test_phase15_artifact_contains_reproducibility_and_limitations(tmp_path) -> None:
    config = RunConfig.from_dict(
        {
            "model_id": "local",
            "model": {"family": "phase15_pathology"},
            "dataset": {"id": "synthetic", "revision": "v1"},
            "task": {"type": "BINARY_CLASSIFICATION"},
            "recipe": {"id": "phase15-test", "family": "tile_classification"},
            "base_model_revision": "offline-random-contract",
            "preprocessing_hash": "pre-v1",
            "dataset_hash": "data-v1",
            "seed": 15,
            "output_dir": str(tmp_path),
        }
    )
    artifact = make_phase15_artifact(
        config,
        {
            "tile/auroc": pathology_segmentation_metrics(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))[
                "tile/dice/class_0"
            ]
        },
    )
    assert isinstance(artifact, EvaluationArtifact)
    assert artifact.reproducibility["config_hash"] == config.config_hash()
    assert artifact.limitations
    serialized = artifact.to_dict()
    assert serialized["schema_version"] == 1
    assert serialized["recipe_id"] == "phase15-test"
