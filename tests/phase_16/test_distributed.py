from __future__ import annotations

import json
from pathlib import Path

import pytest

from medfm.evaluation.distributed import (
    BackendTolerance,
    assert_shared_evaluation_seed,
    compare_backend_metrics,
    gather_host_metadata,
    reduce_metric_counts,
    reduce_metric_mapping,
    remove_padded_duplicates,
)


def test_distributed_reductions_use_true_counts_and_host_metadata() -> None:
    reduced = reduce_metric_counts([{"numerator": 2, "denominator": 2}, {"numerator": 1, "denominator": 3}])
    assert reduced.numerator == 3
    assert reduced.denominator == 5
    mapping = reduce_metric_mapping(
        [{"accuracy": {"numerator": 1, "denominator": 2}}, {"accuracy": {"numerator": 3, "denominator": 4}}]
    )
    assert mapping["accuracy"].value == pytest.approx(4 / 6)
    metadata = gather_host_metadata([{"rows": [{"sample_id": "b"}]}, {"rows": [{"sample_id": "a"}]}])
    assert metadata[0]["rank"] == 0 and metadata[1]["rank"] == 1


def test_distributed_seed_and_backend_tolerance_contracts(prediction_artifact) -> None:
    assert assert_shared_evaluation_seed([17, 17]) == 17
    with pytest.raises(ValueError, match="seeds"):
        assert_shared_evaluation_seed([17, 18])
    parity = compare_backend_metrics(
        {"auroc": 1.0},
        {"auroc": 1.00001},
        backend="cuda",
        tolerance=BackendTolerance(1e-4, 1e-4),
    )
    assert parity.within_tolerance
    assert remove_padded_duplicates(prediction_artifact.predictions) == prediction_artifact.valid_predictions


def test_prediction_schema_json_is_machine_valid(prediction_artifact) -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(Path("docs/evaluation/prediction_schema.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(prediction_artifact.to_dict()))
    assert not errors
