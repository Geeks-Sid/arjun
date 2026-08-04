"""Local-first tracking: protocol, JSON tracker, redaction, optional extras."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medfm.training.tracking import (
    REDACTED,
    MLflowTracker,
    TensorBoardTracker,
    Tracker,
    WandBTracker,
    create_tracker,
    redact_mapping,
)


def test_redact_mapping_recurses_and_matches_case_insensitively():
    payload = {
        "hf_token": "abc",
        "learning_rate": 1e-4,
        "nested": {"WANDB_API_KEY": "xyz", "patient_id": "p1", "ok": 1},
    }
    redacted = redact_mapping(payload)
    assert redacted["hf_token"] == REDACTED
    assert redacted["learning_rate"] == 1e-4
    assert redacted["nested"]["WANDB_API_KEY"] == REDACTED
    assert redacted["nested"]["patient_id"] == REDACTED
    assert redacted["nested"]["ok"] == 1
    assert payload["hf_token"] == "abc"  # input not mutated


def test_local_json_tracker_round_trip(tmp_path: Path):
    tracker = create_tracker("local_json", log_dir=tmp_path / "run", sensitive_fragments=("token", "patient"))
    assert isinstance(tracker, Tracker)
    tracker.log_params({"model": "tiny", "api_token": "secret"})
    tracker.log_metrics({"loss": 0.5}, step=0)
    tracker.log_metrics({"loss": 0.25}, step=1)
    tracker.close()

    params = json.loads((tmp_path / "run" / "params.json").read_text())
    assert params == {"model": "tiny", "api_token": REDACTED}

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["metrics"] == {"loss": 0.5}
    assert records[1]["step"] == 1


def test_unknown_tracker_rejected():
    with pytest.raises(ValueError, match="unknown tracker"):
        create_tracker("nonexistent")


def test_external_trackers_behind_extras():
    # Neither mlflow nor wandb is part of the dev environment; both adapters
    # must fail with an actionable error, never a bare ModuleNotFoundError.
    with pytest.raises((ImportError, NotImplementedError), match="extra|reserved"):
        MLflowTracker()
    with pytest.raises((ImportError, NotImplementedError), match="extra|reserved"):
        WandBTracker()


def test_tensorboard_tracker_optional(tmp_path: Path):
    pytest.importorskip("tensorboard", reason="tracking extra not installed")
    tracker = TensorBoardTracker(log_dir=tmp_path / "tb")
    tracker.log_params({"hf_token": "secret", "model": "tiny"})
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.close()
    assert list((tmp_path / "tb").glob("events.*"))
