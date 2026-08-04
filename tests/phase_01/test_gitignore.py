"""Data/weight/cache paths must be git-ignored; source files must not be."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IGNORED_PATHS = [
    "artifacts/run_001/metrics.jsonl",
    "model_cache/hf/hub/model.safetensors",
    "dataset_cache/preprocessed/scan.pt",
    ".cache/models/checkpoint.ckpt",
    "data/patient/scan.nii.gz",
    "data/patient/slice.dcm",
    "data/pathology/slide.svs",
    "data/volume.mha",
    "weights/base_model.bin",
    "wandb/run-123/logs",
    ".env",
]

TRACKED_PATHS = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "Makefile",
    "medfm/tools/doctor.py",
    "tests/phase_01/conftest.py",
    "artifacts/.gitkeep",
]


def _check_ignore(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_data_weights_and_caches_are_ignored():
    ignored = set(_check_ignore(IGNORED_PATHS))
    missing = [p for p in IGNORED_PATHS if p not in ignored]
    assert not missing, f"paths not covered by .gitignore: {missing}"


def test_source_and_lockfile_are_not_ignored():
    ignored = set(_check_ignore(TRACKED_PATHS))
    leaked = [p for p in TRACKED_PATHS if p in ignored]
    assert not leaked, f"source paths wrongly ignored: {leaked}"
