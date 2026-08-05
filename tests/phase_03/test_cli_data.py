"""``medfm data`` CLI: fingerprint (Phase 03 smoke), inspect, migrate, split."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from medfm.tools import data_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "manifests" / "mixed_synthetic.parquet"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "medfm.cli.data", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_committed_fixture_exists_and_is_valid() -> None:
    assert FIXTURE.is_file(), "run tests/phase_03/generate_fixture.py to regenerate the smoke fixture"


def test_smoke_command_fingerprints_committed_fixture(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = _run_cli(["fingerprint", "--manifest", str(FIXTURE), "--output", str(output)])
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"]["samples"] > 0
    assert report["fingerprint_hash"]
    assert report["split_leakage"]["ok"] is True


def test_fingerprint_is_deterministic_across_runs(tmp_path: Path) -> None:
    hashes = []
    for i in range(2):
        output = tmp_path / f"report_{i}.json"
        result = _run_cli(["fingerprint", "--manifest", str(FIXTURE), "--output", str(output)])
        assert result.returncode == 0, result.stderr
        hashes.append(json.loads(output.read_text(encoding="utf-8"))["fingerprint_hash"])
    assert hashes[0] == hashes[1]


def test_fingerprint_missing_manifest_fails_cleanly() -> None:
    result = _run_cli(["fingerprint", "--manifest", "does/not/exist.parquet"])
    assert result.returncode != 0


def test_inspect_command_reports_schema(tmp_path: Path) -> None:
    output = tmp_path / "inspect.json"
    result = _run_cli(["inspect", "--manifest", str(FIXTURE), "--output", str(output)])
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["row_count"] > 0
    assert summary["columns_missing"] == []


def test_split_command_writes_assignment_and_report(tmp_path: Path) -> None:
    out_manifest = tmp_path / "split.parquet"
    out_report = tmp_path / "split_report.json"
    result = _run_cli(
        [
            "split",
            "--manifest",
            str(FIXTURE),
            "--output",
            str(out_manifest),
            "--report",
            str(out_report),
            "--policy",
            "patient",
            "--seed",
            "42",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert out_manifest.is_file() and out_report.is_file()
    report = json.loads(out_report.read_text(encoding="utf-8"))
    assert report["seed"] == 42
    assert report["policy"] == "PATIENT"
    assert report["report_hash"]


def test_migrate_command_round_trips(tmp_path: Path) -> None:
    out_manifest = tmp_path / "migrated.parquet"
    result = _run_cli(["migrate", "--input", str(FIXTURE), "--output", str(out_manifest)])
    assert result.returncode == 0, result.stderr
    assert out_manifest.is_file()


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        data_tools.main([])
