"""Doctor diagnostics: schema conformance, redaction, backend reporting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from medfm.tools import doctor

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "medfm" / "tools" / "doctor_schema.json").read_text())


def _validate(report: dict) -> None:
    errors = sorted(e.message for e in Draft202012Validator(SCHEMA).iter_errors(report))
    assert not errors, f"doctor report failed schema: {errors}"


def test_cpu_report_conforms_to_schema():
    report = doctor.collect(backend="cpu")
    _validate(report)
    assert report["backend"] == "cpu"
    assert report["torch"]["cuda_initialized"] is False


def test_json_cli_conforms_to_schema():
    result = subprocess.run(
        [sys.executable, "-m", "medfm.tools.doctor", "--backend", "cpu", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    _validate(json.loads(result.stdout))


def test_cuda_backend_reports_expected_fields():
    report = doctor.collect(backend="cuda")
    _validate(report)
    acc = report["accelerator"]
    for key in (
        "available",
        "device_count",
        "cuda_runtime_version",
        "bf16_supported",
        "sdpa_available",
        "flash_attention",
        "nccl_available",
        "devices",
    ):
        assert key in acc, f"missing CUDA field: {key}"


def test_tpu_selection_reports_incompatible_packages():
    report = doctor.collect(backend="xla_tpu")
    _validate(report)
    acc = report["accelerator"]
    assert "incompatible_packages" in acc
    assert "spmd_available" in acc
    assert acc["torch_xla_version"] == report["packages"]["torch_xla"]


def test_no_credentials_or_absolute_home_paths_leak():
    env = {
        **os.environ,
        "HF_TOKEN": "supersecret-token-value",
        "WANDB_API_KEY": "supersecret-wandb-value",
        "MEDFM_MODEL_CACHE": str(Path.home() / ".cache" / "medfm" / "models"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "medfm.tools.doctor", "--backend", "cpu", "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "supersecret" not in result.stdout
    assert str(Path.home()) not in result.stdout, "absolute home path leaked"


def test_package_versions_reported_without_import():
    report = doctor.collect(backend="cpu")
    packages = report["packages"]
    assert packages["torch"] is not None
    # bitsandbytes must be reported by version lookup only — never imported
    assert "bitsandbytes" not in sys.modules
    assert "torch_xla" not in sys.modules
