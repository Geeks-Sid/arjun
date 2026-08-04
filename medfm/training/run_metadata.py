"""Deterministic run-metadata capture for reproducibility.

Every training run must be describable by a canonical JSON document covering:
source state (commit, dirty tree, lockfile hash), runtime (Python, PyTorch,
CUDA/XLA, driver, accelerator topology), and configuration (seed, precision,
batch geometry, model/adapter hashes). Runtime-measured fields (peak memory,
XLA metrics) are captured but excluded from the configuration hash.

Dataset-manifest, preprocessing, and base-model hashes are placeholders here:
they are passed in by later phases and recorded verbatim.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE_PATH = REPO_ROOT / "uv.lock"


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunMetadata:
    """Canonical description of one training/evaluation run."""

    # Source state
    git_commit: str | None
    git_dirty: bool | None
    lockfile_sha256: str | None

    # Runtime
    python_version: str
    torch_version: str
    cuda_version: str | None
    driver_version: str | None
    gpu_models: list[str]
    accelerator_backend: str  # cpu | cuda | xla_tpu
    device_count: int
    world_size: int
    topology: str | None
    compiler_flags: dict[str, str]
    xla_metrics_report: str | None
    nccl_available: bool | None

    # Run configuration
    seed: int
    precision: str  # fp32 | bf16 | fp16 | bf16_mixed
    microbatch_per_device: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    shape_buckets: dict[str, Any]

    # Model / data provenance (placeholders filled by later phases)
    base_model_revision: str | None
    adapter_config_sha256: str | None
    dataset_manifest_sha256: str | None
    preprocessing_config_sha256: str | None

    # Measured
    trainable_parameters: int | None
    total_parameters: int | None
    peak_memory_bytes: int | None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Deterministic serialization: sorted keys, fixed separators."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def config_hash(self) -> str:
        """Hash of configuration-relevant fields only (not measured values)."""
        data = self.to_dict()
        for measured in ("peak_memory_bytes", "xla_metrics_report"):
            data.pop(measured, None)
        return _stable_hash(data)


def _accelerator_details(backend: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "cuda_version": None,
        "driver_version": None,
        "gpu_models": [],
        "device_count": 0,
        "topology": None,
        "xla_metrics_report": None,
        "peak_memory_bytes": None,
        "nccl_available": None,
    }
    if backend == "cuda":
        import torch

        details["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            details["device_count"] = torch.cuda.device_count()
            details["gpu_models"] = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
            details["nccl_available"] = bool(torch.distributed.is_available() and torch.distributed.is_nccl_available())
            if torch.cuda.is_initialized():
                details["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            details["driver_version"] = out.stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    elif backend == "xla_tpu":
        try:
            import torch_xla.debug.metrics as met
            import torch_xla.runtime as xr

            details["device_count"] = xr.global_runtime_device_count()
            details["topology"] = xr.device_type()
            report = met.short_metrics_report()
            details["xla_metrics_report"] = report if report else None
        except Exception:  # no XLA runtime attached
            pass
    return details


def capture_run_metadata(
    *,
    accelerator_backend: str,
    seed: int,
    precision: str,
    microbatch_per_device: int,
    gradient_accumulation_steps: int = 1,
    world_size: int | None = None,
    model: Any | None = None,
    base_model_revision: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    dataset_manifest_sha256: str | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    shape_buckets: dict[str, Any] | None = None,
    compiler_flags: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> RunMetadata:
    """Capture run metadata. ``world_size`` defaults to the local device count."""
    if accelerator_backend not in {"cpu", "cuda", "xla_tpu"}:
        raise ValueError(f"unknown accelerator backend: {accelerator_backend}")
    if microbatch_per_device < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch geometry values must be positive")

    import torch

    details = _accelerator_details(accelerator_backend)
    devices = int(details["device_count"] or 0)
    if world_size is None:
        world_size = max(devices, 1)

    trainable: int | None = None
    total: int | None = None
    if model is not None:
        params = list(model.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)

    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])

    return RunMetadata(
        git_commit=commit,
        git_dirty=(len(status) > 0) if status is not None else None,
        lockfile_sha256=_sha256_file(LOCKFILE_PATH),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=details["cuda_version"],
        driver_version=details["driver_version"],
        gpu_models=details["gpu_models"],
        accelerator_backend=accelerator_backend,
        device_count=devices,
        world_size=world_size,
        topology=details["topology"],
        compiler_flags=dict(compiler_flags or {}),
        xla_metrics_report=details["xla_metrics_report"],
        nccl_available=details["nccl_available"],
        seed=seed,
        precision=precision,
        microbatch_per_device=microbatch_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=microbatch_per_device * world_size * gradient_accumulation_steps,
        shape_buckets=dict(shape_buckets or {}),
        base_model_revision=base_model_revision,
        adapter_config_sha256=_stable_hash(adapter_config) if adapter_config is not None else None,
        dataset_manifest_sha256=dataset_manifest_sha256,
        preprocessing_config_sha256=(_stable_hash(preprocessing_config) if preprocessing_config is not None else None),
        trainable_parameters=trainable,
        total_parameters=total,
        peak_memory_bytes=details["peak_memory_bytes"],
        extra=dict(extra or {}),
    )
