"""Local smoke runner: tiny forward pass on the requested backend.

Smoke status is distinct from metadata validation and local-weight
validation: this module actually constructs the model (via its plugin) and
runs one tiny backend-specific input. A successful smoke records per-backend
evidence (ModelRegistry.record_backend_result) and writes a run artifact
carrying the exact model ID and pinned revision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medfm.registry.core import ModelRegistry
from medfm.registry.plugins import get_plugin
from medfm.registry.schema import ModelSpec, ModelStatus


class NoAdapterError(RuntimeError):
    """Raised when a model has no registered plugin (adapter) yet."""


class BackendUnavailableError(RuntimeError):
    """Raised when the requested accelerator backend is not available locally."""


@dataclass(frozen=True)
class SmokeResult:
    model_id: str
    revision: str
    backend: str
    success: bool
    detail: str
    artifact_path: str | None


def _device_for_backend(backend: str) -> Any:
    import torch

    if backend == "cpu":
        return torch.device("cpu")
    if backend in ("cuda_single", "cuda_distributed"):
        if not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA is not available on this host")
        return torch.device("cuda")
    if backend in ("tpu_single_host", "tpu_multi_host"):
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as e:
            raise BackendUnavailableError("torch_xla is not installed") from e
        return xm.xla_device()
    raise ValueError(f"unknown backend: {backend}")


def run_smoke(
    model_id: str,
    backend: str = "cpu",
    artifact_dir: Path | str | None = None,
    record: bool = True,
) -> SmokeResult:
    """Run a tiny forward pass for ``model_id`` on ``backend``.

    Uses the exact registered revision: evidence gathered here applies only to
    that revision (record_backend_result enforces the match).
    """
    import torch

    spec = ModelRegistry.get(model_id)
    if spec.status == ModelStatus.BLOCKED:
        raise RuntimeError(f"{model_id} is BLOCKED: {spec.blocked_reason}")

    plugin = get_plugin(model_id)
    if plugin is None:
        raise NoAdapterError(
            f"no adapter plugin registered for {model_id}; adapter phases 06-08 "
            f"provide one (see medfm/registry/plugins.py)"
        )

    device = _device_for_backend(backend)
    model = plugin.build(spec).to(device)
    model.eval()

    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in plugin.tiny_input(spec).items()}
    with torch.no_grad():
        output = model(**inputs) if isinstance(inputs, dict) else model(inputs)

    detail = f"forward ok; output shape {getattr(output, 'shape', None)}"
    artifact_path = None
    if artifact_dir is not None:
        artifact_path = str(_write_run_artifact(spec, backend, detail, Path(artifact_dir)))

    if record:
        ModelRegistry.record_backend_result(
            model_id,
            backend,
            revision=spec.revision,
            success=True,
            date=datetime.now(UTC).date().isoformat(),
        )

    return SmokeResult(
        model_id=spec.model_id,
        revision=spec.revision,
        backend=backend,
        success=True,
        detail=detail,
        artifact_path=artifact_path,
    )


def _write_run_artifact(spec: ModelSpec, backend: str, detail: str, artifact_dir: Path) -> Path:
    """Every run artifact carries the exact model ID and pinned revision."""
    artifact = {
        "artifact_type": "model_smoke",
        "model_id": spec.model_id,
        "revision": spec.revision,
        "backend": backend,
        "success": True,
        "detail": detail,
        "created_at": datetime.now(UTC).isoformat(),
    }
    out_dir = artifact_dir / spec.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"smoke_{backend}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return path
