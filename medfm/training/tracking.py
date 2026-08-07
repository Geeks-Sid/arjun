"""Local-first experiment tracking.

A small ``Tracker`` protocol with a JSON-lines local tracker as the default.
TensorBoard is supported but optional; MLflow and W&B adapters exist only
behind their extras. External trackers are never required because medical
run metadata can be sensitive.

All trackers redact configured sensitive keys before anything is written.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from medfm.core.serialization import canonical_json

REDACTED = "<redacted>"

#: Key fragments (case-insensitive substring match) redacted before logging.
DEFAULT_SENSITIVE_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "credential",
    "patient",
    "mrn",
)


def redact_mapping(
    values: dict[str, Any],
    sensitive_fragments: tuple[str, ...] = DEFAULT_SENSITIVE_FRAGMENTS,
) -> dict[str, Any]:
    """Return a copy with sensitive keys redacted, including nested containers."""

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return redact_mapping(value, sensitive_fragments)
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(redact(item) for item in value)
        return value

    redacted: dict[str, Any] = {}
    for key, value in values.items():
        lowered = str(key).lower()
        redacted[key] = REDACTED if any(fragment in lowered for fragment in sensitive_fragments) else redact(value)
    return redacted


class NonFiniteTrainingError(RuntimeError):
    """Loss or gradients became NaN/Inf; collectives must not proceed."""


def assert_finite_loss(loss: torch.Tensor, *, step: int | None = None) -> None:
    """Fail before distributed reductions when the loss is not finite."""
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise NonFiniteTrainingError(f"training loss must be a scalar tensor (step={step})")
    if not bool(torch.isfinite(loss.detach()).all()):
        raise NonFiniteTrainingError(f"non-finite training loss at step {step}")


def gradient_finite_report(model: torch.nn.Module) -> dict[str, Any]:
    """Return a detached gradient audit without synchronizing accelerator state."""
    nonfinite: list[str] = []
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad.detach()).all()):
            nonfinite.append(name)
    return {
        "missing_gradient_names": missing,
        "nonfinite_gradient_names": nonfinite,
        "finite": not nonfinite,
    }


def assert_finite_gradients(model: torch.nn.Module, *, step: int | None = None) -> dict[str, Any]:
    report = gradient_finite_report(model)
    if report["nonfinite_gradient_names"]:
        raise NonFiniteTrainingError(
            f"non-finite gradients at step {step}: "
            + ", ".join(report["nonfinite_gradient_names"][:8])
        )
    return report


class FailureReporter:
    """Persist actionable failure context next to the last safe checkpoint."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        error: BaseException,
        *,
        config: dict[str, Any] | None = None,
        command: list[str] | None = None,
        last_safe_checkpoint: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "config": redact_mapping(dict(config or {})),
            "command": list(command or ()),
            "last_safe_checkpoint": str(last_safe_checkpoint) if last_safe_checkpoint is not None else None,
            "extra": redact_mapping(dict(extra or {})),
        }
        path = self.run_dir / "failure.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return path


@runtime_checkable
class Tracker(Protocol):
    """Minimal experiment-tracking protocol."""

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class LocalJSONTracker:
    """Default tracker: writes params.json and metrics.jsonl under a run dir."""

    def __init__(
        self,
        log_dir: str | Path,
        sensitive_fragments: tuple[str, ...] = DEFAULT_SENSITIVE_FRAGMENTS,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._fragments = sensitive_fragments
        self._metrics_path = self._log_dir / "metrics.jsonl"
        self._closed = False

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def log_params(self, params: dict[str, Any]) -> None:
        payload = redact_mapping(params, self._fragments)
        (self._log_dir / "params.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        record = {
            "step": step,
            "time": time.time(),
            "metrics": redact_mapping(metrics, self._fragments),
        }
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        self._closed = True


class TensorBoardTracker:
    """TensorBoard adapter. Requires the ``tracking`` extra; never mandatory."""

    def __init__(
        self,
        log_dir: str | Path,
        sensitive_fragments: tuple[str, ...] = DEFAULT_SENSITIVE_FRAGMENTS,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError("TensorBoardTracker requires the 'tracking' extra: uv sync --extra tracking") from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._fragments = sensitive_fragments

    def log_params(self, params: dict[str, Any]) -> None:
        payload = redact_mapping(params, self._fragments)
        self._writer.add_text("params", json.dumps(payload, indent=2, sort_keys=True, default=str))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for key, value in redact_mapping(metrics, self._fragments).items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        self._writer.close()


class MLflowTracker:
    """Reserved adapter behind the ``mlflow`` extra. Disabled by default."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        try:
            import mlflow  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError("MLflowTracker requires the 'mlflow' extra: uv sync --extra mlflow") from exc
        raise NotImplementedError("MLflow adapter is reserved for a later phase")


class WandBTracker:
    """Reserved adapter behind the ``wandb`` extra. Disabled by default."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        try:
            import wandb  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError("WandBTracker requires the 'wandb' extra: uv sync --extra wandb") from exc
        raise NotImplementedError("W&B adapter is reserved for a later phase")


_TRACKERS = {
    "local_json": LocalJSONTracker,
    "tensorboard": TensorBoardTracker,
    "mlflow": MLflowTracker,
    "wandb": WandBTracker,
}


def create_tracker(kind: str, **kwargs: Any) -> Tracker:
    """Create a tracker by name: local_json | tensorboard | mlflow | wandb."""
    try:
        cls = _TRACKERS[kind]
    except KeyError:
        raise ValueError(f"unknown tracker '{kind}'; expected one of {sorted(_TRACKERS)}") from None
    tracker = cls(**kwargs)
    assert isinstance(tracker, Tracker), f"{cls.__name__} does not satisfy the Tracker protocol"
    return tracker
