"""Task and loss contracts.

Task modules (Phase 11+) turn model outputs into losses and metrics. The
contract keeps training backend-neutral and distributed-correct:

- ``LossOutput.total`` is the scalar used for ``backward()``; ``components``
  carry named, unreduced-per-component scalars for logging.
- ``sample_count`` is the number of *real* supervised examples contributing
  to the loss (padded bucket samples excluded). Distributed trainers must
  reduce losses by the summed true count across ranks, never by averaging
  per-rank means of uneven batches
  (``implementation_plan/accelerator_training_strategy.md``).

Metric lifecycle: ``reset_metrics()`` at epoch start, ``update_metrics()``
per batch, ``compute_metrics()`` at epoch end. Implementations must keep
running sufficient statistics (sums + counts) so metrics can be reduced
across distributed ranks by summation before the final ratio is computed —
``compute_metrics`` on partially reduced state is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class LossOutput:
    """Loss result of a task module.

    - ``total``: scalar tensor used for backward.
    - ``components``: named scalar tensors (e.g. ``{"ce": ..., "dice": ...}``);
      ``total`` must equal their weighted sum, with weights in ``diagnostics``
      when non-uniform.
    - ``sample_count``: real (non-padded) supervised examples in this batch.
    - ``diagnostics``: non-tensor extras (grad norms, queue sizes, weights).
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_count: int = 0
    token_count: int | None = None  # supervised text tokens, for LM losses
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total.ndim != 0:
            raise ShapeContractError(f"LossOutput.total must be a scalar tensor; got shape {tuple(self.total.shape)}")
        for name, component in self.components.items():
            if not name:
                raise ShapeContractError("loss component names must be non-empty")
            if not isinstance(component, torch.Tensor) or component.ndim != 0:
                raise ShapeContractError(f"loss component {name!r} must be a scalar tensor")
        if self.sample_count < 0:
            raise ShapeContractError("sample_count must be >= 0")
        if self.token_count is not None and self.token_count < 0:
            raise ShapeContractError("token_count must be >= 0")

    def component_dict(self) -> dict[str, float]:
        """Detached float view of components for trackers."""
        return {name: float(value.detach().cpu()) for name, value in self.components.items()}


@runtime_checkable
class TaskModule(Protocol):
    """Contract every task head/loss implements (Phase 11+).

    Implementations declare the modalities and task types they support up
    front; an unsupported combination must raise
    :class:`UnsupportedModalityError` / :class:`UnsupportedTaskError` from
    ``check_supported`` (trainers call it before the first batch).
    """

    @property
    def task_type(self) -> TaskType: ...

    @property
    def supported_modalities(self) -> tuple[Modality, ...]: ...

    def check_supported(self, modality: Modality) -> None: ...

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput: ...

    def reset_metrics(self) -> None:
        """Clear accumulated metric state (epoch start)."""
        ...

    def update_metrics(self, model_output: Any, batch: MedicalBatch) -> None:
        """Accumulate sufficient statistics for one batch (no reduction yet)."""
        ...

    def compute_metrics(self) -> dict[str, float]:
        """Finalize metrics from accumulated state (after any cross-rank sum)."""
        ...
