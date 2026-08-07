"""Validation/evaluation loop kept separate from optimizer orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.task import TaskModule
from medfm.training.backend import AcceleratorBackend
from medfm.training.steps import TrainingStep
from medfm.training.tracking import assert_finite_loss


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    batches: int
    samples: int
    step: int
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "batches": self.batches,
            "samples": self.samples,
            "step": self.step,
            "backend": self.backend,
            "diagnostics": dict(self.diagnostics),
        }


class Evaluator:
    """Run task metrics without retaining computation graphs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        task: TaskModule,
        step: TrainingStep,
        backend: AcceleratorBackend,
    ) -> None:
        self.model = model
        self.task = task
        self.step = step
        self.backend = backend

    def evaluate(
        self,
        dataloader: Any,
        *,
        global_step: int = 0,
        max_batches: int | None = None,
    ) -> EvaluationResult:
        was_training = self.model.training
        self.model.eval()
        self.task.reset_metrics()
        batches = 0
        local_samples = 0
        try:
            with torch.no_grad():
                for raw_batch in dataloader:
                    if max_batches is not None and batches >= max_batches:
                        break
                    batch = self.backend.prepare_batch(raw_batch)
                    with self.backend.autocast():
                        output = self.step.forward_model(self.model, batch)
                        loss_output = self.step.ensure_loss_output(self.task.compute_loss(output, batch))
                    assert_finite_loss(loss_output.total, step=global_step)
                    self.task.update_metrics(output, batch)
                    local_samples += int(loss_output.sample_count)
                    batches += 1
                    self.backend.mark_step()
            local_metrics = self.task.compute_metrics()
        finally:
            if was_training:
                self.model.train()
        # A rank with only padded samples contributes zero weight; using one
        # here would bias the cross-rank denominator.
        local_count = torch.tensor(float(local_samples), device=self.backend.device)
        global_count = self.backend.reduce_sum(local_count)
        metrics: dict[str, float] = {}
        for name, value in local_metrics.items():
            scalar = torch.tensor(float(value), device=self.backend.device) * local_count
            reduced = self.backend.reduce_sum(scalar)
            metrics[name] = float((reduced / global_count.clamp_min(1.0)).detach().cpu())
        reduced_samples = self.backend.reduce_sum(torch.tensor(float(local_samples), device=self.backend.device))
        return EvaluationResult(
            metrics=metrics,
            batches=batches,
            samples=int(reduced_samples.detach().cpu()),
            step=global_step,
            backend=self.backend.name,
        )

    __call__ = evaluate


__all__ = ["EvaluationResult", "Evaluator"]
