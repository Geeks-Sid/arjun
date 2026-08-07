"""Visual-dependence checks shared by native and external VLM recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.task import TaskModule


@dataclass(frozen=True)
class VisualDependenceResult:
    """Loss comparison for image, no-image, and shuffled-image conditions."""

    image_loss: float
    no_image_loss: float
    shuffled_loss: float
    no_image_delta: float
    shuffled_delta: float
    passed: bool
    criteria: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_loss": self.image_loss,
            "no_image_loss": self.no_image_loss,
            "shuffled_loss": self.shuffled_loss,
            "no_image_delta": self.no_image_delta,
            "shuffled_delta": self.shuffled_delta,
            "passed": self.passed,
            "criteria": dict(self.criteria),
        }


def _loss_for_mode(model: nn.Module, task: TaskModule, batch: MedicalBatch, mode: str) -> float:
    forward_mode = getattr(model, "forward_mode", None)
    if not callable(forward_mode):
        raise TypeError("visual-dependence model must expose forward_mode(batch, mode=...)")
    output = forward_mode(batch, mode=mode)
    loss = task.compute_loss(output, batch)
    return float(loss.total.detach().float().cpu())


def run_visual_dependence_ablation(
    model: nn.Module,
    task: TaskModule,
    batch: MedicalBatch,
    *,
    minimum_no_image_delta: float = 1e-5,
    minimum_shuffled_delta: float = 0.0,
) -> VisualDependenceResult:
    """Run deterministic no-image/shuffled-image checks without mutating data.

    Loss is lower-is-better.  A shuffled-image loss that is equal to the
    visual loss therefore fails the gate: the model did not demonstrate
    measurable visual dependence within the predeclared margin.
    """

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            image_loss = _loss_for_mode(model, task, batch, "image")
            no_image_loss = _loss_for_mode(model, task, batch, "none")
            shuffled_loss = _loss_for_mode(model, task, batch, "shuffle")
    finally:
        model.train(was_training)
    no_image_delta = no_image_loss - image_loss
    shuffled_delta = shuffled_loss - image_loss
    criteria = {
        "minimum_no_image_delta": minimum_no_image_delta,
        "minimum_shuffled_delta": minimum_shuffled_delta,
        "requires_no_image_degradation": no_image_delta >= minimum_no_image_delta,
        "requires_shuffled_degradation": shuffled_delta > minimum_shuffled_delta,
    }
    return VisualDependenceResult(
        image_loss=image_loss,
        no_image_loss=no_image_loss,
        shuffled_loss=shuffled_loss,
        no_image_delta=no_image_delta,
        shuffled_delta=shuffled_delta,
        passed=bool(criteria["requires_no_image_degradation"] and criteria["requires_shuffled_degradation"]),
        criteria=criteria,
    )


__all__ = ["VisualDependenceResult", "run_visual_dependence_ablation"]
