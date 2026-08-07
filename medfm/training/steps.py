"""Task-specific forward/loss adapters behind one trainer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput, TaskModule

ModelForward = Callable[[nn.Module, MedicalBatch], Any]


class TrainingStep(ABC):
    """One task family's model forward and loss contract.

    The trainer owns accumulation, placement, clipping, and reductions.  This
    object owns only the task-specific model input/output semantics, keeping
    classification, segmentation, and VLM code out of the loop itself.
    """

    task_family: str = "generic"

    def __init__(self, task: TaskModule, *, model_forward: ModelForward | None = None):
        self.task = task
        self.model_forward = model_forward
        self.last_model_output: Any | None = None

    @abstractmethod
    def forward_and_loss(self, model: nn.Module, batch: MedicalBatch) -> LossOutput: ...

    def forward_model(self, model: nn.Module, batch: MedicalBatch) -> Any:
        return invoke_model(model, batch, forward=self.model_forward)

    @staticmethod
    def ensure_loss_output(value: LossOutput) -> LossOutput:
        if not isinstance(value, LossOutput):
            raise ShapeContractError("task.compute_loss must return LossOutput")
        # Numerically sensitive losses are accumulated in FP32.  Casting here
        # preserves the graph while avoiding low-precision scalar reductions.
        total = value.total.float() if value.total.dtype in (torch.float16, torch.bfloat16) else value.total
        components = {
            name: component.float() if component.dtype in (torch.float16, torch.bfloat16) else component
            for name, component in value.components.items()
        }
        components_changed = any(components[name] is not value.components[name] for name in components)
        if total is value.total and not components_changed:
            return value
        return LossOutput(
            total=total,
            components=components,
            sample_count=value.sample_count,
            token_count=value.token_count,
            diagnostics=value.diagnostics,
        )


class TaskTrainingStep(TrainingStep):
    """Generic task adapter used by custom recipe modules."""

    task_family = "task"

    def forward_and_loss(self, model: nn.Module, batch: MedicalBatch) -> LossOutput:
        output = self.forward_model(model, batch)
        self.last_model_output = output
        return self.ensure_loss_output(self.task.compute_loss(output, batch))


class ClassificationTrainingStep(TaskTrainingStep):
    """Classification head/loss step; model output stays task-owned."""

    task_family = "classification"


class SegmentationTrainingStep(TaskTrainingStep):
    """Shared 2D/3D segmentation decoder/loss step."""

    task_family = "segmentation"


class VLMTrainingStep(TaskTrainingStep):
    """2D visual-language step with no assumptions about a concrete VLM."""

    task_family = "vlm_2d"


class ThreeDVLMTrainingStep(VLMTrainingStep):
    """3D visual-language step; only the declared modality differs."""

    task_family = "vlm_3d"


# Alternate spelling used by recipe code.
VLM2DTrainingStep = VLMTrainingStep
VLM3DTrainingStep = ThreeDVLMTrainingStep


def invoke_model(model: nn.Module, batch: MedicalBatch, *, forward: ModelForward | None = None) -> Any:
    """Call a model through a backend-neutral batch interface.

    Recipe code can provide ``forward`` for unusual model signatures.  The
    default supports modules that accept ``MedicalBatch``, ``forward_batch``,
    or a single tensor field for tiny/local models.
    """
    if forward is not None:
        return forward(model, batch)
    forward_batch = getattr(model, "forward_batch", None)
    if callable(forward_batch):
        return forward_batch(batch)
    try:
        return model(batch)
    except (TypeError, AttributeError) as first_error:
        tensor = batch.pixel_values
        if tensor is None:
            tensor = batch.input_ids
        if tensor is None:
            visual = batch.task_targets.get("visual_tokens")
            tensor = visual if isinstance(visual, torch.Tensor) else None
        if tensor is None:
            raise ShapeContractError(
                "model does not accept MedicalBatch and no tensor input is available for fallback"
            ) from first_error
        try:
            return model(tensor)
        except (TypeError, AttributeError) as second_error:
            raise ShapeContractError("model must accept MedicalBatch or its primary tensor input") from second_error


def _task_name(task: TaskModule) -> str:
    value = getattr(task, "task_type", None)
    return str(getattr(value, "value", value or "")).lower()


def make_training_step(
    task: TaskModule,
    *,
    model_forward: ModelForward | None = None,
    family: str | None = None,
) -> TrainingStep:
    """Select a task-step family without embedding recipe model choices."""
    selected = (family or _task_name(task)).lower()
    if selected in {
        "classification",
        "binary_classification",
        "multiclass_classification",
        "multilabel_classification",
        "ordinal_classification",
    }:
        return ClassificationTrainingStep(task, model_forward=model_forward)
    if "segmentation" in selected:
        return SegmentationTrainingStep(task, model_forward=model_forward)
    if "3d" in selected and ("vlm" in selected or "generation" in selected):
        return ThreeDVLMTrainingStep(task, model_forward=model_forward)
    if any(token in selected for token in ("vlm", "generation", "retrieval", "alignment", "language")):
        return VLMTrainingStep(task, model_forward=model_forward)
    return TaskTrainingStep(task, model_forward=model_forward)


__all__ = [
    "ClassificationTrainingStep",
    "ModelForward",
    "SegmentationTrainingStep",
    "TaskTrainingStep",
    "ThreeDVLMTrainingStep",
    "TrainingStep",
    "VLM2DTrainingStep",
    "VLM3DTrainingStep",
    "VLMTrainingStep",
    "invoke_model",
    "make_training_step",
]
