"""Named multitask loss composition with fixed/scheduled weight extension points."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput

from .base import TaskModuleBase, ensure_scalar


class LossComputingTask(Protocol):
    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput: ...


_torch_assert = cast(Callable[[Any, str], None], torch._assert)


@runtime_checkable
class WeightSchedule(Protocol):
    """Schedule boundary; trainer owns the global step and static task set."""

    def __call__(self, step: int) -> float: ...


@dataclass(frozen=True)
class FixedWeight:
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0:
            raise ShapeContractError("fixed multitask weight must be finite and non-negative")

    def __call__(self, step: int) -> float:
        del step
        return self.value


@dataclass(frozen=True)
class LinearWeightSchedule:
    start: float
    end: float
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.start, self.end)):
            raise ShapeContractError("scheduled weights must be finite and non-negative")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ShapeContractError("weight schedule steps must be increasing")

    def __call__(self, step: int) -> float:
        if step <= self.start_step:
            return self.start
        if step >= self.end_step:
            return self.end
        ratio = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start + ratio * (self.end - self.start)


class WeightingExtension(Protocol):
    """Reserved interface for uncertainty/GradNorm weighting implementations."""

    def weights(self, losses: Mapping[str, LossOutput], step: int) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class TaskLossConfig:
    name: str
    weight: float | WeightSchedule
    required: bool = True

    def weight_at(self, step: int) -> float:
        value = self.weight(step) if callable(self.weight) else self.weight
        if not math.isfinite(value) or value < 0:
            raise ShapeContractError(f"weight for task {self.name!r} must be finite and non-negative")
        return float(value)


class MultiTaskLossComposer(nn.Module):
    """Compose a fixed task signature without task-dependent graph branching."""

    def __init__(
        self,
        task_weights: Mapping[str, float | WeightSchedule],
        *,
        step: int = 0,
        weighting_extension: WeightingExtension | None = None,
    ) -> None:
        super().__init__()
        if not task_weights:
            raise ShapeContractError("multitask composer requires at least one named task")
        self.task_names = tuple(task_weights)
        self.weights = {
            name: (FixedWeight(weight) if isinstance(weight, int | float) else weight)
            for name, weight in task_weights.items()
        }
        self.step = int(step)
        self.weighting_extension = weighting_extension
        self._validate_weights(self.step)

    def _validate_weights(self, step: int) -> dict[str, float]:
        values = {name: schedule(step) for name, schedule in self.weights.items()}
        if any(not math.isfinite(weight) or weight <= 0 for weight in values.values()):
            bad = [name for name, weight in values.items() if not math.isfinite(weight) or weight <= 0]
            raise ShapeContractError(f"active multitask weights must be finite and > 0; invalid: {bad}")
        return values

    def forward(
        self,
        losses: Mapping[str, LossOutput],
        *,
        step: int | None = None,
    ) -> LossOutput:
        if not isinstance(losses, Mapping):
            raise ShapeContractError("multitask losses must be a mapping")
        active_step = self.step if step is None else int(step)
        weights = self._validate_weights(active_step)
        missing = [name for name in self.task_names if name not in losses]
        extra = [name for name in losses if name not in self.task_names]
        if missing or extra:
            raise ShapeContractError(
                f"multitask signature mismatch; missing={missing or []}, unexpected={extra or []}. "
                "Use a fixed task signature per static bucket."
            )
        extension_weights: Mapping[str, torch.Tensor] = {}
        if self.weighting_extension is not None:
            extension_weights = self.weighting_extension.weights(losses, active_step)
        total: torch.Tensor | None = None
        components: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        token_counts: dict[str, int] = {}
        for name in self.task_names:
            output = losses[name]
            scalar = ensure_scalar(output.total, f"loss {name}")
            _torch_assert(torch.isfinite(scalar).all(), f"loss {name!r} is non-finite")
            weight_tensor = extension_weights.get(name)
            coefficient = weight_tensor if weight_tensor is not None else scalar.new_tensor(weights[name])
            _torch_assert(coefficient.ndim == 0, f"multitask weight for {name!r} must be scalar")
            _torch_assert(
                torch.isfinite(coefficient).all() & (coefficient > 0).all(),
                f"multitask extension weight for {name!r} must be finite and > 0",
            )
            weighted = coefficient * scalar
            total = weighted if total is None else total + weighted
            components[name] = scalar
            counts[name] = output.sample_count
            if output.token_count is not None:
                token_counts[name] = output.token_count
        assert total is not None
        total_count = sum(counts.values())
        diagnostics: dict[str, Any] = {
            "task_names": self.task_names,
            "weights": weights,
            "sample_counts": counts,
            "token_counts": token_counts,
            "reduction": "sum(weight * per-task-mean); reduce each task by its true count before composition",
        }
        return LossOutput(
            total=total,
            components=components,
            sample_count=total_count,
            token_count=sum(token_counts.values()) if token_counts else None,
            diagnostics=diagnostics,
        )

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ShapeContractError("multitask step must be non-negative")
        self.step = int(step)


class MultiTaskTask(TaskModuleBase):
    """TaskModule facade for a fixed set of child task modules."""

    def __init__(
        self,
        tasks: Mapping[str, nn.Module],
        composer: MultiTaskLossComposer,
        *,
        supported_modalities: tuple[Modality, ...] | None = None,
    ) -> None:
        super().__init__(TaskType.MULTITASK, supported_modalities or tuple(m for m in Modality if not m.is_text_only))
        if tuple(tasks) != composer.task_names:
            raise ShapeContractError("child task names must exactly match the multitask composer signature")
        self.tasks = nn.ModuleDict(dict(tasks))
        self.composer = composer

    def compute_loss(self, model_output: Mapping[str, Any], batch: MedicalBatch) -> LossOutput:
        if not isinstance(model_output, Mapping):
            raise ShapeContractError("multitask model output must be keyed by the fixed task signature")
        losses: dict[str, LossOutput] = {}
        for name in self.composer.task_names:
            task = self.tasks[name]
            if not hasattr(task, "compute_loss"):
                raise ShapeContractError(f"child task {name!r} does not implement compute_loss")
            child_task = cast(LossComputingTask, task)
            losses[name] = child_task.compute_loss(model_output[name], batch)
        return cast(LossOutput, self.composer(losses))

    def update_metrics(self, model_output: Any, batch: MedicalBatch) -> None:
        output = self.compute_loss(model_output, batch)
        self._update_loss_metric(output)
        for name, value in output.components.items():
            self._metric_diagnostics[f"task/{name}"] = float(value.detach().cpu())


# More explicit name for trainer imports.
MultiTaskLoss = MultiTaskLossComposer

__all__ = [
    "WeightSchedule",
    "FixedWeight",
    "LinearWeightSchedule",
    "WeightingExtension",
    "TaskLossConfig",
    "MultiTaskLossComposer",
    "MultiTaskLoss",
    "MultiTaskTask",
]
