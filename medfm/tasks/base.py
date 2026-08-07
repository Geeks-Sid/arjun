"""Shared task-module lifecycle and accelerator-neutral reduction utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError, UnsupportedModalityError
from medfm.core.task import LossOutput

VISUAL_MODALITIES: tuple[Modality, ...] = tuple(modality for modality in Modality if not modality.is_text_only)


def batch_size(batch: MedicalBatch) -> int:
    if batch.pixel_values is not None:
        return int(batch.pixel_values.shape[0])
    if batch.input_ids is not None:
        return int(batch.input_ids.shape[0])
    visual = batch.task_targets.get("visual_tokens")
    if isinstance(visual, torch.Tensor):
        return int(visual.shape[0])
    return len(batch.sample_ids)


def valid_sample_count(batch: MedicalBatch, *, default: int | None = None) -> int:
    """Return true non-padding sample count for reporting/reduction.

    The tensor mask is also returned in ``LossOutput.diagnostics`` by task
    implementations so distributed trainers can reduce counts on-device.  The
    integer is required by the Phase 02 public ``LossOutput`` contract and is
    only materialized at the task boundary.
    """

    mask = batch.task_targets.get("sample_mask")
    if mask is None:
        mask = batch.image_mask if batch.image_mask is not None and batch.image_mask.ndim == 1 else None
    if isinstance(mask, torch.Tensor):
        if mask.ndim != 1 or int(mask.shape[0]) != batch_size(batch):
            raise ShapeContractError("sample_mask must have shape [B]")
        return int(mask.to(dtype=torch.int64).sum().detach().cpu())
    return batch_size(batch) if default is None else int(default)


def valid_sample_mask(batch: MedicalBatch) -> torch.Tensor | None:
    mask = batch.task_targets.get("sample_mask")
    if mask is None and batch.image_mask is not None and batch.image_mask.ndim == 1:
        mask = batch.image_mask
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
        raise ShapeContractError("sample_mask must be a rank-1 tensor")
    return mask.to(dtype=torch.bool)


def target_from_batch(batch: MedicalBatch, key: str, *, allow_labels: bool = True) -> torch.Tensor:
    value = batch.task_targets.get(key)
    if value is None and allow_labels:
        value = batch.labels
    if not isinstance(value, torch.Tensor):
        raise ShapeContractError(f"batch is missing tensor target {key!r}")
    return value


class TaskModuleBase(nn.Module):
    """Base implementation of the Phase 02 ``TaskModule`` lifecycle."""

    def __init__(self, task_type: TaskType, supported_modalities: tuple[Modality, ...] = VISUAL_MODALITIES) -> None:
        super().__init__()
        self._task_type = task_type
        self._supported_modalities = tuple(supported_modalities)
        self.reset_metrics()

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def supported_modalities(self) -> tuple[Modality, ...]:
        return self._supported_modalities

    def check_supported(self, modality: Modality) -> None:
        if modality not in self.supported_modalities:
            raise UnsupportedModalityError(
                f"{self.__class__.__name__} does not support modality {modality}; "
                f"supported: {[entry.value for entry in self.supported_modalities]}"
            )

    def reset_metrics(self) -> None:
        self._metric_loss_sum = 0.0
        self._metric_count = 0
        self._metric_diagnostics: dict[str, float] = {}

    def _update_loss_metric(self, output: LossOutput) -> None:
        self._metric_loss_sum += float(output.total.detach().cpu()) * max(1, output.sample_count)
        self._metric_count += output.sample_count
        for name, value in output.components.items():
            self._metric_diagnostics[name] = self._metric_diagnostics.get(name, 0.0) + float(value.detach().cpu())

    def update_metrics(self, model_output: Any, batch: MedicalBatch) -> None:
        output = self.compute_loss(model_output, batch)
        self._update_loss_metric(output)

    def compute_metrics(self) -> dict[str, float]:
        denominator = max(1, self._metric_count)
        return {
            "loss": self._metric_loss_sum / denominator,
            "sample_count": float(self._metric_count),
            **{f"loss/{name}": value / denominator for name, value in self._metric_diagnostics.items()},
        }

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        raise NotImplementedError


def ensure_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ShapeContractError(f"{name} must be a scalar tensor")
    return value


def detached_count_tensor(count: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(count, dtype=torch.float32, device=reference.device)


def ensure_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShapeContractError(f"{name} must be a mapping keyed by task name")
    return value
