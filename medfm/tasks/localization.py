"""Bounding-box localization task wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import CoordinateSystem, Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput
from medfm.models.heads import BoxL1Loss, BoxOutput, GIoULoss

from .base import TaskModuleBase, detached_count_tensor, target_from_batch, valid_sample_count


class LocalizationTask(TaskModuleBase):
    def __init__(
        self,
        head: nn.Module,
        *,
        l1_loss: nn.Module | None = None,
        iou_loss: nn.Module | None = None,
        l1_weight: float = 1.0,
        iou_weight: float = 1.0,
        supported_modalities: tuple[Modality, ...] | None = None,
    ) -> None:
        super().__init__(
            TaskType.BOUNDING_BOX_LOCALIZATION, supported_modalities or tuple(m for m in Modality if not m.is_text_only)
        )
        if l1_weight < 0 or iou_weight < 0 or l1_weight + iou_weight <= 0:
            raise ShapeContractError("localization loss weights must be non-negative and not both zero")
        self.head = head
        self.l1_loss = l1_loss or BoxL1Loss()
        self.iou_loss = iou_loss or GIoULoss()
        self.l1_weight = float(l1_weight)
        self.iou_weight = float(iou_weight)

    def forward(self, value: Any) -> BoxOutput:
        output = self.head(value)
        if isinstance(output, BoxOutput):
            return output
        if isinstance(output, torch.Tensor):
            return BoxOutput(output, coordinate_system=CoordinateSystem.NORMALIZED_IMAGE)
        raise ShapeContractError("localization head must return BoxOutput")

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        predicted = model_output if isinstance(model_output, BoxOutput) else self.head(model_output)
        if not isinstance(predicted, BoxOutput):
            raise ShapeContractError("localization head must return BoxOutput")
        target = target_from_batch(batch, "boxes", allow_labels=False)
        box_mask = batch.task_targets.get("box_mask")
        if box_mask is not None and not isinstance(box_mask, torch.Tensor):
            raise ShapeContractError("box_mask must be a tensor")
        l1 = (
            self.l1_loss(predicted, target, valid_mask=box_mask)
            if box_mask is not None
            else self.l1_loss(predicted, target)
        )
        iou = (
            self.iou_loss(predicted, target, valid_mask=box_mask)
            if box_mask is not None
            else self.iou_loss(predicted, target)
        )
        total = self.l1_weight * l1 + self.iou_weight * iou
        count = valid_sample_count(batch)
        return LossOutput(
            total=total,
            components={"box_l1": l1, "box_iou": iou},
            sample_count=count,
            diagnostics={
                "task": self.task_type.value,
                "l1_weight": self.l1_weight,
                "iou_weight": self.iou_weight,
                "valid_count": detached_count_tensor(count, predicted.boxes),
            },
        )
