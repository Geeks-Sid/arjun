"""2D/3D segmentation task wrappers sharing one interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput
from medfm.models.decoders import SegmentationOutput

from .base import (
    TaskModuleBase,
    detached_count_tensor,
    target_from_batch,
    valid_sample_count,
    valid_sample_mask,
)
from .losses import DeepSupervisionLoss, DiceBCELoss, DiceCELoss

SEGMENTATION_MODALITIES: tuple[Modality, ...] = (
    Modality.XRAY_2D,
    Modality.CT_2D_SLICE,
    Modality.MRI_2D_SLICE,
    Modality.PATHOLOGY_TILE,
    Modality.CT_3D,
    Modality.MRI_3D,
    Modality.MULTI_SERIES_3D,
)


class SegmentationTask(TaskModuleBase):
    """Run a decoder over feature maps and compute Dice+CE/BCE by default."""

    def __init__(
        self,
        decoder: nn.Module,
        loss: nn.Module | None = None,
        *,
        binary: bool = False,
        deep_supervision_weights: Sequence[float] | None = None,
        supported_modalities: tuple[Modality, ...] = SEGMENTATION_MODALITIES,
    ) -> None:
        super().__init__(TaskType.SEMANTIC_SEGMENTATION, supported_modalities)
        self.decoder = decoder
        base_loss = loss or (DiceBCELoss() if binary else DiceCELoss())
        self.loss = (
            DeepSupervisionLoss(base_loss, deep_supervision_weights)
            if deep_supervision_weights is not None
            else base_loss
        )
        self.binary = bool(binary)

    def forward(self, model_input: Any) -> SegmentationOutput:
        if isinstance(model_input, SegmentationOutput):
            return model_input
        if isinstance(model_input, dict):
            if "segmentation" in model_input:
                candidate = model_input["segmentation"]
                if isinstance(candidate, SegmentationOutput):
                    return candidate
                if isinstance(candidate, torch.Tensor):
                    return SegmentationOutput(logits=candidate)
            for key in ("feature_maps", "visual_features", "encoder_output"):
                if key in model_input:
                    model_input = model_input[key]
                    break
        value = self.decoder(model_input)
        if isinstance(value, SegmentationOutput):
            return value
        if isinstance(value, torch.Tensor):
            return SegmentationOutput(logits=value)
        if isinstance(value, dict) and isinstance(value.get("logits"), torch.Tensor):
            return SegmentationOutput(
                logits=value["logits"],
                deep_supervision=tuple(value.get("deep_supervision", ())),
                native_outputs=value.get("native_outputs"),
            )
        if hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
            return SegmentationOutput(
                logits=value.logits,
                deep_supervision=tuple(getattr(value, "deep_supervision", ())),
                native_outputs=getattr(value, "native_outputs", None),
            )
        raise ShapeContractError("segmentation decoder must return SegmentationOutput or logits tensor")

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        output = self.forward(model_output)
        target = target_from_batch(batch, "segmentation", allow_labels=False)
        valid_mask = batch.task_targets.get("voxel_mask")
        if valid_mask is not None and not isinstance(valid_mask, torch.Tensor):
            raise ShapeContractError("voxel_mask must be a tensor")
        if isinstance(valid_mask, torch.Tensor) and valid_mask.ndim == target.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)
        sample_mask = valid_sample_mask(batch)
        if sample_mask is not None:
            sample_mask = sample_mask.reshape(sample_mask.shape[0], *([1] * (target.ndim - 1)))
            valid_mask = sample_mask if valid_mask is None else valid_mask.to(dtype=torch.bool) & sample_mask
        if isinstance(self.loss, DeepSupervisionLoss):
            values: tuple[torch.Tensor, ...] | torch.Tensor = output.deep_supervision or (output.logits,)
            total = self.loss(values, target, valid_mask=valid_mask)
            components = {"segmentation": total}
        else:
            total = self.loss(output.logits, target, valid_mask=valid_mask)
            components = {"segmentation": total}
        count = valid_sample_count(batch)
        return LossOutput(
            total=total,
            components=components,
            sample_count=count,
            diagnostics={
                "task": self.task_type.value,
                "spatial_rank": output.logits.ndim - 2,
                "valid_count": detached_count_tensor(count, output.logits),
            },
        )


class BinarySegmentationTask(SegmentationTask):
    def __init__(self, decoder: nn.Module, loss: nn.Module | None = None, **kwargs: Any) -> None:
        super().__init__(decoder, loss or DiceBCELoss(), binary=True, **kwargs)
