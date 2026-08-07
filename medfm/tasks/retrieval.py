"""Contrastive image/text task wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput
from medfm.models.heads import ImageTextProjectionHead, RetrievalOutput, SymmetricContrastiveLoss

from .base import TaskModuleBase, detached_count_tensor, valid_sample_count


class RetrievalTask(TaskModuleBase):
    """Project image/text representations and compute symmetric alignment loss."""

    def __init__(
        self,
        head: ImageTextProjectionHead,
        loss: nn.Module | None = None,
        *,
        supported_modalities: tuple[Modality, ...] | None = None,
        filter_same_patient: bool = False,
    ) -> None:
        super().__init__(
            TaskType.CONTRASTIVE_ALIGNMENT, supported_modalities or tuple(m for m in Modality if not m.is_text_only)
        )
        self.head = head
        self.loss = loss or SymmetricContrastiveLoss(require_negative=filter_same_patient)
        self.filter_same_patient = bool(filter_same_patient)

    def forward(self, image: Any, text: torch.Tensor, **kwargs: Any) -> RetrievalOutput:
        return cast(RetrievalOutput, self.head(image, text, **kwargs))

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        if isinstance(model_output, RetrievalOutput):
            result = model_output
        elif isinstance(model_output, dict):
            image = model_output.get("image", model_output.get("encoder_output"))
            text = model_output.get("text_embeddings")
            if text is None:
                text = batch.task_targets.get("text_embeddings")
            if image is None or not isinstance(text, torch.Tensor):
                raise ShapeContractError("retrieval output needs image representation and text_embeddings")
            result = self.head(
                image,
                text,
                text_mask=model_output.get("text_mask"),
                patient_ids=model_output.get("patient_ids"),
                negative_provider=model_output.get("negative_provider"),
            )
        else:
            raise ShapeContractError("retrieval compute_loss expects RetrievalOutput or mapping")
        ids = result.auxiliary.get("patient_ids")
        patient_ids = ids if isinstance(ids, Sequence) and not isinstance(ids, str | bytes) else None
        if patient_ids is None:
            batch_ids = batch.task_targets.get("patient_ids")
            patient_ids = (
                batch_ids if isinstance(batch_ids, Sequence) and not isinstance(batch_ids, str | bytes) else None
            )
        total = cast(torch.Tensor, self.loss(result, patient_ids=patient_ids if self.filter_same_patient else None))
        count = valid_sample_count(batch, default=int(result.logits_per_image.shape[0]))
        return LossOutput(
            total=total,
            components={"alignment": total},
            sample_count=count,
            diagnostics={
                "task": self.task_type.value,
                "same_patient_filter": self.filter_same_patient,
                "valid_count": detached_count_tensor(count, result.logits_per_image),
            },
        )
