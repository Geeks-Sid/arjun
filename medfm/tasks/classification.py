"""Classification task wrappers over encoder-independent heads and losses."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderOutput
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.task import LossOutput

from .base import TaskModuleBase, detached_count_tensor, target_from_batch, valid_sample_count, valid_sample_mask
from .losses import (
    AsymmetricMultilabelLoss,
    BinaryCrossEntropyWithLogitsLoss,
    CrossEntropyClassificationLoss,
    OrdinalCumulativeLinkLoss,
)


class ClassificationTask(TaskModuleBase):
    """Attach a classification head and a named baseline/optional loss."""

    def __init__(
        self,
        head: nn.Module,
        loss: nn.Module | None = None,
        *,
        task_type: TaskType = TaskType.MULTICLASS_CLASSIFICATION,
        supported_modalities: tuple[Modality, ...] | None = None,
    ) -> None:
        super().__init__(task_type, supported_modalities or tuple(m for m in Modality if not m.is_text_only))
        self.head = head
        self.loss = loss or CrossEntropyClassificationLoss()

    def forward(self, encoder_output: EncoderOutput | torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(encoder_output))

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        logits = (
            self.forward(model_output)
            if isinstance(model_output, EncoderOutput | torch.Tensor)
            else self._extract_logits(model_output)
        )
        targets = target_from_batch(batch, "classification")
        if isinstance(self.loss, BinaryCrossEntropyWithLogitsLoss | AsymmetricMultilabelLoss):
            if targets.shape != logits.shape:
                raise ShapeContractError("multi-label classification targets must match logits shape")
            value = cast(torch.Tensor, self.loss(logits, targets, valid_mask=valid_sample_mask(batch)))
        elif isinstance(self.loss, OrdinalCumulativeLinkLoss):
            value = cast(torch.Tensor, self.loss(logits, targets, valid_mask=valid_sample_mask(batch)))
        else:
            value = cast(torch.Tensor, self.loss(logits, targets, valid_mask=valid_sample_mask(batch)))
        value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
        count = valid_sample_count(batch)
        count_tensor = detached_count_tensor(count, logits)
        output = LossOutput(
            total=value,
            components={"classification": value},
            sample_count=count,
            diagnostics={"task": self.task_type.value, "valid_count": count_tensor},
        )
        return output

    @staticmethod
    def _extract_logits(value: Any) -> torch.Tensor:
        if isinstance(value, dict) and isinstance(value.get("logits"), torch.Tensor):
            return cast(torch.Tensor, value["logits"])
        if hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
            return value.logits
        raise ShapeContractError("classification model output must be EncoderOutput, logits tensor, or .logits")


class BinaryClassificationTask(ClassificationTask):
    def __init__(self, head: nn.Module, loss: nn.Module | None = None, **kwargs: Any) -> None:
        super().__init__(
            head, loss or BinaryCrossEntropyWithLogitsLoss(), task_type=TaskType.BINARY_CLASSIFICATION, **kwargs
        )


class MultiLabelClassificationTask(ClassificationTask):
    def __init__(self, head: nn.Module, loss: nn.Module | None = None, **kwargs: Any) -> None:
        super().__init__(
            head, loss or BinaryCrossEntropyWithLogitsLoss(), task_type=TaskType.MULTILABEL_CLASSIFICATION, **kwargs
        )


class OrdinalClassificationTask(ClassificationTask):
    def __init__(self, head: nn.Module, loss: nn.Module | None = None, **kwargs: Any) -> None:
        super().__init__(head, loss or OrdinalCumulativeLinkLoss(), task_type=TaskType.ORDINAL_CLASSIFICATION, **kwargs)
