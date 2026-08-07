"""Structured-generation task scoring and invalid-output accounting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.language import GeneratedText, LanguageOutput
from medfm.core.task import LossOutput

from .base import TaskModuleBase, detached_count_tensor, valid_sample_count
from .structured import StructuredFindingsValidator, StructuredValidationReport


class StructuredGenerationTask(TaskModuleBase):
    """Validate generated findings before scoring and expose parse/schema errors."""

    def __init__(
        self,
        *,
        validator: StructuredFindingsValidator | None = None,
        supported_modalities: tuple[Modality, ...] | None = None,
        language_loss: nn.Module | None = None,
    ) -> None:
        super().__init__(
            TaskType.STRUCTURED_FINDING_GENERATION,
            supported_modalities or tuple(m for m in Modality if not m.is_text_only),
        )
        self.validator = validator or StructuredFindingsValidator()
        self.language_loss = language_loss

    def validate(self, generated: Sequence[str | dict[str, Any]]) -> StructuredValidationReport:
        return self.validator.validate_batch(generated)

    @staticmethod
    def _generated_values(model_output: Any) -> tuple[Sequence[str | dict[str, Any]], torch.Tensor | None]:
        if isinstance(model_output, GeneratedText):
            return model_output.texts, None
        if isinstance(model_output, dict):
            values = model_output.get("generated_texts", model_output.get("texts"))
            if not isinstance(values, Sequence) or isinstance(values, str | bytes):
                raise ShapeContractError("structured generation output needs generated_texts sequence")
            return values, model_output.get("language_loss")
        if isinstance(model_output, Sequence) and not isinstance(model_output, str | bytes):
            return model_output, None
        if isinstance(model_output, LanguageOutput):
            generated = model_output.auxiliary.get("generated_texts")
            if not isinstance(generated, Sequence) or isinstance(generated, str | bytes):
                raise ShapeContractError(
                    "LanguageOutput auxiliary['generated_texts'] is required for structured scoring"
                )
            return generated, model_output.loss
        raise ShapeContractError("unsupported structured generation model output")

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        values, supplied_loss = self._generated_values(model_output)
        report = self.validate(values)
        if supplied_loss is not None:
            total = supplied_loss
        elif self.language_loss is not None:
            target = batch.task_targets.get("language_labels")
            logits = model_output.get("logits") if isinstance(model_output, dict) else None
            if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise ShapeContractError("language_loss requires logits and language_labels")
            total = self.language_loss(logits, target)
        else:
            reference = batch.labels if isinstance(batch.labels, torch.Tensor) else torch.zeros((), dtype=torch.float32)
            total = reference.float().sum() * 0.0 if reference.ndim else reference.float() * 0.0
        count = valid_sample_count(batch, default=len(values))
        reference_tensor = total if isinstance(total, torch.Tensor) else torch.zeros((), dtype=torch.float32)
        if not isinstance(total, torch.Tensor) or total.ndim != 0:
            raise ShapeContractError("structured generation language loss must be a scalar tensor")
        return LossOutput(
            total=total,
            components={"language": total},
            sample_count=count,
            token_count=int(batch.task_targets.get("language_token_count", 0)),
            diagnostics={
                "task": self.task_type.value,
                "structured_schema_version": self.validator.version,
                "valid_output_count": report.valid,
                "invalid_output_count": report.invalid,
                "parse_error_count": report.parse_errors,
                "schema_error_count": report.schema_errors,
                "valid_count": detached_count_tensor(count, reference_tensor),
            },
        )
