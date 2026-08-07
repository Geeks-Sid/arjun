"""Text-query encoding and spatial language-conditioned segmentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.language import LanguageOutput
from medfm.core.task import LossOutput
from medfm.models.decoders import LanguageConditionedMaskDecoder, SegmentationOutput

from .base import TaskModuleBase, detached_count_tensor, target_from_batch, valid_sample_count
from .losses import DiceBCELoss, DiceCELoss
from .segmentation import SEGMENTATION_MODALITIES


class LanguageQueryEncoder(nn.Module):
    """Small adapter boundary that keeps text encoding separate from masks."""

    def __init__(self, encoder: nn.Module | Callable[..., Any]) -> None:
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else _CallableModule(encoder)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        try:
            result = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            result = self.encoder(input_ids, attention_mask)
        if isinstance(result, LanguageOutput):
            if result.hidden_states:
                return result.hidden_states[-1]
            if result.logits is not None:
                return result.logits
            raise ShapeContractError("language query encoder returned LanguageOutput without hidden states/logits")
        if isinstance(result, dict):
            for key in ("last_hidden_state", "hidden_states", "embeddings", "text_embeddings"):
                value = result.get(key)
                if isinstance(value, tuple | list) and value and isinstance(value[-1], torch.Tensor):
                    return value[-1]
                if isinstance(value, torch.Tensor):
                    return value
        if hasattr(result, "last_hidden_state") and isinstance(result.last_hidden_state, torch.Tensor):
            return result.last_hidden_state
        if not isinstance(result, torch.Tensor):
            raise ShapeContractError("language query encoder must return a tensor or LanguageOutput")
        return result


class _CallableModule(nn.Module):
    def __init__(self, function: Callable[..., Any]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


class LanguageConditionedSegmentationTask(TaskModuleBase):
    """Encode text queries separately, cross-attend, then decode spatial masks."""

    def __init__(
        self,
        decoder: LanguageConditionedMaskDecoder,
        text_encoder: nn.Module | Callable[..., Any] | None = None,
        loss: nn.Module | None = None,
        *,
        binary: bool = True,
        supported_modalities: tuple[Modality, ...] = SEGMENTATION_MODALITIES,
    ) -> None:
        super().__init__(TaskType.LANGUAGE_CONDITIONED_SEGMENTATION, supported_modalities)
        self.decoder = decoder
        self.text_encoder = LanguageQueryEncoder(text_encoder) if text_encoder is not None else None
        self.loss = loss or (DiceBCELoss() if binary else DiceCELoss())

    def encode_query(
        self, batch: MedicalBatch, model_output: Any | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(model_output, dict) and isinstance(model_output.get("text_embeddings"), torch.Tensor):
            return model_output["text_embeddings"], model_output.get("text_mask")
        if isinstance(model_output, tuple) and model_output and isinstance(model_output[0], torch.Tensor):
            return model_output[0], model_output[1] if len(model_output) > 1 else None
        if self.text_encoder is None:
            embeddings = batch.task_targets.get("text_embeddings")
            if not isinstance(embeddings, torch.Tensor):
                raise ShapeContractError(
                    "language-conditioned task requires text_encoder or task_targets['text_embeddings']"
                )
            return embeddings, batch.task_targets.get("text_mask")
        if batch.input_ids is None:
            raise ShapeContractError("language-conditioned task requires batch.input_ids")
        return self.text_encoder(batch.input_ids, batch.attention_mask), batch.attention_mask

    def forward(self, model_output: Any, *, visual_features: Any | None = None) -> SegmentationOutput:
        if isinstance(model_output, SegmentationOutput):
            return model_output
        if isinstance(model_output, dict):
            visual = model_output.get("visual_features", visual_features)
            text = model_output.get("text_embeddings")
            if visual is None or not isinstance(text, torch.Tensor):
                raise ShapeContractError("language segmentation mapping needs visual_features and text_embeddings")
            return self.decoder(
                visual,
                text,
                text_mask=model_output.get("text_mask"),
                query_mask=model_output.get("query_mask"),
                output_size=model_output.get("output_size"),
            )
        raise ShapeContractError("language segmentation forward expects a mapping or SegmentationOutput")

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        if not isinstance(model_output, dict) or "visual_features" not in model_output:
            raise ShapeContractError("language-conditioned compute_loss needs visual_features in model_output")
        visual = model_output["visual_features"]
        text, text_mask = self.encode_query(batch, model_output)
        output = self.decoder(
            visual,
            text,
            text_mask=model_output.get("text_mask", text_mask),
            query_mask=model_output.get("query_mask", batch.task_targets.get("query_mask")),
            output_size=model_output.get("output_size"),
        )
        target = target_from_batch(batch, "segmentation", allow_labels=False)
        voxel_mask = batch.task_targets.get("voxel_mask")
        if voxel_mask is not None and not isinstance(voxel_mask, torch.Tensor):
            raise ShapeContractError("voxel_mask must be a tensor")
        total = self.loss(output.logits, target, valid_mask=voxel_mask)  # type: ignore[call-arg]
        count = valid_sample_count(batch)
        return LossOutput(
            total=total,
            components={"language_segmentation": total},
            sample_count=count,
            diagnostics={
                "task": self.task_type.value,
                "text_conditioned": True,
                "valid_count": detached_count_tensor(count, output.logits),
            },
        )
