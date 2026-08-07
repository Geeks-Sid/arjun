"""Image/text projection and contrastive alignment modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError


@dataclass(frozen=True, eq=False)
class RetrievalOutput:
    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor
    logit_scale: torch.Tensor
    auxiliary: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class DistributedNegativeProvider(Protocol):
    """Future distributed-negative boundary.

    Implementations may return globally gathered embeddings and patient IDs;
    the local baseline deliberately uses no process-group or backend imports.
    """

    def gather(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        patient_ids: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Sequence[str] | None]: ...


class ImageTextProjectionHead(nn.Module):
    """Project normalized image/text representations into a shared space."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        projection_dim: int = 256,
        *,
        hidden_dim: int | None = None,
        logit_scale_init: float = 1 / 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if image_dim <= 0 or text_dim <= 0 or projection_dim <= 0:
            raise ShapeContractError("projection dimensions must be positive")
        if logit_scale_init <= 0 or max_logit_scale <= 0:
            raise ShapeContractError("logit scales must be positive")
        self.image_dim = int(image_dim)
        self.text_dim = int(text_dim)
        self.projection_dim = int(projection_dim)
        self.max_logit_scale = float(max_logit_scale)
        if hidden_dim is None:
            self.image_projection = nn.Linear(image_dim, projection_dim)
            self.text_projection = nn.Linear(text_dim, projection_dim)
        else:
            if hidden_dim <= 0:
                raise ShapeContractError("projection hidden_dim must be positive")
            self.image_projection = nn.Sequential(
                nn.Linear(image_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim)
            )
            self.text_projection = nn.Sequential(
                nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim)
            )
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)).log())

    def _image_input(self, image: EncoderOutput | torch.Tensor) -> torch.Tensor:
        if isinstance(image, EncoderOutput):
            if image.pooled_embedding is None:
                raise UnsupportedCapabilityError("retrieval image projection requires pooled_embedding")
            image = image.pooled_embedding
        if not isinstance(image, torch.Tensor) or image.ndim != 2 or image.shape[-1] != self.image_dim:
            raise ShapeContractError(f"image representation must be [B, {self.image_dim}]")
        return image

    def _text_input(self, text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        if text.ndim == 3:
            if text.shape[-1] != self.text_dim:
                raise ShapeContractError(f"text representation must end in {self.text_dim}")
            if text_mask is None:
                text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            if tuple(text_mask.shape) != tuple(text.shape[:2]):
                raise ShapeContractError("text_mask must be [B, L]")
            weights = text_mask.to(device=text.device, dtype=text.dtype).unsqueeze(-1)
            text = (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if not isinstance(text, torch.Tensor) or text.ndim != 2 or text.shape[-1] != self.text_dim:
            raise ShapeContractError(f"text representation must be [B, {self.text_dim}] or [B, L, {self.text_dim}]")
        return text

    def forward(
        self,
        image: EncoderOutput | torch.Tensor,
        text: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        patient_ids: Sequence[str] | None = None,
        negative_provider: DistributedNegativeProvider | None = None,
    ) -> RetrievalOutput:
        image_input = self._image_input(image)
        text_input = self._text_input(text, text_mask)
        if image_input.shape[0] != text_input.shape[0]:
            raise ShapeContractError("image and text batch sizes must match")
        image_embeddings = F.normalize(self.image_projection(image_input), dim=-1)
        text_embeddings = F.normalize(self.text_projection(text_input), dim=-1)
        ids = patient_ids
        if negative_provider is not None:
            image_embeddings, text_embeddings, ids = negative_provider.gather(image_embeddings, text_embeddings, ids)
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        logits = scale * image_embeddings @ text_embeddings.transpose(0, 1)
        return RetrievalOutput(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            logits_per_image=logits,
            logits_per_text=logits.transpose(0, 1),
            logit_scale=scale,
            auxiliary={"patient_ids": ids} if ids is not None else {},
        )


class ImageTextRetrievalHead(ImageTextProjectionHead):
    """Configuration-friendly alias for the dual projection head."""


def _patient_validity(
    count: int,
    *,
    patient_ids: Sequence[str] | None,
    device: torch.device,
    require_negative: bool = True,
) -> torch.Tensor:
    valid_rows: list[list[bool]] = [[True] * count for _ in range(count)]
    if patient_ids is not None:
        if len(patient_ids) != count:
            raise ShapeContractError("patient_ids length must equal the contrastive batch size")
        valid_rows = [
            [row == column or left != right for column, right in enumerate(patient_ids)]
            for row, left in enumerate(patient_ids)
        ]
    if require_negative:
        if count < 2:
            raise ShapeContractError("contrastive batch must contain at least one negative per sample")
        if any(
            sum(valid and column != row for column, valid in enumerate(row_values)) == 0
            for row, row_values in enumerate(valid_rows)
        ):
            raise ShapeContractError(
                "contrastive batch has no valid negative for at least one sample after patient filtering"
            )
    return torch.tensor(valid_rows, dtype=torch.bool, device=device)


def symmetric_contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor | None = None,
    *,
    patient_ids: Sequence[str] | None = None,
    require_negative: bool = True,
) -> torch.Tensor:
    """Symmetric CLIP-style loss with optional same-patient negative masking."""

    if logits_per_image.ndim != 2 or logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ShapeContractError("contrastive logits must be square [B, B]")
    if logits_per_text is None:
        logits_per_text = logits_per_image.transpose(0, 1)
    if logits_per_text.shape != logits_per_image.shape:
        raise ShapeContractError("image-to-text and text-to-image logits must have equal square shapes")
    count = int(logits_per_image.shape[0])
    valid = _patient_validity(
        count,
        patient_ids=patient_ids,
        device=logits_per_image.device,
        require_negative=require_negative,
    )
    # Positive diagonal is always retained; only non-positive same-patient
    # entries are filtered.  Cross entropy remains defined for every row.
    masked_i = logits_per_image.masked_fill(~valid, torch.finfo(logits_per_image.dtype).min)
    masked_t = logits_per_text.masked_fill(~valid.transpose(0, 1), torch.finfo(logits_per_text.dtype).min)
    labels = torch.arange(count, device=logits_per_image.device)
    image_loss = F.cross_entropy(masked_i, labels)
    text_loss = F.cross_entropy(masked_t, labels)
    return (image_loss + text_loss) * 0.5


class SymmetricContrastiveLoss(nn.Module):
    def __init__(self, *, require_negative: bool = True) -> None:
        super().__init__()
        self.require_negative = bool(require_negative)

    def forward(
        self,
        logits_per_image: torch.Tensor | RetrievalOutput,
        logits_per_text: torch.Tensor | None = None,
        *,
        patient_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if isinstance(logits_per_image, RetrievalOutput):
            output = logits_per_image
            logits_per_image = output.logits_per_image
            logits_per_text = output.logits_per_text
            if patient_ids is None:
                maybe_ids = output.auxiliary.get("patient_ids")
                patient_ids = (
                    maybe_ids if isinstance(maybe_ids, Sequence) and not isinstance(maybe_ids, str | bytes) else None
                )
        return symmetric_contrastive_loss(
            logits_per_image,
            logits_per_text,
            patient_ids=patient_ids,
            require_negative=self.require_negative,
        )


__all__ = [
    "RetrievalOutput",
    "DistributedNegativeProvider",
    "ImageTextProjectionHead",
    "ImageTextRetrievalHead",
    "symmetric_contrastive_loss",
    "SymmetricContrastiveLoss",
]
