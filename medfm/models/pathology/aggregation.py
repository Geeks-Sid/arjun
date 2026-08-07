"""Bounded slide-level aggregators for cached pathology tile embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SlideAggregation:
    """Static-shape slide representation and evidence metadata."""

    embedding: torch.Tensor
    attention: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    evidence_indices: tuple[tuple[int, ...], ...] = ()
    evidence_tiles: tuple[tuple[Any, ...], ...] = ()

    @property
    def slide_embedding(self) -> torch.Tensor:
        return self.embedding


class SlideAggregator(nn.Module):
    """Backend-neutral slide aggregation interface.

    Inputs are ``embeddings [B,T,D]`` and a real-tile mask ``[B,T]``. Padding
    is always masked before any reduction, making the operation safe for TPU
    static-shape buckets and for batches with different slide sizes.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = int(embedding_dim)

    def _validate(self, embeddings: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if embeddings.ndim != 3 or int(embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"embeddings must have shape [B,T,{self.embedding_dim}]; got {tuple(embeddings.shape)}")
        if mask is None:
            return torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        if mask.shape != embeddings.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(embeddings.shape[:2])}; got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask != 0
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every slide must contain at least one valid tile")
        return mask

    def aggregate(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        evidence_indices: tuple[tuple[int, ...], ...] = (),
        evidence_tiles: tuple[tuple[Any, ...], ...] = (),
    ) -> SlideAggregation:
        raise NotImplementedError

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.aggregate(embeddings, mask).embedding


class MeanPoolingAggregator(SlideAggregator):
    """Masked mean pooling baseline."""

    def aggregate(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        evidence_indices: tuple[tuple[int, ...], ...] = (),
        evidence_tiles: tuple[tuple[Any, ...], ...] = (),
    ) -> SlideAggregation:
        valid = self._validate(embeddings, mask)
        weights = valid.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return SlideAggregation(
            pooled, valid_mask=valid, evidence_indices=evidence_indices, evidence_tiles=evidence_tiles
        )


class AttentionMILAggregator(SlideAggregator):
    """Gated attention MIL with a mask-safe softmax."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__(embedding_dim)
        hidden = int(hidden_dim or max(8, min(256, embedding_dim // 2)))
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def aggregate(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        evidence_indices: tuple[tuple[int, ...], ...] = (),
        evidence_tiles: tuple[tuple[Any, ...], ...] = (),
    ) -> SlideAggregation:
        valid = self._validate(embeddings, mask)
        logits = self.attention_net(embeddings).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
        pooled = torch.sum(embeddings * attention.unsqueeze(-1), dim=1)
        return SlideAggregation(
            pooled,
            attention=attention,
            valid_mask=valid,
            evidence_indices=evidence_indices,
            evidence_tiles=evidence_tiles,
        )


class GigaPathFlashAggregator(MeanPoolingAggregator):
    """GigaPath slide-level boundary; native Flash weights are optional."""


class TITANAggregator(AttentionMILAggregator):
    """TITAN-compatible slide representation boundary with attention evidence."""


# Names used by earlier planning notes and downstream recipes.
MeanPoolAggregator = MeanPoolingAggregator
AttentionMIL = AttentionMILAggregator
GenericMeanPoolingAggregator = MeanPoolingAggregator

__all__ = [
    "AttentionMIL",
    "AttentionMILAggregator",
    "GenericMeanPoolingAggregator",
    "GigaPathFlashAggregator",
    "MeanPoolAggregator",
    "MeanPoolingAggregator",
    "SlideAggregation",
    "SlideAggregator",
    "TITANAggregator",
]
