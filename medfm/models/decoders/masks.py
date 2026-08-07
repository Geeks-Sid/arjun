"""Promptable, transformer-mask, and native decoder interfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from medfm.core.errors import ShapeContractError

from .base import SegmentationOutput, as_feature_maps


def _flatten_visual(features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if features.ndim not in (4, 5):
        raise ShapeContractError("mask decoder visual features must be rank 4 or 5")
    spatial = tuple(int(v) for v in features.shape[2:])
    tokens = features.flatten(2).transpose(1, 2)
    return tokens, spatial


class TransformerMaskDecoder(nn.Module):
    """Dot-product mask decoder driven by visual tokens and mask queries."""

    def __init__(
        self,
        visual_dim: int,
        query_dim: int,
        *,
        mask_dim: int | None = None,
        num_masks: int = 1,
    ) -> None:
        super().__init__()
        if visual_dim <= 0 or query_dim <= 0 or num_masks <= 0:
            raise ShapeContractError("TransformerMaskDecoder dimensions must be positive")
        self.visual_dim = int(visual_dim)
        self.query_dim = int(query_dim)
        self.mask_dim = int(mask_dim or query_dim)
        self.num_masks = int(num_masks)
        self.visual_projection = nn.Linear(visual_dim, self.mask_dim)
        self.query_projection = nn.Linear(query_dim, self.mask_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_masks))

    def forward(
        self,
        visual_features: torch.Tensor | Sequence[torch.Tensor],
        mask_queries: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        output_size: Sequence[int] | None = None,
    ) -> SegmentationOutput:
        maps = as_feature_maps(visual_features)
        visual = maps[-1]
        visual_tokens, spatial = _flatten_visual(visual)
        if mask_queries.ndim == 2:
            mask_queries = mask_queries.unsqueeze(1)
        if mask_queries.ndim != 3 or mask_queries.shape[0] != visual_tokens.shape[0]:
            raise ShapeContractError("mask_queries must be [B, Q, D]")
        if int(mask_queries.shape[1]) != self.num_masks:
            raise ShapeContractError(f"expected {self.num_masks} mask queries, got {mask_queries.shape[1]}")
        if query_mask is None:
            query_mask = torch.ones(mask_queries.shape[:2], dtype=torch.bool, device=mask_queries.device)
        else:
            if tuple(query_mask.shape) != tuple(mask_queries.shape[:2]):
                raise ShapeContractError("query_mask must have shape [B, Q]")
            query_mask = query_mask.to(device=mask_queries.device, dtype=torch.bool)
        visual_projected = self.visual_projection(visual_tokens)
        queries = self.query_projection(mask_queries)
        scores = torch.einsum("bqd,bsd->bqs", queries, visual_projected) / self.mask_dim**0.5
        scores = scores + self.output_bias.reshape(1, -1, 1).to(dtype=scores.dtype, device=scores.device)
        scores = scores * query_mask.unsqueeze(-1).to(dtype=scores.dtype)
        size = tuple(int(v) for v in (output_size or spatial))
        logits = scores.reshape(scores.shape[0], scores.shape[1], *spatial)
        if size != spatial:
            mode = "bilinear" if len(size) == 2 else "trilinear"
            logits = torch.nn.functional.interpolate(logits, size=size, mode=mode, align_corners=False)
        return SegmentationOutput(logits=logits, auxiliary={"decoder": "transformer_mask", "query_mask": query_mask})


class PromptableMaskDecoder(TransformerMaskDecoder):
    """Transformer mask decoder with prompt embeddings as mask queries."""

    def forward(
        self,
        visual_features: torch.Tensor | Sequence[torch.Tensor],
        prompts: torch.Tensor,
        *,
        prompt_mask: torch.Tensor | None = None,
        output_size: Sequence[int] | None = None,
    ) -> SegmentationOutput:
        return super().forward(visual_features, prompts, query_mask=prompt_mask, output_size=output_size)


@runtime_checkable
class NativeMaskDecoder(Protocol):
    """Protocol for native model decoders whose output semantics stay opaque."""

    def __call__(self, features: Any, *args: Any, **kwargs: Any) -> Any: ...


class NativeModelDecoderWrapper(nn.Module):
    """Adapt a native decoder without flattening its native result."""

    def __init__(self, decoder: NativeMaskDecoder | Callable[..., Any]) -> None:
        super().__init__()
        if isinstance(decoder, nn.Module):
            self.decoder = decoder
        else:
            self.decoder = _CallableDecoder(decoder)

    def forward(self, features: Any, *args: Any, **kwargs: Any) -> SegmentationOutput:
        native = self.decoder(features, *args, **kwargs)
        if isinstance(native, SegmentationOutput):
            return native
        if isinstance(native, torch.Tensor):
            logits = native
        elif isinstance(native, dict) and isinstance(native.get("logits"), torch.Tensor):
            logits = native["logits"]
        elif hasattr(native, "logits") and isinstance(native.logits, torch.Tensor):
            logits = native.logits
        else:
            raise ShapeContractError(
                "native decoder must expose a spatial tensor as logits while its complete "
                "result is retained in native_outputs"
            )
        return SegmentationOutput(logits=logits, native_outputs=native, auxiliary={"decoder": "native"})


class _CallableDecoder(nn.Module):
    def __init__(self, function: Callable[..., Any]) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


__all__ = [
    "TransformerMaskDecoder",
    "PromptableMaskDecoder",
    "NativeMaskDecoder",
    "NativeModelDecoderWrapper",
]
