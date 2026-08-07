"""Model-neutral placement of visual tokens and causal loss masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from medfm.core.errors import ShapeContractError
from medfm.core.language import ProjectedVisualTokens

IGNORE_INDEX = -100
PlacementMode = Literal["prefix", "suffix"]


@dataclass(frozen=True)
class TokenPlacementConfig:
    """Versioned placement policy shared by external language adapters."""

    mode: PlacementMode = "prefix"
    use_boundary_embeddings: bool = True
    config_name: str = "external-prefix-v1"

    def __post_init__(self) -> None:
        if self.mode not in ("prefix", "suffix"):
            raise ShapeContractError("placement mode must be 'prefix' or 'suffix'")
        if not self.config_name:
            raise ShapeContractError("placement config_name must be non-empty")


class VisualBoundaryEmbeddings(nn.Module):
    """Trainable begin/end markers independent of tokenizer placeholder IDs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ShapeContractError("hidden_size must be positive")
        self.embeddings = nn.Parameter(torch.empty(2, hidden_size))
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.embeddings.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)


def mask_causal_labels(
    labels: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    prompt_token_mask: torch.Tensor | None = None,
    visual_prefix_tokens: int = 0,
) -> torch.Tensor:
    """Mask padding, prompt, and visual positions with the HF ignore index.

    Phase 04 labels already identify assistant output positions.  This helper
    applies the additional masks introduced when a visual span is placed into
    the language sequence, without mutating the caller's tensor.
    """
    if labels.ndim != 2 or labels.dtype not in (torch.int32, torch.int64):
        raise ShapeContractError("labels must be integer [B,L]")
    result = labels.to(dtype=torch.long).clone()
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(labels.shape):
            raise ShapeContractError("attention_mask must align with labels")
        result = result.masked_fill(~attention_mask.bool(), IGNORE_INDEX)
    if prompt_token_mask is not None:
        if tuple(prompt_token_mask.shape) != tuple(labels.shape):
            raise ShapeContractError("prompt_token_mask must align with labels")
        result = result.masked_fill(prompt_token_mask.bool(), IGNORE_INDEX)
    if visual_prefix_tokens < 0 or visual_prefix_tokens > int(labels.shape[1]):
        raise ShapeContractError("visual_prefix_tokens is outside the label sequence")
    if visual_prefix_tokens:
        result[:, :visual_prefix_tokens] = IGNORE_INDEX
    return result


def _validate_visual(batch: int, visual: ProjectedVisualTokens) -> tuple[torch.Tensor, torch.Tensor]:
    if int(visual.tokens.shape[0]) != batch:
        raise ShapeContractError("text and visual token batches must match")
    visual_mask = visual.token_mask
    if visual_mask is None:
        visual_mask = torch.ones(visual.tokens.shape[:2], dtype=torch.bool, device=visual.tokens.device)
    visual_mask = visual_mask.to(device=visual.tokens.device, dtype=torch.bool)
    return visual.tokens, visual_mask


def place_visual_tokens(
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
    visual_tokens: ProjectedVisualTokens | None,
    *,
    labels: torch.Tensor | None = None,
    config: TokenPlacementConfig | None = None,
    boundary_embeddings: VisualBoundaryEmbeddings | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, int]]:
    """Build static multimodal embeddings, masks, and aligned labels.

    Visual tokens are prefixed or suffixed as a single framework-owned span.
    The caller supplies no image placeholder IDs; datasets remain independent
    from a specific tokenizer or VLM chat template.
    """
    policy = config or TokenPlacementConfig()
    if text_embeddings.ndim != 3:
        raise ShapeContractError("text_embeddings must be [B,L,D]")
    if tuple(text_attention_mask.shape) != tuple(text_embeddings.shape[:2]):
        raise ShapeContractError("text_attention_mask must align with text_embeddings")
    if labels is not None and tuple(labels.shape) != tuple(text_embeddings.shape[:2]):
        raise ShapeContractError("labels must align with text embeddings before placement")
    batch, text_len, hidden = text_embeddings.shape
    if visual_tokens is None:
        result_labels = None if labels is None else mask_causal_labels(labels, attention_mask=text_attention_mask)
        return (
            text_embeddings,
            text_attention_mask.bool(),
            result_labels,
            {"visual_start": text_len, "visual_tokens": 0},
        )

    visual, visual_mask = _validate_visual(batch, visual_tokens)
    if int(visual.shape[-1]) != int(hidden):
        raise ShapeContractError("visual and text embedding widths must match")
    pieces: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    label_pieces: list[torch.Tensor] = []
    boundary_count = 2 if policy.use_boundary_embeddings else 0
    begin: torch.Tensor | None = None
    end: torch.Tensor | None = None
    boundary_mask: torch.Tensor | None = None
    boundary_labels: torch.Tensor | None = None
    if policy.use_boundary_embeddings:
        if boundary_embeddings is None:
            raise ShapeContractError("boundary embeddings are required by this placement config")
        boundaries = boundary_embeddings(batch, device=text_embeddings.device, dtype=text_embeddings.dtype)
        begin = boundaries[:, 0:1, :]
        end = boundaries[:, 1:2, :]
        boundary_mask = torch.ones((batch, 1), dtype=torch.bool, device=text_embeddings.device)
        boundary_labels = torch.full((batch, 1), IGNORE_INDEX, dtype=torch.long, device=text_embeddings.device)

    visual_labels = torch.full(
        (batch, int(visual.shape[1])), IGNORE_INDEX, dtype=torch.long, device=text_embeddings.device
    )
    text_labels = None if labels is None else mask_causal_labels(labels, attention_mask=text_attention_mask)
    if policy.mode == "prefix":
        if begin is not None and end is not None and boundary_mask is not None and boundary_labels is not None:
            pieces.extend((begin, visual, end))
            masks.extend((boundary_mask, visual_mask, boundary_mask))
            label_pieces.extend((boundary_labels, visual_labels, boundary_labels))
            visual_start = 1
        else:
            pieces.append(visual)
            masks.append(visual_mask)
            label_pieces.append(visual_labels)
            visual_start = 0
        pieces.append(text_embeddings)
        masks.append(text_attention_mask.bool())
        if text_labels is not None:
            label_pieces.append(text_labels)
    else:
        pieces.append(text_embeddings)
        masks.append(text_attention_mask.bool())
        if text_labels is not None:
            label_pieces.append(text_labels)
        visual_start = text_len + (1 if begin is not None else 0)
        if begin is not None and end is not None and boundary_mask is not None and boundary_labels is not None:
            pieces.extend((begin, visual, end))
            masks.extend((boundary_mask, visual_mask, boundary_mask))
            label_pieces.extend((boundary_labels, visual_labels, boundary_labels))
        else:
            pieces.append(visual)
            masks.append(visual_mask)
            label_pieces.append(visual_labels)

    embedded = torch.cat(pieces, dim=1)
    attention = torch.cat(masks, dim=1)
    placed_labels = None if labels is None else torch.cat(label_pieces, dim=1)
    return (
        embedded,
        attention,
        placed_labels,
        {
            "visual_start": int(visual_start),
            "visual_tokens": int(visual.shape[1]),
            "boundary_tokens": boundary_count,
        },
    )


class VisualTokenPlacementAdapter(nn.Module):
    """Reusable placement module for external language adapters."""

    def __init__(
        self,
        hidden_size: int,
        *,
        config: TokenPlacementConfig | None = None,
        boundary_embeddings: VisualBoundaryEmbeddings | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TokenPlacementConfig()
        self.boundary_embeddings = boundary_embeddings or VisualBoundaryEmbeddings(hidden_size)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_attention_mask: torch.Tensor,
        visual_tokens: ProjectedVisualTokens | None,
        *,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, int]]:
        return place_visual_tokens(
            text_embeddings,
            text_attention_mask,
            visual_tokens,
            labels=labels,
            config=self.config,
            boundary_embeddings=self.boundary_embeddings,
        )


__all__ = [
    "IGNORE_INDEX",
    "PlacementMode",
    "TokenPlacementConfig",
    "VisualBoundaryEmbeddings",
    "VisualTokenPlacementAdapter",
    "mask_causal_labels",
    "place_visual_tokens",
]
