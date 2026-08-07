"""Static-query visual-token resampling for external encoders."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from medfm.core.language import ProjectedVisualTokens
from medfm.models.bridges.base import BridgeError, VisionLanguageBridge


class PerceiverResamplerBridge(VisionLanguageBridge):
    """Fixed-query Perceiver resampler with explicit key-padding masks.

    The query count is a constructor constant.  Every sample emits the same
    number of queries; samples with no valid visual inputs receive an all-false
    output mask.  No data-dependent pruning or Python control flow occurs in
    ``forward``.
    """

    bridge_type = "perceiver_resampler"

    def __init__(self, *, query_count: int | None = None, heads: int = 4, dropout: float = 0.0, **kwargs: Any) -> None:
        if query_count is not None:
            if "output_tokens" in kwargs and int(kwargs["output_tokens"]) != int(query_count):
                raise BridgeError("query_count and output_tokens must agree")
            kwargs["output_tokens"] = int(query_count)
        super().__init__(**kwargs)
        if heads <= 0 or self.target_dim % heads:
            raise BridgeError("heads must be positive and divide target_dim")
        if not 0.0 <= dropout < 1.0:
            raise BridgeError("dropout must be in [0, 1)")
        self.heads = int(heads)
        self.query = nn.Parameter(torch.randn(self.output_tokens, self.target_dim) * 0.02)
        self.input_projection = nn.Linear(self.source_dim, self.target_dim)
        self.attention = nn.MultiheadAttention(self.target_dim, self.heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(self.target_dim), nn.Linear(self.target_dim, self.target_dim))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
    ) -> ProjectedVisualTokens:
        mask = self._validate_inputs(visual_tokens, token_mask)
        projected_input = self.input_projection(visual_tokens)
        batch = int(visual_tokens.shape[0])
        queries = self.query.unsqueeze(0).expand(batch, -1, -1)
        # MultiheadAttention's key_padding_mask uses True for ignored keys.
        safe_mask = mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | ~mask.any(dim=1)
        attended, _ = self.attention(queries, projected_input, projected_input, key_padding_mask=~safe_mask)
        output = self.output(attended)
        has_visual = mask.any(dim=1, keepdim=True).expand(-1, self.output_tokens)
        output = output * has_visual.unsqueeze(-1).to(output.dtype)
        return self._projected(output, has_visual)


PerceiverBridge = PerceiverResamplerBridge

__all__ = ["PerceiverBridge", "PerceiverResamplerBridge"]
