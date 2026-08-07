"""Encoder-independent pooling operators for task heads.

All operators consume the shared :class:`~medfm.core.encoder.EncoderOutput`
contract.  Spatial pooling is deliberately strict: a pooled embedding is never
silently substituted for missing spatial tokens.
"""

from __future__ import annotations

from typing import Literal, TypedDict, cast

import torch
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError


def _tokens_and_mask(output: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, EncoderOutput):
        raise ShapeContractError(
            f"pooling expects EncoderOutput, got {type(output).__name__}; "
            "adapters must expose the shared output contract"
        )
    tokens = output.spatial_tokens
    if tokens is None:
        raise UnsupportedCapabilityError(
            "this pooling operator requires EncoderOutput.spatial_tokens; "
            "refusing to fabricate spatial features from a pooled embedding"
        )
    mask = output.token_mask
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    else:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
    return tokens, mask


def _safe_masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with a finite zero result for rows containing no valid tokens."""

    mask = mask.to(dtype=torch.bool, device=scores.device)
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return weights / denominator


class CLSPooling(nn.Module):
    """Use the encoder-declared pooled/CLS representation."""

    requires_pooled = True
    requires_spatial = False

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        if not isinstance(output, EncoderOutput):
            raise ShapeContractError(f"CLSPooling expects EncoderOutput, got {type(output).__name__}")
        if output.pooled_embedding is None:
            raise UnsupportedCapabilityError(
                "CLS pooling requires EncoderOutput.pooled_embedding; "
                "the adapter did not declare/provide a pooled representation"
            )
        return output.pooled_embedding


class MaskedMeanPooling(nn.Module):
    """Mean over real spatial tokens, preserving zero for an empty row."""

    requires_pooled = False
    requires_spatial = True

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        tokens, mask = _tokens_and_mask(output)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(torch.finfo(tokens.dtype).eps)
        return (tokens * weights).sum(dim=1) / denominator


class AttentionPooling(nn.Module):
    """Learned attention pooling over valid spatial tokens."""

    requires_pooled = False
    requires_spatial = True

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ShapeContractError("AttentionPooling input_dim must be positive")
        hidden = hidden_dim or max(1, input_dim // 2)
        self.score = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        tokens, mask = _tokens_and_mask(output)
        scores = self.score(tokens).squeeze(-1)
        weights = _safe_masked_softmax(scores, mask)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


class GeneralizedMeanPooling(nn.Module):
    """Generalized-mean (GeM) pooling with a positive learnable exponent."""

    requires_pooled = False
    requires_spatial = True

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        if p <= 0 or eps <= 0:
            raise ShapeContractError("GeM requires p and eps > 0")
        value = torch.tensor(float(p)).log().exp()
        self.p = nn.Parameter(value) if learnable else value
        self.eps = float(eps)

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        tokens, mask = _tokens_and_mask(output)
        p = self.p.to(dtype=tokens.dtype, device=tokens.device).clamp_min(self.eps)
        positive = tokens.clamp_min(self.eps).pow(p)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (positive * weights).sum(dim=1) / denom
        return pooled.clamp_min(self.eps).pow(1.0 / p)


class TopKPooling(nn.Module):
    """Average the top-k valid tokens according to feature magnitude."""

    requires_pooled = False
    requires_spatial = True

    def __init__(self, k: int = 1, score: Literal["norm", "mean"] = "norm") -> None:
        super().__init__()
        if k <= 0:
            raise ShapeContractError("TopKPooling k must be positive")
        if score not in ("norm", "mean"):
            raise ShapeContractError("TopKPooling score must be 'norm' or 'mean'")
        self.k = int(k)
        self.score_kind = score

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        tokens, mask = _tokens_and_mask(output)
        scores = tokens.norm(dim=-1) if self.score_kind == "norm" else tokens.mean(dim=-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        k = min(self.k, int(tokens.shape[1]))
        values, indices = torch.topk(scores, k=k, dim=1)
        del values
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        selected_mask = torch.gather(mask, 1, indices)
        weights = selected_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (selected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MILAttentionPooling(nn.Module):
    """Gated attention MIL pooling for tile/patch bags."""

    requires_pooled = False
    requires_spatial = True

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ShapeContractError("MILAttentionPooling input_dim must be positive")
        hidden = hidden_dim or max(1, input_dim // 2)
        self.value = nn.Linear(input_dim, hidden)
        self.gate = nn.Linear(input_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        tokens, mask = _tokens_and_mask(output)
        scores = self.score(torch.tanh(self.value(tokens)) * torch.sigmoid(self.gate(tokens))).squeeze(-1)
        weights = _safe_masked_softmax(scores, mask)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


# Short names are useful in configuration files and preserve the terminology
# used by the phase plan.
CLSPool = CLSPooling
MaskedMeanPool = MaskedMeanPooling
AttentionPool = AttentionPooling
GeMPooling = GeneralizedMeanPooling
TopKPool = TopKPooling
MILPool = MILAttentionPooling


class _AttentionPoolingKwargs(TypedDict, total=False):
    hidden_dim: int | None


class _GeneralizedMeanPoolingKwargs(TypedDict, total=False):
    p: float
    eps: float
    learnable: bool


class _TopKPoolingKwargs(TypedDict, total=False):
    k: int
    score: Literal["norm", "mean"]


class _MILAttentionPoolingKwargs(TypedDict, total=False):
    hidden_dim: int | None


def build_pooling(name: str, *, input_dim: int | None = None, **kwargs: object) -> nn.Module:
    """Build a pooling operator from a stable config name."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"cls", "pooled", "classification_token"}:
        return CLSPooling()
    if normalized in {"mean", "masked_mean", "mean_valid", "valid_mean"}:
        return MaskedMeanPooling()
    if normalized in {"attention", "attn"}:
        if input_dim is None:
            raise ShapeContractError("attention pooling requires input_dim")
        return AttentionPooling(input_dim=input_dim, **cast(_AttentionPoolingKwargs, kwargs))
    if normalized in {"gem", "generalized_mean", "generalised_mean"}:
        return GeneralizedMeanPooling(**cast(_GeneralizedMeanPoolingKwargs, kwargs))
    if normalized in {"topk", "top_k"}:
        return TopKPooling(**cast(_TopKPoolingKwargs, kwargs))
    if normalized in {"mil", "mil_attention", "gated_mil"}:
        if input_dim is None:
            raise ShapeContractError("MIL pooling requires input_dim")
        return MILAttentionPooling(input_dim=input_dim, **cast(_MILAttentionPoolingKwargs, kwargs))
    raise ShapeContractError(f"unknown pooling operator {name!r}")
