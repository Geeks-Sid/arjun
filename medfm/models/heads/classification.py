"""Encoder-independent classification heads.

Each head accepts an :class:`EncoderOutput`.  Heads that pool spatial tokens
require that field explicitly and fail with a typed capability error when an
adapter only returns a pooled representation.
"""

from __future__ import annotations

import torch
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError

from .pooling import (
    AttentionPooling,
    GeneralizedMeanPooling,
    MaskedMeanPooling,
    MILAttentionPooling,
    TopKPooling,
)


def _dimension_alias(
    input_dim: int | None,
    in_features: int | None,
    num_classes: int | None,
    out_features: int | None,
) -> tuple[int, int]:
    dim = input_dim if input_dim is not None else in_features
    classes = num_classes if num_classes is not None else out_features
    if dim is None or classes is None or dim <= 0 or classes <= 0:
        raise ShapeContractError(
            "classification heads require positive input_dim/in_features and num_classes/out_features"
        )
    return int(dim), int(classes)


def _as_pooled(value: EncoderOutput | torch.Tensor) -> torch.Tensor:
    if isinstance(value, EncoderOutput):
        if value.pooled_embedding is None:
            raise UnsupportedCapabilityError(
                "classification head requires EncoderOutput.pooled_embedding; "
                "use a spatial pooling head when only spatial_tokens are available"
            )
        return value.pooled_embedding
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ShapeContractError(f"pooled classification input must be [B, D], got {type(value).__name__}")
    return value


class _LinearClassifier(nn.Module):
    output_kind = "classification_logits"

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def _logits(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError(f"classification head expects [B, {self.input_dim}], got {tuple(pooled.shape)}")
        return self.classifier(pooled)

    def forward(self, value: EncoderOutput | torch.Tensor) -> torch.Tensor:
        return self._logits(_as_pooled(value))


class LinearClassificationHead(_LinearClassifier):
    """Frozen-encoder linear probe baseline."""

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        *,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        dim, classes = _dimension_alias(input_dim, in_features, num_classes, out_features)
        super().__init__(dim, classes)


class MLPClassificationHead(_LinearClassifier):
    """Two-layer MLP classification head over an encoder pooled embedding."""

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        dim, classes = _dimension_alias(input_dim, in_features, num_classes, out_features)
        if hidden_dim is None:
            hidden_dim = max(dim, classes)
        if hidden_dim <= 0 or not 0.0 <= dropout < 1.0:
            raise ShapeContractError("MLP hidden_dim must be positive and dropout must be in [0, 1)")
        if activation == "gelu":
            act: nn.Module = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise ShapeContractError(f"unsupported MLP activation {activation!r}")
        nn.Module.__init__(self)
        self.input_dim = dim
        self.num_classes = classes
        self.hidden_dim = int(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, classes),
        )

    def _logits(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError(f"MLP head expects [B, {self.input_dim}], got {tuple(pooled.shape)}")
        return self.classifier(pooled)


class AttentionPoolingClassificationHead(nn.Module):
    """Attention-pool spatial tokens and classify the resulting representation."""

    output_kind = "classification_logits"

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        *,
        hidden_dim: int | None = None,
        pooling: nn.Module | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        dim, classes = _dimension_alias(input_dim, in_features, num_classes, out_features)
        super().__init__()
        self.input_dim = dim
        self.num_classes = classes
        self.pool = pooling or AttentionPooling(dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        if not isinstance(output, EncoderOutput):
            raise ShapeContractError("AttentionPoolingClassificationHead requires EncoderOutput")
        pooled = self.pool(output)
        if int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError("attention pooling output dimension does not match classifier")
        return self.classifier(pooled)


class MultiLabelClassificationHead(LinearClassificationHead):
    """Independent-logit head for multi-label targets."""

    output_kind = "multilabel_logits"


class OrdinalClassificationHead(nn.Module):
    """Cumulative-link ordinal head with ``num_classes - 1`` logits.

    A shared evidence score is compared against ordered learnable thresholds,
    which makes the output interpretable as cumulative ``P(y > k)`` logits.
    """

    output_kind = "ordinal_logits"

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        *,
        in_features: int | None = None,
    ) -> None:
        dim, classes = _dimension_alias(input_dim, in_features, num_classes, None)
        if classes < 2:
            raise ShapeContractError("ordinal classification requires at least two classes")
        super().__init__()
        self.input_dim = dim
        self.num_classes = classes
        self.evidence = nn.Linear(dim, 1)
        self.threshold_deltas = nn.Parameter(torch.zeros(classes - 1))

    def thresholds(self) -> torch.Tensor:
        # Positive deltas ensure ordered thresholds while keeping the operation
        # differentiable on all accelerators.
        return torch.cumsum(torch.nn.functional.softplus(self.threshold_deltas), dim=0)

    def forward(self, value: EncoderOutput | torch.Tensor) -> torch.Tensor:
        pooled = _as_pooled(value)
        if int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError(f"ordinal head expects [B, {self.input_dim}], got {tuple(pooled.shape)}")
        evidence = self.evidence(pooled)
        return evidence - self.thresholds().to(device=evidence.device, dtype=evidence.dtype)


class MILClassificationHead(nn.Module):
    """Gated-attention multiple-instance classification head."""

    output_kind = "classification_logits"

    def __init__(
        self,
        input_dim: int | None = None,
        num_classes: int | None = None,
        *,
        hidden_dim: int | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        dim, classes = _dimension_alias(input_dim, in_features, num_classes, out_features)
        super().__init__()
        self.input_dim = dim
        self.num_classes = classes
        self.pool = MILAttentionPooling(dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        if not isinstance(output, EncoderOutput):
            raise ShapeContractError("MILClassificationHead requires EncoderOutput with spatial_tokens")
        pooled = self.pool(output)
        return self.classifier(pooled)


class PooledClassificationHead(LinearClassificationHead):
    """Compatibility name for the mandatory pooled linear baseline."""


class MeanPoolingClassificationHead(nn.Module):
    """Masked-mean spatial pooling followed by a linear classifier."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.pool = MaskedMeanPooling()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        pooled = self.pool(output)
        if int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError("mean pooling output dimension does not match classifier")
        return self.classifier(pooled)


class GeMClassificationHead(nn.Module):
    """Generalized-mean spatial pooling followed by a linear classifier."""

    def __init__(self, input_dim: int, num_classes: int, *, p: float = 3.0) -> None:
        super().__init__()
        self.pool = GeneralizedMeanPooling(p=p)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        pooled = self.pool(output)
        if int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError("GeM pooling output dimension does not match classifier")
        return self.classifier(pooled)


class TopKClassificationHead(nn.Module):
    """Top-k spatial pooling followed by a linear classifier."""

    def __init__(self, input_dim: int, num_classes: int, *, k: int = 1) -> None:
        super().__init__()
        self.pool = TopKPooling(k=k)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, output: EncoderOutput) -> torch.Tensor:
        pooled = self.pool(output)
        if int(pooled.shape[-1]) != self.input_dim:
            raise ShapeContractError("top-k pooling output dimension does not match classifier")
        return self.classifier(pooled)


__all__ = [
    "LinearClassificationHead",
    "MLPClassificationHead",
    "AttentionPoolingClassificationHead",
    "MultiLabelClassificationHead",
    "OrdinalClassificationHead",
    "MILClassificationHead",
    "PooledClassificationHead",
    "MeanPoolingClassificationHead",
    "GeMClassificationHead",
    "TopKClassificationHead",
]
