"""Shared segmentation decoder output contract and feature coercion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError


@dataclass(frozen=True, eq=False)
class SegmentationOutput:
    """Decoder result with logits, optional deep supervision, and native output.

    ``logits`` is always the primary spatial tensor: ``[B, K, H, W]`` for 2D
    or ``[B, K, D, H, W]`` for 3D.  ``native_outputs`` is intentionally kept
    opaque so a native model's semantics are not flattened into generic maps.
    """

    logits: torch.Tensor
    deep_supervision: tuple[torch.Tensor, ...] = ()
    native_outputs: Any | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim not in (4, 5):
            raise ShapeContractError("segmentation logits must be rank 4 (2D) or rank 5 (3D)")
        for index, tensor in enumerate(self.deep_supervision):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != self.logits.ndim:
                raise ShapeContractError(f"deep_supervision[{index}] must have the same rank as logits")
            if int(tensor.shape[0]) != int(self.logits.shape[0]) or int(tensor.shape[1]) != int(self.logits.shape[1]):
                raise ShapeContractError("deep-supervision batch/channel dimensions must match primary logits")

    @property
    def shape(self) -> torch.Size:
        """Tensor-like convenience for callers that only need output shape."""

        return self.logits.shape

    @property
    def ndim(self) -> int:
        return self.logits.ndim

    def __getattr__(self, name: str) -> Any:
        # Keep decoder results ergonomic without pretending native outputs are
        # generic tensors.  ``shape``/``ndim`` are explicit above.
        if name not in {"logits", "deep_supervision", "native_outputs", "auxiliary"} and "logits" in self.__dict__:
            return getattr(self.__dict__["logits"], name)
        raise AttributeError(name)


DecoderOutput = SegmentationOutput


def as_feature_maps(features: EncoderOutput | torch.Tensor | Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Extract dense feature maps without silently reshaping spatial tokens."""

    if isinstance(features, EncoderOutput):
        if not features.feature_maps:
            raise ShapeContractError(
                "segmentation decoder requires EncoderOutput.feature_maps; "
                "spatial tokens cannot be reshaped without declared feature-map semantics"
            )
        return tuple(features.feature_maps)
    if isinstance(features, torch.Tensor):
        if features.ndim not in (4, 5):
            raise ShapeContractError("dense decoder input tensor must be rank 4 or 5")
        return (features,)
    maps = tuple(features)
    if not maps:
        raise ShapeContractError("segmentation decoder received no feature maps")
    for index, feature in enumerate(maps):
        if not isinstance(feature, torch.Tensor) or feature.ndim not in (4, 5):
            raise ShapeContractError(f"feature map {index} must be rank 4 or 5 tensor")
    return maps


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Small normalization/activation block shared by 2D and 3D decoders."""

    def __init__(self, dimension: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        conv: type[nn.Conv2d] | type[nn.Conv3d]
        if dimension == 2:
            conv = nn.Conv2d
        elif dimension == 3:
            conv = nn.Conv3d
        else:
            raise ShapeContractError("decoder dimension must be 2 or 3")
        self.block = nn.Sequential(
            conv(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            conv(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(value))


def _interpolate(value: torch.Tensor, size: Sequence[int], dimension: int) -> torch.Tensor:
    mode = "bilinear" if dimension == 2 else "trilinear"
    return torch.nn.functional.interpolate(value, size=tuple(int(v) for v in size), mode=mode, align_corners=False)
