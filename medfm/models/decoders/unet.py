"""Minimal static-shape-friendly 2D and 3D UNet-style decoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError

from .base import ConvBlock, SegmentationOutput, _group_count, _interpolate, as_feature_maps


class _UNetDecoder(nn.Module):
    dimension: int

    def __init__(
        self,
        in_channels: int | Sequence[int] | None,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if out_channels <= 0 or hidden_channels <= 0:
            raise ShapeContractError("decoder channel counts must be positive")
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.deep_supervision_enabled = bool(deep_supervision)
        if in_channels is None:
            channel_list: list[int | None] = [None]
        elif isinstance(in_channels, int):
            if in_channels <= 0:
                raise ShapeContractError("in_channels must be positive")
            channel_list = [int(in_channels)]
        else:
            channel_list = [int(c) for c in in_channels]
            if not channel_list or any(c is None or c <= 0 for c in channel_list):
                raise ShapeContractError("in_channels sequence must contain positive entries")
        self.declared_channels = tuple(channel_list)
        self.projections = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.auxiliary_heads = nn.ModuleList()
        for channels in channel_list:
            if channels is None:
                self.projections.append(self._lazy_projection())
            else:
                self.projections.append(self._projection(channels))
            self.blocks.append(ConvBlock(self.dimension, self.hidden_channels, self.hidden_channels))
            self.auxiliary_heads.append(self._head())
        self.head = self._head()

    def _projection(self, channels: int) -> nn.Module:
        conv = nn.Conv2d if self.dimension == 2 else nn.Conv3d
        return nn.Sequential(
            conv(channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )

    def _lazy_projection(self) -> nn.Module:
        conv = nn.LazyConv2d if self.dimension == 2 else nn.LazyConv3d
        return nn.Sequential(
            conv(self.hidden_channels, kernel_size=1),
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.GELU(),
        )

    def _head(self) -> nn.Module:
        conv = nn.Conv2d if self.dimension == 2 else nn.Conv3d
        return conv(self.hidden_channels, self.out_channels, kernel_size=1)

    def forward(
        self,
        features: EncoderOutput | torch.Tensor | Sequence[torch.Tensor],
        *,
        output_size: Sequence[int] | None = None,
    ) -> SegmentationOutput:
        maps = as_feature_maps(features)  # finest resolution is last by contract
        if any(feature.ndim != self.dimension + 2 for feature in maps):
            raise ShapeContractError(f"{self.__class__.__name__} expects rank {self.dimension + 2} feature maps")
        if len(self.projections) == 1 and len(maps) != 1:
            # A single declared channel count is a shorthand for a repeated
            # same-width pyramid. Reuse registered modules so parameters are
            # visible to optimizers; a changing-width pyramid must declare all
            # channels explicitly.
            declared = self.declared_channels[0]
            if declared is not None and any(int(feature.shape[1]) != declared for feature in maps):
                raise ShapeContractError(
                    "a single in_channels value can only decode same-width maps; "
                    "pass a channel sequence for a changing-width pyramid"
                )
            if declared is None and any(int(feature.shape[1]) != int(maps[0].shape[1]) for feature in maps):
                raise ShapeContractError(
                    "in_channels=None with multiple maps requires equal channel widths; "
                    "pass a channel sequence for a changing-width pyramid"
                )
            projection = self.projections[0]
            block = self.blocks[0]
            auxiliary_head = self.auxiliary_heads[0]
            projected = [block(projection(feature)) for feature in maps]
            auxiliary_heads = [auxiliary_head] * len(maps)
        else:
            if len(maps) != len(self.projections):
                raise ShapeContractError(
                    f"decoder was configured for {len(self.projections)} feature maps but received {len(maps)}"
                )
            projected = [
                block(projection(feature))
                for projection, block, feature in zip(self.projections, self.blocks, maps, strict=True)
            ]
            auxiliary_heads = list(self.auxiliary_heads)
        finest_size = tuple(int(v) for v in maps[-1].shape[2:])
        if output_size is not None:
            finest_size = tuple(int(v) for v in output_size)
        fused = projected[-1]
        intermediate: list[torch.Tensor] = []
        for index, value in enumerate(projected[:-1]):
            upsampled = _interpolate(value, finest_size, self.dimension)
            fused = fused + upsampled
            if self.deep_supervision_enabled:
                intermediate.append(_interpolate(auxiliary_heads[index](value), finest_size, self.dimension))
        fused = self.blocks[-1](fused) if len(projected) != 1 and len(self.blocks) == len(projected) else fused
        logits = self.head(fused)
        if self.deep_supervision_enabled:
            deep = tuple(intermediate + [logits])
        else:
            deep = ()
        return SegmentationOutput(logits=logits, deep_supervision=deep, auxiliary={"decoder": "unet"})


class UNetDecoder2D(_UNetDecoder):
    """UNet-style decoder for ``[B, C, H, W]`` feature pyramids."""

    dimension = 2

    def __init__(
        self,
        in_channels: int | Sequence[int] | None = None,
        out_channels: int = 1,
        *,
        num_classes: int | None = None,
        hidden_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            num_classes if num_classes is not None else out_channels,
            hidden_channels=hidden_channels,
            deep_supervision=deep_supervision,
        )


class UNetDecoder3D(_UNetDecoder):
    """UNet-style decoder for ``[B, C, D, H, W]`` feature pyramids."""

    dimension = 3

    def __init__(
        self,
        in_channels: int | Sequence[int] | None = None,
        out_channels: int = 1,
        *,
        num_classes: int | None = None,
        hidden_channels: int = 16,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            num_classes if num_classes is not None else out_channels,
            hidden_channels=hidden_channels,
            deep_supervision=deep_supervision,
        )
