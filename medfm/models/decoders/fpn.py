"""Feature-pyramid segmentation decoders for 2D and 3D encoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import ShapeContractError

from .base import SegmentationOutput, _group_count, _interpolate, as_feature_maps


class _FPNDecoder(nn.Module):
    dimension: int

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *,
        pyramid_channels: int = 64,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in in_channels)
        if not channels or any(c <= 0 for c in channels) or out_channels <= 0 or pyramid_channels <= 0:
            raise ShapeContractError("FPN channels must be positive")
        self.in_channels = channels
        self.out_channels = int(out_channels)
        self.pyramid_channels = int(pyramid_channels)
        self.deep_supervision_enabled = bool(deep_supervision)
        conv = nn.Conv2d if self.dimension == 2 else nn.Conv3d
        self.lateral = nn.ModuleList([conv(c, pyramid_channels, kernel_size=1) for c in channels])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    conv(pyramid_channels, pyramid_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(pyramid_channels), pyramid_channels),
                    nn.GELU(),
                )
                for _ in channels
            ]
        )
        self.head = conv(pyramid_channels, out_channels, kernel_size=1)
        self.auxiliary_heads = nn.ModuleList([conv(pyramid_channels, out_channels, kernel_size=1) for _ in channels])

    def forward(
        self,
        features: EncoderOutput | torch.Tensor | Sequence[torch.Tensor],
        *,
        output_size: Sequence[int] | None = None,
    ) -> SegmentationOutput:
        maps = as_feature_maps(features)
        if len(maps) != len(self.lateral):
            raise ShapeContractError(f"FPN configured for {len(self.lateral)} maps but received {len(maps)}")
        if any(feature.ndim != self.dimension + 2 for feature in maps):
            raise ShapeContractError(f"{self.__class__.__name__} expects rank {self.dimension + 2} feature maps")
        lateral = [layer(feature) for layer, feature in zip(self.lateral, maps, strict=True)]
        pyramid: list[torch.Tensor] = [lateral[-1]]
        for value in reversed(lateral[:-1]):
            pyramid.append(value + _interpolate(pyramid[-1], value.shape[2:], self.dimension))
        pyramid.reverse()
        smoothed = [block(value) for block, value in zip(self.smooth, pyramid, strict=True)]
        target_size = tuple(int(v) for v in (output_size or smoothed[-1].shape[2:]))
        fused = smoothed[-1]
        for value in smoothed[:-1]:
            fused = fused + _interpolate(value, smoothed[-1].shape[2:], self.dimension)
        logits = self.head(
            _interpolate(fused, target_size, self.dimension) if tuple(fused.shape[2:]) != target_size else fused
        )
        deep = ()
        if self.deep_supervision_enabled:
            deep = tuple(
                self.head(
                    _interpolate(value, target_size, self.dimension) if tuple(value.shape[2:]) != target_size else value
                )
                for value in smoothed
            )
        return SegmentationOutput(logits=logits, deep_supervision=deep, auxiliary={"decoder": "fpn"})


class FPNDecoder2D(_FPNDecoder):
    dimension = 2

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 1,
        *,
        num_classes: int | None = None,
        pyramid_channels: int = 64,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            num_classes if num_classes is not None else out_channels,
            pyramid_channels=pyramid_channels,
            deep_supervision=deep_supervision,
        )


class FPNDecoder3D(_FPNDecoder):
    dimension = 3

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 1,
        *,
        num_classes: int | None = None,
        pyramid_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            num_classes if num_classes is not None else out_channels,
            pyramid_channels=pyramid_channels,
            deep_supervision=deep_supervision,
        )
