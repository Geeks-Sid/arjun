"""Coordinate features for 2D, native 3D, and WSI visual tokens.

Dataset records carry modality-neutral tensors plus metadata.  These encoders
turn that representation into learned additive features without introducing
model-specific placeholder syntax or data-dependent branches in compiled
forward.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError
from medfm.core.language import ProjectedVisualTokens
from medfm.models.bridges.base import BridgeError, VisionLanguageBridge


def _batch_feature(value: Any, *, batch: int, tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast a scalar/[B]/[B,N] feature to ``[B,N,1]``."""
    if value is None:
        return torch.zeros((batch, tokens, 1), device=device, dtype=dtype)
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch, tokens)
    elif result.ndim == 1:
        if result.shape[0] == batch:
            result = result[:, None].expand(batch, tokens)
        elif result.shape[0] == tokens and batch == 1:
            result = result[None, :]
        else:
            raise ShapeContractError("coordinate metadata [B] or [N] does not match token shape")
    elif result.ndim == 2 and tuple(result.shape) == (batch, tokens):
        pass
    else:
        raise ShapeContractError("coordinate metadata must be scalar, [B], or [B,N]")
    return result.unsqueeze(-1)


def _positions(value: torch.Tensor, dimensions: int) -> tuple[torch.Tensor, int, int]:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[-1]) != dimensions:
        raise ShapeContractError(f"positions must be [B,N,{dimensions}] or [N,{dimensions}]")
    return value, int(value.shape[0]), int(value.shape[1])


class CoordinateEncoder(nn.Module):
    """Base encoder with a stable output width and explicit feature layout."""

    modality: Modality
    coordinate_system: CoordinateSystem

    def __init__(
        self, *, output_dim: int = 16, input_dim: int, modality: Modality, coordinate_system: CoordinateSystem
    ) -> None:
        super().__init__()
        if output_dim <= 0 or input_dim <= 0:
            raise ShapeContractError("coordinate encoder dimensions must be positive")
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        self.modality = modality
        self.coordinate_system = coordinate_system
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def _features(self, positions: torch.Tensor, metadata: Mapping[str, Any] | None) -> torch.Tensor:
        pos, batch, tokens = _positions(positions, self.position_dimensions)
        projection = cast(nn.Linear, self.projection[0])
        pos = pos.to(dtype=projection.weight.dtype, device=projection.weight.device)
        features = [pos]
        metadata = metadata or {}
        for name in self.feature_names:
            features.append(
                _batch_feature(
                    metadata.get(name),
                    batch=batch,
                    tokens=tokens,
                    device=pos.device,
                    dtype=pos.dtype,
                )
            )
        return torch.cat(features, dim=-1)

    @property
    def position_dimensions(self) -> int:
        return self.input_dim

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ()

    def forward(self, positions: torch.Tensor, metadata: Mapping[str, Any] | None = None) -> torch.Tensor:
        features = self._features(positions, metadata)
        if int(features.shape[-1]) != self.input_dim:
            raise ShapeContractError(
                f"coordinate feature width {int(features.shape[-1])} != declared input_dim {self.input_dim}"
            )
        return cast(torch.Tensor, self.projection(features))

    encode = forward


class TwoDCoordinateEncoder(CoordinateEncoder):
    """Normalized image position plus image/view/timepoint identity features."""

    def __init__(self, *, output_dim: int = 16) -> None:
        super().__init__(
            output_dim=output_dim,
            input_dim=6,
            modality=Modality.MULTI_IMAGE_2D,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        )

    @property
    def position_dimensions(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("image_index", "view", "timepoint", "slice_index")


class ThreeDCoordinateEncoder(CoordinateEncoder):
    """Normalized/physical 3D positions, spacing, and series identity."""

    def __init__(self, *, output_dim: int = 16) -> None:
        super().__init__(
            output_dim=output_dim,
            input_dim=10,
            modality=Modality.CT_3D,
            coordinate_system=CoordinateSystem.MILLIMETERS,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("physical_x", "physical_y", "physical_z", "spacing", "series_index")

    def _features(self, positions: torch.Tensor, metadata: Mapping[str, Any] | None) -> torch.Tensor:
        # positions are normalized xyz; physical_position and spacing are
        # supplied as vector-valued metadata and flattened into the contract.
        pos, batch, tokens = _positions(positions, 3)
        metadata = metadata or {}
        projection = cast(nn.Linear, self.projection[0])
        dtype = projection.weight.dtype
        device = projection.weight.device
        pos = pos.to(device=device, dtype=dtype)
        physical = metadata.get("physical_position", metadata.get("physical_xyz"))
        spacing = metadata.get("spacing", metadata.get("voxel_spacing"))
        physical_tensor = (
            torch.zeros((batch, tokens, 3), device=device, dtype=dtype)
            if physical is None
            else torch.as_tensor(physical, device=device, dtype=dtype)
        )
        spacing_tensor = (
            torch.zeros((batch, tokens, 3), device=device, dtype=dtype)
            if spacing is None
            else torch.as_tensor(spacing, device=device, dtype=dtype)
        )
        if physical_tensor.ndim == 1:
            physical_tensor = physical_tensor.view(1, 1, 3).expand(batch, tokens, 3)
        elif physical_tensor.ndim == 2:
            physical_tensor = physical_tensor.unsqueeze(0).expand(batch, -1, -1)
        if spacing_tensor.ndim == 1:
            spacing_tensor = spacing_tensor.view(1, 1, 3).expand(batch, tokens, 3)
        elif spacing_tensor.ndim == 2:
            spacing_tensor = spacing_tensor.unsqueeze(0).expand(batch, -1, -1)
        if tuple(physical_tensor.shape) != (batch, tokens, 3) or tuple(spacing_tensor.shape) != (batch, tokens, 3):
            raise ShapeContractError("physical_position and spacing must broadcast to [B,N,3]")
        series = _batch_feature(metadata.get("series_index"), batch=batch, tokens=tokens, device=device, dtype=dtype)
        return torch.cat((pos, physical_tensor, spacing_tensor, series), dim=-1)


class WSICoordinateEncoder(CoordinateEncoder):
    """Slide position in microns with MPP, pyramid level, and slide identity."""

    def __init__(self, *, output_dim: int = 16) -> None:
        super().__init__(
            output_dim=output_dim,
            input_dim=7,
            modality=Modality.PATHOLOGY_WSI,
            coordinate_system=CoordinateSystem.MICRONS,
        )

    @property
    def position_dimensions(self) -> int:
        return 2

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("slide_x", "slide_y", "mpp", "pyramid_level", "slide_index")


# Descriptive aliases used by model/data integration code.
ImageCoordinateEncoder = TwoDCoordinateEncoder
VolumeCoordinateEncoder = ThreeDCoordinateEncoder
SlideCoordinateEncoder = WSICoordinateEncoder
CoordinateEncoder2D = TwoDCoordinateEncoder
CoordinateEncoder3D = ThreeDCoordinateEncoder
CoordinateEncoderWSI = WSICoordinateEncoder


class CoordinateAwareBridge(nn.Module):
    """Add learned coordinate features before a normal visual-language bridge."""

    def __init__(self, bridge: VisionLanguageBridge, encoder: CoordinateEncoder) -> None:
        super().__init__()
        if bridge.source_dim <= 0:
            raise BridgeError("bridge source dimension must be positive")
        self.bridge = bridge
        self.encoder = encoder
        self.coordinate_projection = nn.Linear(encoder.output_dim, bridge.source_dim)

    @property
    def output_tokens(self) -> int:
        return self.bridge.output_tokens

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor,
        coordinate_metadata: Mapping[str, Any] | None = None,
    ) -> ProjectedVisualTokens:
        if tuple(coordinates.shape[:2]) != tuple(visual_tokens.shape[:2]) and not (
            coordinates.ndim == 2 and int(visual_tokens.shape[0]) == 1
        ):
            raise BridgeError("coordinates must align with visual token batch and count")
        encoded = self.coordinate_projection(self.encoder(coordinates, coordinate_metadata))
        if encoded.shape[0] != visual_tokens.shape[0]:
            encoded = encoded.expand(visual_tokens.shape[0], -1, -1)
        return cast(
            ProjectedVisualTokens,
            self.bridge(visual_tokens + encoded.to(device=visual_tokens.device, dtype=visual_tokens.dtype), token_mask),
        )


__all__ = [
    "CoordinateAwareBridge",
    "CoordinateEncoder",
    "CoordinateEncoder2D",
    "CoordinateEncoder3D",
    "CoordinateEncoderWSI",
    "ImageCoordinateEncoder",
    "SlideCoordinateEncoder",
    "ThreeDCoordinateEncoder",
    "TwoDCoordinateEncoder",
    "VolumeCoordinateEncoder",
    "WSICoordinateEncoder",
]
