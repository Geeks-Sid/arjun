"""Coordinate-aware, fixed-budget bridges from visual encoders to LMs.

The bridge layer owns the external-encoder pathway.  It accepts framework
visual tokens (never model-specific placeholder strings), validates static
bucket dimensions, and returns the core ``ProjectedVisualTokens`` contract.
All operations are ordinary PyTorch modules so the same code is usable on CPU,
CUDA, and PyTorch/XLA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.core.language import ProjectedVisualTokens


class BridgeError(ShapeContractError):
    """Malformed bridge input or configuration."""


class BridgeCapabilityError(UnsupportedCapabilityError):
    """A bridge cannot satisfy a requested visual-token contract."""


class VisionLanguageBridge(nn.Module, ABC):
    """Base contract for visual-token projectors.

    ``forward`` consumes ``[B, N, source_dim]`` tokens and an optional
    ``[B, N]`` mask.  ``max_input_tokens`` and ``output_tokens`` are fixed
    bounds, making compiled forward shapes explicit.  Padding is preserved in
    the returned mask and never influences masked pooling or attention.
    """

    bridge_type = "base"

    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        output_tokens: int,
        max_input_tokens: int | None = None,
        source_modality: Modality = Modality.MULTI_IMAGE_2D,
        coordinate_system: CoordinateSystem | None = None,
    ) -> None:
        super().__init__()
        if source_dim <= 0 or target_dim <= 0:
            raise BridgeError("source_dim and target_dim must be positive")
        if output_tokens <= 0:
            raise BridgeError("output_tokens must be positive")
        if max_input_tokens is not None and max_input_tokens <= 0:
            raise BridgeError("max_input_tokens must be positive when provided")
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.output_tokens = int(output_tokens)
        self.max_input_tokens = None if max_input_tokens is None else int(max_input_tokens)
        self.source_modality = source_modality
        self.coordinate_system = coordinate_system

    def _validate_inputs(self, visual_tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise BridgeError(f"visual_tokens must be [B,N,D]; got {tuple(visual_tokens.shape)}")
        if int(visual_tokens.shape[-1]) != self.source_dim:
            raise BridgeError(
                f"visual token dimension {int(visual_tokens.shape[-1])} != bridge source_dim {self.source_dim}"
            )
        if self.max_input_tokens is not None and int(visual_tokens.shape[1]) != self.max_input_tokens:
            raise BridgeError(
                f"visual token bucket must use N={self.max_input_tokens}; got {int(visual_tokens.shape[1])}"
            )
        if token_mask is None:
            return torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
        if tuple(token_mask.shape) != tuple(visual_tokens.shape[:2]):
            raise BridgeError("token_mask must have shape [B,N] matching visual_tokens")
        return token_mask.to(device=visual_tokens.device, dtype=torch.bool)

    def _projected(self, tokens: torch.Tensor, mask: torch.Tensor) -> ProjectedVisualTokens:
        if tuple(tokens.shape[:2]) != (int(mask.shape[0]), self.output_tokens):
            raise BridgeError("bridge output does not match its fixed output token budget")
        return ProjectedVisualTokens(
            tokens=tokens,
            source_modality=self.source_modality,
            token_mask=mask,
            coordinate_system=self.coordinate_system,
        )

    @abstractmethod
    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
    ) -> ProjectedVisualTokens:
        """Project visual tokens into the language embedding space."""

    def contract(self) -> dict[str, Any]:
        """Return a serializable declaration used by registry/training phases."""
        return {
            "bridge_type": self.bridge_type,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "output_tokens": self.output_tokens,
            "max_input_tokens": self.max_input_tokens,
            "source_modality": self.source_modality.value,
            "coordinate_system": None if self.coordinate_system is None else self.coordinate_system.value,
        }


class LinearVisionLanguageBridge(VisionLanguageBridge):
    """Single linear projection used for smoke tests and simple baselines."""

    bridge_type = "linear"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projection = nn.Linear(self.source_dim, self.target_dim)
        self.output_norm = nn.LayerNorm(self.target_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
    ) -> ProjectedVisualTokens:
        mask = self._validate_inputs(visual_tokens, token_mask)
        projected = self.output_norm(self.projection(visual_tokens))
        # Padding is zeroed after projection so a padded token cannot leak into
        # an LM through a buggy consumer that ignores the mask.
        projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        return self._projected(projected, mask)


class MLPVisionLanguageBridge(VisionLanguageBridge):
    """Default two-layer GELU bridge; all parameters remain fully trainable."""

    bridge_type = "mlp"

    def __init__(self, *, intermediate_dim: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        width = int(intermediate_dim or max(self.source_dim, self.target_dim))
        if width <= 0:
            raise BridgeError("intermediate_dim must be positive")
        self.intermediate_dim = width
        self.projection = nn.Sequential(
            nn.Linear(self.source_dim, width),
            nn.GELU(),
            nn.Linear(width, self.target_dim),
            nn.LayerNorm(self.target_dim),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        coordinates: torch.Tensor | None = None,
    ) -> ProjectedVisualTokens:
        mask = self._validate_inputs(visual_tokens, token_mask)
        projected = self.projection(visual_tokens)
        projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        return self._projected(projected, mask)


# Short names are part of the public Phase 09 API.
LinearBridge = LinearVisionLanguageBridge
MLPBridge = MLPVisionLanguageBridge

__all__ = [
    "BridgeCapabilityError",
    "BridgeError",
    "LinearBridge",
    "LinearVisionLanguageBridge",
    "MLPBridge",
    "MLPVisionLanguageBridge",
    "VisionLanguageBridge",
]
