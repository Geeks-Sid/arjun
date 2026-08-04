"""Visual encoder contract.

Output tensor semantics (documented contract for every adapter):

- ``pooled_embedding [B, Dp]``: one vector per image / volume / slide.
  Produced by explicit pooling over spatial tokens or a dedicated CLS /
  aggregation pathway. Never a silent substitute for requested spatial output.
- ``spatial_tokens [B, N, Dv]``: patch- (2D), volume-patch- (3D), slice-
  (slice-sequence), or tile-level (WSI) tokens in row-major spatial order.
- ``feature_maps``: pyramid of dense maps for segmentation decoders —
  ``[B, C_l, H_l, W_l]`` (2D) or ``[B, C_l, D_l, H_l, W_l]`` (3D), finest
  resolution last unless the adapter documents otherwise.
- ``token_mask [B, N]``: True/1 = real token, False/0 = padding (buckets).
- ``token_coordinates``: one coordinate per spatial token, interpreted in
  ``token_coordinate_system`` — NORMALIZED_IMAGE (x, y in [0, 1]),
  MILLIMETERS (patient space, radiology), MICRONS or SLIDE_PIXELS
  (pathology). Shape ``[B, N, 2]`` (2D / slide) or ``[B, N, 3]`` (3D).
- ``logits``: adapter-native head outputs, when the backbone ships one.
- ``native_outputs``: escape hatch for backbone-specific structures; presence
  must be declared in ``auxiliary["native_outputs_kind"]``.

An adapter asked for output it cannot produce must raise
:class:`UnsupportedCapabilityError` — it must never fabricate output (e.g.
silently pooling tokens when spatial tokens were requested).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from medfm.core.batch import MedicalBatch
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError


@dataclass(frozen=True)
class EncoderCapabilities:
    """What an encoder adapter can do. Declared, never inferred."""

    model_id: str
    modalities: tuple[Modality, ...]
    supports_pooled: bool = True
    supports_spatial_tokens: bool = True
    supports_feature_maps: bool = False
    supports_token_coordinates: bool = False
    token_coordinate_systems: tuple[CoordinateSystem, ...] = ()
    max_images_per_sample: int | None = None  # multi-image bound
    max_tiles_per_slide: int | None = None  # WSI bound
    native_visual_connector: bool = False  # feeds an LM directly (e.g. MedGemma visual pathway)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ShapeContractError("EncoderCapabilities.model_id must be non-empty")
        if not self.modalities:
            raise ShapeContractError("EncoderCapabilities.modalities must be non-empty")
        if self.supports_token_coordinates and not self.token_coordinate_systems:
            raise ShapeContractError("supports_token_coordinates requires declaring token_coordinate_systems")
        if not self.supports_spatial_tokens and self.supports_feature_maps:
            raise ShapeContractError("feature maps are built from spatial tokens; declare both or neither")

    def require_modality(self, modality: Modality) -> None:
        from medfm.core.errors import UnsupportedModalityError

        if modality not in self.modalities:
            raise UnsupportedModalityError(
                f"encoder {self.model_id!r} does not support modality {modality}; "
                f"supported: {[m.value for m in self.modalities]}"
            )


@dataclass(frozen=True)
class PreprocessSpec:
    """Deterministic preprocessing an encoder expects (Phase 04 implements it)."""

    image_size: tuple[int, ...]  # (H, W) or (D, H, W)
    channels: int
    mean: tuple[float, ...] | None = None  # per-channel, post-scaling
    std: tuple[float, ...] | None = None
    value_range: tuple[float, float] = (0.0, 1.0)  # range before mean/std normalization
    resize_policy: str = "letterbox"  # letterbox | stretch | center_crop
    canonical_dtype: str = "float32"  # accelerator-neutral dtype name

    def __post_init__(self) -> None:
        if any(d <= 0 for d in self.image_size) or len(self.image_size) not in (2, 3):
            raise ShapeContractError(f"image_size must be 2 or 3 positive dims; got {self.image_size}")
        if self.channels <= 0:
            raise ShapeContractError("channels must be positive")
        for name, stats in (("mean", self.mean), ("std", self.std)):
            if stats is not None and len(stats) != self.channels:
                raise ShapeContractError(f"{name} must have one entry per channel ({self.channels})")
        if self.std is not None and any(s <= 0 for s in self.std):
            raise ShapeContractError("std entries must be positive")
        if self.value_range[0] >= self.value_range[1]:
            raise ShapeContractError("value_range must be increasing")


@dataclass(frozen=True)
class InputSpec:
    """What a batch must contain for this encoder."""

    modality: Modality
    requires_tile_coordinates: bool = False
    requires_spatial_metadata: bool = False
    extra_batch_keys: tuple[str, ...] = ()  # required task_targets keys

    def validate_batch(self, batch: MedicalBatch) -> None:
        from medfm.core.errors import UnsupportedModalityError

        if batch.modality is not self.modality:
            raise UnsupportedModalityError(
                f"encoder input spec is for {self.modality}; batch declares {batch.modality}"
            )
        if self.requires_tile_coordinates and batch.tile_coordinates is None:
            raise ShapeContractError(f"{self.modality} encoder requires batch.tile_coordinates")
        if self.requires_spatial_metadata and not any(batch.spatial_metadata):
            raise ShapeContractError(f"{self.modality} encoder requires batch.spatial_metadata")
        missing = [k for k in self.extra_batch_keys if k not in batch.task_targets]
        if missing:
            raise ShapeContractError(f"batch is missing required task_targets keys: {missing}")


@dataclass(frozen=True)
class OutputSpec:
    """Which outputs an encode() call must produce."""

    pooled: bool = True
    spatial_tokens: bool = False
    feature_maps: bool = False
    token_coordinates: bool = False

    def check_supported(self, capabilities: EncoderCapabilities) -> None:
        """Raise if the encoder cannot produce a requested output (no fabrication)."""
        if self.pooled and not capabilities.supports_pooled:
            raise UnsupportedCapabilityError(f"encoder {capabilities.model_id!r} cannot produce pooled embeddings")
        if self.spatial_tokens and not capabilities.supports_spatial_tokens:
            raise UnsupportedCapabilityError(
                f"encoder {capabilities.model_id!r} cannot produce spatial tokens; refusing to silently pool instead"
            )
        if self.feature_maps and not capabilities.supports_feature_maps:
            raise UnsupportedCapabilityError(f"encoder {capabilities.model_id!r} cannot produce feature maps")
        if self.token_coordinates and not capabilities.supports_token_coordinates:
            raise UnsupportedCapabilityError(f"encoder {capabilities.model_id!r} cannot produce token coordinates")


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class EncoderOutput:
    """Result of a VisualEncoder.encode call. Field semantics in module docstring."""

    pooled_embedding: torch.Tensor | None = None
    spatial_tokens: torch.Tensor | None = None
    feature_maps: tuple[torch.Tensor, ...] | None = None
    token_mask: torch.Tensor | None = None
    token_coordinates: torch.Tensor | None = None
    token_coordinate_system: CoordinateSystem | None = None
    logits: torch.Tensor | None = None
    native_outputs: Any | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spatial_tokens is not None and self.spatial_tokens.ndim != 3:
            raise ShapeContractError(
                f"spatial_tokens must have shape [B, N, Dv]; got {tuple(self.spatial_tokens.shape)}"
            )
        if self.pooled_embedding is not None and self.pooled_embedding.ndim != 2:
            raise ShapeContractError(
                f"pooled_embedding must have shape [B, Dp]; got {tuple(self.pooled_embedding.shape)}"
            )
        if self.token_mask is not None:
            if self.spatial_tokens is None:
                raise ShapeContractError("token_mask requires spatial_tokens")
            if tuple(self.token_mask.shape) != tuple(self.spatial_tokens.shape[:2]):
                raise ShapeContractError(
                    f"token_mask shape {tuple(self.token_mask.shape)} must equal spatial_tokens [B, N] = "
                    f"{tuple(self.spatial_tokens.shape[:2])}"
                )
        if self.token_coordinates is not None:
            if self.spatial_tokens is None:
                raise ShapeContractError("token_coordinates require spatial_tokens")
            if self.token_coordinate_system is None:
                raise ShapeContractError(
                    "token_coordinates require an explicit token_coordinate_system "
                    "(NORMALIZED_IMAGE | MILLIMETERS | MICRONS | SLIDE_PIXELS)"
                )
            coords = self.token_coordinates
            if coords.ndim != 3 or tuple(coords.shape[:2]) != tuple(self.spatial_tokens.shape[:2]):
                raise ShapeContractError(f"token_coordinates must have shape [B, N, 2|3]; got {tuple(coords.shape)}")
            if coords.shape[2] not in (2, 3):
                raise ShapeContractError(f"token_coordinates last dim must be 2 or 3; got {coords.shape[2]}")
        if self.native_outputs is not None and "native_outputs_kind" not in self.auxiliary:
            raise ShapeContractError("native_outputs present but auxiliary['native_outputs_kind'] is undeclared")

    def check_against(self, spec: OutputSpec) -> None:
        """Verify this output satisfies a request; missing requested output is an error."""
        missing: list[str] = []
        if spec.pooled and self.pooled_embedding is None:
            missing.append("pooled_embedding")
        if spec.spatial_tokens and self.spatial_tokens is None:
            missing.append("spatial_tokens")
        if spec.feature_maps and not self.feature_maps:
            missing.append("feature_maps")
        if spec.token_coordinates and self.token_coordinates is None:
            missing.append("token_coordinates")
        if missing:
            raise UnsupportedCapabilityError(
                f"encoder output is missing requested fields {missing}; "
                "adapters must fail rather than fabricate or silently substitute outputs"
            )


@runtime_checkable
class VisualEncoder(Protocol):
    """Contract every visual adapter implements (Phases 06–08)."""

    @property
    def capabilities(self) -> EncoderCapabilities: ...

    def preprocess_spec(self) -> PreprocessSpec: ...

    def encode(self, batch: MedicalBatch, output_hidden_states: bool = False) -> EncoderOutput: ...
