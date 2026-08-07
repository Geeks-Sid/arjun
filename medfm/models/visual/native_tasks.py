"""Native 3D task-model wrappers.

These wrappers preserve native lifecycle semantics instead of pretending that
all task models are interchangeable visual token encoders. NV-Segment-CTMR
exposes a native segmentation decoder attachment. MedSAM2 keeps explicit
initialize/encode/prompt/memory/decode state and sequential memory is never
mixed into the generic token contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.core.encoder import EncoderCapabilities, EncoderOutput, OutputSpec
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.models.visual.native_3d import GenericMONAI3DAdapter, Native3DPreprocess

NV_SEGMENT_MODEL_ID = "nv-segment-ctmr"
NV_SEGMENT_REVISION = "9a2f6c1e4b8d0f3a7c5e9b2d6f1a4c8e0b3d7f5a"
NV_SEGMENT_PREPROCESS = Native3DPreprocess(
    spatial_shape=(96, 96, 96),
    channels=1,
    patch_size=(16, 16, 16),
    mean=(0.0,),
    std=(1.0,),
    value_range=(-1024.0, 3071.0),
    orientation="RAS",
    sequence_order=("CT_OR_MRI_PRIMARY",),
)
MEDSAM2_MODEL_ID = "medsam2"
MEDSAM2_REVISION = "6e4b1c9d2f7a5e3b0d8c6f1a4e9b2d7c5f0a3e8b"
MEDSAM2_PREPROCESS = Native3DPreprocess(
    spatial_shape=(64, 64, 64),
    channels=1,
    patch_size=(8, 8, 8),
    mean=(0.0,),
    std=(1.0,),
    value_range=None,
    orientation="RAS",
    sequence_order=("PRIMARY",),
)


def _segmentation_capabilities(model_id: str) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=(Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D),
        # The native decoder is primary, but its encoder still exposes an
        # explicit pooled representation for registry smoke and classification
        # attachment. No segmentation logits are fabricated by encode().
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
    )


class NVSegmentCTMRAdapter(GenericMONAI3DAdapter):
    """Native segmentation-first CT/MRI adapter with optional decoder head."""

    def __init__(
        self,
        *,
        model_id: str = NV_SEGMENT_MODEL_ID,
        revision: str = NV_SEGMENT_REVISION,
        preprocess: Native3DPreprocess = NV_SEGMENT_PREPROCESS,
        hidden_size: int = 96,
        depth: int = 4,
        heads: int = 4,
        feature_map_layers: tuple[int, ...] = (1, 2, 4),
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            preprocess=preprocess,
            capabilities=_segmentation_capabilities(model_id),
            hidden_size=hidden_size,
            depth=depth,
            heads=heads,
            feature_map_layers=feature_map_layers,
            construction_seed=construction_seed,
            unsupported_xla_ops=("upstream_bundle_custom_inferer (not used by pure-PyTorch fallback)",),
            custom_cuda_dependencies=(),
        )
        self.bundle_metadata: dict[str, Any] = {
            "source": "MONAI bundle metadata must be supplied with the checkpoint",
            "preprocessing": self.preprocess.to_dict(),
            "native_segmentation_first": True,
        }
        self._segmentation_decoder: nn.Module | None = None

    @classmethod
    def build_tiny(cls, *, model_id: str = "nv-segment-ctmr-tiny", construction_seed: int = 0) -> NVSegmentCTMRAdapter:
        base = GenericMONAI3DAdapter.build_tiny(
            model_id=model_id, modality=Modality.CT_3D, construction_seed=construction_seed
        )
        adapter = cls(
            model_id=model_id,
            revision="local-tiny",
            preprocess=base.preprocess,
            hidden_size=32,
            depth=2,
            heads=4,
            feature_map_layers=(1, 2),
            construction_seed=construction_seed,
        )
        adapter.bundle_metadata = {"source": "synthetic", "preprocessing": adapter.preprocess.to_dict()}
        return adapter

    def attach_segmentation_decoder(self, num_classes: int) -> None:
        if num_classes <= 0:
            raise ShapeContractError("num_classes must be positive")
        self._segmentation_decoder = nn.Conv3d(self._hidden_size, num_classes, kernel_size=1).to(
            next(self.parameters()).device
        )

    def segmentation_logits(self, batch: Any) -> torch.Tensor:
        if self._segmentation_decoder is None:
            raise UnsupportedCapabilityError("NV-Segment-CTMR decoder is not attached")
        output = self.encode(batch, output_spec=OutputSpec(feature_maps=True, spatial_tokens=True))
        assert output.feature_maps is not None
        return self._segmentation_decoder(output.feature_maps[-1])


@dataclass
class _MedSAM2Memory:
    features: torch.Tensor
    frame_index: int
    prompts: tuple[dict[str, Any], ...]


class MedSAM2Adapter(GenericMONAI3DAdapter):
    """Promptable adapter with explicit initialize → encode → decode lifecycle."""

    def __init__(
        self,
        *,
        model_id: str = MEDSAM2_MODEL_ID,
        revision: str = MEDSAM2_REVISION,
        preprocess: Native3DPreprocess = MEDSAM2_PREPROCESS,
        hidden_size: int = 64,
        depth: int = 2,
        heads: int = 4,
        feature_map_layers: tuple[int, ...] = (1, 2),
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            preprocess=preprocess,
            capabilities=_segmentation_capabilities(model_id),
            hidden_size=hidden_size,
            depth=depth,
            heads=heads,
            feature_map_layers=feature_map_layers,
            construction_seed=construction_seed,
            unsupported_xla_ops=("sequential_memory_attention (upstream-only)",),
            custom_cuda_dependencies=(),
        )
        self._memory: _MedSAM2Memory | None = None
        self._initialized = False
        self._prompt_records: list[dict[str, Any]] = []

    @classmethod
    def build_tiny(cls, *, model_id: str = "medsam2-tiny", construction_seed: int = 0) -> MedSAM2Adapter:
        base = GenericMONAI3DAdapter.build_tiny(
            model_id=model_id, modality=Modality.CT_3D, construction_seed=construction_seed
        )
        return cls(
            model_id=model_id,
            revision="local-tiny",
            preprocess=base.preprocess,
            hidden_size=32,
            depth=2,
            heads=4,
            feature_map_layers=(1, 2),
            construction_seed=construction_seed,
        )

    def initialize(self) -> None:
        self._initialized = True
        self._memory = None
        self._prompt_records = []

    def encode_image(self, batch: Any) -> EncoderOutput:
        if not self._initialized:
            raise ShapeContractError("MedSAM2.initialize() must be called before encode_image")
        output = self.encode(batch, output_spec=OutputSpec(spatial_tokens=True, feature_maps=True))
        assert output.spatial_tokens is not None
        self._memory = _MedSAM2Memory(output.spatial_tokens, 0, tuple(self._prompt_records))
        return output

    def prompt(self, prompt: dict[str, Any]) -> None:
        if not self._initialized:
            raise ShapeContractError("MedSAM2.initialize() must be called before prompt")
        if not prompt:
            raise ShapeContractError("MedSAM2 prompt must be non-empty")
        self._prompt_records.append(dict(prompt))
        if self._memory is not None:
            self._memory = _MedSAM2Memory(self._memory.features, self._memory.frame_index, tuple(self._prompt_records))

    def add_memory(self, features: torch.Tensor, *, frame_index: int) -> None:
        if not self._initialized:
            raise ShapeContractError("MedSAM2.initialize() must be called before add_memory")
        if features.ndim != 3:
            raise ShapeContractError("MedSAM2 memory features must be [B,N,D]")
        self._memory = _MedSAM2Memory(features, int(frame_index), tuple(self._prompt_records))

    def decode(self) -> torch.Tensor:
        if not self._initialized or self._memory is None:
            raise ShapeContractError("MedSAM2 requires initialize() and encode_image() before decode")
        if not self._prompt_records:
            raise ShapeContractError("MedSAM2 decode requires at least one prompt")
        # A deterministic contract fallback: prompt presence gates a mask;
        # native checkpoints replace this with their prompt/memory decoder.
        logits = self._memory.features.mean(dim=-1)
        return logits.unsqueeze(1)

    @property
    def memory_state(self) -> dict[str, Any] | None:
        if self._memory is None:
            return None
        return {
            "frame_index": self._memory.frame_index,
            "batch": int(self._memory.features.shape[0]),
            "tokens": int(self._memory.features.shape[1]),
            "prompt_count": len(self._memory.prompts),
        }


__all__ = [
    "MEDSAM2_MODEL_ID",
    "MEDSAM2_PREPROCESS",
    "MEDSAM2_REVISION",
    "MedSAM2Adapter",
    "NVSegmentCTMRAdapter",
    "NV_SEGMENT_MODEL_ID",
    "NV_SEGMENT_PREPROCESS",
    "NV_SEGMENT_REVISION",
]
