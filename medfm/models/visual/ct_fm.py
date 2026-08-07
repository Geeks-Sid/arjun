"""CT-FM and FlexiCT native 3D adapter declarations.

The upstream checkpoints remain license/repository gated in the Phase-05
catalog. These adapters therefore provide a pinned-identity contract and an
offline pure-PyTorch path; ``from_pretrained_dir`` is intentionally explicit
and only accepts a caller-supplied local checkpoint directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.models.visual.native_3d import GenericMONAI3DAdapter, Native3DPreprocess

CTFM_MODEL_ID = "ct-fm"
CTFM_REVISION = "7b4e1c9d2a6f3e8b1c0d4a5e6f7b8c9d0e1f2a3b"
CTFM_PREPROCESS = Native3DPreprocess(
    spatial_shape=(96, 96, 96),
    channels=1,
    patch_size=(16, 16, 16),
    mean=(0.0,),
    std=(1.0,),
    value_range=(-1024.0, 3071.0),
    resize_policy="crop_or_pad",
    orientation="RAS",
    sequence_order=("CT_HU",),
)

FLEXICT_MODEL_ID = "flexict-3d"
FLEXICT_REVISION = "4d2e9c7b1a6f3e8d0c5b2a9f7e1d4c6b8a0f2e3d"
FLEXICT_PREPROCESS = Native3DPreprocess(
    spatial_shape=(96, 96, 96),
    channels=1,
    patch_size=(16, 16, 16),
    mean=(0.0,),
    std=(1.0,),
    value_range=(-1024.0, 3071.0),
    resize_policy="crop_or_pad",
    orientation="RAS",
    sequence_order=("CT_HU",),
)


def _ct_capabilities(model_id: str) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=(Modality.CT_3D,),
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
    )


class CTFMAdapter(GenericMONAI3DAdapter):
    """CT-FM contract adapter with pooled, patch, and decoder features."""

    def __init__(
        self,
        *,
        model_id: str = CTFM_MODEL_ID,
        revision: str = CTFM_REVISION,
        preprocess: Native3DPreprocess = CTFM_PREPROCESS,
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
            capabilities=_ct_capabilities(model_id),
            hidden_size=hidden_size,
            depth=depth,
            heads=heads,
            feature_map_layers=feature_map_layers,
            construction_seed=construction_seed,
            unsupported_xla_ops=(),
            custom_cuda_dependencies=(),
        )

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "ct-fm-tiny",
        modality: Modality = Modality.CT_3D,
        channels: int = 1,
        construction_seed: int = 0,
    ) -> CTFMAdapter:
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

    @classmethod
    def from_pretrained_dir(cls, directory: str | Path, *, revision: str = CTFM_REVISION) -> CTFMAdapter:
        import torch

        path = Path(directory)
        config = torch.load(path / "config.pt", map_location="cpu", weights_only=True)
        adapter = cls(revision=revision, **dict(config))
        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        adapter.backbone.load_state_dict(state, strict=True)
        adapter.eval()
        return adapter


class FlexiCT3DAdapter(CTFMAdapter):
    """FlexiCT 3D checkpoint family; capability declarations stay separate."""

    def __init__(self, *, model_id: str = FLEXICT_MODEL_ID, revision: str = FLEXICT_REVISION, **kwargs: Any) -> None:
        kwargs.setdefault("preprocess", FLEXICT_PREPROCESS)
        super().__init__(model_id=model_id, revision=revision, **kwargs)

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "flexict-3d-tiny",
        modality: Modality = Modality.CT_3D,
        channels: int = 1,
        construction_seed: int = 0,
    ) -> FlexiCT3DAdapter:
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


class FlexiCTVLMAdapter(FlexiCT3DAdapter):
    """FlexiCT 3D-VLM identity; no language generation is fabricated here."""

    def __init__(self, *, model_id: str = "flexict-3d-vlm", **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.native_visual_connector = False
        self.language_limitation = "Upstream VLM bridge is gated; this adapter exposes visual features only."

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "flexict-3d-vlm-tiny",
        modality: Modality = Modality.CT_3D,
        channels: int = 1,
        construction_seed: int = 0,
    ) -> FlexiCTVLMAdapter:
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


__all__ = [
    "CTFMAdapter",
    "CTFM_MODEL_ID",
    "CTFM_PREPROCESS",
    "CTFM_REVISION",
    "FlexiCT3DAdapter",
    "FlexiCTVLMAdapter",
    "FLEXICT_MODEL_ID",
    "FLEXICT_PREPROCESS",
    "FLEXICT_REVISION",
]
