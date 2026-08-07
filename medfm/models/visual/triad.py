"""Triad native MRI adapters (MAE and SimMIM variants)."""

from __future__ import annotations

from typing import Any

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.models.visual.native_3d import GenericMONAI3DAdapter, Native3DPreprocess

TRIAD_MODEL_ID = "triad"
TRIAD_MAE_MODEL_ID = "triad-mae"
TRIAD_SIMMIM_MODEL_ID = "triad-simmim"
TRIAD_REVISION = "8c1e4a7d9b2f6e0c3a5d7f1b9e2c4a6d8f0b1e3c"
TRIAD_PREPROCESS = Native3DPreprocess(
    spatial_shape=(64, 96, 96),
    channels=2,
    patch_size=(4, 8, 8),
    mean=(0.0, 0.0),
    std=(1.0, 1.0),
    value_range=None,
    resize_policy="crop_or_pad",
    orientation="RAS",
    sequence_order=("T1", "T2"),
)


def triad_capabilities(model_id: str = TRIAD_MODEL_ID) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=(Modality.MRI_3D, Modality.MULTI_SERIES_3D),
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
    )


class TriadAdapter(GenericMONAI3DAdapter):
    """MRI Swin-style contract adapter preserving sequence channels."""

    variant = "mae"

    def __init__(
        self,
        *,
        model_id: str = TRIAD_MODEL_ID,
        revision: str = TRIAD_REVISION,
        preprocess: Native3DPreprocess = TRIAD_PREPROCESS,
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
            capabilities=triad_capabilities(model_id),
            hidden_size=hidden_size,
            depth=depth,
            heads=heads,
            feature_map_layers=feature_map_layers,
            construction_seed=construction_seed,
            unsupported_xla_ops=(),
            custom_cuda_dependencies=(),
        )

    @classmethod
    def build_tiny(cls, *, model_id: str = "triad-tiny", construction_seed: int = 0) -> TriadAdapter:
        base = GenericMONAI3DAdapter.build_tiny(
            model_id=model_id, modality=Modality.MRI_3D, channels=2, construction_seed=construction_seed
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


class TriadMAEAdapter(TriadAdapter):
    variant = "mae"

    def __init__(self, *, model_id: str = TRIAD_MAE_MODEL_ID, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)


class TriadSimMIMAdapter(TriadAdapter):
    variant = "simmim"

    def __init__(self, *, model_id: str = TRIAD_SIMMIM_MODEL_ID, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)


__all__ = [
    "TRIAD_MAE_MODEL_ID",
    "TRIAD_MODEL_ID",
    "TRIAD_PREPROCESS",
    "TRIAD_REVISION",
    "TRIAD_SIMMIM_MODEL_ID",
    "TriadAdapter",
    "TriadMAEAdapter",
    "TriadSimMIMAdapter",
    "triad_capabilities",
]
