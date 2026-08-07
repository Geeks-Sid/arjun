"""Separately gated research 3D adapters.

Merlin, M3D-CLIP, and M3D-LaMed intentionally retain different capability
surfaces: retrieval/alignment is not language generation. Local tiny builders
exercise the visual contract without downloading or claiming upstream support.
"""

from __future__ import annotations

from typing import Any

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.models.visual.native_3d import GenericMONAI3DAdapter, Native3DPreprocess

MERLIN_MODEL_ID = "merlin"
MERLIN_REVISION = "3a7e1c9d5f2b8e0a4d6c1f9b3e7a2d5c8f0b4e6a"
M3D_CLIP_MODEL_ID = "m3d-clip"
M3D_LAMED_MODEL_ID = "m3d-lamed"
M3D_REVISION = "5f1b8d3a7e2c9f0a4d6b1e8c3f7a5d2b9e0c4f6a"

RESEARCH_3D_PREPROCESS = Native3DPreprocess(
    spatial_shape=(64, 64, 64),
    channels=1,
    patch_size=(8, 8, 8),
    mean=(0.0,),
    std=(1.0,),
    value_range=None,
    orientation="RAS",
    sequence_order=("PRIMARY",),
)

_RESEARCH_PREPROCESS = RESEARCH_3D_PREPROCESS


def _caps(model_id: str) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=(Modality.CT_3D,),
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=False,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
    )


def _tiny_kwargs(model_id: str, construction_seed: int) -> dict[str, Any]:
    base = GenericMONAI3DAdapter.build_tiny(
        model_id=model_id, modality=Modality.CT_3D, construction_seed=construction_seed
    )
    return {
        "model_id": model_id,
        "revision": "local-tiny",
        "preprocess": base.preprocess,
        "hidden_size": 32,
        "depth": 2,
        "heads": 4,
        "construction_seed": construction_seed,
    }


class MerlinAdapter(GenericMONAI3DAdapter):
    """Gated Merlin visual adapter; report/text pathways remain upstream-only."""

    integration_gate = "checkpoint and license review required"

    def __init__(self, *, model_id: str = MERLIN_MODEL_ID, revision: str = MERLIN_REVISION, **kwargs: Any) -> None:
        kwargs.setdefault("preprocess", _RESEARCH_PREPROCESS)
        super().__init__(model_id=model_id, revision=revision, capabilities=_caps(model_id), **kwargs)
        self.native_text_capabilities = False
        self.integration_limitations = ("image-text/report pathway not checkpoint-compatible in local adapter",)

    @classmethod
    def build_tiny(cls, *, model_id: str = "merlin-tiny", construction_seed: int = 0) -> MerlinAdapter:
        return cls(**_tiny_kwargs(model_id, construction_seed))


class M3DCLIPAdapter(GenericMONAI3DAdapter):
    """M3D-CLIP retrieval/alignment only; generation is not exposed."""

    integration_gate = "research-only checkpoint and memory review"

    def __init__(self, *, model_id: str = M3D_CLIP_MODEL_ID, revision: str = M3D_REVISION, **kwargs: Any) -> None:
        kwargs.setdefault("preprocess", _RESEARCH_PREPROCESS)
        super().__init__(model_id=model_id, revision=revision, capabilities=_caps(model_id), **kwargs)
        self.supported_research_tasks = ("retrieval", "contrastive_alignment")

    @classmethod
    def build_tiny(cls, *, model_id: str = "m3d-clip-tiny", construction_seed: int = 0) -> M3DCLIPAdapter:
        return cls(**_tiny_kwargs(model_id, construction_seed))


class M3DLaMedAdapter(M3DCLIPAdapter):
    """M3D-LaMed visual connector; language generation is a separate gate."""

    def __init__(self, *, model_id: str = M3D_LAMED_MODEL_ID, **kwargs: Any) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.language_component_gate = {
            "requires_qlora_memory_profile": True,
            "supports_tpu_bf16_lora": False,
            "status": "gated",
        }

    @classmethod
    def build_tiny(cls, *, model_id: str = "m3d-lamed-tiny", construction_seed: int = 0) -> M3DLaMedAdapter:
        return cls(**_tiny_kwargs(model_id, construction_seed))

    def generate(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("M3D-LaMed generation is gated; attach a reviewed language component explicitly")


__all__ = [
    "M3DCLIPAdapter",
    "M3DLaMedAdapter",
    "M3D_CLIP_MODEL_ID",
    "M3D_LAMED_MODEL_ID",
    "M3D_REVISION",
    "MERLIN_MODEL_ID",
    "MERLIN_REVISION",
    "MerlinAdapter",
    "RESEARCH_3D_PREPROCESS",
]
