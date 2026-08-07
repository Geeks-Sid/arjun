"""Phase 06 adapter registry records and smoke plugins.

The Phase 05 catalog owns roster loading from ``model_registry/*.yaml``;
this module supplies the *adapter-verified* overrides for the 2D visual
models it covers: pinned revisions, real ``PreprocessSpec`` declarations,
verified output/spatial capability statuses, declared LoRA target modules,
and realistic parameter counts. ``load_v1_catalog`` applies these overrides
when registering the roster so the registry record reflects what the
adapters actually implement — registry internals do not change.

License gates are unchanged: the catalog still derives READY/BLOCKED from
``licenses.yaml``. As of Phase 06:

- ``rad-dino`` — MIT weights license verified at the pinned revision
  (``LICENSE`` file, Microsoft copyright) → approved_commercial → READY.
- ``medsiglip`` / ``h-optimus-0`` / ``medgemma-1.5-4b`` — gated HAI-DEF /
  Bioptimus terms; remain BLOCKED with structured license reasons. Their
  revisions are pinned (public hub metadata) but weights do not move until
  named-individual acceptance.
- ``conch`` — intentionally no adapter and no override: kept unavailable
  pending license (non-commercial) and repository-behavior review.

Each adapter also registers a smoke plugin building a tiny offline instance
(``build_tiny``), so ``medfm models smoke <id> --backend cpu`` exercises the
real adapter code path without network access once the model is READY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medfm.data.transforms.specs import NormalizationSpec
from medfm.data.transforms.specs import PreprocessSpec as RegistryPreprocessSpec
from medfm.models.pathology import GigaPathTileEncoder
from medfm.models.visual.ct_fm import (
    CTFM_MODEL_ID,
    CTFM_PREPROCESS,
    CTFM_REVISION,
    FLEXICT_MODEL_ID,
    FLEXICT_PREPROCESS,
    FLEXICT_REVISION,
    CTFMAdapter,
    FlexiCT3DAdapter,
)
from medfm.models.visual.hoptimus0 import HOPTIMUS_MODEL_ID, HOPTIMUS_PREPROCESS, HOPTIMUS_REVISION, HOptimus0Adapter
from medfm.models.visual.medgemma_vision import (
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_PREPROCESS,
    MEDGEMMA_REVISION,
    MedGemmaVisionAdapter,
)
from medfm.models.visual.medsiglip import MEDSIGLIP_MODEL_ID, MEDSIGLIP_PREPROCESS, MEDSIGLIP_REVISION, MedSigLIPAdapter
from medfm.models.visual.native_tasks import (
    MEDSAM2_MODEL_ID,
    MEDSAM2_PREPROCESS,
    MEDSAM2_REVISION,
    NV_SEGMENT_MODEL_ID,
    NV_SEGMENT_PREPROCESS,
    NV_SEGMENT_REVISION,
    MedSAM2Adapter,
    NVSegmentCTMRAdapter,
)
from medfm.models.visual.raddino import RADDINO_MODEL_ID, RADDINO_PREPROCESS, RADDINO_REVISION, RADDINOAdapter
from medfm.models.visual.research_3d import (
    M3D_LAMED_MODEL_ID,
    M3D_REVISION,
    MERLIN_MODEL_ID,
    MERLIN_REVISION,
    RESEARCH_3D_PREPROCESS,
    M3DLaMedAdapter,
    MerlinAdapter,
)
from medfm.models.visual.triad import (
    TRIAD_MODEL_ID,
    TRIAD_PREPROCESS,
    TRIAD_REVISION,
    TriadAdapter,
)
from medfm.registry.plugins import ModelPlugin, get_plugin, register_plugin
from medfm.registry.schema import FeatureStatus


@dataclass(frozen=True)
class AdapterSpecOverride:
    """Adapter-verified fields the catalog applies to a roster record."""

    revision: str
    preprocess: RegistryPreprocessSpec
    spatial_tokens_status: FeatureStatus
    peft_known_target_modules: tuple[str, ...]
    parameters_b: float
    aliases: tuple[str, ...] = ()
    pure_pytorch_fallback: bool = True
    custom_operators: tuple[str, ...] = ()
    compile_risk_note: str | None = None


#: LoRA target module names in registry-friendly form (regexes live on the
#: adapters; these are the human/CLI-facing module groups).
_MEDSIGLIP_PEFT = (
    "vision_model self_attn q/k/v/out_proj + mlp fc1/fc2 (vision-LoRA)",
    "text_model self_attn q/k/v/out_proj + mlp fc1/fc2 (contrastive LoRA)",
)
_RADDINO_PEFT = (
    "encoder.layer attention query/key/value + output.dense",
    "encoder.layer mlp fc1/fc2",
)
_HOPTIMUS_PEFT = (
    "blocks attn qkv/proj (gated behind frozen baseline)",
    "blocks mlp fc1/fc2 (gated behind frozen baseline)",
)
_MEDGEMMA_PEFT = (
    "vision_tower self_attn q/k/v/out_proj + mlp fc1/fc2",
    "multi_modal_projector",
)
_NATIVE_3D_PEFT = (
    r"blocks\.\d+\.self_attn\.out_proj (attention projection; convolution stem excluded)",
    r"blocks\.\d+\.linear[12] (transformer MLP projection; convolution stem excluded)",
)

_PATHOLOGY_TILE_PREPROCESS = RegistryPreprocessSpec(
    model_id="h-optimus-0",
    spatial_shape=(224, 224),
    channels=3,
    dtype="float32",
    value_range=(0.0, 1.0),
    normalization=NormalizationSpec(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
)
_GIGAPATH_REVISION = "local-fallback-pending-license-acceptance"
_TITAN_REVISION = "local-fallback-pending-license-acceptance"
_PATHOLOGY_PEFT = (
    "tile_encoder frozen extraction (slide aggregation is parameter-free by default)",
    "visual projector attention/MLP (when a pathology VLM is enabled)",
)


def adapter_spec_overrides() -> dict[str, AdapterSpecOverride]:
    """Adapter-verified registry fields, keyed by roster model id."""
    return {
        MEDSIGLIP_MODEL_ID: AdapterSpecOverride(
            revision=MEDSIGLIP_REVISION,
            preprocess=MEDSIGLIP_PREPROCESS.registry_spec(MEDSIGLIP_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_MEDSIGLIP_PEFT,
            parameters_b=0.4,
            aliases=("medsiglip_448",),
            compile_risk_note="SDPA attention; static 448x448 grid; no custom operators",
        ),
        RADDINO_MODEL_ID: AdapterSpecOverride(
            revision=RADDINO_REVISION,
            preprocess=RADDINO_PREPROCESS.registry_spec(RADDINO_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_RADDINO_PEFT,
            parameters_b=0.086,
            compile_risk_note="SDPA attention; static 518x518 grid; no custom operators",
        ),
        HOPTIMUS_MODEL_ID: AdapterSpecOverride(
            revision=HOPTIMUS_REVISION,
            preprocess=HOPTIMUS_PREPROCESS.registry_spec(HOPTIMUS_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_HOPTIMUS_PEFT,
            parameters_b=1.1,
            compile_risk_note="timm ViT-g/14; SDPA attention; static 224x224 grid; frozen BF16 default",
        ),
        MEDGEMMA_MODEL_ID: AdapterSpecOverride(
            revision=MEDGEMMA_REVISION,
            preprocess=MEDGEMMA_PREPROCESS.registry_spec(MEDGEMMA_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_MEDGEMMA_PEFT,
            parameters_b=4.0,
            compile_risk_note="Gemma3 SigLIP tower + projector; static 896x896 grid",
        ),
        CTFM_MODEL_ID: AdapterSpecOverride(
            revision=CTFM_REVISION,
            preprocess=CTFM_PREPROCESS.registry_spec(CTFM_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.1,
            aliases=("ct_fm",),
            compile_risk_note="pure PyTorch Conv3d + TransformerEncoder; fixed 96^3 bucket; no custom CUDA ops",
        ),
        FLEXICT_MODEL_ID: AdapterSpecOverride(
            revision=FLEXICT_REVISION,
            preprocess=FLEXICT_PREPROCESS.registry_spec(FLEXICT_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.1,
            aliases=("flexict_2d", "flexict_3d", "flexict_3d_vlm"),
            compile_risk_note="3D fallback is pure PyTorch; 2D/VLM upstream variants remain separately gated",
        ),
        TRIAD_MODEL_ID: AdapterSpecOverride(
            revision=TRIAD_REVISION,
            preprocess=TRIAD_PREPROCESS.registry_spec(TRIAD_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.1,
            aliases=("triad_mae", "triad_simmim"),
            compile_risk_note=(
                "sequence-preserving 2-channel fixed MRI bucket; Swin upstream custom ops not used by fallback"
            ),
        ),
        NV_SEGMENT_MODEL_ID: AdapterSpecOverride(
            revision=NV_SEGMENT_REVISION,
            preprocess=NV_SEGMENT_PREPROCESS.registry_spec(NV_SEGMENT_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.5,
            compile_risk_note=(
                "MONAI bundle metadata required for native checkpoint; pure PyTorch fallback has no custom CUDA ops"
            ),
        ),
        MEDSAM2_MODEL_ID: AdapterSpecOverride(
            revision=MEDSAM2_REVISION,
            preprocess=MEDSAM2_PREPROCESS.registry_spec(MEDSAM2_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.2,
            compile_risk_note="sequential prompt/memory lifecycle is gated; generic token path is fallback only",
        ),
        MERLIN_MODEL_ID: AdapterSpecOverride(
            revision=MERLIN_REVISION,
            preprocess=RESEARCH_3D_PREPROCESS.registry_spec(MERLIN_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=0.2,
            compile_risk_note="image-text/report pathway remains checkpoint-gated; visual fallback is pure PyTorch",
        ),
        M3D_LAMED_MODEL_ID: AdapterSpecOverride(
            revision=M3D_REVISION,
            preprocess=RESEARCH_3D_PREPROCESS.registry_spec(M3D_LAMED_MODEL_ID),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_NATIVE_3D_PEFT,
            parameters_b=8.0,
            compile_risk_note="language component requires QLoRA/memory profile; visual fallback only",
        ),
        "gigapath-flash": AdapterSpecOverride(
            revision=_GIGAPATH_REVISION,
            preprocess=RegistryPreprocessSpec(
                model_id="gigapath-flash",
                spatial_shape=(224, 224),
                channels=3,
                dtype="float32",
                value_range=(0.0, 1.0),
                normalization=NormalizationSpec(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_PATHOLOGY_PEFT,
            parameters_b=1.1,
            pure_pytorch_fallback=True,
            compile_risk_note=(
                "native slide weights are gated; CPU decode plus fixed-token fallback has no custom CUDA ops"
            ),
        ),
        "titan": AdapterSpecOverride(
            revision=_TITAN_REVISION,
            preprocess=RegistryPreprocessSpec(
                model_id="titan",
                spatial_shape=(224, 224),
                channels=3,
                dtype="float32",
                value_range=(0.0, 1.0),
                normalization=NormalizationSpec(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ),
            spatial_tokens_status=FeatureStatus.NATIVE,
            peft_known_target_modules=_PATHOLOGY_PEFT,
            parameters_b=0.7,
            pure_pytorch_fallback=True,
            compile_risk_note=(
                "TITAN checkpoint is research-gated; text alignment is exposed through a fixed-token bridge"
            ),
        ),
    }


# --------------------------------------------------------------------------- #
# Smoke plugins (tiny offline instances)
# --------------------------------------------------------------------------- #


class MedSigLIPPlugin:
    def build(self, spec: Any) -> Any:
        return MedSigLIPAdapter.build_tiny(model_id=spec.model_id)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        return {"pixel_values": torch.zeros(1, 3, 32, 32)}


class RADDINOPlugin:
    def build(self, spec: Any) -> Any:
        return RADDINOAdapter.build_tiny(model_id=spec.model_id)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        return {"pixel_values": torch.zeros(1, 3, 32, 32)}


class PathologyPlugin:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def build(self, spec: Any) -> Any:
        if self._model_id == "titan":
            from medfm.models.pathology import TITANAdapter

            return TITANAdapter(embedding_dim=32, visual_dim=32, max_tokens=32)
        return GigaPathTileEncoder(embedding_dim=32)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        if self._model_id == "titan":
            return {"visual_tokens": torch.zeros(1, 32, 32), "visual_token_mask": torch.ones(1, 32, dtype=torch.bool)}
        return {"tiles": torch.zeros(1, 3, 32, 32)}


class HOptimus0Plugin:
    def build(self, spec: Any) -> Any:
        return HOptimus0Adapter.build_tiny(model_id=spec.model_id)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        return {"pixel_values": torch.zeros(1, 3, 64, 64)}


class MedGemmaVisionPlugin:
    def build(self, spec: Any) -> Any:
        return MedGemmaVisionAdapter.build_tiny(model_id=spec.model_id)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        return {"pixel_values": torch.zeros(1, 3, 32, 32)}


class Native3DPlugin:
    def __init__(self, builder: Any, *, channels: int = 1, shape: tuple[int, int, int] = (16, 16, 16)) -> None:
        self._builder = builder
        self._channels = channels
        self._shape = shape

    def build(self, spec: Any) -> Any:
        return self._builder(model_id=spec.model_id, construction_seed=0)

    def tiny_input(self, spec: Any) -> dict[str, Any]:
        import torch

        return {"pixel_values": torch.zeros(1, self._channels, *self._shape)}


_PLUGINS: dict[str, ModelPlugin] = {
    MEDSIGLIP_MODEL_ID: MedSigLIPPlugin(),
    RADDINO_MODEL_ID: RADDINOPlugin(),
    HOPTIMUS_MODEL_ID: HOptimus0Plugin(),
    "gigapath-flash": PathologyPlugin("gigapath-flash"),
    "titan": PathologyPlugin("titan"),
    MEDGEMMA_MODEL_ID: MedGemmaVisionPlugin(),
    CTFM_MODEL_ID: Native3DPlugin(CTFMAdapter.build_tiny),
    FLEXICT_MODEL_ID: Native3DPlugin(FlexiCT3DAdapter.build_tiny),
    TRIAD_MODEL_ID: Native3DPlugin(TriadAdapter.build_tiny, channels=2),
    NV_SEGMENT_MODEL_ID: Native3DPlugin(NVSegmentCTMRAdapter.build_tiny),
    MEDSAM2_MODEL_ID: Native3DPlugin(MedSAM2Adapter.build_tiny),
    MERLIN_MODEL_ID: Native3DPlugin(MerlinAdapter.build_tiny),
    M3D_LAMED_MODEL_ID: Native3DPlugin(M3DLaMedAdapter.build_tiny),
}


def register_2d_plugins() -> None:
    """Idempotent plugin registration for the 2D adapters (catalog calls this)."""
    for model_id, plugin in _PLUGINS.items():
        if get_plugin(model_id) is None:
            register_plugin(model_id, plugin)


__all__ = [
    "AdapterSpecOverride",
    "adapter_spec_overrides",
    "register_2d_plugins",
    "MedSigLIPPlugin",
    "RADDINOPlugin",
    "HOptimus0Plugin",
    "MedGemmaVisionPlugin",
    "PathologyPlugin",
]
