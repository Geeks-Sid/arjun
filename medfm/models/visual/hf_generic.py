"""Generic Hugging Face vision-tower adapter.

``GenericHFVisionAdapter`` is the first vertical slice of Phase 06 and the
shared substrate for the family-specific adapters:

- ``RADDINOAdapter`` fixes ``family="dinov2"`` with RAD-DINO's preprocessing;
- ``MedSigLIPAdapter`` wraps the full ``SiglipModel`` (vision + text towers);
- ``MedGemmaVisionAdapter`` wraps the Gemma 3 vision tower + projector;
- the generic adapter itself is also the fallback path for other HF vision
  architectures whose *declared* capabilities match one of the families —
  it never upgrades an architecture's capabilities beyond what the family
  registry declares.

Backbones are constructed either from a local config dict (tiny/offline
tests: random weights, ``construction_seed`` recorded in the checkpoint
manifest) or from a local directory of pinned weights
(:meth:`from_pretrained_dir`). Weight download stays explicit and belongs to
Phase 05's ``medfm.registry`` surface, never to adapter construction.

Attention defaults to PyTorch SDPA (``_attn_implementation = "sdpa"``);
CUDA-only custom attention kernels are out of scope for the contract: every
family runs eager/SDPA in pure PyTorch, which keeps the TPU path eligible
(no custom operators declared in the registry).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import UnsupportedCapabilityError
from medfm.models.visual.base import AdapterPreprocess, BackboneResult, BaseVisualAdapter2D, LoraTargetSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FamilySpec:
    """Declarative description of one HF vision-tower family."""

    model_type: str
    pooled_source: str  # "cls" | "attention_pool" | "mean"
    prefix_token_fn: Callable[[dict[str, Any]], int]
    lora_targets: tuple[LoraTargetSpec, ...]


def _siglip_prefix(config: dict[str, Any]) -> int:
    return 0


def _vit_prefix(config: dict[str, Any]) -> int:
    return 1


def _dinov2_prefix(config: dict[str, Any]) -> int:
    return 1 + int(config.get("num_register_tokens", 0))


#: Family registry. Patterns are full-path regexes scoped to the encoder so
#: LoRA injection can never leak into unrelated submodels.
FAMILIES: dict[str, FamilySpec] = {
    "siglip-vision": FamilySpec(
        model_type="siglip",
        pooled_source="attention_pool",
        prefix_token_fn=_siglip_prefix,
        lora_targets=(
            LoraTargetSpec(
                pattern=r"encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
                reason="SigLIP vision-tower attention projections (4 matrices per layer)",
            ),
            LoraTargetSpec(
                pattern=r"encoder\.layers\.\d+\.mlp\.(fc1|fc2)",
                reason="SigLIP vision-tower MLP projections",
            ),
        ),
    ),
    "dinov2": FamilySpec(
        model_type="dinov2",
        pooled_source="cls",
        prefix_token_fn=_dinov2_prefix,
        lora_targets=(
            LoraTargetSpec(
                pattern=r"encoder\.layer\.\d+\.attention\.attention\.(query|key|value)",
                reason="DINOv2 attention Q/K/V projections",
            ),
            LoraTargetSpec(
                pattern=r"encoder\.layer\.\d+\.attention\.output\.dense",
                reason="DINOv2 attention output projection",
            ),
            LoraTargetSpec(
                pattern=r"encoder\.layer\.\d+\.mlp\.(fc1|fc2)",
                reason="DINOv2 MLP projections",
            ),
        ),
    ),
    "vit": FamilySpec(
        model_type="vit",
        pooled_source="cls",
        prefix_token_fn=_vit_prefix,
        lora_targets=(
            LoraTargetSpec(
                pattern=r"encoder\.layer\.\d+\.attention\.attention\.(query|key|value)",
                reason="ViT attention Q/K/V projections",
            ),
            LoraTargetSpec(
                pattern=r"encoder\.layer\.\d+\.(intermediate\.dense|output\.dense)",
                reason="ViT MLP intermediate/output projections",
            ),
        ),
    ),
}


class GenericHFVisionAdapter(BaseVisualAdapter2D):
    """Adapter over an HF vision tower selected by ``family``.

    Construction is fully offline: ``hf_config`` is a JSON-able config dict
    for the family's HF config class. ``from_pretrained_dir`` instead loads
    pinned weights from a local directory (Phase 05 download/verify output).
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        family: str,
        hf_config: dict[str, Any],
        capabilities: EncoderCapabilities,
        preprocess: AdapterPreprocess,
        feature_map_layers: tuple[int, ...] = (),
        lora_targets: tuple[LoraTargetSpec, ...] | None = None,
        construction_seed: int | None = None,
    ) -> None:
        if family not in FAMILIES:
            raise UnsupportedCapabilityError(
                f"unknown HF vision family {family!r}; supported: {sorted(FAMILIES)}. Generic fallback adapters "
                "must match a declared family — capabilities are never inferred."
            )
        spec = FAMILIES[family]
        super().__init__(
            model_id=model_id,
            revision=revision,
            capabilities=capabilities,
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
            lora_targets=lora_targets if lora_targets is not None else spec.lora_targets,
            construction_seed=construction_seed,
        )
        self._family = family
        self._hf_config_dict = dict(hf_config)
        self.backbone = self._build_backbone(hf_config)
        self.eval()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_backbone(self, hf_config: dict[str, Any]) -> nn.Module:
        from transformers import AutoConfig, AutoModel

        if self._construction_seed is not None:
            torch.manual_seed(self._construction_seed)
        config = AutoConfig.for_model(FAMILIES[self._family].model_type, **hf_config)
        # SDPA is the preferred pure-PyTorch attention path; custom CUDA
        # attention kernels are never required by this contract.
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"  # type: ignore[attr-defined]
        return AutoModel.from_config(config)

    @classmethod
    def from_pretrained_dir(
        cls,
        directory: str | Path,
        *,
        model_id: str,
        revision: str,
        family: str,
        capabilities: EncoderCapabilities,
        preprocess: AdapterPreprocess,
        feature_map_layers: tuple[int, ...] = (),
    ) -> GenericHFVisionAdapter:
        """Load pinned weights from a local directory (no network).

        The directory is Phase-05 download output: config.json plus
        safetensors at the pinned revision. Construction still declares SDPA.
        """
        from transformers import AutoConfig, AutoModel

        path = Path(directory)
        hf_config_dict = AutoConfig.from_pretrained(path).to_dict()
        adapter = cls(
            model_id=model_id,
            revision=revision,
            family=family,
            hf_config=hf_config_dict,
            capabilities=capabilities,
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
        )
        weights_model = AutoModel.from_pretrained(path, attn_implementation="sdpa")
        incompatible = adapter.backbone.load_state_dict(weights_model.state_dict(), strict=True)
        del weights_model
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise UnsupportedCapabilityError(
                f"pinned weights at {path} do not match the declared family architecture: "
                f"missing={incompatible.missing_keys[:3]} unexpected={incompatible.unexpected_keys[:3]}"
            )
        adapter.eval()
        return adapter

    # ------------------------------------------------------------------ #
    # Backbone contract
    # ------------------------------------------------------------------ #

    def _prefix_token_count(self) -> int:
        return FAMILIES[self._family].prefix_token_fn(self._hf_config_dict)

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        last_hidden = outputs.last_hidden_state
        source = FAMILIES[self._family].pooled_source
        pooled: torch.Tensor | None
        if source == "cls":
            pooled = last_hidden[:, 0, :]
        elif source == "attention_pool":
            pooled = getattr(outputs, "pooler_output", None)
        elif source == "mean":
            pooled = last_hidden[:, self._prefix_token_count() :, :].mean(dim=1)
        else:  # pragma: no cover - registry is closed
            raise UnsupportedCapabilityError(f"unknown pooled_source {source!r}")
        hidden = tuple(outputs.hidden_states) if output_hidden_states else None
        return BackboneResult(last_hidden_state=last_hidden, pooled=pooled, hidden_states=hidden, raw=outputs)

    # ------------------------------------------------------------------ #
    # Checkpoint config record
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict[str, Any]:
        base = super()._config_dict()
        base["family"] = self._family
        base["hf_config"] = self._hf_config_dict
        return base

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> GenericHFVisionAdapter:
        """Rebuild an adapter from a serialized manifest config (tiny path).

        Only used for locally constructed (non-pretrained) adapters whose
        ``hf_config`` and ``construction_seed`` are recorded in the manifest;
        production adapters rebuild from the pinned base revision instead.
        """
        if config.get("construction_seed") is None:
            raise UnsupportedCapabilityError(
                "from_config_dict rebuild requires a recorded construction_seed (tiny local models only); "
                "production adapters rebuild from the pinned base revision"
            )
        capabilities = EncoderCapabilities(
            model_id=str(config["model_id"]),
            modalities=tuple(Modality.from_value(m) for m in config.get("modalities", ["XRAY_2D"])),
            supports_pooled=True,
            supports_spatial_tokens=True,
            supports_feature_maps=bool(config.get("feature_map_layers")),
            supports_token_coordinates=True,
            token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
        )
        return cls(
            model_id=str(config["model_id"]),
            revision=str(config["revision"]),
            family=str(config["family"]),
            hf_config=dict(config["hf_config"]),
            capabilities=capabilities,
            preprocess=AdapterPreprocess.from_dict(config["preprocess"]),
            feature_map_layers=tuple(int(i) for i in config.get("feature_map_layers", ())),
            construction_seed=config["construction_seed"],
        )
