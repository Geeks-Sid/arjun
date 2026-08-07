"""MedGemma native visual pathway adapter (vision tower + MM projector only).

MedGemma 1.5 4B (Google, HAI-DEF gated) is a Gemma 3 multimodal model: a
SigLIP vision tower, a multi-modal projector (avg-pool -> RMSNorm ->
projection into the LM hidden size), and a Gemma language model. This adapter
exposes **only the native visual pathway** — vision tower + projector — so
its projected tokens can feed an LM bridge while staying strictly separate
from full native-VLM behavior (generation, chat templates, tool use), which
is Phase 09's MedGemma adapter, not this one.

Contract facts (declared; the license is blocked so real checkpoints are
inaccessible until terms acceptance):

- projected spatial tokens: ``get_image_features`` pools the vision grid
  (``mm_tokens_per_image`` tokens at the LM hidden size) — these are the
  tokens the native connector hands the LM. ``native_visual_connector=True``
  is declared so capability queries see the pathway's role.
- pooled output: mean over projected spatial tokens (the pathway has no
  dedicated CLS/attention pool; the aggregation is declared, not hidden);
- feature maps: unavailable — the projector pools and re-projects tokens,
  so dense maps are not faithfully recoverable; requesting them raises a
  typed capability error (never fabricated);
- token coordinates: NORMALIZED_IMAGE centers of the *projected* grid.

Preprocessing (declared, never executed here): 896x896 RGB, SigLIP
normalization (mean/std 0.5), per the MedGemma processor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.models.visual.base import AdapterPreprocess, BackboneResult, BaseVisualAdapter2D, LoraTargetSpec

#: Pinned upstream revision (google/medgemma-1.5-4b-it, HAI-DEF gated).
MEDGEMMA_MODEL_ID = "medgemma-1.5-4b"
MEDGEMMA_REPOSITORY = "https://huggingface.co/google/medgemma-1.5-4b-it"
#: SHA is public hub metadata; *loading* stays blocked until terms acceptance.
MEDGEMMA_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"

#: Declared preprocessing (per the MedGemma processor: 896x896 SigLIP norm).
MEDGEMMA_PREPROCESS = AdapterPreprocess(
    image_size=(896, 896),
    channels=3,
    patch_size=14,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    value_range=(0.0, 1.0),
    resize_policy="stretch",
    color_space="RGB",
)

MEDGEMMA_MODALITIES = (Modality.XRAY_2D, Modality.MULTI_IMAGE_2D)

MEDGEMMA_LORA_TARGETS = (
    LoraTargetSpec(
        pattern=r"model\.vision_tower\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))",
        reason="MedGemma vision tower: attention projections + MLP (native pathway LoRA)",
    ),
    # The multi-modal projector uses a raw `mm_input_projection_weight`
    # parameter (torch.matmul), not an nn.Linear module; LoRA-injectable
    # only via modules_to_save (which wraps the whole projector). Documented
    # but not declared as a target until inject_lora supports modules_to_save.
)


def medgemma_capabilities(model_id: str = MEDGEMMA_MODEL_ID) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=MEDGEMMA_MODALITIES,
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=False,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
        native_visual_connector=True,
    )


#: Tiny offline construction config (random weights; contract tests only).
#: mm_tokens_per_image=4 matches the 32/16=2x2=4 patch grid so the projector's
#: avg-pool kernel is 1 (valid tiny geometry).
TINY_MEDGEMMA_CONFIG: dict[str, Any] = {
    "text_config": {
        "hidden_size": 48,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 24,
        "vocab_size": 256,
        "max_position_embeddings": 64,
    },
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": 32,
        "patch_size": 16,
    },
    "mm_tokens_per_image": 4,
}
TINY_MEDGEMMA_PREPROCESS = AdapterPreprocess(
    image_size=(32, 32), channels=3, patch_size=16, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)


class MedGemmaVisionAdapter(BaseVisualAdapter2D):
    """Native visual pathway of MedGemma: vision tower + MM projector only.

    The full language model is *not* part of this adapter; generation and
    chat behavior belong to Phase 09. ``native_visual_connector`` marks that
    the projected tokens are the LM-consumable surface.
    """

    def __init__(
        self,
        *,
        model_id: str = MEDGEMMA_MODEL_ID,
        revision: str = MEDGEMMA_REVISION,
        hf_config: dict[str, Any],
        preprocess: AdapterPreprocess = MEDGEMMA_PREPROCESS,
        feature_map_layers: tuple[int, ...] = (),
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            capabilities=medgemma_capabilities(model_id),
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
            lora_targets=MEDGEMMA_LORA_TARGETS,
            construction_seed=construction_seed,
        )
        self._hf_config_dict = dict(hf_config)
        self.backbone = self._build_backbone(hf_config)
        self.eval()

    def _build_backbone(self, hf_config: dict[str, Any]) -> torch.nn.Module:
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration

        if self._construction_seed is not None:
            torch.manual_seed(self._construction_seed)
        config = Gemma3Config(**hf_config)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"  # type: ignore[attr-defined]
        return Gemma3ForConditionalGeneration(config)

    @classmethod
    def from_pretrained_dir(
        cls,
        directory: str | Path,
        *,
        model_id: str = MEDGEMMA_MODEL_ID,
        revision: str = MEDGEMMA_REVISION,
    ) -> MedGemmaVisionAdapter:
        """Load pinned weights (license-gated; accessible only after acceptance)."""
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration

        path = Path(directory)
        config = Gemma3Config.from_pretrained(path)
        weights_model = Gemma3ForConditionalGeneration.from_pretrained(path, attn_implementation="sdpa")
        adapter = cls(model_id=model_id, revision=revision, hf_config=config.to_dict())
        adapter.backbone.load_state_dict(weights_model.state_dict(), strict=True)
        del weights_model
        adapter.eval()
        return adapter

    @classmethod
    def build_tiny(cls, *, model_id: str = "medgemma-vision-tiny", construction_seed: int = 0) -> MedGemmaVisionAdapter:
        """Offline tiny instance for contract/smoke tests (no network)."""
        return cls(
            model_id=model_id,
            revision="local-tiny",
            hf_config=TINY_MEDGEMMA_CONFIG,
            preprocess=TINY_MEDGEMMA_PREPROCESS,
            construction_seed=construction_seed,
        )

    # ------------------------------------------------------------------ #
    # Backbone contract
    # ------------------------------------------------------------------ #

    def _prefix_token_count(self) -> int:
        return 0  # projected tokens carry no CLS prefix

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        outputs = self.backbone.get_image_features(pixel_values=pixel_values)
        projected = outputs.pooler_output  # [B, mm_tokens, lm_hidden]
        pooled = projected.mean(dim=1)  # declared mean aggregation
        hidden: tuple[torch.Tensor, ...] | None = None
        if output_hidden_states and getattr(outputs, "hidden_states", None) is not None:
            hidden = tuple(outputs.hidden_states)
        return BackboneResult(last_hidden_state=projected, pooled=pooled, hidden_states=hidden, raw=outputs)

    @property
    def projector(self) -> torch.nn.Module:
        """The multi-modal projector (bridge surface for Phase 09)."""
        return self.backbone.model.multi_modal_projector

    # ------------------------------------------------------------------ #
    # Checkpoint config record
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict[str, Any]:
        base = super()._config_dict()
        base["hf_config"] = self._hf_config_dict
        return base
