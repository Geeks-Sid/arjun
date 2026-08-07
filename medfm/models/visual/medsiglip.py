"""MedSigLIP adapter: SigLIP 400M vision + text towers at 448x448.

MedSigLIP (Google Health AI Developer Foundations) is a SigLIP variant
trained on medical image-text pairs. This adapter wraps the full
``SiglipModel`` so both towers and the learned ``logit_scale``/``logit_bias``
are available:

- pooled image embeddings (attention-pooled vision tower output);
- dense patch tokens: 448 / patch 14 -> 32x32 = 1024 tokens of the vision
  hidden size, row-major — the shared spatial-token surface the external-VLM
  bridge (Phase 09, ADR 0003) consumes;
- text embeddings and normalized image-text similarity using the backbone's
  own temperature and bias (SigLIP sigmoid-loss parameterization);
- hidden states and the native HF output retained for debugging.

Preprocessing (declared, never executed here): 448x448, RGB, rescale to
[0, 1], per-channel mean/std (0.5, 0.5, 0.5)/(0.5, 0.5, 0.5) — the SigLIP
normalization. Single-channel radiology arrives as grayscale repeated to RGB
(Phase 04 radiology2d pipeline). Validate the pinned revision's processor
against these values before enabling real checkpoints (the license is gated:
HAI-DEF terms, named-individual acceptance).

Training modes supported:
- frozen extraction (default; zero trainable backbone parameters);
- contrastive (both towers trainable / LoRA'd — recipe-owned, Phase 11/13);
- classification-head attachment via :meth:`attach_head`;
- vision-only LoRA (``inject_lora`` with the vision-tower target).

are scoped so vision LoRA never touches the text tower and vice versa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.models.visual.base import AdapterPreprocess, BackboneResult, BaseVisualAdapter2D, LoraTargetSpec

logger = logging.getLogger(__name__)

#: Pinned upstream revision (google/medsiglip-448). Loading is license-gated:
#: HAI-DEF terms require named-individual acceptance before weights move.
MEDSIGLIP_MODEL_ID = "medsiglip"
MEDSIGLIP_REPOSITORY = "https://huggingface.co/google/medsiglip-448"
MEDSIGLIP_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"

#: Declared preprocessing; cross-checked against the pinned processor when the
#: license gate opens (model card: 448x448, SigLIP normalization).
MEDSIGLIP_PREPROCESS = AdapterPreprocess(
    image_size=(448, 448),
    channels=3,
    patch_size=14,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    value_range=(0.0, 1.0),
    resize_policy="center_crop",
    color_space="RGB",
)

#: Feature-map hook layers for the 27-layer vision tower (shallow -> deep).
MEDSIGLIP_FEATURE_MAP_LAYERS: tuple[int, ...] = (7, 14, 21, 27)

MEDSIGLIP_LORA_VISION = LoraTargetSpec(
    pattern=r"vision_model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))",
    reason="MedSigLIP vision tower: attention projections + MLP (vision-LoRA mode)",
)
MEDSIGLIP_LORA_TEXT = LoraTargetSpec(
    pattern=r"text_model\.encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))",
    reason="MedSigLIP text tower: attention projections + MLP (contrastive LoRA mode)",
)

#: Tiny offline construction config (random weights; contract tests only).
TINY_MEDSIGLIP_CONFIG: dict[str, Any] = {
    "vision_config": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "image_size": 32,
        "patch_size": 16,
    },
    "text_config": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "max_position_embeddings": 16,
        "vocab_size": 128,
    },
}
TINY_MEDSIGLIP_PREPROCESS = AdapterPreprocess(
    image_size=(32, 32), channels=3, patch_size=16, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)

MEDSIGLIP_MODALITIES = (Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE, Modality.MULTI_IMAGE_2D)


def medsiglip_capabilities(model_id: str = MEDSIGLIP_MODEL_ID) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=MEDSIGLIP_MODALITIES,
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
    )


class MedSigLIPAdapter(BaseVisualAdapter2D):
    """Adapter over the full SiglipModel (vision + text towers)."""

    def __init__(
        self,
        *,
        model_id: str = MEDSIGLIP_MODEL_ID,
        revision: str = MEDSIGLIP_REVISION,
        hf_config: dict[str, Any],
        preprocess: AdapterPreprocess = MEDSIGLIP_PREPROCESS,
        feature_map_layers: tuple[int, ...] = MEDSIGLIP_FEATURE_MAP_LAYERS,
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            capabilities=medsiglip_capabilities(model_id),
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
            lora_targets=(MEDSIGLIP_LORA_VISION, MEDSIGLIP_LORA_TEXT),
            construction_seed=construction_seed,
        )
        self._hf_config_dict = dict(hf_config)
        self.backbone = self._build_backbone(hf_config)
        self.eval()

    def _build_backbone(self, hf_config: dict[str, Any]) -> torch.nn.Module:
        from transformers import SiglipConfig, SiglipModel

        if self._construction_seed is not None:
            torch.manual_seed(self._construction_seed)
        config = SiglipConfig(**hf_config)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        return SiglipModel(config)

    @classmethod
    def from_pretrained_dir(
        cls,
        directory: str | Path,
        *,
        model_id: str = MEDSIGLIP_MODEL_ID,
        revision: str = MEDSIGLIP_REVISION,
    ) -> MedSigLIPAdapter:
        """Load pinned weights from a local directory (no network).

        Requires license acceptance upstream (HAI-DEF gated repository);
        the directory is Phase-05 download output at the pinned revision.
        """
        from transformers import SiglipConfig, SiglipModel

        path = Path(directory)
        config = SiglipConfig.from_pretrained(path)
        weights_model = SiglipModel.from_pretrained(path, attn_implementation="sdpa")
        adapter = cls(
            model_id=model_id,
            revision=revision,
            hf_config=config.to_dict(),
        )
        adapter.backbone.load_state_dict(weights_model.state_dict(), strict=True)
        del weights_model
        adapter.eval()
        return adapter

    @classmethod
    def build_tiny(cls, *, model_id: str = "medsiglip-tiny", construction_seed: int = 0) -> MedSigLIPAdapter:
        """Offline tiny instance for contract/smoke tests (no network)."""
        return cls(
            model_id=model_id,
            revision="local-tiny",
            hf_config=TINY_MEDSIGLIP_CONFIG,
            preprocess=TINY_MEDSIGLIP_PREPROCESS,
            feature_map_layers=(),
            construction_seed=construction_seed,
        )

    # ------------------------------------------------------------------ #
    # Backbone contract
    # ------------------------------------------------------------------ #

    def _prefix_token_count(self) -> int:
        return 0  # SigLIP has no CLS token; attention pooling produces the embedding

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        from transformers import SiglipModel
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        backbone = cast(SiglipModel, self.backbone)
        outputs = cast(
            BaseModelOutputWithPooling,
            backbone.get_image_features(pixel_values=pixel_values, output_hidden_states=output_hidden_states),
        )
        hidden = tuple(outputs.hidden_states) if output_hidden_states and outputs.hidden_states is not None else None
        return BackboneResult(
            last_hidden_state=cast(torch.Tensor, outputs.last_hidden_state),
            pooled=cast(torch.Tensor, outputs.pooler_output),
            hidden_states=hidden,
            raw=outputs,
        )

    # ------------------------------------------------------------------ #
    # Text tower and similarity
    # ------------------------------------------------------------------ #

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalized text embeddings [B, D] from the MedSigLIP text tower."""
        from transformers import SiglipModel
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        backbone = cast(SiglipModel, self.backbone)
        outputs = cast(
            BaseModelOutputWithPooling,
            backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
        )
        return torch.nn.functional.normalize(cast(torch.Tensor, outputs.pooler_output), dim=-1)

    def encode_image_normalized(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized image embeddings [B, D] (retrieval/contrastive input)."""
        from transformers import SiglipModel
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        backbone = cast(SiglipModel, self.backbone)
        outputs = cast(
            BaseModelOutputWithPooling,
            backbone.get_image_features(pixel_values=pixel_values),
        )
        return torch.nn.functional.normalize(cast(torch.Tensor, outputs.pooler_output), dim=-1)

    def image_text_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Normalized image-text similarity [B_img, B_txt].

        Uses the backbone's learned temperature and bias (SigLIP sigmoid-loss
        parameterization): ``exp(logit_scale) * cos(img, txt) + logit_bias``.
        """
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        backbone = cast(Any, self.backbone)
        scale = cast(torch.Tensor, backbone.logit_scale).exp()
        bias = cast(torch.Tensor, backbone.logit_bias)
        return scale * (image_embeddings @ text_embeddings.transpose(0, 1)) + bias

    # ------------------------------------------------------------------ #
    # Checkpoint config record
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict[str, Any]:
        base = super()._config_dict()
        base["hf_config"] = self._hf_config_dict
        return base
