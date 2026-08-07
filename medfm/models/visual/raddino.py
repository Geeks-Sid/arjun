"""RAD-DINO adapter: DINOv2 ViT-B/14 distilled for chest radiography.

RAD-DINO (Microsoft) is a DINOv2-family ViT-B/14 adapted to chest X-rays.
Registry facts pinned at revision ``110cbc18...`` (config.json +
preprocessor_config.json verified against the hub):

- architecture: ``Dinov2Model`` — no register tokens in the pinned config, so
  the token layout is 1 CLS + 1369 patch tokens at 518x518 (37x37 grid,
  patch 14);
- pooled output: the CLS token (classification/retrieval);
- dense patch features: spatial tokens for retrieval/bridging/segmentation;
- preprocessing (declared, never executed here): resize shortest edge 518 +
  center crop 518, RGB (single-channel CXR is repeated to 3 channels by the
  Phase 04 radiology2d pipeline), rescale to [0, 1], mean/std 0.5307/0.2583
  per channel.

Hidden-state hooks (pinned to the revision in the registry record and the
checkpoint manifest): layers (3, 6, 9, 12) of the 12-layer tower feed the
feature-map pyramid for segmentation decoders — ordered shallow -> deep,
deepest last. If the upstream revision ever changes depth or register-token
count, ``_prefix_token_count`` and the hook layers must be re-verified; the
adapter validates the patch-token count against the declared grid at every
encode, so a layout drift fails loudly instead of silently.

Supported representations: pooled (classification/retrieval), spatial tokens
+ feature maps (segmentation, bridging), hidden states (hooks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medfm.core.encoder import EncoderCapabilities
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import UnsupportedCapabilityError
from medfm.models.visual.base import AdapterPreprocess
from medfm.models.visual.hf_generic import GenericHFVisionAdapter

#: Pinned upstream revision (microsoft/rad-dino on the HF hub).
RADDINO_MODEL_ID = "rad-dino"
RADDINO_REPOSITORY = "https://huggingface.co/microsoft/rad-dino"
RADDINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"

#: Declared preprocessing, verified against the pinned preprocessor_config.json.
RADDINO_PREPROCESS = AdapterPreprocess(
    image_size=(518, 518),
    channels=3,
    patch_size=14,
    mean=(0.5307, 0.5307, 0.5307),
    std=(0.2583, 0.2583, 0.2583),
    value_range=(0.0, 1.0),
    resize_policy="center_crop",
    color_space="GRAYSCALE_REPEATED_TO_RGB",
)

#: Hidden-state hook layers for the 12-layer tower (shallow -> deep; deepest
#: last). Indexed into the HF hidden_states tuple (index 0 = embeddings).
RADDINO_FEATURE_MAP_LAYERS: tuple[int, ...] = (3, 6, 9, 12)

RADDINO_MODALITIES = (Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE)


def raddino_capabilities(model_id: str = RADDINO_MODEL_ID) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=RADDINO_MODALITIES,
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
    )


#: Tiny offline construction config (random weights; contract tests only).
TINY_RADDINO_CONFIG: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "image_size": 32,
    "patch_size": 16,
}
TINY_RADDINO_PREPROCESS = AdapterPreprocess(
    image_size=(32, 32), channels=3, patch_size=16, mean=(0.5307, 0.5307, 0.5307), std=(0.2583, 0.2583, 0.2583)
)


class RADDINOAdapter(GenericHFVisionAdapter):
    """RAD-DINO: DINOv2 ViT-B/14 with chest-X-ray preprocessing declared."""

    def __init__(
        self,
        *,
        model_id: str = RADDINO_MODEL_ID,
        revision: str = RADDINO_REVISION,
        hf_config: dict[str, Any],
        preprocess: AdapterPreprocess = RADDINO_PREPROCESS,
        feature_map_layers: tuple[int, ...] = RADDINO_FEATURE_MAP_LAYERS,
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            family="dinov2",
            hf_config=hf_config,
            capabilities=raddino_capabilities(model_id),
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
            construction_seed=construction_seed,
        )

    @classmethod
    def from_pretrained_dir(
        cls,
        directory: str | Path,
        *,
        model_id: str = RADDINO_MODEL_ID,
        revision: str = RADDINO_REVISION,
    ) -> RADDINOAdapter:
        """Load the pinned RAD-DINO checkpoint from a local directory.

        The hub repository ships extra files (``dino_head.safetensors``,
        ``backbone_compatible.safetensors``, training configs); only the
        transformers-compatible ``model.safetensors`` + ``config.json`` are
        used — Phase 05's download allowlist keeps the directory to the
        transformers file set.
        """
        from transformers import AutoConfig, AutoModel

        path = Path(directory)
        hf_config_dict = AutoConfig.from_pretrained(path).to_dict()
        adapter = cls(
            model_id=model_id,
            revision=revision,
            hf_config=hf_config_dict,
        )
        weights_model = AutoModel.from_pretrained(path, attn_implementation="sdpa")
        adapter.backbone.load_state_dict(weights_model.state_dict(), strict=True)
        del weights_model
        adapter.eval()
        return adapter

    @classmethod
    def build_tiny(
        cls, *, model_id: str = "rad-dino-tiny", construction_seed: int = 0, revision: str = "local-tiny"
    ) -> RADDINOAdapter:
        """Offline tiny instance for contract/smoke tests (no network)."""
        return cls(
            model_id=model_id,
            revision=revision,
            hf_config=TINY_RADDINO_CONFIG,
            preprocess=TINY_RADDINO_PREPROCESS,
            feature_map_layers=(1, 2, 3, 4),
            construction_seed=construction_seed,
        )

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> RADDINOAdapter:
        """Rebuild from a serialized manifest config (tiny path only)."""
        if config.get("construction_seed") is None:
            raise UnsupportedCapabilityError("from_config_dict rebuild requires a recorded construction_seed")
        return cls(
            model_id=str(config["model_id"]),
            revision=str(config["revision"]),
            hf_config=dict(config["hf_config"]),
            preprocess=AdapterPreprocess.from_dict(config["preprocess"]),
            feature_map_layers=tuple(int(i) for i in config.get("feature_map_layers", ())),
            construction_seed=config["construction_seed"],
        )
