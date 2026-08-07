"""H-Optimus-0 adapter: pathology tile encoder (timm ViT-g/14, shared w/ P08).

H-Optimus-0 (Bioptimus) is a ~1.1 B-parameter ViT-g/14 trained on pathology
tiles, published on the HF hub via timm (``bioptimus/H-optimus-0``,
``timm.create_model("hf-hub:bioptimus/H-optimus-0", ...)``). This adapter is
shared between Phase 06 (2D tile encoding) and Phase 08 (WSI/MIL): here it
delivers the tile-level encoder contract; Phase 08 adds slide aggregation.

Capability facts (declared, verified against the upstream architecture):

- pooled output: the CLS token (tile embedding) — the primary surface for
  classification and MIL aggregation;
- patch tokens: 224 / patch 14 -> 16x16 = 256 spatial tokens of embed 1536,
  row-major — used by dense tile tasks and bridging;
- intermediate hidden states: timm ``get_intermediate_layers`` patch tokens
  (no CLS) feed the feature-map pyramid, shallow -> deep, deepest last;
- token coordinates: NORMALIZED_IMAGE patch centers.

Default loading mode is **frozen BF16** (per the phase plan); embeddings are
cheap to cache, and :meth:`generate_embedding_cache` writes them with full
model/preprocess metadata so a WSI pipeline can skip the backbone entirely.
LoRA is gated: :meth:`inject_lora` raises until a frozen baseline has been
accepted and measured (:meth:`accept_frozen_baseline`) — ADR 0002 orders
frozen extraction before LoRA, and H-Optimus is large enough that this gate
is enforced, not advisory.

``timm`` is an optional dependency (``medfm[hf]`` extra). Without it the
adapter raises :class:`OptionalDependencyError` at construction, and the
registry keeps H-Optimus BLOCKED on license (gated repository; acceptance
required) regardless — Phase 08 owns the real-checkpoint acceptance.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderCapabilities, OutputSpec
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError
from medfm.core.serialization import canonical_dtype_name
from medfm.models.visual.base import (
    AdapterPreprocess,
    BackboneResult,
    BaseVisualAdapter2D,
    LoRAGateError,
    LoraTargetSpec,
    OptionalDependencyError,
)

logger = logging.getLogger(__name__)


class _TimmBackbone(Protocol):
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

    def get_intermediate_layers(self, pixel_values: torch.Tensor, n: tuple[int, ...]) -> tuple[torch.Tensor, ...]: ...


#: Pinned upstream revision (bioptimus/H-optimus-0 on the HF hub).
HOPTIMUS_MODEL_ID = "h-optimus-0"
HOPTIMUS_REPOSITORY = "https://huggingface.co/bioptimus/H-optimus-0"
HOPTIMUS_REVISION = "b145cc1e6c6b30d3251aa8b1f844e6974188a743"

#: Declared preprocessing: 224x224 RGB pathology tiles, ImageNet normalization
#: (H-Optimus is trained with the standard timm ImageNet statistics).
HOPTIMUS_PREPROCESS = AdapterPreprocess(
    image_size=(224, 224),
    channels=3,
    patch_size=14,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    value_range=(0.0, 1.0),
    resize_policy="stretch",
    color_space="RGB",
)

#: Hidden-state hook layers for the 40-layer tower (shallow -> deep; deepest
#: last). Indexed into timm intermediate layers (0-based block index).
HOPTIMUS_FEATURE_MAP_LAYERS: tuple[int, ...] = (10, 20, 30, 39)

HOPTIMUS_MODALITIES = (Modality.PATHOLOGY_TILE,)

HOPTIMUS_LORA_TARGETS = (
    LoraTargetSpec(
        pattern=r"blocks\.\d+\.attn\.(qkv|proj)",
        reason="H-Optimus attention fused QKV and output projection",
    ),
    LoraTargetSpec(
        pattern=r"blocks\.\d+\.mlp\.(fc1|fc2)",
        reason="H-Optimus MLP projections",
    ),
)


def hoptimus_capabilities(model_id: str = HOPTIMUS_MODEL_ID) -> EncoderCapabilities:
    return EncoderCapabilities(
        model_id=model_id,
        modalities=HOPTIMUS_MODALITIES,
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=True,
        supports_token_coordinates=True,
        token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
    )


#: Tiny offline construction config (random weights; contract tests only).
TINY_HOPTIMUS_CONFIG: dict[str, Any] = {
    "arch": "vit_tiny_patch16_224",
    "img_size": 64,
    "num_classes": 0,
}
TINY_HOPTIMUS_PREPROCESS = AdapterPreprocess(
    image_size=(64, 64), channels=3, patch_size=16, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)


def _require_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise OptionalDependencyError(
            "timm is required for H-Optimus-0 (medfm[hf] extra); the adapter is unavailable without it"
        ) from exc
    return timm


class HOptimus0Adapter(BaseVisualAdapter2D):
    """Pathology tile encoder over a timm ViT-g/14 backbone."""

    def __init__(
        self,
        *,
        model_id: str = HOPTIMUS_MODEL_ID,
        revision: str = HOPTIMUS_REVISION,
        timm_model_name: str = "hf-hub:bioptimus/H-optimus-0",
        pretrained: bool = False,
        preprocess: AdapterPreprocess = HOPTIMUS_PREPROCESS,
        feature_map_layers: tuple[int, ...] = HOPTIMUS_FEATURE_MAP_LAYERS,
        load_dtype: str = "bfloat16",
        construction_seed: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            revision=revision,
            capabilities=hoptimus_capabilities(model_id),
            preprocess=preprocess,
            feature_map_layers=feature_map_layers,
            lora_targets=HOPTIMUS_LORA_TARGETS,
            construction_seed=construction_seed,
        )
        self._timm_model_name = timm_model_name
        self._load_dtype = load_dtype
        self._frozen_baseline_accepted = False
        self._frozen_baseline_peak_bytes: int | None = None
        self.backbone = self._build_backbone(pretrained)
        self.to(dtype=torch.bfloat16 if load_dtype == "bfloat16" else torch.float32)
        self.freeze_backbone()  # frozen extraction is the default mode
        self.eval()

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        timm = _require_timm()
        if self._construction_seed is not None:
            torch.manual_seed(self._construction_seed)
        img_size = self._preprocess.image_size[0]
        model = timm.create_model(
            self._timm_model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )
        return cast(nn.Module, model)

    @classmethod
    def build_tiny(cls, *, model_id: str = "h-optimus-0-tiny", construction_seed: int = 0) -> HOptimus0Adapter:
        """Offline tiny ViT for contract/smoke tests (no network, no hub)."""
        return cls(
            model_id=model_id,
            revision="local-tiny",
            timm_model_name=TINY_HOPTIMUS_CONFIG["arch"],
            pretrained=False,
            preprocess=TINY_HOPTIMUS_PREPROCESS,
            feature_map_layers=(1, 3, 5, 11),
            load_dtype="float32",
            construction_seed=construction_seed,
        )

    # ------------------------------------------------------------------ #
    # Backbone contract
    # ------------------------------------------------------------------ #

    def _prefix_token_count(self) -> int:
        return 1  # timm ViT: one CLS token leads the feature sequence

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        pixel_values = pixel_values.to(dtype=self.compute_dtype())
        backbone = cast(_TimmBackbone, self.backbone)
        features = backbone.forward_features(pixel_values)  # [B, 1+N, D]
        pooled = features[:, 0, :]
        hidden: tuple[torch.Tensor, ...] | None = None
        if output_hidden_states and self._feature_map_layers:
            # timm intermediate layers return patch tokens only (no CLS).
            hidden = backbone.get_intermediate_layers(pixel_values, n=self._feature_map_layers)
        return BackboneResult(last_hidden_state=features, pooled=pooled, hidden_states=hidden, raw=None)

    def _build_feature_maps(self, result: BackboneResult, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # timm intermediate layers already exclude the CLS token, so the hook
        # states are used as-is (no prefix strip).
        rows, cols = self._preprocess.patch_grid
        hidden = result.hidden_states
        if hidden is None or len(hidden) != len(self._feature_map_layers):
            from medfm.core.errors import UnsupportedCapabilityError

            raise UnsupportedCapabilityError(
                f"{self._model_id} feature-map hooks require {len(self._feature_map_layers)} intermediate "
                f"layers; got {len(hidden or ())}"
            )
        maps = []
        for layer_tokens in hidden:
            b, n, d = layer_tokens.shape
            maps.append(layer_tokens.transpose(1, 2).reshape(b, d, rows, cols).contiguous())
        return tuple(maps)

    # ------------------------------------------------------------------ #
    # LoRA gate (frozen-baseline-first, ADR 0002)
    # ------------------------------------------------------------------ #

    def accept_frozen_baseline(self, *, measured_peak_bytes: int) -> None:
        """Record an accepted frozen-extraction baseline with its memory cost."""
        if measured_peak_bytes <= 0:
            raise ShapeContractError("measured_peak_bytes must be positive")
        self._frozen_baseline_accepted = True
        self._frozen_baseline_peak_bytes = measured_peak_bytes

    def check_lora_allowed(self) -> None:
        if not self._frozen_baseline_accepted:
            raise LoRAGateError(
                f"{self._model_id} LoRA is gated behind an accepted frozen baseline (ADR 0002). Run the frozen "
                "extraction baseline, measure peak memory, then call accept_frozen_baseline(measured_peak_bytes=...)."
            )

    # ------------------------------------------------------------------ #
    # Embedding cache (frozen-mode artifact for WSI pipelines, Phase 08)
    # ------------------------------------------------------------------ #

    def generate_embedding_cache(self, batches: list[MedicalBatch], out_dir: str | Path) -> Path:
        """Write frozen pooled embeddings with full model/preprocess metadata.

        Each batch's pooled embeddings are concatenated into one safetensors
        payload plus a metadata sidecar recording the model id, pinned
        revision, preprocess spec hash, dtype, coordinate system, and sample
        ids — everything a downstream WSI pipeline needs to trust the cache.
        """
        from safetensors.torch import save_file

        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        spec = OutputSpec(pooled=True)
        embeddings: list[torch.Tensor] = []
        sample_ids: list[str] = []
        for batch in batches:
            output = self.extract(batch, output_spec=spec)
            assert output.pooled_embedding is not None
            embeddings.append(output.pooled_embedding.detach().cpu())
            sample_ids.extend(batch.sample_ids)
        pooled = torch.cat(embeddings, dim=0)
        save_file({"pooled_embedding": pooled}, str(directory / "embeddings.safetensors"))
        metadata = {
            "artifact_type": "frozen_embedding_cache",
            "model_id": self._model_id,
            "revision": self._revision,
            "load_dtype": self._load_dtype,
            "embedding_dtype": canonical_dtype_name(pooled.dtype),
            "embedding_dim": int(pooled.shape[1]),
            "num_samples": len(sample_ids),
            "sample_ids": sample_ids,
            "preprocess": self._preprocess.to_dict(),
            "preprocess_spec_hash": self._preprocess.registry_spec(self._model_id).spec_hash(),
            "token_coordinate_system": CoordinateSystem.NORMALIZED_IMAGE.value,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return directory

    # ------------------------------------------------------------------ #
    # Checkpoint config record
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict[str, Any]:
        base = super()._config_dict()
        base["timm_model_name"] = self._timm_model_name
        base["load_dtype"] = self._load_dtype
        return base
