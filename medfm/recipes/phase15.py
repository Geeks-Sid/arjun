"""Bounded-memory pathology tile, WSI, VLM, and segmentation recipes.

Phase 15 keeps the expensive part of WSI processing explicit: tiles are read
and encoded in bounded chunks, embeddings are versioned in the Phase 08 HDF5
store, and slide batches carry masks plus level-0 coordinates.  The offline
recipes in this module use deterministic local networks and synthetic data as
contract fixtures only.  Production checkpoints remain registry/license
 gated and are never downloaded implicitly by a recipe builder.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.core.enums import CoordinateSystem, Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.language import LanguageOutput, ProjectedVisualTokens, TokenizedText
from medfm.core.task import LossOutput
from medfm.evaluation.metrics import MetricValue, classification_metrics, segmentation_metrics
from medfm.evaluation.report import EvaluationArtifact, make_artifact
from medfm.models.bridges import (
    CoordinateAwareBridge,
    LinearVisionLanguageBridge,
    PerceiverResamplerBridge,
    WSICoordinateEncoder,
)
from medfm.models.decoders import UNetDecoder2D
from medfm.models.heads.classification import LinearClassificationHead, MLPClassificationHead
from medfm.models.language.base import GenericHFCausalLMAdapter
from medfm.models.pathology import (
    AttentionMILAggregator,
    EmbeddingStore,
    GigaPathFlashAggregator,
    MeanPoolingAggregator,
    PathologyTileEncoder,
    TileEmbeddingMetadata,
    TITANAggregator,
    TokenBudget,
    WSITokenSelector,
)
from medfm.models.pathology.selection import (
    DiversityTileSampler,
    GridTileSampler,
    MultiResolutionTileSampler,
    QualityWeightedTileSampler,
    RandomTileSampler,
    TextConditionedTileSampler,
    TopKAttentionTileSampler,
)
from medfm.peft import LoRAConfig, inject_language_lora, inject_lora
from medfm.recipes.pathology_stitching import (
    COORDINATE_SYSTEM,
    EVIDENCE_SCHEMA_VERSION,
    StitchedSlide,
    TilePrediction,
    evidence_json_is_valid,
    evidence_payload,
    evidence_tiles_from_scores,
    evidence_tiles_to_json,
    extract_evidence_tiles,
    level_to_level0_geometry,
    make_evidence_tiles,
    map_evidence_coordinates,
    map_normalized_coordinates,
    map_normalized_coordinates_to_wsi,
    normalize_level0_geometry,
    normalized_to_level0_geometry,
    serialize_evidence_json,
    stitch_predictions,
    stitch_tile_predictions,
    stitch_wsi_predictions,
    validate_evidence_json,
    validate_evidence_tiles,
)
from medfm.tasks.base import (
    TaskModuleBase,
    detached_count_tensor,
    target_from_batch,
    valid_sample_count,
    valid_sample_mask,
)
from medfm.tasks.classification import BinaryClassificationTask, ClassificationTask, MultiLabelClassificationTask
from medfm.tasks.segmentation import BinarySegmentationTask
from medfm.training.backend import AcceleratorBackend
from medfm.training.config import RunConfig
from medfm.training.optimizer import OptimizerBundle, build_optimizer
from medfm.training.pipeline import ComponentBuilders
from medfm.training.steps import make_training_step
from medfm.training.trainer import Trainer

PHASE15_RECIPE_VERSION = "phase15-1"
PATHOLOGY_MODALITIES = (Modality.PATHOLOGY_TILE, Modality.PATHOLOGY_WSI)
PATHOLOGY_LORA_TARGETS = (r"late_block",)
LANGUAGE_LORA_TARGETS = (r"layers\.\d+\.self_attn\.out_proj", r"layers\.\d+\.linear[12]")
_ALLOWED_FAMILIES = {"tile_classification", "wsi_classification", "wsi_vlm", "pathology_segmentation"}
_ALLOWED_STAGES = {"1", "2", "3", "4"}
_ALLOWED_AGGREGATORS = {"mean", "attention_mil", "gated_attention_mil", "transformer", "gigapath_flash", "titan"}
_ALLOWED_BRIDGES = {"linear", "perceiver"}
_ALLOWED_BLEND_MODES = {"constant", "gaussian"}


class PathologyRecipeConfigurationError(ValueError):
    """A Phase 15 recipe is malformed or requests a gated production path."""


# Compatibility spelling used by recipe clients.
RecipeConfigurationError = PathologyRecipeConfigurationError


@dataclass(frozen=True)
class PatientDisjointSplit:
    """Deterministic slide IDs split without crossing patient boundaries."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    patient_by_slide: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "patient_by_slide": dict(self.patient_by_slide),
            "patient_disjoint": True,
        }

    def all_slides(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test


@dataclass(frozen=True)
class SelectedWSITokens:
    """Fixed-width cached tile tokens and their source evidence metadata."""

    tokens: torch.Tensor
    mask: torch.Tensor
    coordinates: torch.Tensor
    records: tuple[tuple[Any, ...], ...]
    indices: tuple[tuple[int, ...], ...]
    actual_counts: tuple[int, ...]

    @property
    def visual_tokens(self) -> torch.Tensor:
        return self.tokens

    @property
    def visual_token_mask(self) -> torch.Tensor:
        return self.mask


@dataclass(frozen=True)
class PathologyRecipeMetadata:
    """Pinned recipe choices and bounded-memory observability fields."""

    family: str
    recipe_id: str
    backbone: str
    modality: str
    stage: str
    mode: str
    dataset_id: str
    dataset_revision: str
    preprocessing_revision: str
    model_revision: str
    embedding_store_schema: int
    embedding_store_revision: str
    tile_shape: tuple[int, int, int]
    tile_shape_buckets: tuple[tuple[int, int, int], ...]
    tile_mpp: float
    magnification: str
    max_tiles_per_slide: int
    sampled_tile_counts: tuple[int, ...]
    visual_token_count: int | None
    precompression_tile_count: int | None
    text_token_count: int | None
    selector: str | None
    selector_revision: str | None
    train_selector: str | None
    eval_selector: str | None
    aggregator: str | None
    bridge_type: str | None
    microbatch_per_device: int
    world_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    memory_cap_gb: float
    tpu_status: str
    shard_unit: str
    split_policy: str
    cache_embeddings: bool
    cache_tokens: bool
    actual_tile_count_logging: bool
    backend_observability: Mapping[str, Any]
    baseline: str
    task_name: str
    slide_reader_revision: str = "phase03-slide-reader-v1"
    tile_index_revision: str = "phase04-tile-index-v1"
    failure_rates: tuple[tuple[str, float], ...] = ()
    evidence_coordinate_system: str = COORDINATE_SYSTEM
    evidence_schema_version: int = EVIDENCE_SCHEMA_VERSION
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": PHASE15_RECIPE_VERSION,
            "family": self.family,
            "recipe_id": self.recipe_id,
            "backbone": self.backbone,
            "modality": self.modality,
            "stage": self.stage,
            "mode": self.mode,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "preprocessing_revision": self.preprocessing_revision,
            "model_revision": self.model_revision,
            "embedding_store": {
                "schema_version": self.embedding_store_schema,
                "revision": self.embedding_store_revision,
                "versioned": True,
            },
            "tile_shape": list(self.tile_shape),
            "tile_shape_buckets": [list(value) for value in self.tile_shape_buckets],
            "tile_mpp": self.tile_mpp,
            "magnification": self.magnification,
            "max_tiles_per_slide": self.max_tiles_per_slide,
            "sampled_tile_counts": list(self.sampled_tile_counts),
            "visual_token_count": self.visual_token_count,
            "precompression_tile_count": self.precompression_tile_count,
            "text_token_count": self.text_token_count,
            "selector": self.selector,
            "selector_revision": self.selector_revision,
            "selection_behavior": {
                "train": self.train_selector,
                "evaluation": self.eval_selector,
                "evaluation_deterministic": True,
            },
            "aggregator": self.aggregator,
            "bridge_type": self.bridge_type,
            "global_batch_semantics": {
                "microbatch_per_device": self.microbatch_per_device,
                "world_size": self.world_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "global_batch_size": self.global_batch_size,
                "formula": "microbatch_per_device * world_size * gradient_accumulation_steps",
            },
            "memory_cap_gb": self.memory_cap_gb,
            "tpu_status": self.tpu_status,
            "shard_unit": self.shard_unit,
            "split_policy": self.split_policy,
            "cache_embeddings": self.cache_embeddings,
            "cache_tokens": self.cache_tokens,
            "actual_tile_count_logging": self.actual_tile_count_logging,
            "backend_observability": dict(self.backend_observability),
            "baseline": self.baseline,
            "slide_reader_revision": self.slide_reader_revision,
            "tile_index_revision": self.tile_index_revision,
            "failure_rates": {name: value for name, value in self.failure_rates},
            "task_name": self.task_name,
            "evidence": {
                "coordinate_system": self.evidence_coordinate_system,
                "schema_version": self.evidence_schema_version,
            },
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class PathologyRecipeBuild:
    """Direct factory result for tests and offline acceptance tooling."""

    model: nn.Module
    task: nn.Module
    train_data: list[MedicalBatch]
    metadata: PathologyRecipeMetadata


RecipeBuild = PathologyRecipeBuild


@dataclass(frozen=True)
class _Phase15Options:
    family: str
    recipe_id: str
    backbone: str
    modality: Modality
    mode: str
    stage: str
    offline_tiny: bool
    embedding_dim: int
    hidden_size: int
    channels: int
    tile_height: int
    tile_width: int
    tile_shape_buckets: tuple[tuple[int, int, int], ...]
    max_tiles_per_slide: int
    tile_count_buckets: tuple[int, ...]
    visual_token_count: int
    precompression_tile_count: int
    text_token_count: int
    selector: str
    train_selector: str
    eval_selector: str
    selector_revision: str
    aggregator: str
    bridge_type: str
    cache_embeddings: bool
    cache_tokens: bool
    dataset_id: str
    dataset_revision: str
    preprocessing_revision: str
    model_revision: str
    slide_reader_revision: str
    tile_index_revision: str
    embedding_store_revision: str
    tile_mpp: float
    magnification: str
    slide_height: int
    slide_width: int
    tpu_status: str
    baseline: str
    split_policy: str
    task_name: str
    construction_seed: int
    use_lora: bool
    text_alignment: bool
    blend_mode: str
    overlap: float
    evidence_top_k: int


def _canonical_family(value: Any) -> str:
    family = str(value or "wsi_classification").strip().lower().replace("-", "_")
    aliases = {
        "tile_cls": "tile_classification",
        "tile_classification_recipe": "tile_classification",
        "pathology_tile_classification": "tile_classification",
        "classification_tile": "tile_classification",
        "slide_classification": "wsi_classification",
        "pathology_wsi_classification": "wsi_classification",
        "wsi": "wsi_classification",
        "slide_vlm": "wsi_vlm",
        "pathology_wsi_vlm": "wsi_vlm",
        "vlm": "wsi_vlm",
        "segmentation": "pathology_segmentation",
        "tiled_segmentation": "pathology_segmentation",
        "pathology_tile_segmentation": "pathology_segmentation",
    }
    family = aliases.get(family, family)
    if family not in _ALLOWED_FAMILIES:
        raise PathologyRecipeConfigurationError(f"unknown Phase 15 recipe family {family!r}")
    return family


def _stage(value: Any) -> str:
    raw = str(value or "1").strip().upper().replace("STAGE", "")
    aliases = {"A": "1", "B": "2", "C": "3", "D": "4"}
    raw = aliases.get(raw, raw)
    if raw not in _ALLOWED_STAGES:
        raise PathologyRecipeConfigurationError("Phase 15 stage must be 1/2/3/4 or A/B/C/D")
    return raw


def _shape(value: Any, name: str, *, channels: int) -> tuple[int, int, int]:
    if isinstance(value, Mapping):
        value = value.get("shape", value.get("tile", value.get("image")))
    if not isinstance(value, (list, tuple)):
        raise PathologyRecipeConfigurationError(f"{name} must be a two- or three-element shape")
    if len(value) == 2:
        result = (channels, int(value[0]), int(value[1]))
    elif len(value) == 3:
        result = tuple(int(item) for item in value)
    else:
        raise PathologyRecipeConfigurationError(f"{name} must be a two- or three-element shape")
    if any(item <= 0 for item in result):
        raise PathologyRecipeConfigurationError(f"{name} must contain positive dimensions")
    return result


def _parse_tile_buckets(raw: Any, fallback: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    if raw is None:
        return (fallback,)
    values: list[tuple[int, int, int]] = []
    if isinstance(raw, Mapping):
        raw = raw.get("tile", raw.get("image", raw.get("2d", ())))
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, (int, float)) for item in raw):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            try:
                values.append(_shape(entry, "tile shape bucket", channels=fallback[0]))
            except (TypeError, ValueError, PathologyRecipeConfigurationError):
                continue
    return tuple(dict.fromkeys(values)) or (fallback,)


def _parse_count_buckets(raw: Any, fallback: int) -> tuple[int, ...]:
    if raw is None:
        return (fallback,)
    values: list[int] = []
    if isinstance(raw, (int, float)):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, Mapping):
                item = item.get("count", item.get("tiles"))
            try:
                count = int(item)
            except (TypeError, ValueError):
                continue
            if count > 0:
                values.append(count)
    return tuple(sorted(set(values))) or (fallback,)


def _options(config: RunConfig) -> _Phase15Options:
    raw = dict(config.recipe)
    family = _canonical_family(raw.get("family", raw.get("type", "wsi_classification")))
    default_modality = (
        Modality.PATHOLOGY_TILE
        if family in {"tile_classification", "pathology_segmentation"}
        else Modality.PATHOLOGY_WSI
    )
    try:
        modality = Modality(str(raw.get("modality", default_modality.value)).upper())
    except ValueError as exc:
        raise PathologyRecipeConfigurationError(f"unknown pathology modality {raw.get('modality')!r}") from exc
    expected = (
        Modality.PATHOLOGY_TILE
        if family in {"tile_classification", "pathology_segmentation"}
        else Modality.PATHOLOGY_WSI
    )
    if modality is not expected:
        raise PathologyRecipeConfigurationError(f"{family} requires modality {expected.value}")
    stage = _stage(raw.get("stage", raw.get("stage_number", "1")))
    mode = str(raw.get("mode", "offline_tiny" if raw.get("offline_tiny", True) else "production")).lower()
    offline = bool(raw.get("offline_tiny", mode in {"offline_tiny", "smoke", "tiny", "contract"}))
    channels = int(raw.get("channels", 3))
    if channels != 3:
        raise PathologyRecipeConfigurationError("pathology recipes require RGB tiles with channels=3")
    default_shape = (channels, 16 if offline else 224, 16 if offline else 224)
    tile_shape = _shape(raw.get("tile_shape", raw.get("image_shape", default_shape)), "tile_shape", channels=channels)
    if offline:
        # Offline fixtures intentionally remain small while retaining the
        # production shape contract in non-offline profiles.
        tile_shape = (channels, min(tile_shape[1], 32), min(tile_shape[2], 32))
    tile_buckets = _parse_tile_buckets(raw.get("tile_shape_buckets", raw.get("shape_buckets")), tile_shape)
    if offline:
        tile_buckets = (tile_shape,)
    embedding_dim = int(raw.get("embedding_dim", 32 if offline else 1536))
    hidden_size = int(raw.get("hidden_size", 32 if offline else 256))
    if embedding_dim <= 0 or hidden_size <= 0:
        raise PathologyRecipeConfigurationError("embedding_dim and hidden_size must be positive")
    max_tiles = int(raw.get("max_tiles_per_slide", raw.get("max_tiles", 8 if offline else 256)))
    if max_tiles <= 0:
        raise PathologyRecipeConfigurationError("max_tiles_per_slide must be positive")
    count_buckets = _parse_count_buckets(raw.get("tile_count_buckets", raw.get("tile_buckets")), max_tiles)
    if max_tiles not in count_buckets:
        count_buckets = tuple(sorted(set(count_buckets + (max_tiles,))))
    visual_default = 32 if offline else 64
    visual_tokens = int(raw.get("visual_tokens", raw.get("visual_token_count", visual_default)))
    if visual_tokens not in (32, 64, 128):
        raise PathologyRecipeConfigurationError("visual token buckets are exactly 32, 64, or 128")
    precompression = int(raw.get("precompression_tiles", raw.get("precompression", max(128, visual_tokens))))
    if not 128 <= precompression <= 1024:
        raise PathologyRecipeConfigurationError("precompression tile budget must be in [128, 1024]")
    text_tokens = int(raw.get("text_tokens", raw.get("text_token_count", 8 if offline else 512)))
    if text_tokens <= 0:
        raise PathologyRecipeConfigurationError("text token bucket must be positive")
    selector = str(raw.get("selector", "grid")).lower().replace("-", "_")
    train_selector = str(raw.get("train_selector", selector)).lower().replace("-", "_")
    eval_selector = str(raw.get("eval_selector", "grid")).lower().replace("-", "_")
    allowed_selectors = {
        "grid",
        "random",
        "seeded_random",
        "quality",
        "quality_weighted",
        "diversity",
        "topk",
        "topk_attention",
        "text",
        "text_conditioned",
        "multiresolution",
    }
    if (
        selector not in allowed_selectors
        or train_selector not in allowed_selectors
        or eval_selector not in allowed_selectors
    ):
        raise PathologyRecipeConfigurationError(f"unknown pathology tile selector in {sorted(allowed_selectors)}")
    aggregator = str(raw.get("aggregator", raw.get("slide_aggregator", "mean"))).lower().replace("-", "_")
    if aggregator not in _ALLOWED_AGGREGATORS:
        raise PathologyRecipeConfigurationError(f"aggregator must be one of {sorted(_ALLOWED_AGGREGATORS)}")
    bridge = str(raw.get("bridge", raw.get("bridge_type", "perceiver"))).lower().replace("_resampler", "")
    if bridge not in _ALLOWED_BRIDGES:
        raise PathologyRecipeConfigurationError(f"bridge must be one of {sorted(_ALLOWED_BRIDGES)}")
    slide_height, slide_width = _shape(raw.get("slide_shape", (64, 64)), "slide_shape", channels=1)[1:]
    blend_mode = str(raw.get("blend_mode", "constant")).lower()
    if blend_mode not in _ALLOWED_BLEND_MODES:
        raise PathologyRecipeConfigurationError(f"blend_mode must be one of {sorted(_ALLOWED_BLEND_MODES)}")
    overlap = float(raw.get("overlap", raw.get("tile_overlap", 0.25)))
    if not 0.0 <= overlap < 1.0:
        raise PathologyRecipeConfigurationError("overlap must be in [0, 1)")
    mpp = float(raw.get("tile_mpp", raw.get("mpp", 0.5)))
    if not math.isfinite(mpp) or mpp <= 0:
        raise PathologyRecipeConfigurationError("tile_mpp must be positive and finite")
    task_name = str(raw.get("task", config.task.get("type", config.task.get("name", "BINARY_CLASSIFICATION")))).upper()
    return _Phase15Options(
        family=family,
        recipe_id=str(raw.get("id", raw.get("name", f"phase15-{family}"))),
        backbone=str(raw.get("backbone", "h-optimus-0")).lower(),
        modality=modality,
        mode=mode,
        stage=stage,
        offline_tiny=offline,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        channels=channels,
        tile_height=tile_shape[1],
        tile_width=tile_shape[2],
        tile_shape_buckets=tile_buckets,
        max_tiles_per_slide=max_tiles,
        tile_count_buckets=count_buckets,
        visual_token_count=visual_tokens,
        precompression_tile_count=precompression,
        text_token_count=text_tokens,
        selector=selector,
        train_selector=train_selector,
        eval_selector=eval_selector,
        selector_revision=str(raw.get("selector_revision", "phase15-selector-v1")),
        aggregator=aggregator,
        bridge_type=bridge,
        cache_embeddings=bool(
            raw.get("cache_embeddings", raw.get("cached_embeddings", family in {"wsi_classification", "wsi_vlm"}))
        ),
        cache_tokens=bool(raw.get("cache_tokens", raw.get("cached_tokens", False))),
        dataset_id=str(raw.get("dataset_id", config.dataset.get("id", "phase15-synthetic-pathology-v1"))),
        dataset_revision=str(raw.get("dataset_revision", config.dataset.get("revision", "synthetic-v1"))),
        preprocessing_revision=str(
            raw.get("preprocessing_revision", config.preprocessing_hash or "phase15-pathology-preprocess-v1")
        ),
        model_revision=str(
            raw.get(
                "model_revision", config.base_model_revision or raw.get("backbone_revision", "offline-random-contract")
            )
        ),
        embedding_store_revision=str(raw.get("embedding_store_revision", "phase08-hdf5-v1")),
        slide_reader_revision=str(raw.get("slide_reader_revision", "phase03-slide-reader-v1")),
        tile_index_revision=str(raw.get("tile_index_revision", "phase04-tile-index-v1")),
        tile_mpp=mpp,
        magnification=str(raw.get("magnification", "20x")),
        slide_height=slide_height,
        slide_width=slide_width,
        tpu_status=str(
            raw.get(
                "tpu_status",
                "cached_embeddings_static_aggregator"
                if family in {"wsi_classification", "wsi_vlm"}
                else "CPU_CONTRACT_ONLY",
            )
        ),
        baseline=str(
            raw.get("baseline", "hoptimus_frozen_linear" if family == "tile_classification" else "mean_pooling")
        ),
        split_policy=str(raw.get("split_policy", config.dataset.get("split_policy", "patient_disjoint"))),
        task_name=task_name,
        construction_seed=int(raw.get("construction_seed", config.seed)),
        use_lora=bool(raw.get("use_lora", config.peft.enabled or stage in {"2", "3", "4"})),
        text_alignment=bool(raw.get("text_alignment", stage == "4")),
        blend_mode=blend_mode,
        overlap=overlap,
        evidence_top_k=int(raw.get("evidence_top_k", 8)),
    )


def _selector(name: str) -> Any:
    normalized = str(name).lower().replace("-", "_")
    if normalized == "grid":
        return GridTileSampler()
    if normalized in {"random", "seeded_random"}:
        return RandomTileSampler()
    if normalized in {"quality", "quality_weighted"}:
        return QualityWeightedTileSampler()
    if normalized == "diversity":
        return DiversityTileSampler()
    if normalized in {"topk", "topk_attention"}:
        return TopKAttentionTileSampler()
    if normalized in {"text", "text_conditioned"}:
        return TextConditionedTileSampler()
    if normalized == "multiresolution":
        return MultiResolutionTileSampler()
    raise PathologyRecipeConfigurationError(f"unknown tile selector {name!r}")


def _records_for_slide(
    slide_id: str,
    count: int,
    *,
    options: _Phase15Options,
    start: int = 0,
) -> tuple[TileEmbeddingMetadata, ...]:
    records: list[TileEmbeddingMetadata] = []
    stride_x = max(1, options.tile_width - int(options.tile_width * options.overlap))
    stride_y = max(1, options.tile_height - int(options.tile_height * options.overlap))
    for index in range(count):
        grid_x = index % max(1, options.slide_width // stride_x)
        grid_y = index // max(1, options.slide_width // stride_x)
        x = min(grid_x * stride_x, max(0, options.slide_width - options.tile_width))
        y = min(grid_y * stride_y, max(0, options.slide_height - options.tile_height))
        records.append(
            TileEmbeddingMetadata(
                slide_id=slide_id,
                tile_id=f"{slide_id}-tile-{start + index}",
                x=x,
                y=y,
                width=options.tile_width,
                height=options.tile_height,
                level=0,
                mpp=options.tile_mpp,
                quality={"tissue_fraction": 0.8, "blur": float(1 + index % 4), "artifact": 0.0},
            )
        )
    return tuple(records)


def _synthetic_slide_payloads(options: _Phase15Options, *, batch_size: int) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for slide_index in range(max(1, int(batch_size))):
        slide_id = f"slide-{slide_index}"
        # Keep at least one padded entry so all mask paths are exercised.
        count = max(1, options.max_tiles_per_slide - (slide_index % min(3, options.max_tiles_per_slide)))
        generator = torch.Generator().manual_seed(options.construction_seed + 100 + slide_index)
        embeddings = torch.randn((count, options.embedding_dim), generator=generator)
        records = _records_for_slide(slide_id, count, options=options)
        payloads.append(
            {
                "slide_id": slide_id,
                "patient_id": f"patient-{slide_index}",
                "site": f"site-{slide_index % 2}",
                "scanner": f"scanner-{slide_index % 2}",
                "organ": "colon" if slide_index % 2 else "breast",
                "label": int(slide_index % 2),
                "embeddings": embeddings,
                "records": records,
            }
        )
    return tuple(payloads)


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def patient_disjoint_split(
    records: Sequence[Any] | Mapping[str, Any],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> PatientDisjointSplit:
    """Split slide records by patient, preserving all slide dependencies."""
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("train_fraction must be in (0,1) and validation_fraction in [0,1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if isinstance(records, Mapping):
        entries = []
        for slide_id, value in records.items():
            if isinstance(value, Mapping):
                entries.append({"slide_id": slide_id, **dict(value)})
            else:
                entries.append({"slide_id": slide_id, "patient_id": value})
    else:
        entries = list(records)
    slide_to_patient: dict[str, str] = {}
    patients: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        if isinstance(entry, Mapping):
            slide_id = str(entry.get("slide_id", entry.get("id", "")))
            patient_id = str(entry.get("patient_id", entry.get("patient", slide_id)))
        else:
            slide_id = str(getattr(entry, "slide_id", getattr(entry, "id", "")))
            patient_id = str(getattr(entry, "patient_id", getattr(entry, "patient", slide_id)))
        if not slide_id:
            raise ValueError("every split record must provide slide_id")
        slide_to_patient[slide_id] = patient_id
        patients[patient_id].append(slide_id)
    order = sorted(patients)
    random.Random(int(seed)).shuffle(order)
    total = len(order)
    train_count = max(1, int(round(total * train_fraction))) if total else 0
    validation_count = int(round(total * validation_fraction))
    if total >= 2 and train_count >= total:
        train_count = total - 1
    if total >= 3 and validation_count == 0:
        validation_count = 1
    if train_count + validation_count > total:
        validation_count = max(0, total - train_count)
    train_patients = set(order[:train_count])
    validation_patients = set(order[train_count : train_count + validation_count])
    test_patients = set(order[train_count + validation_count :])

    def slides(group: set[str]) -> tuple[str, ...]:
        return tuple(sorted(slide for patient in group for slide in patients[patient]))

    return PatientDisjointSplit(
        slides(train_patients), slides(validation_patients), slides(test_patients), slide_to_patient
    )


# Explicit aliases make the split policy discoverable to recipe callers.
make_patient_disjoint_splits = patient_disjoint_split
deterministic_patient_split = patient_disjoint_split


def _select_indices(
    records: Sequence[Any],
    budget: int,
    *,
    selector: str,
    embeddings: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    seed: int = 0,
) -> list[int]:
    return list(
        _selector(selector).select(
            records,
            budget,
            embeddings=embeddings,
            attention=attention,
            seed=seed,
        )
    )


def pad_slide_embeddings(
    slide_embeddings: Sequence[torch.Tensor],
    slide_records: Sequence[Sequence[Any]],
    *,
    max_tiles: int,
    selector: str = "grid",
    seed: int = 0,
) -> SelectedWSITokens:
    """Select bounded tiles, pad to one bucket, and retain coordinates."""
    if max_tiles <= 0:
        raise ValueError("max_tiles must be positive")
    if len(slide_embeddings) != len(slide_records):
        raise ValueError("slide embeddings and records must have equal batch length")
    if not slide_embeddings:
        raise ValueError("at least one slide is required")
    first = slide_embeddings[0]
    if first.ndim != 2:
        raise ValueError("slide embeddings must be [T,D]")
    dim = int(first.shape[1])
    tokens = torch.zeros((len(slide_embeddings), max_tiles, dim), dtype=first.dtype, device=first.device)
    mask = torch.zeros((len(slide_embeddings), max_tiles), dtype=torch.bool, device=first.device)
    coordinates = torch.zeros((len(slide_embeddings), max_tiles, 4), dtype=torch.int64, device=first.device)
    selected_records: list[tuple[Any, ...]] = []
    selected_indices: list[tuple[int, ...]] = []
    counts: list[int] = []
    for batch_index, (embeddings, records) in enumerate(zip(slide_embeddings, slide_records, strict=True)):
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != len(records) or int(embeddings.shape[1]) != dim:
            raise ValueError("each slide embedding matrix must align with its records and embedding dimension")
        indices = _select_indices(
            records, min(max_tiles, len(records)), selector=selector, embeddings=embeddings, seed=seed + batch_index
        )
        count = len(indices)
        if count:
            tokens[batch_index, :count] = embeddings[indices]
            mask[batch_index, :count] = True
            coordinates[batch_index, :count] = torch.tensor(
                [[int(_record_value(records[i], name)) for name in ("x", "y", "width", "height")] for i in indices],
                dtype=torch.int64,
                device=coordinates.device,
            )
        selected_records.append(tuple(records[i] for i in indices))
        selected_indices.append(tuple(indices))
        counts.append(count)
    return SelectedWSITokens(tokens, mask, coordinates, tuple(selected_records), tuple(selected_indices), tuple(counts))


def select_wsi_visual_tokens(
    slide_embeddings: Sequence[torch.Tensor],
    slide_records: Sequence[Sequence[Any]],
    *,
    budget: TokenBudget | None = None,
    selector: str = "grid",
    seed: int = 0,
) -> SelectedWSITokens:
    """Select/pad the fixed visual-token budget before bridge resampling."""
    resolved = budget or TokenBudget()
    # The precompression limit bounds candidate selection; visual_tokens is the
    # fixed output width consumed by the language bridge.
    capped_embeddings: list[torch.Tensor] = []
    capped_records: list[tuple[Any, ...]] = []
    for index, (embeddings, records) in enumerate(zip(slide_embeddings, slide_records, strict=True)):
        candidate = _select_indices(
            records,
            min(resolved.precompression, len(records)),
            selector=selector,
            embeddings=embeddings,
            seed=seed + index,
        )
        capped_embeddings.append(embeddings[candidate])
        capped_records.append(tuple(records[i] for i in candidate))
    return pad_slide_embeddings(
        capped_embeddings,
        capped_records,
        max_tiles=resolved.visual_tokens,
        selector="grid",
        seed=seed,
    )


# Descriptive aliases for selection code outside recipes.
bucket_slide_tiles = pad_slide_embeddings
select_cached_wsi_tokens = select_wsi_visual_tokens


class GatedAttentionMILAggregator(nn.Module):
    """Ilse-style gated attention MIL with explicit padding masks."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        hidden = int(hidden_dim or max(8, min(256, embedding_dim // 2)))
        self.embedding_dim = int(embedding_dim)
        self.value = nn.Linear(embedding_dim, hidden)
        self.gate = nn.Linear(embedding_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def aggregate(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        evidence_indices: tuple[tuple[int, ...], ...] = (),
        evidence_tiles: tuple[tuple[Any, ...], ...] = (),
    ) -> Any:
        if embeddings.ndim != 3 or int(embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"embeddings must be [B,T,{self.embedding_dim}]")
        valid = (
            torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
            if mask is None
            else mask.bool()
        )
        if valid.shape != embeddings.shape[:2] or not bool(valid.any(dim=1).all()):
            raise ValueError("mask must align with embeddings and every slide needs one real tile")
        logits = self.score(torch.tanh(self.value(embeddings)) * torch.sigmoid(self.gate(embeddings))).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
        pooled = (embeddings * attention.unsqueeze(-1)).sum(dim=1)
        from medfm.models.pathology.aggregation import SlideAggregation

        return SlideAggregation(
            pooled,
            attention=attention,
            valid_mask=valid,
            evidence_indices=evidence_indices,
            evidence_tiles=evidence_tiles,
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.aggregate(embeddings, mask).embedding


class TransformerSlideAggregator(nn.Module):
    """Bounded transformer slide path with a key-padding mask."""

    def __init__(self, embedding_dim: int, *, heads: int = 4, layers: int = 1) -> None:
        super().__init__()
        if embedding_dim <= 0 or heads <= 0 or embedding_dim % heads:
            raise ValueError("embedding_dim must be positive and divisible by heads")
        self.embedding_dim = int(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=max(embedding_dim * 2, 32),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)

    def aggregate(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        evidence_indices: tuple[tuple[int, ...], ...] = (),
        evidence_tiles: tuple[tuple[Any, ...], ...] = (),
    ) -> Any:
        if embeddings.ndim != 3 or int(embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"embeddings must be [B,T,{self.embedding_dim}]")
        valid = (
            torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
            if mask is None
            else mask.bool()
        )
        if valid.shape != embeddings.shape[:2] or not bool(valid.any(dim=1).all()):
            raise ValueError("mask must align with embeddings and every slide needs one real tile")
        encoded = self.norm(self.encoder(embeddings, src_key_padding_mask=~valid))
        weights = valid.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        from medfm.models.pathology.aggregation import SlideAggregation

        return SlideAggregation(
            pooled,
            valid_mask=valid,
            evidence_indices=evidence_indices,
            evidence_tiles=evidence_tiles,
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.aggregate(embeddings, mask).embedding


def build_slide_aggregator(name: str, embedding_dim: int) -> nn.Module:
    normalized = str(name).lower().replace("-", "_")
    if normalized == "mean":
        return MeanPoolingAggregator(embedding_dim)
    if normalized == "attention_mil":
        return AttentionMILAggregator(embedding_dim)
    if normalized == "gated_attention_mil":
        return GatedAttentionMILAggregator(embedding_dim)
    if normalized == "transformer":
        return TransformerSlideAggregator(embedding_dim)
    if normalized == "gigapath_flash":
        return GigaPathFlashAggregator(embedding_dim)
    if normalized == "titan":
        return TITANAggregator(embedding_dim)
    raise PathologyRecipeConfigurationError(f"unknown slide aggregator {name!r}")


class _TinyTileVision(nn.Module, PathologyTileEncoder):
    """Small deterministic RGB encoder for offline tile recipes."""

    def __init__(self, *, embedding_dim: int, hidden_size: int, seed: int, model_id: str = "pathology-tiny") -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.stem = nn.Sequential(
                nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.late_block = nn.Linear(hidden_size, embedding_dim)
        self.model_id = model_id
        self.revision = "offline-random-contract"
        self.preprocess_hash = hashlib.sha256(f"{model_id}:rgb:{seed}".encode()).hexdigest()[:16]
        self.embedding_dim = int(embedding_dim)
        self.dtype = torch.float32

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.ndim != 4 or int(tiles.shape[1]) != 3:
            raise ValueError("pathology tile encoder expects [T,3,H,W]")
        hidden = self.stem(tiles.float())
        return self.late_block(hidden.mean(dim=(-2, -1)))

    forward = encode_tiles


class _TileClassificationModel(nn.Module):
    def __init__(self, vision: _TinyTileVision, *, text_alignment: bool = False) -> None:
        super().__init__()
        self.vision = vision
        self.text_alignment = bool(text_alignment)

    def forward(self, batch: MedicalBatch) -> torch.Tensor | dict[str, torch.Tensor]:
        if batch.pixel_values is None:
            raise ShapeContractError("tile classification requires pixel_values")
        embeddings = self.vision.encode_tiles(batch.pixel_values)
        return {"embeddings": embeddings} if self.text_alignment else embeddings

    def forward_mode(
        self,
        batch: MedicalBatch,
        *,
        mode: str = "image",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if batch.pixel_values is None:
            raise ShapeContractError("tile classification requires pixel_values")
        values = batch.pixel_values
        if mode == "none":
            values = torch.zeros_like(values)
        elif mode == "shuffle" and int(values.shape[0]) > 1:
            values = torch.roll(values, shifts=1, dims=0)
        elif mode != "image":
            raise ValueError(f"unknown tile visual mode {mode!r}")
        embeddings = self.vision.encode_tiles(values)
        return {"embeddings": embeddings} if self.text_alignment else embeddings


class _TileContrastiveClassificationTask(BinaryClassificationTask):
    """Binary tile head with optional image/text alignment loss."""

    def __init__(self, head: nn.Module, *, contrastive_weight: float, **kwargs: Any) -> None:
        super().__init__(head, **kwargs)
        if contrastive_weight < 0:
            raise ValueError("contrastive_weight must be non-negative")
        self.contrastive_weight = float(contrastive_weight)

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        if not isinstance(model_output, Mapping) or not isinstance(model_output.get("embeddings"), torch.Tensor):
            raise ShapeContractError("contrastive tile model must return embeddings")
        embeddings = model_output["embeddings"]
        logits = self.head(embeddings)
        target = target_from_batch(batch, "classification")
        classification = self.loss(logits, target, valid_mask=valid_sample_mask(batch))
        text_embeddings = batch.task_targets.get("text_embeddings")
        if not isinstance(text_embeddings, torch.Tensor):
            raise ShapeContractError("contrastive tile batches require text_embeddings")
        alignment = contrastive_alignment_loss(
            embeddings,
            text_embeddings,
            valid_mask=valid_sample_mask(batch),
        )
        total = classification + self.contrastive_weight * alignment
        count = valid_sample_count(batch)
        return LossOutput(
            total=total,
            components={"classification": classification, "contrastive_alignment": alignment},
            sample_count=count,
            diagnostics={
                "task": self.task_type.value,
                "valid_count": detached_count_tensor(count, logits),
                "contrastive_weight": self.contrastive_weight,
            },
        )


class _WSIClassificationModel(nn.Module):
    def __init__(self, aggregator: nn.Module) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.last_aggregation: Any | None = None

    @staticmethod
    def _source(batch: MedicalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = batch.task_targets.get("tile_embeddings", batch.task_targets.get("visual_tokens"))
        if not isinstance(embeddings, torch.Tensor):
            raise ShapeContractError("WSI classification requires cached tile_embeddings [B,T,D]")
        mask = batch.task_targets.get("tile_mask", batch.task_targets.get("visual_token_mask"))
        if not isinstance(mask, torch.Tensor):
            raise ShapeContractError("cached WSI embeddings require tile_mask")
        return embeddings, mask.bool()

    def aggregate_batch(self, batch: MedicalBatch, *, mode: str = "image") -> Any:
        embeddings, mask = self._source(batch)
        if mode == "none":
            embeddings = torch.zeros_like(embeddings)
            mask = torch.zeros_like(mask)
            mask[:, 0] = True
        elif mode == "shuffle":
            embeddings = torch.roll(embeddings, shifts=1, dims=1)
        elif mode != "image":
            raise ValueError(f"unknown WSI visual mode {mode!r}")
        return self.aggregator.aggregate(embeddings, mask)

    def forward(self, batch: MedicalBatch) -> torch.Tensor:
        self.last_aggregation = self.aggregate_batch(batch)
        return self.last_aggregation.embedding

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> torch.Tensor:
        self.last_aggregation = self.aggregate_batch(batch, mode=mode)
        return self.last_aggregation.embedding


@dataclass(frozen=True, eq=False)
class PathologyVLMOutput:
    language: LanguageOutput
    visual_tokens: ProjectedVisualTokens
    source_coordinates: torch.Tensor
    evidence_tiles: tuple[tuple[dict[str, Any], ...], ...]
    mode: str
    family: str = "wsi_vlm"

    @property
    def logits(self) -> torch.Tensor:
        return self.language.logits


class _WSIVLMModel(nn.Module):
    def __init__(
        self,
        *,
        language: GenericHFCausalLMAdapter,
        bridge: nn.Module,
        selector: WSITokenSelector,
        visual_token_count: int,
        slide_shape: tuple[int, int],
        evidence_top_k: int,
        tile_mpp: float,
        selector_revision: str,
    ) -> None:
        super().__init__()
        self.language = language
        self.bridge = bridge
        self.selector = selector
        self.visual_token_count = int(visual_token_count)
        self.slide_shape = tuple(int(v) for v in slide_shape)
        self.evidence_top_k = int(evidence_top_k)
        self.tile_mpp = float(tile_mpp)
        self.family = "wsi_vlm"
        self.selector_name = "grid"
        self.selector_revision = str(selector_revision)
        self._last_evidence: tuple[tuple[dict[str, Any], ...], ...] = ()

    def _text(self, batch: MedicalBatch) -> tuple[TokenizedText, torch.Tensor]:
        if batch.input_ids is None or batch.attention_mask is None:
            raise ShapeContractError("WSI VLM batches require input_ids and attention_mask")
        labels = batch.task_targets.get("language_labels")
        if not isinstance(labels, torch.Tensor):
            raise ShapeContractError("WSI VLM batch is missing language_labels")
        return TokenizedText(batch.input_ids, batch.attention_mask), labels

    def _source(
        self,
        batch: MedicalBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[Any, ...], ...]]:
        embeddings = batch.task_targets.get("tile_embeddings", batch.task_targets.get("visual_tokens"))
        mask = batch.task_targets.get("tile_mask", batch.task_targets.get("visual_token_mask"))
        raw_records = batch.task_targets.get("tile_records")
        if not isinstance(embeddings, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ShapeContractError("WSI VLM requires cached tile embeddings and tile mask")
        if not isinstance(raw_records, (tuple, list)) or len(raw_records) != int(embeddings.shape[0]):
            raise ShapeContractError("WSI VLM must retain one tile-record sequence per slide")
        records = tuple(tuple(item) for item in raw_records)
        selected = select_wsi_visual_tokens(
            [embeddings[index][mask[index]] for index in range(int(embeddings.shape[0]))],
            [records[index] for index in range(int(embeddings.shape[0]))],
            budget=self.selector.budget,
            selector=self.selector_name,
        )
        return selected.tokens, selected.mask, selected.coordinates, selected.records

    def _visual(
        self,
        batch: MedicalBatch,
        *,
        mode: str,
    ) -> tuple[ProjectedVisualTokens, torch.Tensor, tuple[tuple[dict[str, Any], ...], ...]]:
        tokens, mask, coordinates, records = self._source(batch)
        if mode == "none":
            tokens = torch.zeros_like(tokens)
            mask = torch.zeros_like(mask)
            coordinates = torch.zeros_like(coordinates)
        elif mode in {"shuffle", "shuffle_tiles"}:
            tokens = torch.roll(tokens, shifts=1, dims=1)
        elif mode == "shuffle_coordinates":
            coordinates = torch.roll(coordinates, shifts=1, dims=1)
        elif mode != "image":
            raise ValueError(f"unknown WSI VLM visual mode {mode!r}")
        height, width = self.slide_shape
        normalized = torch.stack(
            (
                coordinates[..., 0].float() / max(width, 1),
                coordinates[..., 1].float() / max(height, 1),
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        mpp = torch.full(mask.shape, self.tile_mpp, dtype=tokens.dtype, device=tokens.device)
        level = torch.zeros_like(mpp)
        slide_index = (
            torch.arange(tokens.shape[0], device=tokens.device, dtype=tokens.dtype).unsqueeze(1).expand_as(mpp)
        )
        coordinate_metadata = {
            "slide_x": coordinates[..., 0].float() * self.tile_mpp,
            "slide_y": coordinates[..., 1].float() * self.tile_mpp,
            "mpp": mpp,
            "pyramid_level": level,
            "slide_index": slide_index,
        }
        if isinstance(self.bridge, CoordinateAwareBridge):
            projected = self.bridge(tokens, mask, coordinates=normalized, coordinate_metadata=coordinate_metadata)
        else:
            projected = self.bridge(tokens, mask, coordinates=normalized)
        evidence: list[tuple[dict[str, Any], ...]] = []
        for row in range(int(tokens.shape[0])):
            row_records = records[row]
            row_scores = mask[row].to(tokens.dtype)
            evidence.append(
                evidence_tiles_from_scores(
                    row_records,
                    row_scores[: len(row_records)],
                    top_k=min(self.evidence_top_k, len(row_records)),
                    slide_shape=self.slide_shape,
                )
            )
        return projected, coordinates, tuple(evidence)

    def forward(self, batch: MedicalBatch) -> PathologyVLMOutput:
        return self.forward_mode(batch, mode="image")

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> PathologyVLMOutput:
        visual, coordinates, evidence = self._visual(batch, mode=mode)
        self._last_evidence = evidence
        text, labels = self._text(batch)
        language = self.language.forward_with_visual_tokens(text, visual, labels)
        auxiliary = dict(language.auxiliary)
        auxiliary.update(
            {
                "source_coordinates": coordinates,
                "evidence_tiles": evidence,
                "visual_mode": mode,
                "visual_token_mask": visual.token_mask,
                "host_embedding_store_input_stall_ms": 0.0,
                "selector_revision": self.selector_revision,
            }
        )
        return PathologyVLMOutput(
            language=LanguageOutput(
                logits=language.logits,
                loss=language.loss,
                hidden_states=language.hidden_states,
                auxiliary=auxiliary,
            ),
            visual_tokens=visual,
            source_coordinates=coordinates,
            evidence_tiles=evidence,
            mode=mode,
        )

    def generate(self, batch: MedicalBatch, *, mode: str = "image", max_new_tokens: int = 8) -> Any:
        visual, _, _ = self._visual(batch, mode=mode)
        text, _ = self._text(batch)
        from medfm.core.language import GenerationConfig

        return self.language.generate(text, visual, GenerationConfig(max_new_tokens=max_new_tokens))

    def evidence_json(self, *, slide_id: str, evidence_index: int = 0) -> dict[str, Any]:
        if not 0 <= evidence_index < len(self._last_evidence):
            raise RuntimeError("run a forward pass before requesting evidence JSON")
        return evidence_payload(
            self._last_evidence[evidence_index],
            slide_id=slide_id,
            slide_shape=self.slide_shape,
        )


class _PathologyVLMTask(TaskModuleBase):
    def __init__(self, task_type: TaskType) -> None:
        super().__init__(task_type, (Modality.PATHOLOGY_WSI,))

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        if not isinstance(model_output, PathologyVLMOutput) or model_output.language.loss is None:
            raise ShapeContractError("WSI VLM model must return a language loss")
        loss = model_output.language.loss
        count = valid_sample_count(batch)
        token_count = int(model_output.language.auxiliary.get("supervised_token_count", 0))
        return LossOutput(
            total=loss,
            components={"language": loss},
            sample_count=count,
            token_count=token_count,
            diagnostics={
                "task": self.task_type.value,
                "valid_count": detached_count_tensor(count, loss),
                "supervised_token_count": token_count,
                "visual_mode": model_output.mode,
                "evidence_tiles": model_output.evidence_tiles,
            },
        )


class _TileSegmentationVision(nn.Module):
    def __init__(self, *, hidden_size: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.stem = nn.Sequential(
                nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.late_block = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = self.stem(pixels.float())
        second = F.gelu(self.late_block(first))
        return (first, second)


class _PathologySegmentationModel(nn.Module):
    def __init__(self, vision: _TileSegmentationVision) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, batch: MedicalBatch) -> dict[str, Any]:
        if batch.pixel_values is None:
            raise ShapeContractError("pathology segmentation requires pixel_values")
        return {"feature_maps": self.vision(batch.pixel_values)}

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> dict[str, Any]:
        if batch.pixel_values is None:
            raise ShapeContractError("pathology segmentation requires pixel_values")
        values = batch.pixel_values if mode == "image" else torch.zeros_like(batch.pixel_values)
        return {"feature_maps": self.vision(values)}


def _text_payload(batch_size: int, text_tokens: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 60, (batch_size, text_tokens), generator=generator, dtype=torch.long)
    ids[:, 0] = 1
    ids[:, -1] = 2
    attention = torch.ones_like(ids, dtype=torch.bool)
    labels = ids.clone()
    if text_tokens > 2:
        labels[:, 0] = -100
    return ids, attention, labels


def _wsi_batch(options: _Phase15Options, config: RunConfig, *, language: bool) -> MedicalBatch:
    batch_size = max(1, int(config.batch.microbatch_per_device))
    payloads = _synthetic_slide_payloads(options, batch_size=batch_size)
    selected = pad_slide_embeddings(
        [value["embeddings"] for value in payloads],
        [value["records"] for value in payloads],
        max_tiles=options.max_tiles_per_slide,
        selector=options.train_selector,
        seed=options.construction_seed,
    )
    labels = torch.tensor([int(value["label"]) for value in payloads], dtype=torch.float32).unsqueeze(1)
    records = tuple(tuple(value) for value in selected.records)
    targets: dict[str, Any] = {
        "visual_tokens": selected.tokens,
        "visual_token_mask": selected.mask,
        "tile_embeddings": selected.tokens,
        "tile_mask": selected.mask,
        "classification": labels,
        "tile_records": records,
        "actual_tile_count": torch.tensor(selected.actual_counts, dtype=torch.int64),
        "patient_ids": tuple(str(value["patient_id"]) for value in payloads),
        "slide_ids": tuple(str(value["slide_id"]) for value in payloads),
        "scanner_ids": tuple(str(value["scanner"]) for value in payloads),
        "site_ids": tuple(str(value["site"]) for value in payloads),
        "organ_ids": tuple(str(value["organ"]) for value in payloads),
        "magnification": tuple(str(options.magnification) for _ in payloads),
        "selection_behavior": {"train": options.train_selector, "evaluation": options.eval_selector},
    }
    kwargs: dict[str, Any] = {
        "modality": Modality.PATHOLOGY_WSI,
        "sample_ids": [str(value["slide_id"]) for value in payloads],
        "tile_coordinates": selected.coordinates,
        "task_targets": targets,
        "bucket": BucketId(BucketKind.VISUAL_TOKENS, (options.max_tiles_per_slide,)),
    }
    if language:
        ids, attention, labels_text = _text_payload(
            batch_size,
            options.text_token_count,
            seed=options.construction_seed + 99,
        )
        targets["language_labels"] = labels_text
        kwargs.update(
            input_ids=ids,
            attention_mask=attention,
            bucket=BucketId(BucketKind.VISUAL_TOKENS, (options.max_tiles_per_slide,)),
        )
    return MedicalBatch(**kwargs)


def _tile_classification_data(options: _Phase15Options, config: RunConfig) -> list[MedicalBatch]:
    batch_size = max(1, int(config.batch.microbatch_per_device))
    generator = torch.Generator().manual_seed(options.construction_seed + 20)
    pixels = torch.randn((batch_size, options.channels, options.tile_height, options.tile_width), generator=generator)
    labels = torch.tensor([index % 2 for index in range(batch_size)], dtype=torch.float32).unsqueeze(1)
    targets: dict[str, Any] = {
        "classification": labels,
        "patient_ids": tuple(f"patient-{index}" for index in range(batch_size)),
        "slide_ids": tuple(f"slide-{index}" for index in range(batch_size)),
        "scanner_ids": tuple(f"scanner-{index % 2}" for index in range(batch_size)),
        "site_ids": tuple(f"site-{index % 2}" for index in range(batch_size)),
        "organ_ids": tuple("breast" if index % 2 else "colon" for index in range(batch_size)),
    }
    if options.text_alignment:
        text_generator = torch.Generator().manual_seed(options.construction_seed + 40)
        targets["text_embeddings"] = torch.randn(
            (batch_size, options.embedding_dim),
            generator=text_generator,
            dtype=torch.float32,
        )
    return [
        MedicalBatch(
            modality=Modality.PATHOLOGY_TILE,
            sample_ids=[f"tile-sample-{index}" for index in range(batch_size)],
            pixel_values=pixels,
            image_mask=torch.ones(batch_size, dtype=torch.bool),
            task_targets=targets,
            bucket=BucketId(BucketKind.IMAGE_2D, (options.tile_height, options.tile_width)),
        )
    ]


def _segmentation_data(options: _Phase15Options, config: RunConfig) -> list[MedicalBatch]:
    batch_size = max(1, int(config.batch.microbatch_per_device))
    generator = torch.Generator().manual_seed(options.construction_seed + 30)
    pixels = torch.randn((batch_size, options.channels, options.tile_height, options.tile_width), generator=generator)
    target = torch.zeros((batch_size, 1, options.tile_height, options.tile_width), dtype=torch.float32)
    y0, y1 = max(1, options.tile_height // 4), max(2, 3 * options.tile_height // 4)
    x0, x1 = max(1, options.tile_width // 4), max(2, 3 * options.tile_width // 4)
    target[:, :, y0:y1, x0:x1] = 1.0
    return [
        MedicalBatch(
            modality=Modality.PATHOLOGY_TILE,
            sample_ids=[f"seg-tile-{index}" for index in range(batch_size)],
            pixel_values=pixels,
            image_mask=torch.ones(batch_size, dtype=torch.bool),
            task_targets={
                "segmentation": target,
                "tile_level": torch.zeros(batch_size, dtype=torch.int64),
                "slide_ids": tuple(f"slide-{index}" for index in range(batch_size)),
                "tile_ids": tuple(f"seg-tile-{index}" for index in range(batch_size)),
                "positive_tile_rate": float(target.mean()),
            },
            bucket=BucketId(BucketKind.IMAGE_2D, (options.tile_height, options.tile_width)),
        )
    ]


def _phase15_data(config: RunConfig, options: _Phase15Options) -> list[MedicalBatch]:
    if options.family == "tile_classification":
        return _tile_classification_data(options, config)
    if options.family == "wsi_classification":
        return [_wsi_batch(options, config, language=False)]
    if options.family == "wsi_vlm":
        return [_wsi_batch(options, config, language=True)]
    return _segmentation_data(options, config)


def _build_tile_vision(options: _Phase15Options) -> _TinyTileVision:
    if not options.offline_tiny:
        raise PathologyRecipeConfigurationError(
            f"production backbone {options.backbone!r} requires an approved local checkpoint/registry integration; "
            "the recipe builder will not download pathology weights"
        )
    return _TinyTileVision(
        embedding_dim=options.embedding_dim,
        hidden_size=options.hidden_size,
        seed=options.construction_seed,
        model_id=options.backbone,
    )


def _inject_tile_lora(vision: nn.Module, config: RunConfig, options: _Phase15Options) -> None:
    if not options.use_lora or options.stage not in {"3", "4"}:
        return
    lora = LoRAConfig(
        enabled=True,
        rank=int(config.recipe.get("lora_rank", config.peft.rank)),
        alpha=float(config.recipe.get("lora_alpha", max(1, config.peft.rank * 2))),
        dropout=0.0,
        target_policy="explicit",
        target_modules=PATHOLOGY_LORA_TARGETS,
        architecture="vision",
        confirm_target_modules=True,
    )
    inject_lora(vision, lora, architecture="vision", confirm_unknown=True)


def _build_language(options: _Phase15Options) -> GenericHFCausalLMAdapter:
    return GenericHFCausalLMAdapter.build_tiny(
        model_id="pathology-language-tiny",
        hidden_size=options.hidden_size,
        vocab_size=64,
        depth=1,
        heads=4,
        max_text_tokens=options.text_token_count,
        visual_token_buckets=(32, 64, 128),
        construction_seed=options.construction_seed + 200,
    )


def _build_wsi_vlm_model(config: RunConfig, options: _Phase15Options) -> _WSIVLMModel:
    language = _build_language(options)
    if options.bridge_type == "perceiver":
        base_bridge: nn.Module = PerceiverResamplerBridge(
            source_dim=options.embedding_dim,
            target_dim=options.hidden_size,
            output_tokens=options.visual_token_count,
            max_input_tokens=options.visual_token_count,
            source_modality=Modality.PATHOLOGY_WSI,
            coordinate_system=CoordinateSystem.MICRONS,
            heads=4,
        )
    else:
        base_bridge = LinearVisionLanguageBridge(
            source_dim=options.embedding_dim,
            target_dim=options.hidden_size,
            output_tokens=options.visual_token_count,
            max_input_tokens=options.visual_token_count,
            source_modality=Modality.PATHOLOGY_WSI,
            coordinate_system=CoordinateSystem.MICRONS,
        )
    bridge = CoordinateAwareBridge(base_bridge, WSICoordinateEncoder(output_dim=max(4, options.hidden_size // 2)))
    selector = WSITokenSelector(
        sampler=_selector(options.selector),
        budget=TokenBudget(precompression=options.precompression_tile_count, visual_tokens=options.visual_token_count),
    )
    model = _WSIVLMModel(
        language=language,
        bridge=bridge,
        selector=selector,
        visual_token_count=options.visual_token_count,
        slide_shape=(options.slide_height, options.slide_width),
        evidence_top_k=options.evidence_top_k,
        tile_mpp=options.tile_mpp,
        selector_revision=options.selector_revision,
    )
    model.selector_name = options.selector
    # Cached-token VLM stages keep the tile encoder outside the optimizer.  The
    # bridge is stage 1; language LoRA is enabled only from stage 2 onward.
    if options.use_lora and options.stage in {"2", "3", "4"}:
        lora = LoRAConfig(
            enabled=True,
            rank=int(config.recipe.get("lora_rank", config.peft.rank)),
            alpha=float(config.recipe.get("lora_alpha", max(1, config.peft.rank * 2))),
            dropout=0.0,
            target_policy="explicit",
            target_modules=LANGUAGE_LORA_TARGETS,
            architecture="llm",
            confirm_target_modules=True,
        )
        inject_language_lora(language, lora, confirm_unknown=True)
    return model


def _phase15_model(config: RunConfig, options: _Phase15Options) -> nn.Module:
    if options.family == "tile_classification":
        vision = _build_tile_vision(options)
        _inject_tile_lora(vision, config, options)
        return _TileClassificationModel(vision, text_alignment=options.text_alignment)
    if options.family == "wsi_classification":
        return _WSIClassificationModel(build_slide_aggregator(options.aggregator, options.embedding_dim))
    if options.family == "wsi_vlm":
        return _build_wsi_vlm_model(config, options)
    if not options.offline_tiny:
        raise PathologyRecipeConfigurationError("production pathology segmentation requires an approved tile encoder")
    vision = _TileSegmentationVision(hidden_size=options.hidden_size, seed=options.construction_seed)
    return _PathologySegmentationModel(vision)


def _classification_task(config: RunConfig, options: _Phase15Options) -> nn.Module:
    task_name = options.task_name
    binary = "BINARY" in task_name or task_name in {"CLASSIFICATION", "BINARY"}
    if "MULTILABEL" in task_name:
        classes = int(config.recipe.get("num_classes", 2))
        return MultiLabelClassificationTask(
            MLPClassificationHead(options.embedding_dim, classes)
            if options.stage == "2"
            else LinearClassificationHead(options.embedding_dim, classes),
            supported_modalities=(Modality.PATHOLOGY_TILE, Modality.PATHOLOGY_WSI),
        )
    classes = 1 if binary else int(config.recipe.get("num_classes", 2))
    head_name = str(config.recipe.get("head", "mlp" if options.stage == "2" else "linear")).lower()
    if head_name in {"mlp", "projection_mlp"}:
        head: nn.Module = MLPClassificationHead(options.embedding_dim, classes)
    else:
        head = LinearClassificationHead(options.embedding_dim, classes)
    if binary:
        if options.text_alignment:
            return _TileContrastiveClassificationTask(
                head,
                contrastive_weight=float(config.recipe.get("contrastive_weight", 0.1)),
                supported_modalities=(options.modality,),
            )
        return BinaryClassificationTask(head, supported_modalities=(options.modality,))
    return ClassificationTask(
        head, task_type=TaskType.MULTICLASS_CLASSIFICATION, supported_modalities=(options.modality,)
    )


def _phase15_task(config: RunConfig, options: _Phase15Options, model: nn.Module) -> nn.Module:
    if options.family == "tile_classification":
        return _classification_task(config, options)
    if options.family == "wsi_classification":
        # Slide aggregators return one embedding per sample; the classification
        # head is kept in the task component so checkpoint roles remain clear.
        return _classification_task(config, options)
    if options.family == "wsi_vlm":
        try:
            task_type = TaskType(options.task_name)
        except ValueError:
            if "REPORT" in options.task_name:
                task_type = TaskType.REPORT_GENERATION
            elif "RETRIEVAL" in options.task_name:
                task_type = TaskType.IMAGE_TEXT_RETRIEVAL
            elif any(
                token in options.task_name
                for token in ("ORGAN", "SITE", "SUBTYPE", "GRADE", "BIOMARKER", "CLASSIFICATION")
            ):
                task_type = TaskType.MULTICLASS_CLASSIFICATION
            else:
                task_type = TaskType.VISUAL_QUESTION_ANSWERING
        return _PathologyVLMTask(task_type)
    decoder = UNetDecoder2D(
        in_channels=(options.hidden_size, options.hidden_size),
        out_channels=1,
        hidden_channels=int(config.recipe.get("decoder_hidden", max(4, options.hidden_size // 2))),
    )
    return BinarySegmentationTask(decoder, supported_modalities=(Modality.PATHOLOGY_TILE,))


def _backend_observability(config: RunConfig, data: Sequence[MedicalBatch], options: _Phase15Options) -> dict[str, Any]:
    counts: list[int] = []
    for batch in data:
        value = batch.task_targets.get("actual_tile_count")
        if isinstance(value, torch.Tensor):
            counts.extend(int(v) for v in value.detach().cpu().reshape(-1).tolist())
    return {
        "backend": config.accelerator.backend,
        "world_size": config.accelerator.world_size,
        "compile_count": 0,
        "input_wait_ms": 0.0,
        "host_embedding_store_input_stall_ms": 0.0,
        "host_to_device_ms": 0.0,
        "throughput_tiles_per_second": 0.0,
        "throughput_slides_per_second": 0.0,
        "peak_vram_gb": None,
        "peak_hbm_gb": None,
        "fallback_operators": (),
        "actual_tile_counts": counts,
        "max_tiles_per_slide": options.max_tiles_per_slide,
        "padded_tiles_excluded_from_loss_and_metrics": True,
        "slide_reader_revision": options.slide_reader_revision,
        "tile_index_revision": options.tile_index_revision,
        "tile_encoder_status": "offline_contract" if options.offline_tiny else "license_gated",
        "aggregator_status": (
            "offline_contract"
            if options.family in {"wsi_classification", "wsi_vlm"} and options.offline_tiny
            else "license_gated"
        ),
        "failure_rates": {
            "corrupt_tiles": 0.0,
            "low_quality_tiles": 0.0,
            "corrupt_slides": 0.0,
            "low_quality_slides": 0.0,
        },
        "shard_unit": "slide",
    }


def _metadata(config: RunConfig, options: _Phase15Options, data: Sequence[MedicalBatch]) -> PathologyRecipeMetadata:
    sampled: list[int] = []
    for batch in data:
        values = batch.task_targets.get("actual_tile_count")
        if isinstance(values, torch.Tensor):
            sampled.extend(int(v) for v in values.detach().cpu().reshape(-1).tolist())
    return PathologyRecipeMetadata(
        family=options.family,
        recipe_id=options.recipe_id,
        backbone=options.backbone,
        modality=options.modality.value,
        stage=options.stage,
        mode=options.mode,
        dataset_id=options.dataset_id,
        dataset_revision=options.dataset_revision,
        preprocessing_revision=options.preprocessing_revision,
        model_revision=options.model_revision,
        embedding_store_schema=EmbeddingStore.schema_version,
        embedding_store_revision=options.embedding_store_revision,
        tile_shape=(options.channels, options.tile_height, options.tile_width),
        tile_shape_buckets=options.tile_shape_buckets,
        tile_mpp=options.tile_mpp,
        magnification=options.magnification,
        max_tiles_per_slide=options.max_tiles_per_slide,
        sampled_tile_counts=tuple(sampled),
        visual_token_count=options.visual_token_count if options.family == "wsi_vlm" else None,
        precompression_tile_count=options.precompression_tile_count if options.family == "wsi_vlm" else None,
        text_token_count=options.text_token_count if options.family == "wsi_vlm" else None,
        selector=options.selector if options.family == "wsi_vlm" else None,
        selector_revision=options.selector_revision if options.family == "wsi_vlm" else None,
        train_selector=options.train_selector if options.family in {"wsi_classification", "wsi_vlm"} else None,
        eval_selector=options.eval_selector if options.family in {"wsi_classification", "wsi_vlm"} else None,
        aggregator=options.aggregator if options.family == "wsi_classification" else None,
        bridge_type=options.bridge_type if options.family == "wsi_vlm" else None,
        microbatch_per_device=config.batch.microbatch_per_device,
        world_size=config.accelerator.world_size,
        gradient_accumulation_steps=config.batch.gradient_accumulation_steps,
        global_batch_size=config.global_batch_size,
        memory_cap_gb=config.memory.max_gpu_memory_gb,
        tpu_status=options.tpu_status,
        shard_unit="slide",
        split_policy=options.split_policy,
        cache_embeddings=options.cache_embeddings,
        cache_tokens=options.cache_tokens,
        actual_tile_count_logging=True,
        backend_observability=_backend_observability(config, data, options),
        baseline=options.baseline,
        slide_reader_revision=options.slide_reader_revision,
        tile_index_revision=options.tile_index_revision,
        failure_rates=(
            ("corrupt_tiles", 0.0),
            ("low_quality_tiles", 0.0),
            ("corrupt_slides", 0.0),
            ("low_quality_slides", 0.0),
        ),
        task_name=options.task_name,
        limitations=(
            "Offline tiny checkpoints and synthetic data are contract evidence, not clinical validation.",
            "Approved pathology weights, de-identified manifests, and external-site evaluation remain required.",
            "TPU status records a static-shape contract unless protected TPU hardware evidence is present.",
        ),
    )


def build_phase15_recipe(config: RunConfig) -> PathologyRecipeBuild:
    """Build a deterministic Phase 15 recipe and its bounded synthetic batch."""
    options = _options(config)
    model = _phase15_model(config, options)
    task = _phase15_task(config, options, model)
    data = _phase15_data(config, options)
    return PathologyRecipeBuild(model=model, task=task, train_data=data, metadata=_metadata(config, options, data))


def _phase15_role_map() -> dict[str, str]:
    return {
        "vision": "vision_lora",
        "aggregator": "aggregator",
        "bridge": "bridge",
        "language": "language_lora",
        "task.head": "task_head",
        "task.decoder": "decoder",
    }


def phase15_builders() -> ComponentBuilders:
    """Return builders compatible with the model-agnostic TrainingPipeline."""

    def dataset(config: RunConfig, *_: Any) -> list[MedicalBatch]:
        return _phase15_data(config, _options(config))

    def model(config: RunConfig, *_: Any) -> nn.Module:
        return _phase15_model(config, _options(config))

    def peft(model_value: nn.Module, *_: Any) -> nn.Module:
        return model_value

    def task(config: RunConfig, model_value: nn.Module, *_: Any) -> nn.Module:
        return _phase15_task(config, _options(config), model_value)

    def optimizer(
        model_value: nn.Module,
        task_value: nn.Module,
        config: RunConfig,
        backend: AcceleratorBackend,
    ) -> OptimizerBundle:
        return build_optimizer(
            model_value,
            config.optimizer,
            backend=backend.name,
            role_map=_phase15_role_map(),
            components={"task": task_value},
        )

    def trainer(
        config: RunConfig,
        backend: AcceleratorBackend,
        model_value: nn.Module,
        optimizer_value: OptimizerBundle,
        task_value: nn.Module,
        dataset_value: list[MedicalBatch],
    ) -> Trainer:
        return Trainer(
            model_value,
            optimizer_value,
            task_value,
            dataset_value,
            config,
            backend=backend,
            training_step=make_training_step(task_value),
            validation_dataloader=dataset_value,
            role_map=_phase15_role_map(),
        )

    return ComponentBuilders(dataset=dataset, model=model, peft=peft, task=task, optimizer=optimizer, trainer=trainer)


def _group_aggregate(
    labels: Sequence[int] | torch.Tensor,
    scores: Sequence[float] | torch.Tensor,
    group_ids: Sequence[str] | None,
) -> tuple[list[int], list[float], list[str]]:
    y = labels.detach().cpu().reshape(-1).tolist() if isinstance(labels, torch.Tensor) else [int(v) for v in labels]
    s = scores.detach().cpu().reshape(-1).tolist() if isinstance(scores, torch.Tensor) else [float(v) for v in scores]
    if len(y) != len(s):
        raise ValueError("labels and scores must have equal length")
    if group_ids is None:
        return y, [float(v) for v in s], [str(index) for index in range(len(y))]
    groups = [str(value) for value in group_ids]
    if len(groups) != len(y):
        raise ValueError("group_ids must align with labels")
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    out_y = [
        int(round(sum(y[index] for index in indices) / len(indices))) for group, indices in sorted(by_group.items())
    ]
    out_s = [float(sum(s[index] for index in indices) / len(indices)) for group, indices in sorted(by_group.items())]
    return out_y, out_s, sorted(by_group)


def pathology_classification_metrics(
    labels: Sequence[int] | torch.Tensor,
    scores: Sequence[float] | torch.Tensor,
    *,
    patient_ids: Sequence[str] | None = None,
    slide_ids: Sequence[str] | None = None,
    scanner_ids: Sequence[str] | None = None,
    site_ids: Sequence[str] | None = None,
    organ_ids: Sequence[str] | None = None,
    valid_mask: Sequence[bool] | torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Return tile/slide/patient metrics while excluding padded entries."""
    y = (
        labels.detach().cpu().reshape(-1)
        if isinstance(labels, torch.Tensor)
        else torch.as_tensor(list(labels), dtype=torch.int64)
    )
    s = (
        scores.detach().cpu().reshape(-1)
        if isinstance(scores, torch.Tensor)
        else torch.as_tensor(list(scores), dtype=torch.float32)
    )
    if y.numel() != s.numel():
        raise ValueError("labels and scores must have equal length")
    if valid_mask is not None:
        valid = (
            valid_mask.detach().cpu().reshape(-1).bool()
            if isinstance(valid_mask, torch.Tensor)
            else torch.as_tensor(list(valid_mask), dtype=torch.bool)
        )
        if valid.numel() != y.numel():
            raise ValueError("valid_mask must align with labels")
        y, s = y[valid], s[valid]

        def filter_ids(values: Sequence[str] | None) -> Sequence[str] | None:
            return (
                None
                if values is None
                else [str(value) for value, keep in zip(values, valid.tolist(), strict=True) if keep]
            )

        patient_ids, slide_ids, scanner_ids, site_ids, organ_ids = (
            filter_ids(patient_ids),
            filter_ids(slide_ids),
            filter_ids(scanner_ids),
            filter_ids(site_ids),
            filter_ids(organ_ids),
        )
    result: dict[str, MetricValue] = {}
    tile = classification_metrics(y, s, unit="per_tile")
    result.update({f"tile/{name}": value for name, value in tile.items()})
    slide_y, slide_s, slide_groups = _group_aggregate(y, s, slide_ids)
    result.update(
        {f"slide/{name}": value for name, value in classification_metrics(slide_y, slide_s, unit="per_slide").items()}
    )
    patient_y, patient_s, _ = _group_aggregate(y, s, patient_ids or slide_ids)
    result.update(
        {
            f"patient/{name}": value
            for name, value in classification_metrics(patient_y, patient_s, unit="per_patient").items()
        }
    )
    for prefix, groups in (("scanner", scanner_ids), ("site", site_ids), ("organ", organ_ids)):
        if groups is not None:
            subgroup = classification_metrics(y, s, group_ids=groups, unit=f"per_{prefix}").get("subgroups")
            if subgroup is not None:
                result[f"{prefix}/subgroups"] = subgroup
    return result


wsi_classification_metrics = pathology_classification_metrics
pathology_tile_metrics = pathology_classification_metrics


def evaluate_wsi_tile_counts_and_magnification(
    labels: Sequence[int] | torch.Tensor,
    scores_by_condition: Mapping[Any, Sequence[float] | torch.Tensor],
    *,
    patient_ids: Sequence[str] | None = None,
    slide_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate deterministic tile-count/magnification rows separately."""
    rows: list[dict[str, Any]] = []
    for condition, scores in scores_by_condition.items():
        if isinstance(condition, tuple) and len(condition) == 2:
            count, magnification = condition
        else:
            count, magnification = condition, None
        metrics = pathology_classification_metrics(labels, scores, patient_ids=patient_ids, slide_ids=slide_ids)
        rows.append(
            {
                "tile_count": None if count is None else int(count),
                "magnification": None if magnification is None else str(magnification),
                "metrics": {name: value.to_dict() for name, value in sorted(metrics.items())},
            }
        )
    rows.sort(key=lambda row: (str(row["magnification"]), -1 if row["tile_count"] is None else row["tile_count"]))
    return tuple(rows)


benchmark_wsi_tile_counts = evaluate_wsi_tile_counts_and_magnification


def tile_segmentation_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, MetricValue]:
    return segmentation_metrics(predicted, target)


def slide_segmentation_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, MetricValue]:
    return segmentation_metrics(predicted, target)


def pathology_segmentation_metrics(
    tile_predicted: torch.Tensor,
    tile_target: torch.Tensor,
    *,
    slide_predicted: torch.Tensor | None = None,
    slide_target: torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Report tile and reconstructed-slide dense metrics as separate units."""
    result = {f"tile/{name}": value for name, value in tile_segmentation_metrics(tile_predicted, tile_target).items()}
    if slide_predicted is not None and slide_target is not None:
        result.update(
            {
                f"slide/{name}": value
                for name, value in slide_segmentation_metrics(slide_predicted, slide_target).items()
            }
        )
    return result


def contrastive_alignment_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric image/text contrastive loss for optional tile Stage 4."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2 or image_embeddings.shape != text_embeddings.shape:
        raise ValueError("image_embeddings and text_embeddings must both be [N,D] with equal shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if valid_mask is None:
        valid_mask = torch.ones(image_embeddings.shape[0], dtype=torch.bool, device=image_embeddings.device)
    if valid_mask.shape != (image_embeddings.shape[0],):
        raise ValueError("valid_mask must be [N]")
    if int(valid_mask.sum()) < 2:
        return (image_embeddings.sum() + text_embeddings.sum()) * 0.0
    image = F.normalize(image_embeddings[valid_mask].float(), dim=-1)
    text = F.normalize(text_embeddings[valid_mask].float(), dim=-1)
    logits = image @ text.T / float(temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2.0


def make_phase15_artifact(
    config: RunConfig,
    metrics: Mapping[str, MetricValue],
    *,
    metadata: PathologyRecipeMetadata | None = None,
    memory: Mapping[str, Any] | None = None,
) -> EvaluationArtifact:
    """Create a provenance-bearing pathology evaluation artifact."""
    chosen = metadata.to_dict() if metadata is not None else {}
    return make_artifact(
        str(config.recipe.get("id", "phase15-pathology")),
        metrics,
        config_hash=config.config_hash(),
        seed=config.seed,
        dataset_hash=config.dataset_hash,
        preprocessing_hash=config.preprocessing_hash,
        model_revision=config.base_model_revision,
        memory={**chosen.get("backend_observability", {}), **dict(memory or {})},
        limitations=(
            "Offline tiny checkpoints and synthetic data are contract evidence, not clinical validation.",
            "Evidence coordinates require protected artifact handling and de-identified slide manifests.",
        ),
    )


__all__ = [
    "COORDINATE_SYSTEM",
    "EVIDENCE_SCHEMA_VERSION",
    "GatedAttentionMILAggregator",
    "PatientDisjointSplit",
    "PathologyRecipeBuild",
    "PathologyRecipeConfigurationError",
    "PathologyRecipeMetadata",
    "RecipeBuild",
    "RecipeConfigurationError",
    "SelectedWSITokens",
    "StitchedSlide",
    "TilePrediction",
    "TransformerSlideAggregator",
    "PHASE15_RECIPE_VERSION",
    "benchmark_wsi_tile_counts",
    "build_phase15_recipe",
    "build_slide_aggregator",
    "contrastive_alignment_loss",
    "deterministic_patient_split",
    "evidence_json_is_valid",
    "evidence_payload",
    "evidence_tiles_from_scores",
    "evidence_tiles_to_json",
    "evaluate_wsi_tile_counts_and_magnification",
    "extract_evidence_tiles",
    "level_to_level0_geometry",
    "make_evidence_tiles",
    "make_patient_disjoint_splits",
    "make_phase15_artifact",
    "map_evidence_coordinates",
    "map_normalized_coordinates",
    "map_normalized_coordinates_to_wsi",
    "normalize_level0_geometry",
    "normalized_to_level0_geometry",
    "pad_slide_embeddings",
    "pathology_classification_metrics",
    "pathology_segmentation_metrics",
    "pathology_tile_metrics",
    "patient_disjoint_split",
    "phase15_builders",
    "select_cached_wsi_tokens",
    "select_wsi_visual_tokens",
    "serialize_evidence_json",
    "slide_segmentation_metrics",
    "stitch_predictions",
    "stitch_tile_predictions",
    "stitch_wsi_predictions",
    "tile_segmentation_metrics",
    "validate_evidence_json",
    "validate_evidence_tiles",
    "wsi_classification_metrics",
]
