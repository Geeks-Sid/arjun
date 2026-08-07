"""Reproducible Phase 14 native-3D and slice-sequence recipe builders.

The module keeps five experiment families explicit:

* native 3D classification over CT/MRI volume tokens;
* native 3D segmentation over frozen encoder feature pyramids;
* native 3D VLMs with physical-coordinate-aware Perceiver bridges;
* slice-sequence VLMs with a 2D tower and host-side selectors; and
* language-conditioned native 3D segmentation.

``offline_tiny`` is a deterministic pure-PyTorch contract fixture.  It is
never presented as a clinical checkpoint or a substitute for a gated upstream
model.  Production recipe entries fail closed unless a caller supplies the
pinned local checkpoint integration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderOutput, OutputSpec
from medfm.core.enums import CoordinateSystem, Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.language import (
    GeneratedText,
    GenerationConfig,
    LanguageOutput,
    ProjectedVisualTokens,
    TokenizedText,
)
from medfm.core.sample import SpatialMetadata
from medfm.data.samplers import ForegroundPatchSampler
from medfm.evaluation.metrics import MetricValue, classification_metrics, segmentation_metrics
from medfm.evaluation.report import EvaluationArtifact, make_artifact
from medfm.models.bridges import (
    CoordinateAwareBridge,
    LinearVisionLanguageBridge,
    PerceiverResamplerBridge,
    ThreeDCoordinateEncoder,
    VisionLanguageBridge,
)
from medfm.models.decoders import LanguageConditionedMaskDecoder, UNetDecoder3D
from medfm.models.heads.classification import AttentionPoolingClassificationHead, LinearClassificationHead
from medfm.models.language.base import GenericHFCausalLMAdapter
from medfm.models.language.medgemma import MedGemmaAdapter
from medfm.models.visual.ct_fm import CTFMAdapter, FlexiCT3DAdapter
from medfm.models.visual.native_3d import GenericMONAI3DAdapter, sliding_window_inference
from medfm.models.visual.triad import TriadMAEAdapter, TriadSimMIMAdapter
from medfm.peft import LoRAConfig, inject_language_lora
from medfm.recipes.phase13 import TinyVisualAdapter
from medfm.recipes.slice_selectors import (
    SLICE_SELECTOR_VERSION,
    SliceRecord,
    build_slice_selector,
    selections_to_metadata,
)
from medfm.tasks.base import TaskModuleBase, detached_count_tensor, valid_sample_count
from medfm.tasks.classification import BinaryClassificationTask, ClassificationTask
from medfm.tasks.language_segmentation import LanguageConditionedSegmentationTask
from medfm.tasks.segmentation import BinarySegmentationTask
from medfm.training.backend import AcceleratorBackend
from medfm.training.config import RunConfig
from medfm.training.optimizer import OptimizerBundle, build_optimizer
from medfm.training.pipeline import ComponentBuilders
from medfm.training.steps import make_training_step
from medfm.training.trainer import Trainer

PHASE14_RECIPE_VERSION = "phase14-1"
NATIVE_3D_MODALITIES = (Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D)
SLICE_SEQUENCE_MODALITY = Modality.MULTI_IMAGE_2D
LANGUAGE_LORA_TARGETS = (r"layers\.\d+\.(self_attn\.out_proj|linear1|linear2)",)
NATIVE_3D_LORA_TARGETS = (
    r"blocks\.layers\.\d+\.self_attn\.out_proj",
    r"blocks\.layers\.\d+\.linear[12]",
)
_ALLOWED_VOLUME_STRATEGIES = {"full_volume", "fixed_crop", "multicrop", "low_resolution_global", "global_local"}
_ALLOWED_BLEND_MODES = {"constant", "gaussian"}


class RecipeConfigurationError(ValueError):
    """A Phase 14 recipe is malformed or requests an unavailable path."""


@dataclass(frozen=True)
class VolumeInputPolicy:
    """Resolved crop/global-local policy derived from a dataset fingerprint."""

    strategy: str
    crop_shape: tuple[int, int, int]
    shape_buckets: tuple[tuple[int, int, int], ...]
    fingerprint_hash: str | None = None
    global_shape: tuple[int, int, int] | None = None
    local_shape: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _ALLOWED_VOLUME_STRATEGIES:
            raise RecipeConfigurationError(f"unknown 3D input strategy {self.strategy!r}")
        if len(self.crop_shape) != 3 or any(int(v) <= 0 for v in self.crop_shape):
            raise RecipeConfigurationError("3D crop_shape must contain three positive dimensions")
        if not self.shape_buckets:
            raise RecipeConfigurationError("at least one fixed 3D shape bucket is required")
        for bucket in self.shape_buckets:
            if len(bucket) != 3 or any(int(v) <= 0 for v in bucket):
                raise RecipeConfigurationError("3D shape buckets must contain positive 3-tuples")
        if self.strategy == "global_local" and (self.global_shape is None or self.local_shape is None):
            raise RecipeConfigurationError("global_local strategy requires global_shape and local_shape")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "crop_shape": list(self.crop_shape),
            "shape_buckets": [list(value) for value in self.shape_buckets],
            "fingerprint_hash": self.fingerprint_hash,
            "global_shape": None if self.global_shape is None else list(self.global_shape),
            "local_shape": None if self.local_shape is None else list(self.local_shape),
        }


def _shape_tuple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise RecipeConfigurationError(f"{name} must be a three-element shape")
    result = (int(value[0]), int(value[1]), int(value[2]))
    if any(item <= 0 for item in result):
        raise RecipeConfigurationError(f"{name} must contain positive dimensions")
    return result


def _bounded_bucket(value: tuple[int, int, int]) -> tuple[int, int, int]:
    # The accepted production bucket set stays intentionally small.  Tiny
    # contract shapes are allowed so CPU tests do not allocate a 96^3 tensor.
    if max(value) <= 32:
        return value
    edge = min(128, max(96, max(value)))
    return (edge, edge, edge)


def _parse_shape_buckets(raw: Any, fallback: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    if isinstance(raw, Mapping):
        raw = raw.get("volume_3d", raw.get("3d_patch", raw.get("volume", ())))
    if raw is None:
        raw = ()
    if isinstance(raw, list | tuple) and raw and all(isinstance(item, int | float) for item in raw):
        raw = [raw]
    values: list[tuple[int, int, int]] = []
    if isinstance(raw, list | tuple):
        for item in raw:
            if isinstance(item, Mapping):
                kind = str(item.get("kind", "")).lower()
                if kind not in {"3d_patch", "volume_3d", "volume"}:
                    continue
                item = item.get("shape")
            try:
                values.append(_shape_tuple(item, "shape bucket"))
            except RecipeConfigurationError:
                continue
    if not values:
        values = [_bounded_bucket(fallback)]
    unique = tuple(dict.fromkeys(values))
    return unique


def select_volume_input_policy(
    fingerprint: Mapping[str, Any] | None = None,
    *,
    strategy: str | None = None,
    crop_shape: Sequence[int] | None = None,
    shape_buckets: Any = None,
) -> VolumeInputPolicy:
    """Resolve a bounded 3D policy from fingerprint recommendations.

    The fingerprint is advisory input selection, never an excuse to create a
    data-dependent accelerator shape.  The returned bucket table is fixed for
    the complete run and is suitable for both CUDA and TPU collation.
    """

    fingerprint = fingerprint or {}
    raw_strategy = str(strategy or fingerprint.get("input_strategy", "fixed_crop")).lower().replace("-", "_")
    aliases = {"crop": "fixed_crop", "low_res": "low_resolution_global", "global_local": "global_local"}
    raw_strategy = aliases.get(raw_strategy, raw_strategy)
    if raw_strategy not in _ALLOWED_VOLUME_STRATEGIES:
        raise RecipeConfigurationError(f"unknown 3D input strategy {raw_strategy!r}")
    recommended = fingerprint.get("recommended_shape_buckets", ())
    recommended_shape: tuple[int, int, int] | None = None
    if isinstance(recommended, list | tuple):
        for entry in recommended:
            if isinstance(entry, Mapping) and str(entry.get("kind", "")) in {"3d_patch", "volume_3d"}:
                try:
                    recommended_shape = _shape_tuple(entry.get("shape"), "fingerprint 3d bucket")
                except RecipeConfigurationError:
                    continue
                break
    resolved_crop = (
        _shape_tuple(crop_shape, "crop_shape") if crop_shape is not None else recommended_shape or (96, 96, 96)
    )
    buckets = _parse_shape_buckets(shape_buckets if shape_buckets is not None else recommended, resolved_crop)
    buckets = tuple(_bounded_bucket(value) for value in buckets)
    global_shape = (
        _shape_tuple(fingerprint["global_shape"], "global_shape") if fingerprint.get("global_shape") else None
    )
    local_shape = _shape_tuple(fingerprint["local_shape"], "local_shape") if fingerprint.get("local_shape") else None
    if raw_strategy == "global_local":
        global_shape = global_shape or buckets[-1]
        local_shape = local_shape or buckets[0]
    return VolumeInputPolicy(
        strategy=raw_strategy,
        crop_shape=resolved_crop,
        shape_buckets=tuple(dict.fromkeys(buckets)),
        fingerprint_hash=(str(fingerprint["fingerprint_hash"]) if fingerprint.get("fingerprint_hash") else None),
        global_shape=global_shape,
        local_shape=local_shape,
    )


@dataclass(frozen=True)
class Phase14RecipeMetadata:
    """Pinned decisions and observability fields carried by every recipe."""

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
    crop_strategy: str
    crop_shape: tuple[int, int, int]
    shape_buckets: tuple[tuple[int, int, int], ...]
    microbatch_per_device: int
    world_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    memory_cap_gb: float
    tpu_status: str
    unsupported_xla_ops: tuple[str, ...] = ()
    custom_cuda_dependencies: tuple[str, ...] = ()
    visual_token_count: int | None = None
    text_token_count: int | None = None
    bridge_type: str | None = None
    cached_tokens: bool = False
    selector: str | None = None
    selector_revision: str | None = None
    slice_count: int | None = None
    query_count: int | None = None
    positive_patch_rate: float | None = None
    positive_sampling_ratio: float | None = None
    deep_supervision: bool = False
    blend_mode: str = "constant"
    backend_observability: dict[str, Any] = field(default_factory=dict)
    baseline: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": PHASE14_RECIPE_VERSION,
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
            "crop_strategy": self.crop_strategy,
            "crop_shape": list(self.crop_shape),
            "shape_buckets": [list(value) for value in self.shape_buckets],
            "global_batch_semantics": {
                "microbatch_per_device": self.microbatch_per_device,
                "world_size": self.world_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "global_batch_size": self.global_batch_size,
                "formula": "microbatch_per_device * world_size * gradient_accumulation_steps",
            },
            "memory_cap_gb": self.memory_cap_gb,
            "tpu_status": self.tpu_status,
            "unsupported_xla_ops": list(self.unsupported_xla_ops),
            "custom_cuda_dependencies": list(self.custom_cuda_dependencies),
            "visual_token_count": self.visual_token_count,
            "text_token_count": self.text_token_count,
            "bridge_type": self.bridge_type,
            "cached_tokens": self.cached_tokens,
            "selector": self.selector,
            "selector_revision": self.selector_revision,
            "slice_count": self.slice_count,
            "query_count": self.query_count,
            "positive_patch_rate": self.positive_patch_rate,
            "positive_sampling_ratio": self.positive_sampling_ratio,
            "deep_supervision": self.deep_supervision,
            "blend_mode": self.blend_mode,
            "backend_observability": dict(self.backend_observability),
            "baseline": self.baseline,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class Phase14RecipeBuild:
    """Direct factory result for offline tests and acceptance tooling."""

    model: nn.Module
    task: nn.Module
    train_data: list[MedicalBatch]
    metadata: Phase14RecipeMetadata


RecipeBuild = Phase14RecipeBuild


@dataclass(frozen=True)
class _Phase14Options:
    family: str
    recipe_id: str
    backbone: str
    stage: str
    mode: str
    offline_tiny: bool
    modality: Modality
    channels: int
    hidden_size: int
    depth: int
    heads: int
    construction_seed: int
    policy: VolumeInputPolicy
    visual_token_count: int
    text_token_count: int
    bridge_type: str
    cache_tokens: bool
    selector: str
    slice_count: int
    slice_source_count: int
    slice_image_size: int
    query_count: int
    positive_ratio: float
    deep_supervision: bool
    blend_mode: str
    overlap: float
    dataset_id: str
    dataset_revision: str
    preprocessing_revision: str
    model_revision: str
    prompt_revision: str | None
    tpu_status: str
    baseline: str
    use_lora: bool
    stage4_features: bool


def _canonical_family(value: Any) -> str:
    family = str(value or "classification").strip().lower().replace("-", "_")
    aliases = {
        "cls": "classification",
        "native_3d": "native_3d_vlm",
        "native_vlm": "native_3d_vlm",
        "3d_vlm": "native_3d_vlm",
        "native_3d_classification": "classification",
        "native_3d_segmentation": "segmentation",
        "slice_vlm": "slice_sequence_vlm",
        "slice_sequence": "slice_sequence_vlm",
        "language_segmentation": "language_conditioned_segmentation",
        "language_conditioned_3d_segmentation": "language_conditioned_segmentation",
        "seg": "segmentation",
    }
    family = aliases.get(family, family)
    allowed = {
        "classification",
        "segmentation",
        "native_3d_vlm",
        "slice_sequence_vlm",
        "language_conditioned_segmentation",
    }
    if family not in allowed:
        raise RecipeConfigurationError(f"unknown Phase 14 recipe family {family!r}")
    return family


def _options(config: RunConfig) -> _Phase14Options:
    raw = dict(config.recipe)
    family = _canonical_family(raw.get("family", raw.get("type", "classification")))
    backbone = str(raw.get("backbone", "ct-fm")).lower()
    default_modality = (
        Modality.MULTI_IMAGE_2D
        if family == "slice_sequence_vlm"
        else (Modality.MRI_3D if "triad" in backbone else Modality.CT_3D)
    )
    try:
        modality = Modality(str(raw.get("modality", default_modality.value)).upper())
    except ValueError as exc:
        raise RecipeConfigurationError(f"unknown Phase 14 modality {raw.get('modality')!r}") from exc
    if family == "slice_sequence_vlm" and modality is not Modality.MULTI_IMAGE_2D:
        raise RecipeConfigurationError("slice-sequence VLM recipes must use MULTI_IMAGE_2D")
    if family != "slice_sequence_vlm" and modality not in NATIVE_3D_MODALITIES:
        raise RecipeConfigurationError("native Phase 14 families must use CT_3D, MRI_3D, or MULTI_SERIES_3D")
    stage = str(raw.get("stage", "A")).upper().replace("STAGE", "")
    if stage not in {"A", "B", "C", "D", "1", "2", "3", "4"}:
        raise RecipeConfigurationError("Phase 14 stage must be A/B/C/D or 1/2/3/4")
    stage = {"1": "A", "2": "B", "3": "C", "4": "D"}.get(stage, stage)
    mode = str(raw.get("mode", "offline_tiny" if raw.get("offline_tiny", True) else "production")).lower()
    offline = bool(raw.get("offline_tiny", mode in {"offline_tiny", "smoke", "tiny", "contract"}))
    default_shape: tuple[int, int, int] = (
        (16, 16, 16) if offline else _shape_tuple(raw.get("crop_shape", (96, 96, 96)), "crop_shape")
    )
    fingerprint = raw.get("dataset_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, Mapping):
        raise RecipeConfigurationError("recipe.dataset_fingerprint must be a mapping")
    policy = select_volume_input_policy(
        fingerprint,
        strategy=raw.get("input_strategy", raw.get("crop_strategy")),
        crop_shape=raw.get("crop_shape", default_shape),
        shape_buckets=raw.get("shape_buckets", raw.get("volume_shape_buckets")),
    )
    if offline:
        policy = VolumeInputPolicy(
            strategy=policy.strategy,
            crop_shape=default_shape,
            shape_buckets=(default_shape,),
            fingerprint_hash=policy.fingerprint_hash,
            global_shape=policy.global_shape,
            local_shape=policy.local_shape,
        )
    visual_default = 64 if family in {"native_3d_vlm", "slice_sequence_vlm"} else 32
    visual_tokens = int(raw.get("visual_tokens", raw.get("visual_token_count", visual_default)))
    if visual_tokens not in (32, 64, 128):
        raise RecipeConfigurationError("Phase 14 visual token buckets are exactly 32, 64, or 128")
    text_default = 8 if offline else 512
    text_tokens = int(raw.get("text_tokens", raw.get("text_token_count", text_default)))
    if text_tokens < 5:
        raise RecipeConfigurationError("Phase 14 language recipes require at least five text tokens")
    bridge = str(raw.get("bridge", raw.get("bridge_type", "perceiver"))).lower()
    if bridge in {"perceiver_resampler", "resampler"}:
        bridge = "perceiver"
    if bridge not in {"linear", "perceiver"}:
        raise RecipeConfigurationError("Phase 14 bridges must be linear or perceiver")
    selector = str(raw.get("selector", raw.get("slice_selector", "uniform"))).lower().replace("-", "_")
    slice_count = int(raw.get("slice_count", 4 if offline else 32))
    source_count = int(raw.get("slice_source_count", max(slice_count, 8 if offline else slice_count * 2)))
    if slice_count <= 0 or source_count < slice_count:
        raise RecipeConfigurationError("slice_source_count must be >= positive slice_count")
    image_size = int(raw.get("slice_image_size", 32 if offline else 448))
    if image_size <= 0:
        raise RecipeConfigurationError("slice_image_size must be positive")
    query_count = int(raw.get("query_count", 4))
    if query_count <= 0:
        raise RecipeConfigurationError("query_count must be positive")
    positive_ratio = float(raw.get("positive_sampling_ratio", raw.get("positive_ratio", 0.75)))
    if not 0.0 <= positive_ratio <= 1.0:
        raise RecipeConfigurationError("positive_sampling_ratio must be in [0, 1]")
    blend_mode = str(raw.get("blend_mode", "constant")).lower()
    if blend_mode not in _ALLOWED_BLEND_MODES:
        raise RecipeConfigurationError(f"blend_mode must be one of {sorted(_ALLOWED_BLEND_MODES)}")
    overlap = float(raw.get("window_overlap", raw.get("overlap", 0.25)))
    if not 0.0 <= overlap < 1.0:
        raise RecipeConfigurationError("window_overlap must be in [0, 1)")
    if "triad" in backbone:
        channels = int(raw.get("channels", 2))
    else:
        channels = int(raw.get("channels", 1))
    if channels <= 0:
        raise RecipeConfigurationError("channels must be positive")
    return _Phase14Options(
        family=family,
        recipe_id=str(raw.get("id", raw.get("name", f"phase14-{family}"))),
        backbone=backbone,
        stage=stage,
        mode=mode,
        offline_tiny=offline,
        modality=modality,
        channels=channels,
        hidden_size=int(raw.get("hidden_size", 32 if offline else 96)),
        depth=int(raw.get("depth", 2 if offline else 4)),
        heads=int(raw.get("heads", 4)),
        construction_seed=int(raw.get("construction_seed", config.seed)),
        policy=policy,
        visual_token_count=visual_tokens,
        text_token_count=text_tokens,
        bridge_type=bridge,
        cache_tokens=bool(raw.get("cache_tokens", raw.get("cached_spatial_tokens", False))),
        selector=selector,
        slice_count=slice_count,
        slice_source_count=source_count,
        slice_image_size=image_size,
        query_count=query_count,
        positive_ratio=positive_ratio,
        deep_supervision=bool(raw.get("deep_supervision", False)),
        blend_mode=blend_mode,
        overlap=overlap,
        dataset_id=str(raw.get("dataset_id", config.dataset.get("id", "phase14-synthetic-v1"))),
        dataset_revision=str(raw.get("dataset_revision", config.dataset.get("revision", "synthetic-v1"))),
        preprocessing_revision=str(
            raw.get("preprocessing_revision", config.preprocessing_hash or "phase14-preprocess-v1")
        ),
        model_revision=str(
            raw.get("model_revision", config.base_model_revision or raw.get("backbone_revision", "local-tiny"))
        ),
        prompt_revision=(str(raw["prompt_revision"]) if raw.get("prompt_revision") is not None else None),
        tpu_status=str(raw.get("tpu_status", "CPU_CONTRACT_ONLY")),
        baseline=str(raw.get("baseline", "native_encoder_decoder")),
        use_lora=bool(raw.get("use_lora", config.peft.enabled)),
        stage4_features=bool(raw.get("stage4_features", stage == "D")),
    )


def _spatial_metadata(
    shape: tuple[int, int, int],
    *,
    original_shape: tuple[int, int, int] | None = None,
    origin: tuple[int, int, int] = (0, 0, 0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> SpatialMetadata:
    affine = torch.eye(4, dtype=torch.float64)
    # SpatialMetadata uses D,H,W for tensor axes while affine exposes x,y,z.
    affine[0, 3] = float(origin[2]) * spacing[2]
    affine[1, 3] = float(origin[1]) * spacing[1]
    affine[2, 3] = float(origin[0]) * spacing[0]
    return SpatialMetadata(
        original_shape=original_shape or shape,
        current_shape=shape,
        affine=affine,
        original_affine=torch.eye(4, dtype=torch.float64),
        spacing_mm=spacing,
        orientation="RAS",
        anatomical_axes=("S", "A", "R"),
    )


def _build_native_adapter(options: _Phase14Options) -> GenericMONAI3DAdapter:
    if not options.offline_tiny:
        raise RecipeConfigurationError(
            "Phase 14 production adapters require an approved local checkpoint; "
            f"offline_tiny=false for {options.backbone!r} is intentionally blocked"
        )
    seed = options.construction_seed
    if "triad-simmim" in options.backbone or options.backbone.endswith("simmim"):
        return TriadSimMIMAdapter.build_tiny(construction_seed=seed)
    if "triad" in options.backbone:
        return TriadMAEAdapter.build_tiny(construction_seed=seed)
    if "flexict" in options.backbone:
        return FlexiCT3DAdapter.build_tiny(construction_seed=seed)
    if "ct-fm" in options.backbone or "ctfm" in options.backbone:
        return CTFMAdapter.build_tiny(construction_seed=seed)
    return GenericMONAI3DAdapter.build_tiny(
        modality=options.modality if options.modality in {Modality.CT_3D, Modality.MRI_3D} else Modality.CT_3D,
        channels=options.channels,
        construction_seed=seed,
    )


def _build_slice_adapter(options: _Phase14Options) -> TinyVisualAdapter:
    if not options.offline_tiny:
        raise RecipeConfigurationError("production slice-sequence towers require a registered local checkpoint")
    return TinyVisualAdapter(
        model_id=f"{options.backbone}-offline-tiny",
        modality=Modality.CT_2D_SLICE,
        image_size=options.slice_image_size,
        channels=1,
        hidden_size=options.hidden_size,
        patch_size=max(1, options.slice_image_size // 4),
        construction_seed=options.construction_seed,
    )


def _text_payload(values: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(values.shape[0])
    input_ids = torch.full((batch, length), 3, dtype=torch.long)
    input_ids[:, 0] = 1
    input_ids[:, 1] = 5
    input_ids[:, 2] = 6
    input_ids[:, 3] = 7
    answer = (values.float().flatten(1).mean(dim=1) > 0).to(torch.long) + 10
    input_ids[:, 4] = answer
    if length > 5:
        input_ids[:, 5:] = 2
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(attention)
    prompt_mask[:, :4] = True
    labels[prompt_mask] = -100
    return input_ids, attention, labels


def _classification_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    adapter = _build_native_adapter(options)
    shape = _shape_tuple(adapter.preprocess.spatial_shape, "adapter spatial_shape")
    channels = adapter.preprocess.channels
    batch_size = int(config.batch.microbatch_per_device)
    generator = torch.Generator().manual_seed(config.seed)
    values = torch.randn((batch_size, channels, *shape), generator=generator)
    labels = (values.float().mean(dim=(1, 2, 3, 4)) > 0).to(torch.long)
    if bool(labels.all()) or not bool(labels.any()):
        labels = torch.arange(batch_size) % 2
        values = values + (labels.float() * 2 - 1).reshape(-1, 1, 1, 1, 1)
    task_name = str(config.task.get("type", config.task.get("name", "BINARY_CLASSIFICATION"))).upper()
    target: torch.Tensor = labels.float().unsqueeze(1) if "BINARY" in task_name else labels
    metadata: list[SpatialMetadata | None] = [_spatial_metadata(shape) for _ in range(batch_size)]
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase14-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            labels=labels,
            spatial_metadata=metadata,
            task_targets={
                "classification": target,
                "patient_ids": tuple(f"patient-{i // 2}" for i in range(batch_size)),
                "study_ids": tuple(f"study-{i}" for i in range(batch_size)),
                "crop_origin": tuple((0, 0, 0) for _ in range(batch_size)),
                "crop_strategy": options.policy.strategy,
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            },
        )
    ]


def _segmentation_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    adapter = _build_native_adapter(options)
    patch_shape = _shape_tuple(adapter.preprocess.spatial_shape, "adapter spatial_shape")
    channels = adapter.preprocess.channels
    batch_size = int(config.batch.microbatch_per_device)
    source_shape: tuple[int, int, int] = (
        patch_shape[0] + max(4, patch_shape[0] // 2),
        patch_shape[1] + max(4, patch_shape[1] // 2),
        patch_shape[2] + max(4, patch_shape[2] // 2),
    )
    generator = torch.Generator().manual_seed(config.seed)
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    metas: list[SpatialMetadata | None] = []
    origins: list[tuple[int, int, int]] = []
    positive_flags: list[bool] = []
    for index in range(batch_size):
        image = torch.rand((channels, *source_shape), generator=generator)
        mask = torch.zeros(source_shape, dtype=torch.float32)
        z0 = 1 + index % 2
        y0 = max(1, source_shape[1] // 3)
        x0 = max(1, source_shape[2] // 3)
        z1 = min(source_shape[0], z0 + max(3, patch_shape[0] // 3))
        y1 = min(source_shape[1], y0 + max(3, patch_shape[1] // 3))
        x1 = min(source_shape[2], x0 + max(3, patch_shape[2] // 3))
        mask[z0:z1, y0:y1, x0:x1] = 1.0
        sampler = ForegroundPatchSampler(patch_shape, positive_ratio=options.positive_ratio, seed=config.seed + index)
        patch = sampler.sample(image, mask=mask, spacing_mm=(1.0, 1.0, 1.0))
        assert patch.mask is not None
        images.append(patch.image)
        masks.append(patch.mask.unsqueeze(0))
        origins.append(patch.info.origin)
        positive_flags.append(bool(patch.info.target_positive))
        metas.append(_spatial_metadata(patch_shape, original_shape=source_shape, origin=patch.info.origin))
    image_values = torch.stack(images)
    segmentation = torch.stack(masks)
    voxel_mask = torch.ones((batch_size, *patch_shape), dtype=torch.bool)
    positive_rate = float(sum(positive_flags) / max(1, len(positive_flags)))
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase14-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=image_values,
            image_mask=torch.ones(batch_size, dtype=torch.bool),
            spatial_metadata=metas,
            task_targets={
                "segmentation": segmentation,
                "voxel_mask": voxel_mask,
                "crop_origin": tuple(origins),
                "positive_patch_rate": positive_rate,
                "positive_sampling_ratio": options.positive_ratio,
                "sampling_policy": "foreground_lesion_centered_mixture",
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            },
        )
    ]


def _native_cache(
    adapter: GenericMONAI3DAdapter,
    values: torch.Tensor,
    modality: Modality,
    metas: list[SpatialMetadata | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adapter.eval()
    with torch.no_grad():
        batch = MedicalBatch(
            modality=modality,
            sample_ids=[f"cache-{i}" for i in range(int(values.shape[0]))],
            pixel_values=values,
            spatial_metadata=metas,
        )
        output = adapter.encode(batch, output_spec=OutputSpec(pooled=True, spatial_tokens=True, token_coordinates=True))
    assert output.spatial_tokens is not None and output.token_mask is not None and output.token_coordinates is not None
    return output.spatial_tokens.detach(), output.token_mask.detach(), output.token_coordinates.detach()


def _native_vlm_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    adapter = _build_native_adapter(options)
    shape = _shape_tuple(adapter.preprocess.spatial_shape, "adapter spatial_shape")
    channels = adapter.preprocess.channels
    batch_size = int(config.batch.microbatch_per_device)
    generator = torch.Generator().manual_seed(config.seed)
    values = torch.randn((batch_size, channels, *shape), generator=generator)
    metas: list[SpatialMetadata | None] = [_spatial_metadata(shape) for _ in range(batch_size)]
    input_ids, attention, labels = _text_payload(values, options.text_token_count)
    visual_metadata = tuple(
        {
            "patient_id": f"patient-{i // 2}",
            "study_id": f"study-{i}",
            "crop_origin": (0, 0, 0),
            "spacing_mm": (1.0, 1.0, 1.0),
            "sequence_order": tuple(adapter.preprocess.sequence_order),
            "architecture_family": "native_3d",
        }
        for i in range(batch_size)
    )
    targets: dict[str, Any] = {
        "language_labels": labels,
        "prompt_token_mask": attention.clone().fill_(False),
        "visual_metadata": visual_metadata,
        "sample_mask": torch.ones(batch_size, dtype=torch.bool),
    }
    targets["prompt_token_mask"][:, :4] = True
    if options.cache_tokens:
        cached, cached_mask, cached_coordinates = _native_cache(adapter, values, options.modality, metas)
        targets.update(
            {
                "cached_visual_tokens": cached,
                "cached_visual_token_mask": cached_mask,
                "cached_token_coordinates": cached_coordinates,
                "cache_revision": options.model_revision,
            }
        )
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase14-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            image_mask=torch.ones(batch_size, dtype=torch.bool),
            spatial_metadata=metas,
            input_ids=input_ids,
            attention_mask=attention,
            task_targets=targets,
        )
    ]


def _slice_records(sample: int, options: _Phase14Options, generator: torch.Generator) -> tuple[SliceRecord, ...]:
    records: list[SliceRecord] = []
    for index in range(options.slice_source_count):
        image = torch.randn((1, options.slice_image_size, options.slice_image_size), generator=generator)
        records.append(
            SliceRecord(
                index=index,
                image=image,
                physical_z_mm=float(index * 2.5),
                series_order=index,
                window="soft" if index % 2 == 0 else "bone",
                mri_sequence="T1" if index % 2 == 0 else "T2",
                anatomy_score=float((index + sample) % 5),
                report_score=float((options.slice_source_count - index + sample) % 7),
                lesion_score=float(index % 3),
            )
        )
    return tuple(records)


def _slice_vlm_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    batch_size = int(config.batch.microbatch_per_device)
    generator = torch.Generator().manual_seed(config.seed)
    images: list[torch.Tensor] = []
    metadata: list[tuple[dict[str, object], ...]] = []
    for sample in range(batch_size):
        records = _slice_records(sample, options, generator)
        selections = build_slice_selector(options.selector, count=options.slice_count).select(records)
        by_index = {record.index: record for record in records}
        images.append(torch.stack([by_index[item.index].image for item in selections]))
        metadata.append(selections_to_metadata(selections))
    values = torch.stack(images)  # [B, I, C, H, W], explicitly MULTI_IMAGE_2D
    input_ids, attention, labels = _text_payload(values, options.text_token_count)
    return [
        MedicalBatch(
            modality=Modality.MULTI_IMAGE_2D,
            sample_ids=[f"phase14-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            image_mask=torch.ones(batch_size, options.slice_count, dtype=torch.bool),
            input_ids=input_ids,
            attention_mask=attention,
            task_targets={
                "language_labels": labels,
                "prompt_token_mask": torch.cat(
                    (
                        torch.ones(batch_size, 4, dtype=torch.bool),
                        torch.zeros(batch_size, max(0, options.text_token_count - 4), dtype=torch.bool),
                    ),
                    dim=1,
                ),
                "slice_metadata": tuple(metadata),
                "visual_metadata": tuple(
                    {
                        "slice_metadata": sample,
                        "selector": options.selector,
                        "selector_revision": SLICE_SELECTOR_VERSION,
                    }
                    for sample in metadata
                ),
                "selector": options.selector,
                "selector_revision": SLICE_SELECTOR_VERSION,
                "slice_count": options.slice_count,
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            },
        )
    ]


def _query_tokens(text: str, length: int, *, vocab_size: int = 64) -> torch.Tensor:
    pieces = [3 + (sum(bytearray(piece.encode("utf-8"))) % (vocab_size - 3)) for piece in text.split()]
    ids = [1, *pieces, 2]
    ids = (ids + [2] * length)[:length]
    return torch.tensor(ids, dtype=torch.long)


def _language_segmentation_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    adapter = _build_native_adapter(options)
    shape = _shape_tuple(adapter.preprocess.spatial_shape, "adapter spatial_shape")
    channels = adapter.preprocess.channels
    batch_size = int(config.batch.microbatch_per_device)
    generator = torch.Generator().manual_seed(config.seed)
    values = torch.rand((batch_size, channels, *shape), generator=generator)
    segmentation = torch.zeros((batch_size, 1, *shape), dtype=torch.float32)
    queries = (
        ("left lung lesion", "laterality"),
        ("right lung lesion", "laterality"),
        ("liver lesion", "multiple_target"),
        ("nonexistent anatomy", "absent_target"),
        ("unclear finding", "ambiguous_query"),
    )
    query_text: list[str] = []
    query_behavior: list[str] = []
    query_mask = torch.zeros(batch_size, dtype=torch.bool)
    for row in range(batch_size):
        text, behavior = queries[row % len(queries)]
        query_text.append(text)
        query_behavior.append(behavior)
        if behavior == "absent_target" or behavior == "ambiguous_query":
            continue
        query_mask[row] = True
        z0, z1 = 2, max(3, shape[0] - 2)
        y0, y1 = 2, max(3, shape[1] - 2)
        x0, x1 = (
            (2, max(3, shape[2] // 2)) if text.startswith("left") else (max(2, shape[2] // 2), max(3, shape[2] - 2))
        )
        segmentation[row, 0, z0:z1, y0:y1, x0:x1] = 1.0
    input_ids = torch.stack([_query_tokens(text, options.text_token_count) for text in query_text])
    attention = input_ids.ne(2) | (torch.arange(options.text_token_count).unsqueeze(0) == 0)
    metas: list[SpatialMetadata | None] = [_spatial_metadata(shape) for _ in range(batch_size)]
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase14-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            image_mask=torch.ones(batch_size, dtype=torch.bool),
            spatial_metadata=metas,
            input_ids=input_ids,
            attention_mask=attention,
            task_targets={
                "segmentation": segmentation,
                "query_mask": query_mask,
                "query_text": tuple(query_text),
                "query_behavior": tuple(query_behavior),
                "query_count": options.query_count,
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            },
        )
    ]


def _language_adapter(options: _Phase14Options, *, native: bool) -> GenericHFCausalLMAdapter:
    max_text = max(128, options.text_token_count)
    if native:
        return MedGemmaAdapter.build_tiny(
            model_id="medgemma-native-3d-offline-tiny",
            hidden_size=options.hidden_size,
            vocab_size=64,
            construction_seed=options.construction_seed,
            max_text_tokens=max_text,
        )
    return GenericHFCausalLMAdapter.build_tiny(
        model_id="generic-slice-vlm-offline-tiny",
        hidden_size=options.hidden_size,
        vocab_size=64,
        construction_seed=options.construction_seed,
        max_text_tokens=max_text,
    )


def _make_bridge(
    *,
    source_dim: int,
    target_dim: int,
    input_tokens: int,
    output_tokens: int,
    modality: Modality,
    bridge_type: str,
) -> VisionLanguageBridge:
    coordinate_system = CoordinateSystem.MILLIMETERS if modality.is_volumetric else CoordinateSystem.NORMALIZED_IMAGE
    if bridge_type == "linear":
        return LinearVisionLanguageBridge(
            source_dim=source_dim,
            target_dim=target_dim,
            output_tokens=output_tokens,
            max_input_tokens=input_tokens,
            source_modality=modality,
            coordinate_system=coordinate_system,
        )
    return PerceiverResamplerBridge(
        query_count=output_tokens,
        heads=4,
        source_dim=source_dim,
        target_dim=target_dim,
        output_tokens=output_tokens,
        max_input_tokens=input_tokens,
        source_modality=modality,
        coordinate_system=coordinate_system,
    )


class _Native3DClassificationModel(nn.Module):
    def __init__(self, vision: GenericMONAI3DAdapter) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, batch: MedicalBatch) -> EncoderOutput:
        return self.vision.encode(batch, output_spec=OutputSpec(pooled=True, spatial_tokens=True))

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> EncoderOutput:
        del mode
        return self.forward(batch)


class _Native3DSegmentationModel(nn.Module):
    def __init__(self, vision: GenericMONAI3DAdapter) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, batch: MedicalBatch) -> dict[str, Any]:
        output = self.vision.encode(batch, output_spec=OutputSpec(pooled=True, spatial_tokens=True, feature_maps=True))
        assert output.feature_maps is not None
        spatial = (
            tuple(int(value) for value in batch.pixel_values.shape[-3:]) if batch.pixel_values is not None else None
        )
        maps = output.feature_maps
        if spatial is not None:
            maps = tuple(F.interpolate(value, size=spatial, mode="trilinear", align_corners=False) for value in maps)
        return {
            "feature_maps": maps,
            "encoder_output": output,
            "crop_origin": batch.task_targets.get("crop_origin"),
            "positive_patch_rate": batch.task_targets.get("positive_patch_rate", 0.0),
            "host_transform_time_ms": batch.task_targets.get("host_transform_time_ms", 0.0),
        }

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> dict[str, Any]:
        del mode
        return self.forward(batch)


@dataclass(frozen=True, eq=False)
class Phase14LanguageOutput:
    language: LanguageOutput
    visual_tokens: ProjectedVisualTokens | None
    source_coordinates: torch.Tensor | None
    visual_metadata: tuple[dict[str, Any], ...]
    mode: str
    family: str


class _Phase14LanguageModelBase(nn.Module):
    def __init__(
        self,
        language: GenericHFCausalLMAdapter,
        *,
        text_tokens: int,
        family: str,
    ) -> None:
        super().__init__()
        self.language = language
        self.text_tokens = int(text_tokens)
        self.family = family

    def _text(self, batch: MedicalBatch) -> tuple[TokenizedText, torch.Tensor]:
        if batch.input_ids is None or batch.attention_mask is None:
            raise ShapeContractError("Phase 14 VLM batches require input_ids and attention_mask")
        labels = batch.task_targets.get("language_labels")
        if not isinstance(labels, torch.Tensor):
            raise ShapeContractError("Phase 14 VLM batch is missing language_labels")
        prompt_mask = batch.task_targets.get("prompt_token_mask")
        metadata: dict[str, Any] = {}
        if isinstance(prompt_mask, torch.Tensor):
            metadata["prompt_token_mask"] = prompt_mask
        return TokenizedText(batch.input_ids, batch.attention_mask, metadata=metadata), labels

    def _forward_language(
        self,
        batch: MedicalBatch,
        visual: ProjectedVisualTokens,
        *,
        mode: str,
        coordinates: torch.Tensor | None,
        metadata: tuple[dict[str, Any], ...],
    ) -> Phase14LanguageOutput:
        text, labels = self._text(batch)
        language = self.language.forward_with_visual_tokens(text, visual, labels)
        if not isinstance(language, LanguageOutput):
            raise ShapeContractError("Phase 14 language adapter returned an invalid LanguageOutput")
        auxiliary = dict(language.auxiliary)
        auxiliary.update(
            {
                "source_coordinates": coordinates,
                "visual_metadata": metadata,
                "visual_mode": mode,
                "visual_token_mask": visual.token_mask,
                "recipe_family": self.family,
            }
        )
        return Phase14LanguageOutput(
            language=LanguageOutput(
                logits=language.logits,
                loss=language.loss,
                hidden_states=language.hidden_states,
                auxiliary=auxiliary,
            ),
            visual_tokens=visual,
            source_coordinates=coordinates,
            visual_metadata=metadata,
            mode=mode,
            family=self.family,
        )

    def _generate_with_visual(
        self,
        batch: MedicalBatch,
        visual: ProjectedVisualTokens,
        generation_config: GenerationConfig,
    ) -> GeneratedText:
        text, _ = self._text(batch)
        return self.language.generate(text, visual, generation_config)


class _Native3DVisualLanguageModel(_Phase14LanguageModelBase):
    def __init__(
        self,
        vision: GenericMONAI3DAdapter,
        language: GenericHFCausalLMAdapter,
        bridge: CoordinateAwareBridge,
        *,
        visual_token_count: int,
        text_tokens: int,
        stage4_features: bool,
    ) -> None:
        super().__init__(language, text_tokens=text_tokens, family="native_3d_vlm")
        self.vision = vision
        self.bridge = bridge
        self.visual_token_count = int(visual_token_count)
        self.stage4_features = bool(stage4_features)
        self.region_projection = nn.Linear(vision._hidden_size, vision._hidden_size) if stage4_features else None

    def _source(self, batch: MedicalBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = batch.task_targets.get("cached_visual_tokens")
        cached_mask = batch.task_targets.get("cached_visual_token_mask")
        cached_coordinates = batch.task_targets.get("cached_token_coordinates")
        if isinstance(cached, torch.Tensor):
            if (
                cached.ndim != 3
                or not isinstance(cached_mask, torch.Tensor)
                or not isinstance(cached_coordinates, torch.Tensor)
            ):
                raise ShapeContractError("cached native 3D tokens require tokens, mask, and coordinates")
            return cached, cached_mask.to(dtype=torch.bool), cached_coordinates
        output = self.vision.encode(
            batch,
            output_spec=OutputSpec(pooled=True, spatial_tokens=True, token_coordinates=True),
        )
        if output.spatial_tokens is None or output.token_mask is None or output.token_coordinates is None:
            raise ShapeContractError("native 3D VLM source must expose tokens, mask, and physical coordinates")
        return output.spatial_tokens, output.token_mask, output.token_coordinates

    @staticmethod
    def _normalized_positions(coordinates: torch.Tensor) -> torch.Tensor:
        scale = coordinates.detach().abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return (coordinates / scale).clamp(-1.0, 1.0)

    def _visual(
        self,
        batch: MedicalBatch,
        *,
        mode: str,
    ) -> tuple[ProjectedVisualTokens, torch.Tensor, tuple[dict[str, Any], ...]]:
        tokens, mask, coordinates = self._source(batch)
        if mode == "none":
            tokens = torch.zeros_like(tokens)
            mask = torch.zeros_like(mask)
            coordinates = torch.zeros_like(coordinates)
        elif mode == "shuffle":
            if int(tokens.shape[0]) > 1:
                tokens = torch.roll(tokens, shifts=1, dims=0)
                mask = torch.roll(mask, shifts=1, dims=0)
                coordinates = torch.roll(coordinates, shifts=1, dims=0)
        elif mode != "image":
            raise ValueError(f"unknown native 3D visual mode {mode!r}")
        spacing = [
            metadata.spacing_mm if metadata is not None and metadata.spacing_mm is not None else (1.0, 1.0, 1.0)
            for metadata in batch.spatial_metadata
        ]
        spacing_tensor = torch.as_tensor(spacing, device=coordinates.device, dtype=coordinates.dtype)
        if spacing_tensor.ndim == 2:
            spacing_tensor = spacing_tensor.unsqueeze(1).expand(-1, coordinates.shape[1], -1)
        coordinate_metadata = {"physical_position": coordinates, "spacing": spacing_tensor}
        projected = self.bridge(
            tokens,
            mask,
            coordinates=self._normalized_positions(coordinates),
            coordinate_metadata=coordinate_metadata,
        )
        metadata = tuple(dict(value) for value in batch.task_targets.get("visual_metadata", ()))
        return projected, coordinates, metadata

    def forward(self, batch: MedicalBatch) -> Phase14LanguageOutput:
        return self.forward_mode(batch, mode="image")

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> Phase14LanguageOutput:
        visual, coordinates, metadata = self._visual(batch, mode=mode)
        result = self._forward_language(batch, visual, mode=mode, coordinates=coordinates, metadata=metadata)
        aux = dict(result.language.auxiliary)
        aux["cached_spatial_tokens"] = isinstance(batch.task_targets.get("cached_visual_tokens"), torch.Tensor)
        aux["stage4_features"] = {
            "region_tokens": self.stage4_features,
            "coordinate_output": self.stage4_features,
            "language_conditioned_masks": self.stage4_features,
        }
        language = LanguageOutput(
            logits=result.language.logits,
            loss=result.language.loss,
            hidden_states=result.language.hidden_states,
            auxiliary=aux,
        )
        return Phase14LanguageOutput(
            language, result.visual_tokens, result.source_coordinates, result.visual_metadata, mode, self.family
        )

    def generate(
        self, batch: MedicalBatch, *, mode: str = "image", generation_config: GenerationConfig | None = None
    ) -> GeneratedText:
        visual, _, _ = self._visual(batch, mode=mode)
        return self._generate_with_visual(batch, visual, generation_config or GenerationConfig(max_new_tokens=8))


class _SliceSequenceVisualLanguageModel(_Phase14LanguageModelBase):
    def __init__(
        self,
        vision: TinyVisualAdapter,
        language: GenericHFCausalLMAdapter,
        bridge: VisionLanguageBridge,
        *,
        visual_token_count: int,
        slice_count: int,
        text_tokens: int,
    ) -> None:
        super().__init__(language, text_tokens=text_tokens, family="slice_sequence_vlm")
        self.vision = vision
        self.bridge = bridge
        self.visual_token_count = int(visual_token_count)
        self.slice_count = int(slice_count)

    def _source(self, batch: MedicalBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.pixel_values is None or batch.pixel_values.ndim != 5:
            raise ShapeContractError("slice-sequence VLM expects [B,I,C,H,W] pixel_values")
        batch_size, slice_count, channels, height, width = batch.pixel_values.shape
        if int(slice_count) != self.slice_count:
            raise ShapeContractError(f"slice count bucket is {self.slice_count}; got {slice_count}")
        flattened = batch.pixel_values.reshape(batch_size * slice_count, channels, height, width)
        image_mask = batch.image_mask
        flat_mask = (
            image_mask.reshape(-1)
            if isinstance(image_mask, torch.Tensor)
            else torch.ones(batch_size * slice_count, dtype=torch.bool, device=flattened.device)
        )
        flat_batch = MedicalBatch(
            modality=Modality.CT_2D_SLICE,
            sample_ids=[f"slice-{row}" for row in range(int(batch_size * slice_count))],
            pixel_values=flattened,
            image_mask=flat_mask,
        )
        output = self.vision.encode(flat_batch, output_spec=OutputSpec(pooled=True))
        if output.pooled_embedding is None:
            raise ShapeContractError("slice 2D tower must expose pooled embeddings")
        tokens = output.pooled_embedding.reshape(batch_size, slice_count, -1)
        mask = flat_mask.reshape(batch_size, slice_count)
        raw_metadata = batch.task_targets.get("slice_metadata")
        if not isinstance(raw_metadata, tuple | list) or len(raw_metadata) != batch_size:
            raise ShapeContractError("slice-sequence batch must preserve per-sample slice_metadata")
        z = torch.tensor(
            [[float(item.get("normalized_z", 0.0)) for item in sample] for sample in raw_metadata],
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(-1)
        # A 2D tower receives normalized z as a position side channel only;
        # the modality and acceptance family remain MULTI_IMAGE_2D.
        coordinates = torch.cat((z, torch.zeros_like(z), torch.zeros_like(z)), dim=-1)
        return tokens, mask, coordinates

    def _visual(
        self, batch: MedicalBatch, *, mode: str
    ) -> tuple[ProjectedVisualTokens, torch.Tensor, tuple[dict[str, Any], ...]]:
        tokens, mask, coordinates = self._source(batch)
        if mode == "none":
            tokens = torch.zeros_like(tokens)
            mask = torch.zeros_like(mask)
            coordinates = torch.zeros_like(coordinates)
        elif mode == "shuffle" and int(tokens.shape[0]) > 1:
            tokens = torch.roll(tokens, shifts=1, dims=0)
            mask = torch.roll(mask, shifts=1, dims=0)
            coordinates = torch.roll(coordinates, shifts=1, dims=0)
        elif mode != "image":
            raise ValueError(f"unknown slice-sequence visual mode {mode!r}")
        projected = self.bridge(tokens, mask, coordinates=coordinates)
        metadata = tuple(dict(value) for value in batch.task_targets.get("visual_metadata", ()))
        return projected, coordinates, metadata

    def forward(self, batch: MedicalBatch) -> Phase14LanguageOutput:
        return self.forward_mode(batch, mode="image")

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> Phase14LanguageOutput:
        visual, coordinates, metadata = self._visual(batch, mode=mode)
        return self._forward_language(batch, visual, mode=mode, coordinates=coordinates, metadata=metadata)

    def generate(
        self, batch: MedicalBatch, *, mode: str = "image", generation_config: GenerationConfig | None = None
    ) -> GeneratedText:
        visual, _, _ = self._visual(batch, mode=mode)
        return self._generate_with_visual(batch, visual, generation_config or GenerationConfig(max_new_tokens=8))


class _LanguageConditioned3DModel(nn.Module):
    def __init__(self, vision: GenericMONAI3DAdapter, *, hidden_size: int, text_dim: int, query_count: int) -> None:
        super().__init__()
        self.vision = vision
        self.text_encoder = nn.Embedding(64, text_dim)
        self.query_projection = nn.Linear(text_dim, text_dim)
        self.query_count = int(query_count)

    def _visual_features(self, batch: MedicalBatch) -> tuple[torch.Tensor, ...]:
        output = self.vision.encode(batch, output_spec=OutputSpec(pooled=True, spatial_tokens=True, feature_maps=True))
        if output.feature_maps is None:
            raise ShapeContractError("language-conditioned 3D model requires a feature pyramid")
        spatial = (
            tuple(int(value) for value in batch.pixel_values.shape[-3:]) if batch.pixel_values is not None else None
        )
        maps = output.feature_maps
        if spatial is not None:
            maps = tuple(F.interpolate(value, size=spatial, mode="trilinear", align_corners=False) for value in maps)
        return maps

    def _text(self, batch: MedicalBatch) -> torch.Tensor:
        if batch.input_ids is None:
            raise ShapeContractError("language-conditioned 3D segmentation requires query input_ids")
        return cast(torch.Tensor, self.query_projection(self.text_encoder(batch.input_ids)))

    def forward(self, batch: MedicalBatch) -> dict[str, Any]:
        return {
            "visual_features": self._visual_features(batch),
            "text_embeddings": self._text(batch),
            "text_mask": batch.attention_mask,
            "query_mask": batch.task_targets.get("query_mask"),
            "output_size": tuple(int(value) for value in batch.pixel_values.shape[-3:])
            if batch.pixel_values is not None
            else None,
            "query_behavior": batch.task_targets.get("query_behavior"),
        }

    def predict_queries(
        self,
        batch: MedicalBatch,
        decoder: LanguageConditionedMaskDecoder,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed ``[B,Q,1,D,H,W]`` masks for query evaluation."""

        if query_input_ids.ndim != 3 or query_attention_mask.shape != query_input_ids.shape:
            raise ShapeContractError("query_input_ids and query_attention_mask must be [B,Q,L]")
        if int(query_input_ids.shape[1]) != self.query_count:
            raise ShapeContractError(f"query bucket is Q={self.query_count}")
        if batch.pixel_values is None:
            raise ShapeContractError("language-conditioned 3D segmentation requires pixel_values")
        pixel_values = batch.pixel_values
        visual = self._visual_features(batch)
        outputs: list[torch.Tensor] = []
        for query in range(self.query_count):
            text = self.query_projection(self.text_encoder(query_input_ids[:, query]))
            valid = None if query_mask is None else query_mask[:, query]
            outputs.append(
                decoder(
                    visual,
                    text,
                    text_mask=query_attention_mask[:, query],
                    query_mask=valid,
                    output_size=tuple(int(value) for value in pixel_values.shape[-3:]),
                ).logits
            )
        return torch.stack(outputs, dim=1)


class _Phase14LanguageTask(TaskModuleBase):
    def __init__(self, *, task_type: TaskType) -> None:
        super().__init__(task_type, NATIVE_3D_MODALITIES + (Modality.MULTI_IMAGE_2D,))

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> Any:
        self.check_supported(batch.modality)
        if not isinstance(model_output, Phase14LanguageOutput):
            raise ShapeContractError("Phase 14 VLM task expects Phase14LanguageOutput")
        language = model_output.language
        if language.loss is None:
            raise ShapeContractError("Phase 14 VLM model must return a language loss")
        count = valid_sample_count(batch)
        token_count = int(language.auxiliary.get("supervised_token_count", 0))
        from medfm.core.task import LossOutput

        return LossOutput(
            total=language.loss,
            components={"language": language.loss},
            sample_count=count,
            token_count=token_count,
            diagnostics={
                "task": self.task_type.value,
                "valid_count": detached_count_tensor(count, language.loss),
                "supervised_token_count": token_count,
                "visual_mode": model_output.mode,
                "recipe_family": model_output.family,
            },
        )


class _Phase14LanguageSegmentationTask(LanguageConditionedSegmentationTask):
    """Named task wrapper so optimizer/reporting identifies Phase 14 decoder roles."""


_PHASE14_ROLE_MAP = {
    "query_projection": "bridge",
    "region_projection": "bridge",
}


def _lora(rank: int, *, architecture: str, targets: tuple[str, ...]) -> LoRAConfig:
    return LoRAConfig(
        enabled=True,
        rank=int(rank),
        alpha=float(max(1, rank * 2)),
        dropout=0.0,
        target_policy="explicit",
        target_modules=targets,
        architecture=architecture,
        confirm_target_modules=True,
    )


def _inject_phase14_lora(model: nn.Module, config: RunConfig, options: _Phase14Options) -> None:
    stage = options.stage
    if not options.use_lora or stage not in {"B", "C", "D"}:
        return
    rank = int(config.recipe.get("lora_rank", config.peft.rank))
    if options.family in {"native_3d_vlm", "slice_sequence_vlm"}:
        language = getattr(model, "language", None)
        if isinstance(language, MedGemmaAdapter | GenericHFCausalLMAdapter):
            inject_language_lora(language, _lora(rank, architecture="llm", targets=LANGUAGE_LORA_TARGETS))
    if stage in {"C", "D"} and options.family in {"classification", "segmentation", "native_3d_vlm"}:
        vision = getattr(model, "vision", None)
        if isinstance(vision, GenericMONAI3DAdapter):
            vision.inject_lora(rank=rank, alpha=max(1, rank * 2), dropout=0.0, targets=NATIVE_3D_LORA_TARGETS)


def _phase14_model(config: RunConfig, options: _Phase14Options) -> nn.Module:
    if options.family == "slice_sequence_vlm":
        slice_vision = _build_slice_adapter(options)
        language = _language_adapter(options, native=False)
        bridge = _make_bridge(
            source_dim=options.hidden_size,
            target_dim=options.hidden_size,
            input_tokens=options.slice_count,
            output_tokens=options.visual_token_count,
            modality=Modality.CT_2D_SLICE,
            bridge_type=options.bridge_type,
        )
        return _SliceSequenceVisualLanguageModel(
            slice_vision,
            language,
            bridge,
            visual_token_count=options.visual_token_count,
            slice_count=options.slice_count,
            text_tokens=options.text_token_count,
        )
    native_vision = _build_native_adapter(options)
    if options.family == "classification":
        return _Native3DClassificationModel(native_vision)
    if options.family == "segmentation":
        return _Native3DSegmentationModel(native_vision)
    if options.family == "language_conditioned_segmentation":
        return _LanguageConditioned3DModel(
            native_vision,
            hidden_size=options.hidden_size,
            text_dim=options.hidden_size,
            query_count=options.query_count,
        )
    language = _language_adapter(options, native=True)
    source_tokens = native_vision.preprocess.num_patches
    base_bridge = _make_bridge(
        source_dim=options.hidden_size,
        target_dim=options.hidden_size,
        input_tokens=source_tokens,
        output_tokens=options.visual_token_count,
        modality=options.modality,
        bridge_type=options.bridge_type,
    )
    coordinate_bridge = CoordinateAwareBridge(
        base_bridge, ThreeDCoordinateEncoder(output_dim=options.hidden_size // 2 or 1)
    )
    return _Native3DVisualLanguageModel(
        native_vision,
        language,
        coordinate_bridge,
        visual_token_count=options.visual_token_count,
        text_tokens=options.text_token_count,
        stage4_features=options.stage4_features,
    )


def _phase14_task(config: RunConfig, options: _Phase14Options, model: nn.Module) -> TaskModuleBase:
    if options.family == "classification":
        task_name = str(config.task.get("type", config.task.get("name", "BINARY_CLASSIFICATION"))).upper()
        binary = "BINARY" in task_name
        head_name = str(config.recipe.get("head", "attention" if options.stage in {"B", "D"} else "linear")).lower()
        classes = 1 if binary else 2
        if head_name in {"attention", "attention_pooling", "attn"}:
            head: nn.Module = AttentionPoolingClassificationHead(
                input_dim=options.hidden_size,
                num_classes=classes,
                hidden_dim=int(config.recipe.get("pool_hidden", max(8, options.hidden_size // 2))),
            )
        else:
            head = LinearClassificationHead(options.hidden_size, classes)
        if binary:
            return BinaryClassificationTask(head, supported_modalities=NATIVE_3D_MODALITIES)
        return ClassificationTask(
            head,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            supported_modalities=NATIVE_3D_MODALITIES,
        )
    if options.family == "segmentation":
        decoder = UNetDecoder3D(
            in_channels=options.hidden_size,
            out_channels=1,
            hidden_channels=int(config.recipe.get("decoder_hidden", 8 if options.offline_tiny else 16)),
            deep_supervision=options.deep_supervision,
        )
        return BinarySegmentationTask(decoder, supported_modalities=NATIVE_3D_MODALITIES)
    if options.family == "language_conditioned_segmentation":
        language_decoder = LanguageConditionedMaskDecoder(
            visual_dim=options.hidden_size,
            text_dim=options.hidden_size,
            hidden_dim=int(config.recipe.get("decoder_hidden", options.hidden_size)),
            out_channels=1,
        )
        return _Phase14LanguageSegmentationTask(
            language_decoder,
            supported_modalities=NATIVE_3D_MODALITIES,
        )
    task_name = str(config.task.get("type", config.task.get("name", "VISUAL_QUESTION_ANSWERING"))).upper()
    if "STRUCTURED" in task_name or "FINDING" in task_name:
        task_type = TaskType.STRUCTURED_FINDING_GENERATION
    elif "REPORT" in task_name:
        task_type = TaskType.REPORT_GENERATION
    else:
        task_type = TaskType.VISUAL_QUESTION_ANSWERING
    return _Phase14LanguageTask(task_type=task_type)


def _phase14_data(config: RunConfig, options: _Phase14Options) -> list[MedicalBatch]:
    if options.family == "classification":
        return _classification_data(config, options)
    if options.family == "segmentation":
        return _segmentation_data(config, options)
    if options.family == "native_3d_vlm":
        return _native_vlm_data(config, options)
    if options.family == "slice_sequence_vlm":
        return _slice_vlm_data(config, options)
    return _language_segmentation_data(config, options)


def _backend_observability(config: RunConfig) -> dict[str, Any]:
    return {
        "backend": config.accelerator.backend,
        "compiler_count": 0,
        "graph_count": 0,
        "input_wait_ms": 0.0,
        "host_to_device_ms": 0.0,
        "throughput_examples_per_second": 0.0,
        "peak_vram_gb": None,
        "peak_hbm_gb": None,
        "fallback_operators": (),
        "host_transform_time_ms": 0.0,
        "host_inversion_time_ms": 0.0,
    }


def _metadata(
    config: RunConfig,
    options: _Phase14Options,
    data: list[MedicalBatch],
) -> Phase14RecipeMetadata:
    positive_rate: float | None = None
    if options.family == "segmentation" and data:
        values = [batch.task_targets.get("positive_patch_rate") for batch in data]
        rates = [float(value) for value in values if isinstance(value, int | float)]
        positive_rate = sum(rates) / len(rates) if rates else None
    adapter: GenericMONAI3DAdapter | None = None
    unsupported: tuple[str, ...] = ()
    custom_cuda: tuple[str, ...] = ()
    if options.family != "slice_sequence_vlm":
        try:
            adapter = _build_native_adapter(options)
            unsupported = tuple(getattr(adapter, "_unsupported_xla_ops", ()))
            custom_cuda = tuple(getattr(adapter, "_custom_cuda_dependencies", ()))
        except RecipeConfigurationError:
            pass
    return Phase14RecipeMetadata(
        family=options.family,
        recipe_id=options.recipe_id,
        backbone=options.backbone,
        modality=options.modality.value,
        stage=options.stage,
        mode=options.mode,
        dataset_id=options.dataset_id,
        dataset_revision=options.dataset_revision,
        preprocessing_revision=options.preprocessing_revision,
        model_revision=options.model_revision if adapter is None else adapter.revision,
        crop_strategy=options.policy.strategy,
        crop_shape=options.policy.crop_shape,
        shape_buckets=options.policy.shape_buckets,
        microbatch_per_device=config.batch.microbatch_per_device,
        world_size=config.accelerator.world_size,
        gradient_accumulation_steps=config.batch.gradient_accumulation_steps,
        global_batch_size=config.global_batch_size,
        memory_cap_gb=config.memory.max_gpu_memory_gb,
        tpu_status=options.tpu_status,
        unsupported_xla_ops=unsupported,
        custom_cuda_dependencies=custom_cuda,
        visual_token_count=options.visual_token_count
        if options.family in {"native_3d_vlm", "slice_sequence_vlm"}
        else None,
        text_token_count=options.text_token_count
        if options.family in {"native_3d_vlm", "slice_sequence_vlm", "language_conditioned_segmentation"}
        else None,
        bridge_type=options.bridge_type if options.family in {"native_3d_vlm", "slice_sequence_vlm"} else None,
        cached_tokens=options.cache_tokens,
        selector=options.selector if options.family == "slice_sequence_vlm" else None,
        selector_revision=SLICE_SELECTOR_VERSION if options.family == "slice_sequence_vlm" else None,
        slice_count=options.slice_count if options.family == "slice_sequence_vlm" else None,
        query_count=options.query_count if options.family == "language_conditioned_segmentation" else None,
        positive_patch_rate=positive_rate,
        positive_sampling_ratio=options.positive_ratio if options.family == "segmentation" else None,
        deep_supervision=options.deep_supervision,
        blend_mode=options.blend_mode,
        backend_observability=_backend_observability(config),
        baseline=options.baseline,
        limitations=(
            "Offline tiny checkpoints and synthetic data are contract evidence, not clinical validation.",
            "Production acceptance requires approved de-identified data, external-site testing, and human review.",
        ),
    )


def build_phase14_recipe(config: RunConfig) -> Phase14RecipeBuild:
    """Build one Phase 14 recipe, including deterministic synthetic data."""

    options = _options(config)
    model = _phase14_model(config, options)
    _inject_phase14_lora(model, config, options)
    task = _phase14_task(config, options, model)
    data = _phase14_data(config, options)
    return Phase14RecipeBuild(model=model, task=task, train_data=data, metadata=_metadata(config, options, data))


def phase14_builders() -> ComponentBuilders:
    """Return builders compatible with the model-agnostic TrainingPipeline."""

    def dataset(config: RunConfig, *_: Any) -> list[MedicalBatch]:
        return _phase14_data(config, _options(config))

    def model(config: RunConfig, *_: Any) -> nn.Module:
        options = _options(config)
        value = _phase14_model(config, options)
        _inject_phase14_lora(value, config, options)
        return value

    def peft(model_value: nn.Module, *_: Any) -> nn.Module:
        return model_value

    def task(config: RunConfig, model_value: nn.Module, *_: Any) -> TaskModuleBase:
        return _phase14_task(config, _options(config), model_value)

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
            role_map=_PHASE14_ROLE_MAP,
            components={"task": task_value},
        )

    def trainer(
        config: RunConfig,
        backend: AcceleratorBackend,
        model_value: nn.Module,
        optimizer_value: OptimizerBundle,
        task_value: TaskModuleBase,
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
            role_map=_PHASE14_ROLE_MAP,
        )

    return ComponentBuilders(dataset=dataset, model=model, peft=peft, task=task, optimizer=optimizer, trainer=trainer)


def restore_volume_mask_to_original(
    mask: torch.Tensor,
    *,
    history: Sequence[Any] | None = None,
    original_size: tuple[int, int, int] | None = None,
    crop_origin: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Invert 3D preprocessing or re-embed a crop in original voxel space."""

    if mask.ndim not in (4, 5):
        raise ShapeContractError("3D mask must be [B,D,H,W] or [B,1,D,H,W]")
    result = mask
    if history is not None:
        from medfm.data.transforms.base import invert_history

        result = invert_history(list(history), result, mode="label", strict=True)
    if original_size is None:
        return result
    target = _shape_tuple(original_size, "original_size")
    if crop_origin is None:
        if tuple(result.shape[-3:]) == target:
            return result
        mode = "nearest"
        value = result if result.ndim == 5 else result.unsqueeze(1)
        restored = F.interpolate(value.float(), size=target, mode=mode)
        return restored if result.ndim == 5 else restored[:, 0]
    origin = _shape_tuple(crop_origin, "crop_origin")
    if any(origin[axis] + int(result.shape[-3 + axis]) > target[axis] for axis in range(3)):
        raise ShapeContractError("crop origin plus mask shape exceeds original_size")
    value = result if result.ndim == 5 else result.unsqueeze(1)
    restored = torch.zeros((*value.shape[:2], *target), dtype=value.dtype, device=value.device)
    restored[
        ...,
        origin[0] : origin[0] + value.shape[-3],
        origin[1] : origin[1] + value.shape[-2],
        origin[2] : origin[2] + value.shape[-1],
    ] = value
    return restored if result.ndim == 5 else restored[:, 0]


def sliding_window_predict(
    volume: torch.Tensor,
    predictor: Callable[..., torch.Tensor],
    *,
    window_shape: tuple[int, int, int],
    overlap: float = 0.25,
    blend_mode: str = "constant",
    metadata: Sequence[SpatialMetadata] | None = None,
) -> torch.Tensor:
    """Run host-orchestrated fixed-window inference with constant/Gaussian blending."""

    mode = str(blend_mode).lower()
    if mode not in _ALLOWED_BLEND_MODES:
        raise ShapeContractError(f"blend_mode must be one of {sorted(_ALLOWED_BLEND_MODES)}")
    if mode == "constant":
        return sliding_window_inference(
            volume,
            predictor,
            window_shape=window_shape,
            overlap=overlap,
            metadata=metadata,
        )
    if volume.ndim != 5 or len(window_shape) != 3:
        raise ShapeContractError("Gaussian sliding-window inference expects [B,C,D,H,W]")
    b, _, depth, height, width = volume.shape
    strides = tuple(max(1, int(window * (1.0 - overlap))) for window in window_shape)
    starts = []
    for size, window, stride in zip((depth, height, width), window_shape, strides, strict=True):
        if window > size:
            raise ShapeContractError("window_shape cannot exceed the volume")
        positions = list(range(0, max(size - window + 1, 1), stride))
        if positions[-1] != size - window:
            positions.append(size - window)
        starts.append(tuple(positions))
    grid = torch.meshgrid(
        torch.linspace(-1.0, 1.0, window_shape[0], device=volume.device),
        torch.linspace(-1.0, 1.0, window_shape[1], device=volume.device),
        torch.linspace(-1.0, 1.0, window_shape[2], device=volume.device),
        indexing="ij",
    )
    sigma = 0.5
    squared_coordinates = tuple(axis.square() for axis in grid)
    weights_window = torch.exp(
        -sum(squared_coordinates, torch.zeros_like(squared_coordinates[0])) / (2.0 * sigma * sigma)
    ).clamp_min(1e-3)
    output: torch.Tensor | None = None
    weights = torch.zeros((b, 1, depth, height, width), device=volume.device, dtype=torch.float32)
    metas = list(metadata) if metadata is not None else [None] * b
    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                crop = volume[:, :, z0 : z0 + window_shape[0], y0 : y0 + window_shape[1], x0 : x0 + window_shape[2]]
                try:
                    prediction = predictor(crop, metas)
                except TypeError:
                    prediction = predictor(crop)
                if prediction.ndim != 5 or tuple(prediction.shape[0:1] + prediction.shape[-3:]) != (b, *window_shape):
                    raise ShapeContractError("sliding-window predictor must return [B,K,*window_shape]")
                if output is None:
                    output = torch.zeros(
                        (b, int(prediction.shape[1]), depth, height, width),
                        device=prediction.device,
                        dtype=prediction.dtype,
                    )
                weighted = prediction * weights_window.to(dtype=prediction.dtype).reshape(1, 1, *window_shape)
                output[:, :, z0 : z0 + window_shape[0], y0 : y0 + window_shape[1], x0 : x0 + window_shape[2]] += (
                    weighted
                )
                weights[:, :, z0 : z0 + window_shape[0], y0 : y0 + window_shape[1], x0 : x0 + window_shape[2]] += (
                    weights_window
                )
    assert output is not None
    return output / weights.to(dtype=output.dtype).clamp_min(1e-3)


def _hd95(predicted: torch.Tensor, target: torch.Tensor, spacing_mm: tuple[float, float, float]) -> float:
    pred = predicted.detach().cpu().numpy().astype(bool).squeeze()
    truth = target.detach().cpu().numpy().astype(bool).squeeze()
    if pred.ndim != 3 or truth.ndim != 3:
        raise ShapeContractError("HD95 expects one 3D foreground mask at a time")
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return float("inf")
    pred_boundary = pred & (~ndimage.binary_erosion(pred) | ~ndimage.binary_dilation(pred))
    truth_boundary = truth & (~ndimage.binary_erosion(truth) | ~ndimage.binary_dilation(truth))
    pred_distance = ndimage.distance_transform_edt(~pred_boundary, sampling=spacing_mm)
    truth_distance = ndimage.distance_transform_edt(~truth_boundary, sampling=spacing_mm)
    distances = list(pred_distance[truth_boundary]) + list(truth_distance[pred_boundary])
    return float(torch.tensor(distances, dtype=torch.float64).quantile(0.95))


def native_3d_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    unit: str = "per_scan",
) -> dict[str, MetricValue]:
    """Report Dice/surface/HD95, lesion recall, FP/scan, and volume error."""

    result = segmentation_metrics(logits, target, unit=unit)
    predicted = torch.sigmoid(logits) >= 0.5
    truth = target > 0.5
    batch = int(logits.shape[0])
    lesion_recalls: list[float] = []
    false_positives: list[float] = []
    hd95_values: list[float] = []
    volume_errors: list[float] = []
    voxel_volume = float(math.prod(spacing_mm))
    for row in range(batch):
        pred = predicted[row, :1]
        true = truth[row, :1]
        true_np = true.squeeze(0).detach().cpu().numpy().astype(bool)
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(bool)
        true_labels, true_count = ndimage.label(true_np)
        pred_labels, pred_count = ndimage.label(pred_np)
        detected = 0
        for lesion in range(1, true_count + 1):
            component = true_labels == lesion
            if bool((pred_np & component).any()):
                detected += 1
        lesion_recalls.append(float(detected / true_count) if true_count else (1.0 if pred_count == 0 else 0.0))
        false_positives.append(float(max(0, pred_count - detected)))
        hd95_values.append(_hd95(pred, true, spacing_mm))
        volume_errors.append(abs(float(pred.sum()) - float(true.sum())) * voxel_volume)
    result["hd95/class_0"] = MetricValue(
        "hd95/class_0", sum(hd95_values) / max(1, batch), unit, batch, {"spacing_mm": spacing_mm}
    )
    result["lesion_recall/class_0"] = MetricValue(
        "lesion_recall/class_0", sum(lesion_recalls) / max(1, batch), unit, batch
    )
    result["false_positives_per_scan/class_0"] = MetricValue(
        "false_positives_per_scan/class_0", sum(false_positives) / max(1, batch), unit, batch
    )
    result["volume_error_mm3/class_0"] = MetricValue(
        "volume_error_mm3/class_0", sum(volume_errors) / max(1, batch), "mm3_per_scan", batch
    )
    return result


def _aggregate_classification_groups(
    labels: list[int],
    scores: list[float],
    group_ids: Iterable[str] | None,
    *,
    unit: str,
) -> dict[str, MetricValue]:
    if group_ids is None:
        return classification_metrics(labels, scores, unit=unit)
    groups = list(group_ids)
    if len(groups) != len(labels):
        raise ValueError("group_ids must align with labels and scores")
    grouped: dict[str, list[int]] = {}
    grouped_scores: dict[str, list[float]] = {}
    for label, score, group in zip(labels, scores, groups, strict=True):
        key = str(group)
        grouped.setdefault(key, []).append(int(label))
        grouped_scores.setdefault(key, []).append(float(score))
    aggregate_labels: list[int] = []
    aggregate_scores: list[float] = []
    for key in sorted(grouped):
        group_labels = grouped[key]
        if len(set(group_labels)) != 1:
            raise ValueError(f"{unit} group {key!r} contains inconsistent binary labels")
        aggregate_labels.append(group_labels[0])
        aggregate_scores.append(sum(grouped_scores[key]) / len(grouped_scores[key]))
    return classification_metrics(aggregate_labels, aggregate_scores, unit=unit)


def native_3d_classification_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    patient_ids: Iterable[str] | None = None,
    study_ids: Iterable[str] | None = None,
) -> dict[str, MetricValue]:
    """Compute metrics after deterministic patient/study score aggregation."""

    labels_list = [int(value) for value in labels]
    scores_list = [float(value) for value in scores]
    if len(labels_list) != len(scores_list):
        raise ValueError("labels and scores must align")
    result: dict[str, MetricValue] = {}
    result.update(
        {
            f"patient/{name}": value
            for name, value in _aggregate_classification_groups(
                labels_list, scores_list, patient_ids, unit="per_patient"
            ).items()
        }
    )
    if study_ids is not None:
        result.update(
            {
                f"study/{name}": value
                for name, value in _aggregate_classification_groups(
                    labels_list, scores_list, study_ids, unit="per_study"
                ).items()
            }
        )
    return result


def native_vlm_grounding_metrics(
    image_logits: torch.Tensor,
    no_image_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
) -> dict[str, MetricValue]:
    """Separate image dependence from language loss/generation reporting."""

    if image_logits.shape != no_image_logits.shape or image_logits.shape != shuffled_logits.shape:
        raise ShapeContractError("VLM ablation logits must have identical shapes")
    image_delta = float((image_logits.float() - no_image_logits.float()).abs().mean())
    shuffle_delta = float((image_logits.float() - shuffled_logits.float()).abs().mean())
    count = int(image_logits.shape[0])
    return {
        "image_dependence": MetricValue("image_dependence", image_delta, "mean_logit_delta", count),
        "shuffled_image_dependence": MetricValue("shuffled_image_dependence", shuffle_delta, "mean_logit_delta", count),
    }


def language_conditioned_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    query_grounding: torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Return mask accuracy and query grounding as separate metric families."""

    if logits.shape != target.shape or logits.ndim not in (5, 6):
        raise ShapeContractError("language-conditioned masks must be [B,1,D,H,W] or [B,Q,1,D,H,W]")
    query_shape: tuple[int, ...]
    if logits.ndim == 5:
        query_shape = (logits.shape[0],)
        spatial_correct = (torch.sigmoid(logits) >= 0.5) == (target > 0.5)
    else:
        query_shape = (logits.shape[0], logits.shape[1])
        spatial_correct = (torch.sigmoid(logits) >= 0.5) == (target > 0.5)
    active = (
        torch.ones(query_shape, dtype=torch.bool, device=logits.device)
        if query_mask is None
        else query_mask.to(device=logits.device, dtype=torch.bool)
    )
    if tuple(active.shape) != query_shape:
        raise ShapeContractError(f"query_mask must be {query_shape}")
    reduce_dims = tuple(range(1, spatial_correct.ndim)) if logits.ndim == 5 else tuple(range(2, spatial_correct.ndim))
    correct = spatial_correct.float().mean(dim=reduce_dims)
    active_values = correct[active]
    mask_accuracy = float(active_values.mean()) if active_values.numel() else 1.0
    result = {
        "mask_accuracy": MetricValue("mask_accuracy", mask_accuracy, "per_query", int(active.sum())),
    }
    if query_grounding is not None:
        grounding = query_grounding.to(device=logits.device, dtype=torch.float32)
        if tuple(grounding.shape) != query_shape:
            raise ShapeContractError("query_grounding must align with query_mask")
        values = grounding[active]
        result["query_grounding"] = MetricValue(
            "query_grounding", float(values.mean()) if values.numel() else 0.0, "per_query", int(active.sum())
        )
    return result


def benchmark_slice_token_budgets(
    *,
    slice_buckets: Iterable[int] = (16, 32, 48, 64),
    visual_token_buckets: Iterable[int] = (32, 64),
    text_token_buckets: Iterable[int] = (256, 512),
    memory_cap_gb: float = 48.0,
) -> tuple[dict[str, Any], ...]:
    """Produce a deterministic CUDA/TPU comparison table under the memory cap."""

    rows: list[dict[str, Any]] = []
    for slices in slice_buckets:
        for visual in visual_token_buckets:
            for text in text_token_buckets:
                estimate = float(slices) * 0.015 + float(visual) * 0.006 + float(text) * 0.0008
                rows.append(
                    {
                        "slice_count": int(slices),
                        "visual_tokens": int(visual),
                        "text_tokens": int(text),
                        "estimated_memory_gb": round(estimate, 6),
                        "within_48gb_cap": estimate < float(memory_cap_gb),
                        "benchmark_backends": ("cuda", "xla_tpu"),
                        "selector_on_host": True,
                    }
                )
    return tuple(rows)


def make_phase14_artifact(
    config: RunConfig,
    metrics: Mapping[str, MetricValue],
    *,
    memory: Mapping[str, Any] | None = None,
    metadata: Phase14RecipeMetadata | None = None,
) -> EvaluationArtifact:
    """Create a provenance-bearing Phase 14 evaluation artifact."""

    limitations = (
        metadata.limitations
        if metadata is not None and metadata.limitations
        else (
            "Offline tiny checkpoints and synthetic data are contract evidence, not clinical validation.",
            "External-site and human-review evidence are required before clinical use.",
        )
    )
    return make_artifact(
        metadata.recipe_id if metadata is not None else str(config.recipe.get("id", "phase14-recipe")),
        metrics,
        config_hash=config.config_hash(),
        seed=config.seed,
        dataset_hash=config.dataset_hash,
        preprocessing_hash=config.preprocessing_hash,
        model_revision=config.base_model_revision,
        memory=dict(memory or (metadata.backend_observability if metadata is not None else {})),
        limitations=tuple(limitations),
    )


__all__ = [
    "LANGUAGE_LORA_TARGETS",
    "NATIVE_3D_LORA_TARGETS",
    "NATIVE_3D_MODALITIES",
    "PHASE14_RECIPE_VERSION",
    "Phase14LanguageOutput",
    "Phase14RecipeBuild",
    "Phase14RecipeMetadata",
    "RecipeBuild",
    "RecipeConfigurationError",
    "SLICE_SEQUENCE_MODALITY",
    "VolumeInputPolicy",
    "benchmark_slice_token_budgets",
    "build_phase14_recipe",
    "language_conditioned_segmentation_metrics",
    "make_phase14_artifact",
    "native_3d_classification_metrics",
    "native_3d_segmentation_metrics",
    "native_vlm_grounding_metrics",
    "phase14_builders",
    "restore_volume_mask_to_original",
    "select_volume_input_policy",
    "sliding_window_predict",
]
