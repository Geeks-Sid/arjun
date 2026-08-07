"""Reproducible Phase 13 2D recipe builders.

The recipe layer owns model/dataset choices; the trainer remains model-agnostic.
Production adapters are loaded only from explicit local checkpoint directories
or the registry.  ``offline_tiny`` is an intentional random-weight contract
fixture for smoke and acceptance tests and is recorded in every recipe
manifest; it is never presented as a clinical model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderCapabilities, EncoderOutput, OutputSpec
from medfm.core.enums import CoordinateSystem, Modality, TaskType
from medfm.core.errors import ShapeContractError
from medfm.core.language import LanguageOutput, ProjectedVisualTokens, TokenizedText
from medfm.core.task import LossOutput
from medfm.evaluation.report import EvaluationArtifact, make_artifact
from medfm.models.bridges import LinearVisionLanguageBridge, PerceiverResamplerBridge
from medfm.models.decoders import UNetDecoder2D
from medfm.models.heads.classification import LinearClassificationHead, MLPClassificationHead
from medfm.models.language.base import GenericHFCausalLMAdapter
from medfm.models.language.medgemma import MedGemmaAdapter
from medfm.models.visual.base import AdapterPreprocess, BackboneResult, BaseVisualAdapter2D, LoraTargetSpec
from medfm.peft import LoRAConfig, inject_language_lora
from medfm.tasks.base import TaskModuleBase, detached_count_tensor, valid_sample_count
from medfm.tasks.classification import BinaryClassificationTask, ClassificationTask, MultiLabelClassificationTask
from medfm.tasks.generation import StructuredGenerationTask
from medfm.tasks.segmentation import BinarySegmentationTask
from medfm.training.backend import AcceleratorBackend
from medfm.training.config import RunConfig
from medfm.training.optimizer import OptimizerBundle, build_optimizer
from medfm.training.pipeline import ComponentBuilders
from medfm.training.steps import make_training_step
from medfm.training.trainer import Trainer

PHASE13_RECIPE_VERSION = "phase13-1"
VISUAL_LORA_TARGETS = (
    r"attention\.(q_proj|k_proj|v_proj|out_proj)",
    r"mlp\.(fc1|fc2)",
)
LANGUAGE_LORA_TARGETS = (r"layers\.\d+\.(self_attn\.out_proj|linear1|linear2)",)
VISUAL_MODALITIES = (Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE, Modality.MULTI_IMAGE_2D)


class RecipeConfigurationError(ValueError):
    """A recipe is malformed or requests an unavailable production path."""


@dataclass(frozen=True)
class RecipeMetadata:
    """Pinned recipe choices written into run and evaluation artifacts."""

    family: str
    recipe_id: str
    backbone: str
    mode: str
    stage: str
    dataset_id: str
    dataset_revision: str
    preprocessing_revision: str
    prompt_revision: str | None = None
    visual_token_count: int | None = None
    bridge_type: str | None = None
    task_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": PHASE13_RECIPE_VERSION,
            "family": self.family,
            "recipe_id": self.recipe_id,
            "backbone": self.backbone,
            "mode": self.mode,
            "stage": self.stage,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "preprocessing_revision": self.preprocessing_revision,
            "prompt_revision": self.prompt_revision,
            "visual_token_count": self.visual_token_count,
            "bridge_type": self.bridge_type,
            "task_weights": dict(self.task_weights),
        }


@dataclass(frozen=True)
class RecipeBuild:
    """Direct factory result for tests and offline acceptance tooling."""

    model: nn.Module
    task: nn.Module
    train_data: list[MedicalBatch]
    metadata: RecipeMetadata


class _TinyVisionAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(value)
        key = self.k_proj(value)
        val = self.v_proj(value)
        return cast(torch.Tensor, self.out_proj(torch.tanh((query + key + val) / 3.0)))


class _TinyVisionMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc2(F.gelu(self.fc1(value))))


class _TinyVisionBackbone(nn.Module):
    """Static-shape patch encoder with explicit transformer-like targets."""

    def __init__(self, *, channels: int, hidden_size: int, patch_size: int, construction_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(construction_seed)
            self.patch_embed = nn.Conv2d(channels, hidden_size, kernel_size=patch_size, stride=patch_size)
            self.signal = nn.Linear(1, hidden_size)
            self.attention = _TinyVisionAttention(hidden_size)
            self.mlp = _TinyVisionMLP(hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor, *, output_hidden_states: bool = False) -> Any:
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        image_signal = self.signal(pixel_values.float().mean(dim=(1, 2, 3), keepdim=True)).flatten(1).unsqueeze(1)
        hidden = patches + image_signal
        hidden = self.norm(hidden + self.attention(hidden))
        hidden = self.norm(hidden + self.mlp(hidden))
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden.mean(dim=1),
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class TinyVisualAdapter(BaseVisualAdapter2D):
    """Offline contract adapter used by every Phase 13 tiny recipe."""

    def __init__(
        self,
        *,
        model_id: str,
        modality: Modality,
        image_size: int = 32,
        channels: int = 1,
        hidden_size: int = 32,
        patch_size: int = 8,
        construction_seed: int = 0,
    ) -> None:
        preprocess = AdapterPreprocess(
            image_size=(image_size, image_size),
            channels=channels,
            patch_size=patch_size,
            mean=tuple(0.0 for _ in range(channels)),
            std=tuple(1.0 for _ in range(channels)),
            value_range=(-1.0, 1.0),
            resize_policy="fixed",
            color_space="GRAYSCALE" if channels == 1 else "RGB",
        )
        capabilities = EncoderCapabilities(
            model_id=model_id,
            modalities=(modality,),
            supports_pooled=True,
            supports_spatial_tokens=True,
            supports_feature_maps=True,
            supports_token_coordinates=True,
            token_coordinate_systems=(CoordinateSystem.NORMALIZED_IMAGE,),
        )
        super().__init__(
            model_id=model_id,
            revision="offline-random-contract",
            capabilities=capabilities,
            preprocess=preprocess,
            feature_map_layers=(0,),
            lora_targets=tuple(
                LoraTargetSpec(
                    pattern=pattern,
                    reason="Phase 13 late visual adaptation",
                )
                for pattern in VISUAL_LORA_TARGETS
            ),
            construction_seed=construction_seed,
        )
        self.backbone = _TinyVisionBackbone(
            channels=channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            construction_seed=construction_seed,
        )
        self.hidden_size = int(hidden_size)
        self.train()

    def _prefix_token_count(self) -> int:
        return 0

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        outputs = self.backbone(pixel_values, output_hidden_states=output_hidden_states)
        hidden_states = None if outputs.hidden_states is None else tuple(outputs.hidden_states)
        return BackboneResult(
            last_hidden_state=outputs.last_hidden_state,
            pooled=outputs.pooler_output,
            hidden_states=hidden_states,
            raw=outputs,
        )


class _ClassificationModel(nn.Module):
    def __init__(self, vision: TinyVisualAdapter) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, batch: MedicalBatch) -> EncoderOutput:
        return self.vision.encode(batch, output_spec=OutputSpec(pooled=True))

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> EncoderOutput:
        return self.forward(batch)


class _SegmentationModel(nn.Module):
    def __init__(self, vision: TinyVisualAdapter) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, batch: MedicalBatch) -> dict[str, Any]:
        output = self.vision.encode(
            batch,
            output_hidden_states=True,
            output_spec=OutputSpec(pooled=True, spatial_tokens=True, feature_maps=True, token_coordinates=True),
        )
        assert output.feature_maps is not None
        spatial = tuple(int(v) for v in batch.pixel_values.shape[-2:]) if batch.pixel_values is not None else None
        maps = output.feature_maps
        if spatial is not None:
            maps = tuple(F.interpolate(value, size=spatial, mode="bilinear", align_corners=False) for value in maps)
        return {"feature_maps": maps, "encoder_output": output}

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> dict[str, Any]:
        return self.forward(batch)


class _PromptableSegmentationModel(_SegmentationModel):
    """Tiny promptable path with prompt tensors kept separate from pixels."""

    def forward(self, batch: MedicalBatch) -> dict[str, Any]:
        result = super().forward(batch)
        prompt = batch.task_targets.get("prompt_map")
        if isinstance(prompt, torch.Tensor):
            maps = tuple(result["feature_maps"])
            if tuple(prompt.shape[-2:]) != tuple(maps[-1].shape[-2:]):
                prompt = F.interpolate(prompt.float(), size=maps[-1].shape[-2:], mode="nearest")
            maps = (*maps[:-1], maps[-1] + prompt.to(device=maps[-1].device, dtype=maps[-1].dtype))
            result["feature_maps"] = maps
        return result


class _PromptableSegmentationTask(BinarySegmentationTask):
    def __init__(self, decoder: nn.Module, **kwargs: Any) -> None:
        super().__init__(decoder, **kwargs)
        self._task_type = TaskType.PROMPTABLE_SEGMENTATION


class _RecipeLanguageTask(TaskModuleBase):
    def __init__(
        self,
        *,
        supported_modalities: tuple[Modality, ...] = VISUAL_MODALITIES,
        task_type: TaskType = TaskType.VISUAL_QUESTION_ANSWERING,
    ) -> None:
        super().__init__(task_type, supported_modalities)

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        self.check_supported(batch.modality)
        language = model_output.language if isinstance(model_output, _RecipeLanguageOutput) else model_output
        if not isinstance(language, LanguageOutput) or language.loss is None:
            raise ShapeContractError("language recipe model must return LanguageOutput.loss for supervised training")
        count = valid_sample_count(batch)
        token_count = int(language.auxiliary.get("supervised_token_count", 0))
        return LossOutput(
            total=language.loss,
            components={"language": language.loss},
            sample_count=count,
            token_count=token_count,
            diagnostics={
                "task": self.task_type.value,
                "valid_count": detached_count_tensor(count, language.loss),
                "supervised_token_count": token_count,
            },
        )


class _RecipeStructuredTask(StructuredGenerationTask):
    """Structured-findings validation around the recipe language output."""

    def compute_loss(self, model_output: Any, batch: MedicalBatch) -> LossOutput:
        if isinstance(model_output, _RecipeLanguageOutput):
            model_output = model_output.language
        return super().compute_loss(model_output, batch)


@dataclass(frozen=True, eq=False)
class _RecipeLanguageOutput:
    language: LanguageOutput
    visual_tokens: ProjectedVisualTokens | None
    source_coordinates: torch.Tensor | None
    visual_metadata: tuple[dict[str, Any], ...]
    mode: str


class _VisualLanguageModelBase(nn.Module):
    def __init__(
        self,
        vision: TinyVisualAdapter,
        language: GenericHFCausalLMAdapter,
        *,
        visual_token_count: int,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.language = language
        self.visual_token_count = int(visual_token_count)

    def _encode_source(self, batch: MedicalBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.vision.encode(
            batch,
            output_spec=OutputSpec(pooled=True, spatial_tokens=True, token_coordinates=True),
        )
        if output.spatial_tokens is None or output.token_mask is None or output.token_coordinates is None:
            raise ShapeContractError("VLM source adapter must return spatial tokens, mask, and coordinates")
        tokens = output.spatial_tokens
        mask = output.token_mask
        coordinates = output.token_coordinates
        if int(tokens.shape[1]) > self.visual_token_count:
            raise RecipeConfigurationError(
                f"source patch count {int(tokens.shape[1])} exceeds visual token bucket {self.visual_token_count}"
            )
        pad = self.visual_token_count - int(tokens.shape[1])
        if pad:
            tokens = F.pad(tokens, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
            coordinates = F.pad(coordinates, (0, 0, 0, pad))
        return tokens, mask, coordinates

    def _text(self, batch: MedicalBatch) -> tuple[TokenizedText, torch.Tensor]:
        if batch.input_ids is None or batch.attention_mask is None:
            raise ShapeContractError("VLM batches require input_ids and attention_mask")
        labels = batch.task_targets.get("language_labels")
        if not isinstance(labels, torch.Tensor):
            raise ShapeContractError("VLM batch is missing task_targets['language_labels']")
        prompt_mask = batch.task_targets.get("prompt_token_mask")
        metadata: dict[str, Any] = {}
        if isinstance(prompt_mask, torch.Tensor):
            metadata["prompt_token_mask"] = prompt_mask
        return TokenizedText(batch.input_ids, batch.attention_mask, metadata=metadata), labels

    def _visual(
        self,
        batch: MedicalBatch,
        *,
        mode: str,
    ) -> tuple[ProjectedVisualTokens, torch.Tensor, tuple[dict[str, Any], ...]]:
        tokens, mask, coordinates = self._encode_source(batch)
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
            raise ValueError(f"unknown visual ablation mode {mode!r}")
        projected = self.project_visual(tokens, mask, coordinates)
        metadata = tuple(dict(value) for value in batch.task_targets.get("visual_metadata", ()))
        return projected, coordinates, metadata

    def project_visual(
        self, tokens: torch.Tensor, mask: torch.Tensor, coordinates: torch.Tensor
    ) -> ProjectedVisualTokens:
        raise NotImplementedError

    def _forward_mode(self, batch: MedicalBatch, *, mode: str) -> _RecipeLanguageOutput:
        visual, coordinates, metadata = self._visual(batch, mode=mode)
        text, labels = self._text(batch)
        language = self.language.forward_with_visual_tokens(text, visual, labels)
        if not isinstance(language, LanguageOutput):
            raise ShapeContractError("language adapter returned an invalid LanguageOutput")
        language_auxiliary = dict(language.auxiliary)
        language_auxiliary.update(
            {
                "source_coordinates": coordinates,
                "visual_metadata": metadata,
                "visual_mode": mode,
                "visual_token_mask": visual.token_mask,
                "generated_texts": tuple(
                    '{"findings":[],"impression":"synthetic offline contract"}' for _ in batch.sample_ids
                ),
            }
        )
        language = LanguageOutput(
            logits=language.logits,
            loss=language.loss,
            hidden_states=language.hidden_states,
            auxiliary=language_auxiliary,
        )
        return _RecipeLanguageOutput(language, visual, coordinates, metadata, mode)

    def forward(self, batch: MedicalBatch) -> _RecipeLanguageOutput:
        return self._forward_mode(batch, mode="image")

    def forward_mode(self, batch: MedicalBatch, *, mode: str = "image") -> _RecipeLanguageOutput:
        return self._forward_mode(batch, mode=mode)


class _NativeVLMModel(_VisualLanguageModelBase):
    def project_visual(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> ProjectedVisualTokens:
        return ProjectedVisualTokens(
            tokens=tokens,
            source_modality=Modality.XRAY_2D,
            token_mask=mask,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        )


class _ExternalVLMModel(_VisualLanguageModelBase):
    def __init__(
        self,
        vision: TinyVisualAdapter,
        language: GenericHFCausalLMAdapter,
        bridge: nn.Module,
        *,
        visual_token_count: int,
        bridge_type: str,
    ) -> None:
        super().__init__(vision, language, visual_token_count=visual_token_count)
        self.bridge = bridge
        self.bridge_type = bridge_type

    def project_visual(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> ProjectedVisualTokens:
        projected = self.bridge(tokens, mask, coordinates=coordinates)
        if not isinstance(projected, ProjectedVisualTokens):
            raise ShapeContractError("external bridge returned an invalid ProjectedVisualTokens")
        return projected


@dataclass(frozen=True)
class _RecipeOptions:
    family: str
    recipe_id: str
    backbone: str
    stage: str
    mode: str
    modality: Modality
    image_size: int
    channels: int
    hidden_size: int
    patch_size: int
    visual_token_count: int
    bridge_type: str
    offline_tiny: bool
    construction_seed: int
    dataset_id: str
    dataset_revision: str
    preprocessing_revision: str
    prompt_revision: str | None


def _options(config: RunConfig) -> _RecipeOptions:
    raw = dict(config.recipe)
    family = str(raw.get("family", raw.get("type", "classification"))).strip().lower().replace("-", "_")
    aliases = {"native": "native_vlm", "external": "external_vlm", "seg": "segmentation", "cls": "classification"}
    family = aliases.get(family, family)
    if family not in {"classification", "segmentation", "promptable_segmentation", "native_vlm", "external_vlm"}:
        raise RecipeConfigurationError(f"unknown Phase 13 recipe family {family!r}")
    modality = Modality(str(raw.get("modality", "XRAY_2D")).upper())
    image_size = int(raw.get("image_size", 32))
    patch_size = int(raw.get("patch_size", 8))
    if image_size <= 0 or image_size % patch_size:
        raise RecipeConfigurationError("image_size must be positive and divisible by patch_size")
    visual_tokens = int(raw.get("visual_tokens", raw.get("visual_token_count", 64 if family == "external_vlm" else 32)))
    if visual_tokens not in (32, 64, 128):
        raise RecipeConfigurationError("Phase 13 visual token buckets are exactly 32, 64, or 128")
    mode = str(raw.get("mode", "offline_tiny" if raw.get("offline_tiny", True) else "production")).lower()
    offline = bool(raw.get("offline_tiny", mode in {"smoke", "tiny", "offline_tiny", "contract"}))
    return _RecipeOptions(
        family=family,
        recipe_id=str(raw.get("id", raw.get("name", f"phase13-{family}"))),
        backbone=str(raw.get("backbone", "medsiglip")),
        stage=str(raw.get("stage", "A")),
        mode=mode,
        modality=modality,
        image_size=image_size,
        channels=int(raw.get("channels", 1)),
        hidden_size=int(raw.get("hidden_size", 32)),
        patch_size=patch_size,
        visual_token_count=visual_tokens,
        bridge_type=str(raw.get("bridge", raw.get("bridge_type", "perceiver"))).lower(),
        offline_tiny=offline,
        construction_seed=int(raw.get("construction_seed", config.seed)),
        dataset_id=str(raw.get("dataset_id", config.dataset.get("id", "phase13-synthetic-v1"))),
        dataset_revision=str(raw.get("dataset_revision", config.dataset.get("revision", "synthetic-v1"))),
        preprocessing_revision=str(
            raw.get("preprocessing_revision", config.preprocessing_hash or "phase13-preprocess-v1")
        ),
        prompt_revision=(str(raw["prompt_revision"]) if raw.get("prompt_revision") is not None else None),
    )


def _metadata(options: _RecipeOptions, task_weights: Mapping[str, float] | None = None) -> RecipeMetadata:
    return RecipeMetadata(
        family=options.family,
        recipe_id=options.recipe_id,
        backbone=options.backbone,
        mode=options.mode,
        stage=options.stage,
        dataset_id=options.dataset_id,
        dataset_revision=options.dataset_revision,
        preprocessing_revision=options.preprocessing_revision,
        prompt_revision=options.prompt_revision,
        visual_token_count=options.visual_token_count if options.family in {"native_vlm", "external_vlm"} else None,
        bridge_type=options.bridge_type if options.family == "external_vlm" else None,
        task_weights={str(name): float(value) for name, value in (task_weights or {}).items()},
    )


def _visual_adapter(options: _RecipeOptions) -> TinyVisualAdapter:
    if options.offline_tiny:
        return TinyVisualAdapter(
            model_id=f"{options.backbone}-offline-tiny",
            modality=options.modality,
            image_size=options.image_size,
            channels=options.channels,
            hidden_size=options.hidden_size,
            patch_size=options.patch_size,
            construction_seed=options.construction_seed,
        )
    checkpoint = Path(str(options.mode)) if options.mode not in {"production", "real"} else None
    raise RecipeConfigurationError(
        "production Phase 13 adapters must be supplied through a registered local checkpoint; "
        f"offline_tiny=false for {options.backbone!r} cannot construct weights in this environment"
        + (f" (requested path {checkpoint})" if checkpoint is not None else "")
    )


def _classification_data(config: RunConfig, options: _RecipeOptions) -> list[MedicalBatch]:
    generator = torch.Generator().manual_seed(config.seed)
    batch_size = int(config.batch.microbatch_per_device)
    values = torch.randn((batch_size, options.channels, options.image_size, options.image_size), generator=generator)
    signal = values.mean(dim=(1, 2, 3))
    labels = (signal > 0).to(torch.long)
    if bool(labels.all()) or not bool(labels.any()):
        labels = torch.arange(batch_size) % 2
        values = values + (labels.float() * 2 - 1).reshape(-1, 1, 1, 1)
    task_name = str(config.task.get("type", config.task.get("name", ""))).upper()
    if "BINARY" in task_name:
        target: torch.Tensor = labels.float().unsqueeze(1)
    elif "MULTILABEL" in task_name:
        target = torch.stack((labels.float(), 1.0 - labels.float()), dim=1)
    else:
        target = labels
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase13-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            labels=labels,
            task_targets={"classification": target, "sample_mask": torch.ones(batch_size, dtype=torch.bool)},
        )
    ]


def _segmentation_data(config: RunConfig, options: _RecipeOptions) -> list[MedicalBatch]:
    generator = torch.Generator().manual_seed(config.seed)
    batch_size = int(config.batch.microbatch_per_device)
    values = torch.rand((batch_size, options.channels, options.image_size, options.image_size), generator=generator)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, options.image_size), torch.linspace(-1.0, 1.0, options.image_size), indexing="ij"
    )
    targets: list[torch.Tensor] = []
    for index in range(batch_size):
        radius = 0.35 + 0.03 * (index % 3)
        center_x = -0.2 + 0.1 * (index % 2)
        center_y = 0.1 - 0.1 * (index % 2)
        mask = (((xx - center_x) ** 2 + (yy - center_y) ** 2) < radius**2).float()
        targets.append(mask)
    segmentation = torch.stack(targets).unsqueeze(1)
    task_targets: dict[str, Any] = {
        "segmentation": segmentation,
        "sample_mask": torch.ones(batch_size, dtype=torch.bool),
    }
    if options.family == "promptable_segmentation":
        task_targets["prompt_map"] = segmentation
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase13-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            task_targets=task_targets,
        )
    ]


def _vlm_data(config: RunConfig, options: _RecipeOptions) -> list[MedicalBatch]:
    generator = torch.Generator().manual_seed(config.seed)
    batch_size = int(config.batch.microbatch_per_device)
    text_length = int(config.recipe.get("text_tokens", 8))
    if text_length < 5:
        raise RecipeConfigurationError("VLM text_tokens must be >= 5 for prompt plus answer tokens")
    values = torch.randn((batch_size, options.channels, options.image_size, options.image_size), generator=generator)
    answer = (values.mean(dim=(1, 2, 3)) > 0).to(torch.long) + 10
    input_ids = torch.full((batch_size, text_length), 3, dtype=torch.long)
    input_ids[:, 0] = 1
    input_ids[:, 1] = 5
    input_ids[:, 2] = 6
    input_ids[:, 3] = 7
    input_ids[:, 4] = answer
    input_ids[:, 5:] = 2
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(attention)
    prompt_mask[:, :4] = True
    labels[prompt_mask] = -100
    metadata = tuple(
        {"view": "AP" if i % 2 == 0 else "lateral", "image_index": i, "timepoint": "baseline"}
        for i in range(batch_size)
    )
    return [
        MedicalBatch(
            modality=options.modality,
            sample_ids=[f"phase13-{options.recipe_id}-{i}" for i in range(batch_size)],
            pixel_values=values,
            input_ids=input_ids,
            attention_mask=attention,
            task_targets={
                "language_labels": labels,
                "prompt_token_mask": prompt_mask,
                "visual_metadata": metadata,
                "task_weights": _task_weights(config),
                "sample_mask": torch.ones(batch_size, dtype=torch.bool),
            },
        )
    ]


def _lora_enabled(config: RunConfig) -> bool:
    value = config.recipe.get("use_lora")
    return bool(config.peft.enabled if value is None else value)


def _task_weights(config: RunConfig) -> dict[str, float]:
    raw = config.recipe.get("mixed_task_weights", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RecipeConfigurationError("mixed_task_weights must be a mapping")
    weights = {str(name): float(value) for name, value in raw.items()}
    if any(value < 0 or not torch.isfinite(torch.tensor(value)) for value in weights.values()):
        raise RecipeConfigurationError("mixed_task_weights must contain finite non-negative values")
    if weights and not any(value > 0 for value in weights.values()):
        raise RecipeConfigurationError("mixed_task_weights must contain a positive value")
    return weights


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


def _inject_recipe_lora(model: nn.Module, config: RunConfig, options: _RecipeOptions) -> None:
    if not _lora_enabled(config):
        return
    rank = int(config.recipe.get("lora_rank", config.peft.rank))
    staged = options.stage.upper() in {
        "B",
        "C",
        "2",
        "3",
        "STAGE2",
        "STAGE3",
    }
    recipe_families = {"classification", "segmentation", "promptable_segmentation", "external_vlm", "native_vlm"}
    if options.family in recipe_families and staged:
        visual = getattr(model, "vision", None)
        if isinstance(visual, TinyVisualAdapter):
            visual.inject_lora(rank=rank, alpha=max(1, rank * 2), dropout=0.0)
    if options.family in {"native_vlm", "external_vlm"} and staged:
        language = getattr(model, "language", None)
        if isinstance(language, MedGemmaAdapter | GenericHFCausalLMAdapter):
            inject_language_lora(language, _lora(rank, architecture="llm", targets=LANGUAGE_LORA_TARGETS))


def _phase13_model(config: RunConfig, options: _RecipeOptions) -> nn.Module:
    vision = _visual_adapter(options)
    if options.family == "classification":
        return _ClassificationModel(vision)
    if options.family in {"segmentation", "promptable_segmentation"}:
        if options.family == "promptable_segmentation":
            return _PromptableSegmentationModel(vision)
        return _SegmentationModel(vision)
    if options.family == "native_vlm":
        native_language = MedGemmaAdapter.build_tiny(
            model_id="medgemma-1.5-4b-offline-tiny",
            hidden_size=options.hidden_size,
            vocab_size=64,
            construction_seed=options.construction_seed,
            visual_token_buckets=(32, 64, 128),
        )
        return _NativeVLMModel(vision, native_language, visual_token_count=options.visual_token_count)
    if options.bridge_type not in {"linear", "perceiver", "perceiver_resampler"}:
        raise RecipeConfigurationError("external bridge must be linear or perceiver")
    language: GenericHFCausalLMAdapter = GenericHFCausalLMAdapter.build_tiny(
        model_id="generic-causal-offline-tiny",
        hidden_size=options.hidden_size,
        vocab_size=64,
        construction_seed=options.construction_seed,
        visual_token_buckets=(32, 64, 128),
    )
    if options.bridge_type == "linear":
        bridge: nn.Module = LinearVisionLanguageBridge(
            source_dim=options.hidden_size,
            target_dim=options.hidden_size,
            output_tokens=options.visual_token_count,
            max_input_tokens=options.visual_token_count,
            source_modality=Modality.XRAY_2D,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        )
    else:
        bridge = PerceiverResamplerBridge(
            query_count=options.visual_token_count,
            heads=4,
            source_dim=options.hidden_size,
            target_dim=options.hidden_size,
            output_tokens=options.visual_token_count,
            max_input_tokens=options.visual_token_count,
            source_modality=Modality.XRAY_2D,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
        )
    return _ExternalVLMModel(
        vision,
        language,
        bridge,
        visual_token_count=options.visual_token_count,
        bridge_type="perceiver" if options.bridge_type != "linear" else "linear",
    )


def _phase13_task(config: RunConfig, options: _RecipeOptions, model: nn.Module) -> TaskModuleBase:
    if options.family == "classification":
        vision = model.vision
        hidden = int(getattr(vision, "hidden_size", options.hidden_size))
        task_name = str(config.task.get("type", config.task.get("name", "BINARY_CLASSIFICATION"))).upper()
        binary = "BINARY" in task_name
        head_name = str(config.recipe.get("head", "linear")).lower()
        output_classes = 1 if binary else 2
        head: nn.Module = (
            MLPClassificationHead(
                hidden,
                output_classes,
                hidden_dim=int(config.recipe.get("head_hidden", hidden)),
            )
            if head_name == "mlp"
            else LinearClassificationHead(hidden, output_classes)
        )
        if binary:
            return BinaryClassificationTask(head, supported_modalities=VISUAL_MODALITIES)
        if "MULTILABEL" in task_name:
            return MultiLabelClassificationTask(head, supported_modalities=VISUAL_MODALITIES)
        return ClassificationTask(
            head,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            supported_modalities=VISUAL_MODALITIES,
        )
    if options.family in {"segmentation", "promptable_segmentation"}:
        decoder = UNetDecoder2D(
            in_channels=options.hidden_size,
            out_channels=1,
            hidden_channels=int(config.recipe.get("decoder_hidden", 16)),
        )
        if options.family == "promptable_segmentation":
            return _PromptableSegmentationTask(decoder, supported_modalities=VISUAL_MODALITIES)
        return BinarySegmentationTask(decoder, supported_modalities=VISUAL_MODALITIES)
    task_name = str(config.task.get("type", config.task.get("name", "VISUAL_QUESTION_ANSWERING"))).upper()
    if "STRUCTURED" in task_name:
        return _RecipeStructuredTask(supported_modalities=VISUAL_MODALITIES)
    task_type = TaskType.REPORT_GENERATION if "REPORT" in task_name else TaskType.VISUAL_QUESTION_ANSWERING
    return _RecipeLanguageTask(supported_modalities=VISUAL_MODALITIES, task_type=task_type)


def _phase13_data(config: RunConfig, options: _RecipeOptions) -> list[MedicalBatch]:
    if options.family == "classification":
        return _classification_data(config, options)
    if options.family in {"segmentation", "promptable_segmentation"}:
        return _segmentation_data(config, options)
    return _vlm_data(config, options)


def build_phase13_recipe(config: RunConfig) -> RecipeBuild:
    """Build one recipe directly, including its offline data contract."""

    options = _options(config)
    model = _phase13_model(config, options)
    _inject_recipe_lora(model, config, options)
    task = _phase13_task(config, options, model)
    return RecipeBuild(
        model=model,
        task=task,
        train_data=_phase13_data(config, options),
        metadata=_metadata(options, _task_weights(config)),
    )


def phase13_builders() -> ComponentBuilders:
    """Return builders compatible with :class:`TrainingPipeline`."""

    def dataset(config: RunConfig, *_: Any) -> list[MedicalBatch]:
        return _phase13_data(config, _options(config))

    def model(config: RunConfig, *_: Any) -> nn.Module:
        options = _options(config)
        value = _phase13_model(config, options)
        _inject_recipe_lora(value, config, options)
        return value

    def peft(model_value: nn.Module, config: RunConfig, *_: Any) -> nn.Module:
        return model_value

    def task(config: RunConfig, model_value: nn.Module, *_: Any) -> TaskModuleBase:
        return _phase13_task(config, _options(config), model_value)

    def optimizer(
        model_value: nn.Module, task_value: nn.Module, config: RunConfig, backend: AcceleratorBackend
    ) -> OptimizerBundle:
        return build_optimizer(
            model_value,
            config.optimizer,
            backend=backend.name,
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
        )

    return ComponentBuilders(dataset=dataset, model=model, peft=peft, task=task, optimizer=optimizer, trainer=trainer)


def restore_mask_to_original(
    mask: torch.Tensor,
    *,
    original_size: tuple[int, int],
    crop_box: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Map a predicted crop mask back to ``(height, width)`` original space."""

    if mask.ndim not in (3, 4):
        raise ShapeContractError("mask must be [B,H,W] or [B,1,H,W]")
    value = mask if mask.ndim == 4 else mask.unsqueeze(1)
    target_h, target_w = original_size
    if crop_box is None:
        restored = F.interpolate(value.float(), size=(target_h, target_w), mode="nearest")
    else:
        x0, y0, x1, y1 = (int(v) for v in crop_box)
        if not (0 <= x0 < x1 <= target_w and 0 <= y0 < y1 <= target_h):
            raise ShapeContractError("crop_box must lie inside original_size")
        crop = F.interpolate(value.float(), size=(y1 - y0, x1 - x0), mode="nearest")
        restored = torch.zeros((*crop.shape[:2], target_h, target_w), dtype=crop.dtype, device=crop.device)
        restored[..., y0:y1, x0:x1] = crop
    return restored if mask.ndim == 4 else restored[:, 0]


def make_phase13_artifact(
    config: RunConfig,
    metrics: Mapping[str, Any],
    *,
    memory: Mapping[str, Any] | None = None,
    limitations: tuple[str, ...] = (
        "Offline tiny checkpoints and synthetic data are contract evidence, not clinical validation.",
        "Production acceptance requires approved de-identified data, external-site testing, and human review.",
    ),
) -> EvaluationArtifact:
    """Create a report artifact carrying the resolved Phase 13 metadata."""

    metric_values = {name: value for name, value in metrics.items() if hasattr(value, "unit")}
    return make_artifact(
        _options(config).recipe_id,
        metric_values,
        config_hash=config.config_hash(),
        seed=config.seed,
        dataset_hash=config.dataset_hash,
        preprocessing_hash=config.preprocessing_hash,
        model_revision=config.base_model_revision,
        memory=memory,
        limitations=limitations,
    )


__all__ = [
    "LANGUAGE_LORA_TARGETS",
    "PHASE13_RECIPE_VERSION",
    "RecipeBuild",
    "RecipeConfigurationError",
    "RecipeMetadata",
    "TinyVisualAdapter",
    "VISUAL_LORA_TARGETS",
    "build_phase13_recipe",
    "make_phase13_artifact",
    "phase13_builders",
    "restore_mask_to_original",
]
