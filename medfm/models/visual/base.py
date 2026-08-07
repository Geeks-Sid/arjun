"""Shared machinery for 2D visual-encoder adapters (Phase 06).

Every 2D adapter (MedSigLIP, RAD-DINO, H-Optimus-0, MedGemma vision pathway,
generic HF/timm fallbacks) is a :class:`BaseVisualAdapter2D` wrapping one
backbone module. The base class owns the cross-cutting contract so individual
adapters only supply backbone construction and token layout:

- honest capability handling: requested outputs the backbone cannot produce
  raise :class:`UnsupportedCapabilityError` (never fabricated);
- preprocessing *declaration* only — ``preprocess_spec()`` describes the
  tensor contract; ``encode`` never preprocesses (Phase 04 pipelines do);
- token semantics: prefix tokens (CLS/registers) are stripped from spatial
  tokens; token coordinates are patch centers in NORMALIZED_IMAGE space;
  token masks cover padded images when multi-image batches are folded;
- training modes: frozen extraction (zero trainable backbone params),
  deterministic evaluation, task-head attachment without importing any
  concrete task, and LoRA injection restricted to declared target modules
  (each with a recorded reason);
- checkpoints: full round-trip saves for development plus the ADR-0006
  canonical adapter-only export (manifest carries base model id, pinned
  revision, and configuration hash; base weights are never re-exported);
- backend neutrality: tensors are created on the input/model device — no
  ``.cuda``, no CUDA-specific constructors, SDPA attention preferred.

Subclass contract: set ``self.backbone`` and implement ``_forward_backbone``
returning a :class:`BackboneResult`. See ``hf_generic.py`` for the reference
implementation over Hugging Face vision towers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import (
    EncoderCapabilities,
    EncoderOutput,
    OutputSpec,
    VisualEncoder,
)
from medfm.core.encoder import (
    PreprocessSpec as CorePreprocessSpec,
)
from medfm.core.enums import CoordinateSystem
from medfm.core.errors import ContractError, ShapeContractError, UnsupportedCapabilityError
from medfm.core.serialization import config_hash, dtype_from_canonical
from medfm.data.transforms.specs import NormalizationSpec
from medfm.data.transforms.specs import PreprocessSpec as RegistryPreprocessSpec

logger = logging.getLogger(__name__)

#: Checkpoint layout version written by save/export methods.
ADAPTER_CHECKPOINT_VERSION = 1


class AdapterCheckpointError(ContractError):
    """A checkpoint or manifest is malformed or missing required provenance."""


class LoRAGateError(ContractError):
    """LoRA was requested before its adapter-specific gate was satisfied."""


class OptionalDependencyError(ContractError):
    """An optional dependency (e.g. timm) required by an adapter is missing."""


class AdapterBuilder(Protocol):
    """Rebuilds an adapter from a serialized config record (checkpoint loads)."""

    def __call__(self, config: dict[str, Any]) -> BaseVisualAdapter2D: ...


@dataclass(frozen=True)
class LoraTargetSpec:
    """One declared LoRA-eligible module group plus *why* it is a target.

    ``pattern`` is a full-module-path regex (PEFT applies ``re.fullmatch`` in
    string mode), scoped to the intended
    submodel so patterns never leak into unrelated towers. Only declared
    patterns may be injected: adapters reject undeclared targets so unknown
    families require explicit confirmation (registry ``PeftCapability``
    policy).
    """

    pattern: str
    reason: str


@dataclass(frozen=True)
class AdapterPreprocess:
    """Single source of truth for an adapter's declared input contract.

    Derives both the core-contract :class:`CorePreprocessSpec` (returned by
    ``preprocess_spec()``) and the registry/pipeline
    :class:`RegistryPreprocessSpec` (Phase 04 validation + registry record)
    so the two can never disagree.

    - ``image_size``: exact (H, W) the backbone consumes (fixed resolution).
    - ``value_range``: expected range *before* mean/std normalization.
    - ``resize_policy``: how Phase 04 reaches ``image_size`` (letterbox,
      stretch, center_crop) — informational, never executed here.
    - ``color_space``: expected color layout (e.g. ``RGB`` or
      ``GRAYSCALE_REPEATED_TO_RGB`` for single-channel radiology).
    """

    image_size: tuple[int, int]
    channels: int
    patch_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    value_range: tuple[float, float] = (0.0, 1.0)
    resize_policy: str = "letterbox"
    color_space: str = "RGB"
    canonical_dtype: str = "float32"

    def __post_init__(self) -> None:
        h, w = self.image_size
        if h <= 0 or w <= 0 or h % self.patch_size or w % self.patch_size:
            raise ShapeContractError(
                f"image_size {self.image_size} must be positive and divisible by patch_size {self.patch_size}"
            )
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ShapeContractError("mean/std must have one entry per channel")
        if self.value_range[0] >= self.value_range[1]:
            raise ShapeContractError("value_range must be increasing")
        dtype_from_canonical(self.canonical_dtype)

    @property
    def patch_grid(self) -> tuple[int, int]:
        """(rows, cols) of the patch grid at the declared image size."""
        return (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)

    @property
    def num_patches(self) -> int:
        rows, cols = self.patch_grid
        return rows * cols

    def core_spec(self) -> CorePreprocessSpec:
        return CorePreprocessSpec(
            image_size=self.image_size,
            channels=self.channels,
            mean=self.mean,
            std=self.std,
            value_range=self.value_range,
            resize_policy=self.resize_policy,
            canonical_dtype=self.canonical_dtype,
        )

    def registry_spec(self, model_id: str) -> RegistryPreprocessSpec:
        return RegistryPreprocessSpec(
            model_id=model_id,
            spatial_shape=self.image_size,
            channels=self.channels,
            dtype=self.canonical_dtype,
            value_range=self.value_range,
            normalization=NormalizationSpec(mean=self.mean, std=self.std),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterPreprocess:
        image_size = (int(data["image_size"][0]), int(data["image_size"][1]))
        value_range = (float(data["value_range"][0]), float(data["value_range"][1]))
        return cls(
            image_size=image_size,
            channels=int(data["channels"]),
            patch_size=int(data["patch_size"]),
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
            value_range=value_range,
            resize_policy=str(data["resize_policy"]),
            color_space=str(data["color_space"]),
            canonical_dtype=str(data["canonical_dtype"]),
        )


@dataclass(frozen=True)
class BackboneResult:
    """Raw result of one backbone forward, in backbone-native layout.

    - ``last_hidden_state``: full token sequence including prefix tokens
      (CLS/registers), shape ``[B, S, D]``.
    - ``pooled``: backbone-native pooled vector ``[B, Dp]`` or ``None`` when
      the backbone has no native pooling (the adapter then mean-pools over
      spatial tokens and declares it explicitly).
    - ``hidden_states``: per-layer states when requested, same layout.
    - ``raw``: the native output object, kept only for debugging.
    """

    last_hidden_state: torch.Tensor
    pooled: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None
    raw: Any


class LinearHead(nn.Module):
    """Minimal linear classification head for adapter-attachment verification.

    This is attachment scaffolding, *not* a task module: real task heads and
    losses arrive in Phase 11 as ``TaskModule`` implementations consuming
    ``EncoderOutput``. Its architecture is serializable so adapter
    checkpoints round-trip without importing any concrete task.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ShapeContractError("LinearHead dimensions must be positive")
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear(pooled))

    def architecture_dict(self) -> dict[str, int | str]:
        return {
            "type": "linear",
            "in_features": int(self.linear.in_features),
            "out_features": int(self.linear.out_features),
        }

    @classmethod
    def from_architecture_dict(cls, data: dict[str, Any]) -> LinearHead:
        if data.get("type") != "linear":
            raise AdapterCheckpointError(f"unsupported head architecture: {data!r}")
        return cls(in_features=int(data["in_features"]), out_features=int(data["out_features"]))


class BaseVisualAdapter2D(nn.Module):
    """Base class for 2D visual-encoder adapters; see module docstring."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        capabilities: EncoderCapabilities,
        preprocess: AdapterPreprocess,
        feature_map_layers: tuple[int, ...] = (),
        lora_targets: tuple[LoraTargetSpec, ...] = (),
        construction_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not model_id:
            raise ShapeContractError("adapter model_id must be non-empty")
        self._model_id = model_id
        self._revision = revision
        self._capabilities = capabilities
        self._preprocess = preprocess
        self._feature_map_layers = tuple(int(i) for i in feature_map_layers)
        self._lora_targets = lora_targets
        self._construction_seed = construction_seed
        self.backbone: nn.Module  # set by subclass constructors
        self._head: nn.Module | None = None
        self._lora_state: dict[str, Any] | None = None
        self._pending_lora_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Contract surface
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> EncoderCapabilities:
        return self._capabilities

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def preprocess(self) -> AdapterPreprocess:
        return self._preprocess

    def preprocess_spec(self) -> CorePreprocessSpec:
        """Declared input contract; encoding performs no preprocessing."""
        return self._preprocess.core_spec()

    def registry_preprocess_spec(self) -> RegistryPreprocessSpec:
        """The Phase-04/registry view of the same declaration."""
        return self._preprocess.registry_spec(self._model_id)

    def satisfies_visual_encoder_protocol(self) -> bool:
        """Explicit protocol check used by contract tests."""
        return isinstance(self, VisualEncoder)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def _forward_backbone(self, pixel_values: torch.Tensor, output_hidden_states: bool) -> BackboneResult:
        raise NotImplementedError

    def _prefix_token_count(self) -> int:
        """Leading non-spatial tokens (CLS/registers) in backbone sequences."""
        raise NotImplementedError

    def encode(
        self,
        batch: MedicalBatch,
        output_hidden_states: bool = False,
        output_spec: OutputSpec | None = None,
    ) -> EncoderOutput:
        """Encode a collated batch into the shared ``EncoderOutput`` contract.

        ``output_spec`` defaults to pooled-only. Requesting anything the
        adapter cannot produce raises :class:`UnsupportedCapabilityError`
        before the backbone runs. ``encode`` never preprocesses: pixels must
        already match :meth:`preprocess_spec`.
        """
        request = output_spec if output_spec is not None else OutputSpec(pooled=True)
        request.check_supported(self._capabilities)
        self._capabilities.require_modality(batch.modality)
        pixel_values, fold = self._extract_pixels(batch)
        want_hidden = output_hidden_states or bool(request.feature_maps and self._feature_map_layers)
        result = self._forward_backbone(pixel_values, want_hidden)

        patch_tokens = self._strip_prefix(result.last_hidden_state)
        b, n, d = patch_tokens.shape
        expected_n = self._preprocess.num_patches
        if n != expected_n:
            raise ShapeContractError(
                f"backbone produced {n} patch tokens but the declared grid "
                f"{self._preprocess.patch_grid} expects {expected_n}; patch-size/preprocess mismatch"
            )

        pooled = result.pooled if result.pooled is not None else patch_tokens.mean(dim=1)
        token_mask = self._token_mask(batch, fold, patch_tokens)
        auxiliary: dict[str, Any] = {
            "patch_grid": self._preprocess.patch_grid,
            "prefix_tokens_stripped": self._prefix_token_count(),
        }
        if fold is not None:
            auxiliary["multi_image_fold"] = {"batch": fold[0], "images_per_sample": fold[1]}

        feature_maps: tuple[torch.Tensor, ...] | None = None
        if request.feature_maps:
            feature_maps = self._build_feature_maps(result, patch_tokens)

        native_outputs: Any | None = None
        hidden_states: tuple[torch.Tensor, ...] | None = None
        if output_hidden_states:
            hidden_states = tuple(self._strip_prefix(h) for h in (result.hidden_states or ()))
            native_outputs = result.raw
            auxiliary["native_outputs_kind"] = f"hf_model_output:{type(result.raw).__name__}"
            auxiliary["hidden_state_layers"] = len(result.hidden_states or ())

        output = EncoderOutput(
            pooled_embedding=pooled if request.pooled else None,
            spatial_tokens=patch_tokens if request.spatial_tokens else None,
            feature_maps=feature_maps if request.feature_maps else None,
            token_mask=token_mask if request.spatial_tokens else None,
            token_coordinates=(
                self._token_coordinates(patch_tokens, token_mask) if request.token_coordinates else None
            ),
            token_coordinate_system=(CoordinateSystem.NORMALIZED_IMAGE if request.token_coordinates else None),
            logits=None,
            native_outputs=native_outputs,
            auxiliary=auxiliary,
        )
        if hidden_states is not None:
            output.auxiliary["hidden_states"] = hidden_states
        output.check_against(request)
        return output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Smoke-test shortcut: pooled embedding for one fixed-shape input.

        The full contract path is :meth:`encode`; this exists so the registry
        smoke runner (``model(**tiny_input)``) can exercise the adapter.
        """
        if self.training:
            raise ShapeContractError("forward() smoke path requires eval mode; call adapter.eval() first")
        batch = MedicalBatch(
            modality=self._capabilities.modalities[0],
            sample_ids=[f"smoke-{i}" for i in range(int(pixel_values.shape[0]))],
            pixel_values=pixel_values,
        )
        pooled = self.encode(batch, output_spec=OutputSpec(pooled=True)).pooled_embedding
        assert pooled is not None
        return pooled

    def _extract_pixels(self, batch: MedicalBatch) -> tuple[torch.Tensor, tuple[int, int] | None]:
        pixel_values = batch.pixel_values
        if pixel_values is None:
            raise ShapeContractError(f"{self._model_id} requires batch.pixel_values; got None")
        fold: tuple[int, int] | None = None
        if pixel_values.ndim == 5:
            # MULTI_IMAGE_2D [B, I, C, H, W] -> fold I into the batch dim.
            b, i, c, h, w = pixel_values.shape
            fold = (b, i)
            pixel_values = pixel_values.reshape(b * i, c, h, w)
        if pixel_values.ndim != 4:
            raise ShapeContractError(
                f"{self._model_id} expects [B, C, H, W] pixels (or [B, I, C, H, W] for MULTI_IMAGE_2D); "
                f"got rank {pixel_values.ndim}"
            )
        _, channels, height, width = pixel_values.shape
        exp_h, exp_w = self._preprocess.image_size
        if int(channels) != self._preprocess.channels or (int(height), int(width)) != (exp_h, exp_w):
            raise ShapeContractError(
                f"preprocess mismatch for {self._model_id}: expects [C={self._preprocess.channels}, "
                f"H={exp_h}, W={exp_w}] per the adapter's preprocess_spec(); got "
                f"[C={int(channels)}, H={int(height)}, W={int(width)}]. Preprocess externally (Phase 04) "
                "to the declared contract."
            )
        return pixel_values, fold

    def _strip_prefix(self, tokens: torch.Tensor) -> torch.Tensor:
        prefix = self._prefix_token_count()
        if prefix == 0:
            return tokens
        return tokens[:, prefix:, :]

    def _token_mask(
        self, batch: MedicalBatch, fold: tuple[int, int] | None, patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        b, n, _ = patch_tokens.shape
        mask = torch.ones(b, n, dtype=torch.bool, device=patch_tokens.device)
        if fold is not None and batch.image_mask is not None:
            # A padded image produces all-padding token rows.
            image_mask = batch.image_mask.to(dtype=torch.bool, device=patch_tokens.device)
            mask = image_mask.reshape(-1, 1).expand(b, n).contiguous()
        return mask

    def _token_coordinates(self, patch_tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Patch-center coordinates in NORMALIZED_IMAGE space, row-major."""
        rows, cols = self._preprocess.patch_grid
        device, dtype = patch_tokens.device, patch_tokens.dtype
        ys = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows
        xs = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return coords.unsqueeze(0).expand(int(patch_tokens.shape[0]), -1, -1).contiguous()

    def _build_feature_maps(self, result: BackboneResult, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Dense maps for segmentation decoders from hidden-state hooks.

        Single-scale ViT pyramids share spatial resolution across entries;
        entries are ordered shallow -> deep, the deepest (final) layer last.
        Hook layer indices are adapter-configured and pinned to the model
        revision via the checkpoint manifest / registry record.
        """
        rows, cols = self._preprocess.patch_grid
        maps: list[torch.Tensor] = []
        if self._feature_map_layers:
            hidden = result.hidden_states
            if hidden is None or len(hidden) <= max(self._feature_map_layers):
                raise UnsupportedCapabilityError(
                    f"{self._model_id} feature-map hooks require hidden states up to layer "
                    f"{max(self._feature_map_layers)}; backbone returned {len(hidden or ())}"
                )
            sources = [hidden[i] for i in self._feature_map_layers]
        else:
            sources = [result.last_hidden_state]
        for layer_tokens in sources:
            patches = self._strip_prefix(layer_tokens)
            b, n, d = patches.shape
            maps.append(patches.transpose(1, 2).reshape(b, d, rows, cols).contiguous())
        return tuple(maps)

    # ------------------------------------------------------------------ #
    # Training modes
    # ------------------------------------------------------------------ #

    def freeze_backbone(self) -> None:
        """Frozen extraction mode: zero trainable backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        if self.lora_active:
            raise LoRAGateError("LoRA is active; unfreezing the full backbone defeats PEFT (ADR 0002)")
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def trainable_backbone_parameters(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def set_deterministic_eval(self) -> None:
        """Deterministic evaluation mode: eval() + frozen dropout-free forward.

        Backbones in this package carry no stochastic layers at eval time, so
        repeated ``encode`` calls on the same input are bit-identical.
        """
        self.eval()

    def extract(self, batch: MedicalBatch, output_spec: OutputSpec | None = None) -> EncoderOutput:
        """Frozen-extraction helper: inference-mode encode (no autograd)."""
        with torch.inference_mode():
            return self.encode(batch, output_hidden_states=False, output_spec=output_spec)

    # ------------------------------------------------------------------ #
    # Task-head attachment (no concrete task import)
    # ------------------------------------------------------------------ #

    def attach_head(self, head: nn.Module) -> None:
        """Attach a task head mapping pooled embeddings [B, Dp] -> [B, K].

        The adapter never runs losses; heads are applied explicitly via
        :meth:`head_logits` so task modules (Phase 11) own the loss path.
        """
        if not isinstance(head, nn.Module):
            raise TypeError(f"head must be an nn.Module; got {type(head).__name__}")
        device = self._device()
        self._head = head.to(device)

    def detach_head(self) -> None:
        self._head = None

    @property
    def attached_head(self) -> nn.Module | None:
        return self._head

    def head_logits(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        if self._head is None:
            raise UnsupportedCapabilityError(f"{self._model_id} has no attached head; call attach_head first")
        return cast(torch.Tensor, self._head(pooled_embedding))

    # ------------------------------------------------------------------ #
    # LoRA
    # ------------------------------------------------------------------ #

    def lora_target_specs(self) -> tuple[LoraTargetSpec, ...]:
        """Declared LoRA-eligible module patterns with reasons."""
        return self._lora_targets

    def lora_target_patterns(self) -> tuple[str, ...]:
        return tuple(t.pattern for t in self._lora_targets)

    @property
    def lora_active(self) -> bool:
        return self._lora_state is not None

    @property
    def lora_state(self) -> dict[str, Any] | None:
        """LoRA construction parameters, once injected."""
        return dict(self._lora_state) if self._lora_state is not None else None

    def check_lora_allowed(self) -> None:
        """Adapter-specific LoRA gate (e.g. H-Optimus frozen-baseline rule)."""

    def inject_lora(
        self,
        *,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        targets: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Inject LoRA into the backbone; returns matched target module names.

        Only declared patterns (``lora_target_specs``) may be injected; the
        backbone is frozen first so gradients flow only to LoRA parameters
        (plus any attached head, which stays trainable independently).
        """
        self.check_lora_allowed()
        if self.lora_active:
            raise LoRAGateError(f"{self._model_id} already has LoRA injected")
        declared = set(self.lora_target_patterns())
        requested = tuple(targets) if targets is not None else self.lora_target_patterns()
        undeclared = sorted(set(requested) - declared)
        if undeclared:
            raise UnsupportedCapabilityError(
                f"{self._model_id} LoRA targets {undeclared} are not declared; declared: {sorted(declared)}. "
                "Unknown-family targets require explicit confirmation (registry PeftCapability policy)."
            )
        from medfm.peft import LoRAConfig, inject_visual_lora

        config = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_policy="explicit",
            target_modules=requested,
            adapter_name="default",
            architecture="vision",
            confirm_target_modules=True,
        )
        result = inject_visual_lora(self, config)
        self._lora_state = {
            "rank": int(rank),
            "alpha": int(alpha),
            "dropout": float(dropout),
            "targets": list(requested),
            "adapter_name": config.adapter_name,
        }
        return list(result.matched_module_names)

    def lora_matched_module_names(self) -> list[str]:
        """Backbone module names that received LoRA adapters.

        After ``get_peft_model`` the module tree is prefixed
        (``base_model.model.…``), so the declared patterns are matched with
        ``re.search`` against the full path (the ``hasattr(module, 'lora_A')``
        filter already restricts candidates to adapted modules).
        """
        if self._lora_state is None:
            return []
        patterns = tuple(str(t) for t in self._lora_state["targets"])
        names = {
            name
            for name, module in self.backbone.named_modules()
            if hasattr(module, "lora_A") and any(re.search(p, name) for p in patterns)
        }
        return sorted(names)

    def restore_lora(self) -> None:
        """Re-inject LoRA with the exact parameters recorded in the
        adapter-only checkpoint manifest (backbone is rebuilt first)."""
        if self.lora_active:
            raise LoRAGateError(f"{self._model_id} already has LoRA injected")
        if self._pending_lora_state is None:
            raise LoRAGateError("no LoRA state recorded to restore")
        state = self._pending_lora_state
        self._pending_lora_state = None
        self.inject_lora(
            rank=int(state["rank"]),
            alpha=int(state["alpha"]),
            dropout=float(state["dropout"]),
            targets=tuple(str(t) for t in state["targets"]),
        )

    def lora_parameters(self) -> dict[str, torch.Tensor]:
        return {n: p for n, p in self.backbone.named_parameters() if "lora_" in n}

    # ------------------------------------------------------------------ #
    # Devices and dtypes (backend-neutral)
    # ------------------------------------------------------------------ #

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def compute_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict[str, Any]:
        """Construction record; subclasses extend with family specifics."""
        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "adapter_class": type(self).__name__,
            "preprocess": self._preprocess.to_dict(),
            "feature_map_layers": list(self._feature_map_layers),
            "construction_seed": self._construction_seed,
            "lora_state": self._lora_state,
        }

    def config_hash(self) -> str:
        return config_hash(self._config_dict())

    def _head_architecture(self) -> dict[str, Any] | None:
        if self._head is None:
            return None
        if isinstance(self._head, LinearHead):
            return self._head.architecture_dict()
        return None  # generic heads re-attach by the caller; state is still saved

    def save_checkpoint(self, directory: str | Path, *, include_backbone: bool = True) -> Path:
        """Development checkpoint: full round-trip (optionally full weights).

        Canonical training artifacts use :meth:`export_adapter_checkpoint`
        instead (ADR 0006). Saved tensors are materialized on CPU; devices
        are never serialized.
        """
        from safetensors.torch import save_file

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        if include_backbone:
            tensors.update({f"backbone.{k}": v.detach().cpu() for k, v in self.backbone.state_dict().items()})
        elif self.lora_active:
            tensors.update({f"backbone.{k}": v.detach().cpu() for k, v in self.lora_parameters().items()})
        if self._head is not None:
            tensors.update({f"head.{k}": v.detach().cpu() for k, v in self._head.state_dict().items()})
        manifest = {
            "checkpoint_version": ADAPTER_CHECKPOINT_VERSION,
            "kind": "full" if include_backbone else "adapter_only",
            "model_id": self._model_id,
            "revision": self._revision,
            "config": self._config_dict(),
            "config_hash": self.config_hash(),
            "lora_active": self.lora_active,
            "head_architecture": self._head_architecture(),
            "tensor_file": "tensors.safetensors",
            "created_at": datetime.now(UTC).isoformat(),
        }
        save_file(tensors, str(out_dir / "tensors.safetensors"))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return out_dir

    def export_adapter_checkpoint(self, directory: str | Path) -> Path:
        """ADR-0006 canonical export: trained params only (LoRA + head).

        The manifest carries the base model id, pinned revision, and config
        hash; a checkpoint without these is not exportable. Base weights are
        never re-exported (license-safe for gated backbones).
        """
        if not re.fullmatch(r"[0-9a-f]{40,64}", self._revision):
            raise AdapterCheckpointError(
                f"canonical export requires a pinned base revision; got {self._revision!r}. "
                "Pin the base checkpoint revision first (docs/reproducibility_policy.md)."
            )
        if not self.lora_active and self._head is None:
            raise AdapterCheckpointError("nothing trained to export: no LoRA and no attached head")
        return self.save_checkpoint(directory, include_backbone=False)

    @staticmethod
    def _read_manifest(directory: str | Path) -> dict[str, Any]:
        path = Path(directory) / "manifest.json"
        if not path.exists():
            raise AdapterCheckpointError(f"no manifest.json in {directory}")
        manifest = json.loads(path.read_text())
        if manifest.get("checkpoint_version") != ADAPTER_CHECKPOINT_VERSION:
            raise AdapterCheckpointError(
                f"unsupported checkpoint version {manifest.get('checkpoint_version')}; "
                f"expected {ADAPTER_CHECKPOINT_VERSION}"
            )
        for required in ("model_id", "revision", "config", "config_hash"):
            if required not in manifest:
                raise AdapterCheckpointError(f"manifest missing required provenance field {required!r}")
        return cast(dict[str, Any], manifest)

    @classmethod
    def load_checkpoint(
        cls,
        directory: str | Path,
        *,
        rebuild: AdapterBuilder,
        device: str | torch.device | None = None,
    ) -> BaseVisualAdapter2D:
        """Reload an adapter from a checkpoint directory.

        ``rebuild(config)`` constructs the adapter (backbone from the pinned
        base revision or from the recorded tiny construction config); the
        checkpoint then restores backbone weights (full checkpoints) or
        re-injects LoRA and loads trained tensors (adapter-only checkpoints,
        ADR 0006). Outputs compare within dtype-specific tolerance.
        """
        from safetensors.torch import load_file

        out_dir = Path(directory)
        manifest = cls._read_manifest(out_dir)
        adapter = rebuild(manifest["config"])
        if adapter.model_id != manifest["model_id"]:
            raise AdapterCheckpointError(
                f"checkpoint belongs to {manifest['model_id']!r}; rebuild produced {adapter.model_id!r}"
            )
        tensors = load_file(str(out_dir / manifest.get("tensor_file", "tensors.safetensors")))
        backbone_state = {k.removeprefix("backbone."): v for k, v in tensors.items() if k.startswith("backbone.")}
        head_state = {k.removeprefix("head."): v for k, v in tensors.items() if k.startswith("head.")}

        if manifest["kind"] == "full":
            incompatible = adapter.backbone.load_state_dict(backbone_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise AdapterCheckpointError(
                    f"backbone state mismatch: missing={incompatible.missing_keys[:4]} "
                    f"unexpected={incompatible.unexpected_keys[:4]}"
                )
        else:
            lora_state = manifest["config"].get("lora_state")
            if lora_state is not None:
                adapter._pending_lora_state = lora_state
                adapter.restore_lora()
                adapter.backbone.load_state_dict(backbone_state, strict=False)
        if head_state:
            arch = manifest.get("head_architecture")
            if arch is None:
                logger.warning("checkpoint carries a non-serializable head's tensors; re-attach the head manually")
            else:
                head = LinearHead.from_architecture_dict(arch)
                head.load_state_dict(head_state)
                adapter.attach_head(head)
        if device is not None:
            adapter = adapter.to(device)
        return adapter

    # ------------------------------------------------------------------ #
    # TPU compilation smoke configuration
    # ------------------------------------------------------------------ #

    def tpu_smoke_config(self) -> dict[str, Any]:
        """Fixed-resolution/static-batch configuration for XLA compilation.

        Static shapes avoid recompilation; BF16 is the TPU compute dtype
        (ADR 0009: NF4/INT8 are never used on xla_tpu).
        """
        h, w = self._preprocess.image_size
        return {
            "model_id": self._model_id,
            "batch_size": 2,
            "channels": self._preprocess.channels,
            "image_size": [h, w],
            "dtype": "bfloat16",
            "static_batch": True,
            "attention": "sdpa",
        }
