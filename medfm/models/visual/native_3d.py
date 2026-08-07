"""Backend-neutral native 3D visual adapter contract.

The adapter deliberately consumes canonical ``[B, C, D, H, W]`` tensors from
Phase 04. It never turns a volume into a batch of unrelated slices. The tiny
local MONAI-compatible architecture is useful for contract tests and as a
pure-PyTorch fallback when an upstream checkpoint uses unavailable operators.
Model-specific modules in this package only declare preprocessing, checkpoint
identity, and capability differences.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderCapabilities, EncoderOutput, OutputSpec, VisualEncoder
from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ContractError, ShapeContractError, UnsupportedCapabilityError
from medfm.core.sample import SpatialMetadata
from medfm.core.serialization import config_hash, dtype_from_canonical
from medfm.data.transforms.specs import NormalizationSpec
from medfm.data.transforms.specs import PreprocessSpec as RegistryPreprocessSpec
from medfm.models.visual.base import LoraTargetSpec

logger = logging.getLogger(__name__)

NATIVE_3D_CHECKPOINT_VERSION = 1


class Native3DCheckpointError(ContractError):
    """A native 3D checkpoint is malformed or lacks provenance."""


class Native3DOptionalDependencyError(ContractError):
    """An optional upstream dependency is unavailable."""


@dataclass(frozen=True)
class Native3DPreprocess:
    """Exact canonical volume contract for a native 3D checkpoint."""

    spatial_shape: tuple[int, int, int]
    channels: int
    patch_size: tuple[int, int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    value_range: tuple[float, float] | None = None
    resize_policy: str = "crop_or_pad"
    orientation: str = "RAS"
    sequence_order: tuple[str, ...] = ()
    canonical_dtype: str = "float32"

    def __post_init__(self) -> None:
        shape = tuple(int(v) for v in self.spatial_shape)
        patch = tuple(int(v) for v in self.patch_size)
        if len(shape) != 3 or len(patch) != 3 or any(v <= 0 for v in shape + patch):
            raise ShapeContractError("native 3D spatial_shape and patch_size must contain three positive values")
        if any(s % p for s, p in zip(shape, patch, strict=True)):
            raise ShapeContractError(f"spatial_shape {shape} must be divisible by patch_size {patch}")
        if self.channels <= 0 or len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ShapeContractError("native 3D channels and mean/std lengths are inconsistent")
        if any(s <= 0 for s in self.std):
            raise ShapeContractError("native 3D standard deviations must be positive")
        if self.value_range is not None and self.value_range[0] >= self.value_range[1]:
            raise ShapeContractError("native 3D value_range must be increasing")
        if self.sequence_order and len(self.sequence_order) != self.channels:
            raise ShapeContractError("sequence_order must have one name per input channel")
        dtype_from_canonical(self.canonical_dtype)
        object.__setattr__(self, "spatial_shape", shape)
        object.__setattr__(self, "patch_size", patch)

    @property
    def patch_grid(self) -> tuple[int, int, int]:
        depth, height, width = self.spatial_shape
        patch_depth, patch_height, patch_width = self.patch_size
        return (depth // patch_depth, height // patch_height, width // patch_width)

    @property
    def num_patches(self) -> int:
        return math.prod(self.patch_grid)

    def core_spec(self) -> Any:
        from medfm.core.encoder import PreprocessSpec

        return PreprocessSpec(
            image_size=self.spatial_shape,
            channels=self.channels,
            mean=self.mean,
            std=self.std,
            value_range=self.value_range or (0.0, 1.0),
            resize_policy=self.resize_policy,
            canonical_dtype=self.canonical_dtype,
        )

    def registry_spec(self, model_id: str) -> RegistryPreprocessSpec:
        return RegistryPreprocessSpec(
            model_id=model_id,
            spatial_shape=self.spatial_shape,
            channels=self.channels,
            dtype=self.canonical_dtype,
            value_range=self.value_range,
            normalization=NormalizationSpec(mean=self.mean, std=self.std),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Native3DPreprocess:
        raw_range = data.get("value_range")
        return cls(
            spatial_shape=cast(tuple[int, int, int], tuple(int(v) for v in data["spatial_shape"])),
            channels=int(data["channels"]),
            patch_size=cast(tuple[int, int, int], tuple(int(v) for v in data["patch_size"])),
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
            value_range=(
                cast(tuple[float, float], tuple(float(v) for v in raw_range)) if raw_range is not None else None
            ),
            resize_policy=str(data.get("resize_policy", "crop_or_pad")),
            orientation=str(data.get("orientation", "RAS")),
            sequence_order=tuple(str(v) for v in data.get("sequence_order", ())),
            canonical_dtype=str(data.get("canonical_dtype", "float32")),
        )


@dataclass(frozen=True)
class _BackboneResult3D:
    tokens: torch.Tensor
    pooled: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class _LocalMONAI3DBackbone(nn.Module):
    """Small transformer over volume patches; no MONAI/CUDA-only operators."""

    def __init__(self, channels: int, hidden_size: int, depth: int, heads: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_embed = nn.Conv3d(channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, volume: torch.Tensor) -> _BackboneResult3D:
        patches = self.patch_embed(volume).flatten(2).transpose(1, 2).contiguous()
        cls = self.cls_token.expand(volume.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        states: list[torch.Tensor] = [tokens]
        for block in self.blocks.layers:
            tokens = block(tokens)
            states.append(tokens)
        tokens = self.norm(tokens)
        states[-1] = tokens
        return _BackboneResult3D(tokens=tokens, pooled=tokens[:, 0], hidden_states=tuple(states))


class GenericMONAI3DAdapter(nn.Module):
    """Generic native-volume adapter with honest spatial/physical semantics."""

    def __init__(
        self,
        *,
        model_id: str = "generic-monai-3d",
        revision: str = "local-generic-3d",
        preprocess: Native3DPreprocess,
        capabilities: EncoderCapabilities | None = None,
        hidden_size: int = 64,
        depth: int = 2,
        heads: int = 4,
        feature_map_layers: tuple[int, ...] = (),
        lora_targets: tuple[str, ...] = (
            r"blocks\.layers\.\d+\.self_attn\.out_proj",
            r"blocks\.layers\.\d+\.linear[12]",
        ),
        construction_seed: int | None = None,
        max_full_volume_voxels: int = 256 * 256 * 256,
        unsupported_xla_ops: tuple[str, ...] = (),
        custom_cuda_dependencies: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or depth <= 0 or heads <= 0 or hidden_size % heads:
            raise ShapeContractError("hidden_size must be positive and divisible by heads; depth must be positive")
        if any(i < 0 or i > depth for i in feature_map_layers):
            raise ShapeContractError("feature_map_layers must be valid transformer layer indices")
        self._model_id = model_id
        self._revision = revision
        self._preprocess = preprocess
        self._hidden_size = int(hidden_size)
        self._depth = int(depth)
        self._heads = int(heads)
        self._feature_map_layers = tuple(feature_map_layers)
        self._lora_targets = tuple(lora_targets)
        self._construction_seed = construction_seed
        self._max_full_volume_voxels = int(max_full_volume_voxels)
        self._unsupported_xla_ops = tuple(unsupported_xla_ops)
        self._custom_cuda_dependencies = tuple(custom_cuda_dependencies)
        if capabilities is None:
            capabilities = EncoderCapabilities(
                model_id=model_id,
                modalities=(Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D),
                supports_pooled=True,
                supports_spatial_tokens=True,
                supports_feature_maps=True,
                supports_token_coordinates=True,
                token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
            )
        self._capabilities = capabilities
        if construction_seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(construction_seed)
                self.backbone = _LocalMONAI3DBackbone(
                    preprocess.channels, hidden_size, depth, heads, preprocess.patch_size
                )
        else:
            self.backbone = _LocalMONAI3DBackbone(preprocess.channels, hidden_size, depth, heads, preprocess.patch_size)
        self._head: nn.Module | None = None
        self._lora_state: dict[str, Any] | None = None

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "generic-monai-3d-tiny",
        modality: Modality = Modality.CT_3D,
        channels: int = 1,
        construction_seed: int = 0,
    ) -> GenericMONAI3DAdapter:
        preprocess = Native3DPreprocess(
            spatial_shape=(16, 16, 16),
            channels=channels,
            patch_size=(4, 4, 4),
            mean=tuple(0.0 for _ in range(channels)),
            std=tuple(1.0 for _ in range(channels)),
            value_range=None,
            sequence_order=tuple(f"sequence_{i}" for i in range(channels)),
        )
        caps = EncoderCapabilities(
            model_id=model_id,
            modalities=(modality,),
            supports_pooled=True,
            supports_spatial_tokens=True,
            supports_feature_maps=True,
            supports_token_coordinates=True,
            token_coordinate_systems=(CoordinateSystem.MILLIMETERS,),
        )
        return cls(
            model_id=model_id,
            revision="local-tiny",
            preprocess=preprocess,
            capabilities=caps,
            hidden_size=32,
            depth=2,
            heads=4,
            feature_map_layers=(1, 2),
            construction_seed=construction_seed,
            max_full_volume_voxels=16 * 16 * 16,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def capabilities(self) -> EncoderCapabilities:
        return self._capabilities

    @property
    def preprocess(self) -> Native3DPreprocess:
        return self._preprocess

    def preprocess_spec(self) -> Any:
        return self._preprocess.core_spec()

    def registry_preprocess_spec(self) -> RegistryPreprocessSpec:
        return self._preprocess.registry_spec(self._model_id)

    def satisfies_visual_encoder_protocol(self) -> bool:
        return isinstance(self, VisualEncoder)

    def _validate_spatial_metadata(self, batch: MedicalBatch, spatial: tuple[int, int, int]) -> None:
        if not batch.spatial_metadata:
            return
        for index, metadata in enumerate(batch.spatial_metadata):
            if metadata is None:
                continue
            if metadata.spatial_rank != 3 or tuple(metadata.current_shape) != spatial:
                raise ShapeContractError(
                    f"{self._model_id} metadata[{index}] current_shape {metadata.current_shape} "
                    f"must match volume {spatial}"
                )
            if metadata.spacing_mm is not None and len(metadata.spacing_mm) != 3:
                raise ShapeContractError(f"{self._model_id} requires three spacing_mm values for native 3D input")
            if metadata.orientation is not None and len(metadata.orientation) != 3:
                raise ShapeContractError(f"{self._model_id} orientation must contain three anatomical axes")
            if metadata.orientation is not None and metadata.orientation != self._preprocess.orientation:
                raise ShapeContractError(
                    f"{self._model_id} expects canonical orientation {self._preprocess.orientation}; "
                    f"got {metadata.orientation}"
                )

    def _extract_volume(self, batch: MedicalBatch) -> torch.Tensor:
        if batch.pixel_values is None or batch.pixel_values.ndim != 5:
            raise ShapeContractError(f"{self._model_id} expects canonical [B, C, D, H, W] pixel_values")
        volume = batch.pixel_values
        expected = (self._preprocess.channels, *self._preprocess.spatial_shape)
        if tuple(volume.shape[1:]) != expected:
            raise ShapeContractError(
                f"preprocess mismatch for {self._model_id}: expects [C,D,H,W]={expected}; got {tuple(volume.shape[1:])}"
            )
        if volume.dtype != dtype_from_canonical(self._preprocess.canonical_dtype):
            raise ShapeContractError(
                f"preprocess mismatch for {self._model_id}: expects dtype "
                f"{self._preprocess.canonical_dtype}; got {volume.dtype}"
            )
        return volume

    def _forward_backbone(self, volume: torch.Tensor) -> _BackboneResult3D:
        return cast(_BackboneResult3D, self.backbone(volume))

    def _coordinates(self, batch: MedicalBatch, tokens: torch.Tensor) -> torch.Tensor:
        if not batch.spatial_metadata or any(m is None for m in batch.spatial_metadata):
            raise ShapeContractError(
                f"{self._model_id} cannot emit MILLIMETERS token_coordinates without affine/spacing metadata"
            )
        pd, ph, pw = self._preprocess.patch_size
        gd, gh, gw = self._preprocess.patch_grid
        # Flatten order follows Conv3d.flatten(2): depth, height, width.
        z, y, x = torch.meshgrid(
            torch.arange(gd, device=tokens.device, dtype=tokens.dtype),
            torch.arange(gh, device=tokens.device, dtype=tokens.dtype),
            torch.arange(gw, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        ijk = torch.stack(((x + 0.5) * pw, (y + 0.5) * ph, (z + 0.5) * pd), dim=-1).reshape(-1, 3)
        result: list[torch.Tensor] = []
        for metadata in batch.spatial_metadata:
            assert metadata is not None
            if metadata.affine is not None:
                affine = metadata.affine.to(device=tokens.device, dtype=tokens.dtype)
                homogeneous = torch.cat((ijk, torch.ones_like(ijk[:, :1])), dim=-1)
                coords = (homogeneous @ affine.transpose(0, 1))[:, :3]
            elif metadata.spacing_mm is not None:
                # SpatialMetadata spacing follows D,H,W for canonical volume tensors.
                spacing = torch.tensor(metadata.spacing_mm, device=tokens.device, dtype=tokens.dtype)
                coords_dhw = torch.stack(((z + 0.5) * pd, (y + 0.5) * ph, (x + 0.5) * pw), dim=-1).reshape(-1, 3)
                coords = coords_dhw * spacing
                coords = coords[:, (2, 1, 0)]  # expose patient x,y,z order
            else:
                raise ShapeContractError(
                    f"{self._model_id} metadata must carry affine or spacing_mm for MILLIMETERS coordinates"
                )
            result.append(coords)
        return torch.stack(result, dim=0)

    def _feature_maps(self, states: tuple[torch.Tensor, ...], batch_size: int) -> tuple[torch.Tensor, ...]:
        layers = self._feature_map_layers or (len(states) - 1,)
        gd, gh, gw = self._preprocess.patch_grid
        maps: list[torch.Tensor] = []
        for layer in layers:
            if layer >= len(states):
                raise UnsupportedCapabilityError(f"{self._model_id} feature layer {layer} is unavailable")
            patch_tokens = states[layer][:, 1:, :]
            maps.append(patch_tokens.transpose(1, 2).reshape(batch_size, -1, gd, gh, gw).contiguous())
        return tuple(maps)

    def encode(
        self,
        batch: MedicalBatch,
        output_hidden_states: bool = False,
        output_spec: OutputSpec | None = None,
    ) -> EncoderOutput:
        request = output_spec or OutputSpec(pooled=True)
        request.check_supported(self._capabilities)
        self._capabilities.require_modality(batch.modality)
        volume = self._extract_volume(batch)
        volume_spatial = (int(volume.shape[-3]), int(volume.shape[-2]), int(volume.shape[-1]))
        self._validate_spatial_metadata(batch, volume_spatial)
        result = self._forward_backbone(volume)
        spatial = result.tokens[:, 1:, :]
        b, n, _ = spatial.shape
        if n != self._preprocess.num_patches:
            raise ShapeContractError(f"{self._model_id} emitted {n} tokens; expected {self._preprocess.num_patches}")
        mask = torch.ones(b, n, dtype=torch.bool, device=spatial.device)
        if batch.image_mask is not None:
            mask = batch.image_mask.to(device=spatial.device, dtype=torch.bool).reshape(b, 1).expand(b, n).contiguous()
        auxiliary: dict[str, Any] = {
            "patch_grid": self._preprocess.patch_grid,
            "patch_size": self._preprocess.patch_size,
            "flatten_order": "depth,height,width row-major; Conv3d flatten order",
            "orientation": self._preprocess.orientation,
            "sequence_order": self._preprocess.sequence_order,
            "physical_coordinate_transform": (
                "voxel centers transformed by SpatialMetadata.affine; spacing fallback is D,H,W"
            ),
            "unsupported_limitations": (),
        }
        maps = self._feature_maps(result.hidden_states, b) if request.feature_maps else None
        coords = self._coordinates(batch, spatial) if request.token_coordinates else None
        if batch.spatial_metadata:
            auxiliary["spatial_metadata"] = tuple(batch.spatial_metadata)
        if output_hidden_states:
            auxiliary["hidden_states"] = tuple(state[:, 1:, :] for state in result.hidden_states)
            auxiliary["native_outputs_kind"] = "local_monai_3d_hidden_states"
        output = EncoderOutput(
            pooled_embedding=result.pooled if request.pooled else None,
            spatial_tokens=spatial if request.spatial_tokens else None,
            feature_maps=maps if request.feature_maps else None,
            token_mask=mask if request.spatial_tokens else None,
            token_coordinates=coords,
            token_coordinate_system=CoordinateSystem.MILLIMETERS if coords is not None else None,
            native_outputs={"hidden_states": result.hidden_states} if output_hidden_states else None,
            auxiliary=auxiliary,
        )
        output.check_against(request)
        return output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ShapeContractError("forward() smoke path requires eval mode")
        if pixel_values.ndim != 5:
            raise ShapeContractError("native 3D forward expects [B,C,D,H,W]")
        metadata: list[SpatialMetadata | None] = [
            SpatialMetadata(
                original_shape=tuple(int(v) for v in pixel_values.shape[-3:]),
                current_shape=tuple(int(v) for v in pixel_values.shape[-3:]),
                affine=torch.eye(4, dtype=torch.float64),
                spacing_mm=(1.0, 1.0, 1.0),
                orientation="RAS",
            )
            for _ in range(int(pixel_values.shape[0]))
        ]
        batch = MedicalBatch(
            modality=self._capabilities.modalities[0],
            sample_ids=[f"smoke-{i}" for i in range(int(pixel_values.shape[0]))],
            pixel_values=pixel_values,
            spatial_metadata=metadata,
        )
        output = self.encode(batch, output_spec=OutputSpec(pooled=True))
        assert output.pooled_embedding is not None
        return output.pooled_embedding

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        if self._lora_state is not None:
            raise ShapeContractError("LoRA is active; unfreezing the full native 3D backbone is forbidden")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def trainable_backbone_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad)

    def set_deterministic_eval(self) -> None:
        self.eval()

    def attach_head(self, head: nn.Module) -> None:
        if not isinstance(head, nn.Module):
            raise TypeError("head must be an nn.Module")
        self._head = head.to(next(self.parameters()).device)

    def detach_head(self) -> None:
        self._head = None

    @property
    def attached_head(self) -> nn.Module | None:
        return self._head

    def head_logits(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        if self._head is None:
            raise UnsupportedCapabilityError(f"{self._model_id} has no attached head")
        return cast(torch.Tensor, self._head(pooled_embedding))

    def lora_target_patterns(self) -> tuple[str, ...]:
        return self._lora_targets

    def lora_target_specs(self) -> tuple[LoraTargetSpec, ...]:
        return tuple(
            LoraTargetSpec(
                pattern=pattern,
                reason="transformer attention/MLP projection; convolution patch stem stays frozen",
            )
            for pattern in self._lora_targets
        )

    @property
    def lora_active(self) -> bool:
        return self._lora_state is not None

    def inject_lora(
        self, *, rank: int = 8, alpha: int = 16, dropout: float = 0.0, targets: tuple[str, ...] | None = None
    ) -> list[str]:
        requested = tuple(targets or self._lora_targets)
        if any(target not in self._lora_targets for target in requested):
            raise UnsupportedCapabilityError("LoRA targets must be declared by the adapter")
        if self.lora_active:
            raise ShapeContractError("LoRA is already active")
        from medfm.peft import LoRAConfig, inject_visual_lora

        config = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_policy="explicit",
            target_modules=requested,
            adapter_name="default",
            architecture="3d_transformer",
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
        if self._lora_state is None:
            return []
        patterns = tuple(self._lora_state["targets"])
        return sorted(
            name
            for name, module in self.backbone.named_modules()
            if hasattr(module, "lora_A") and any(re.search(pattern, name) for pattern in patterns)
        )

    def _config_dict(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "revision": self._revision,
            "adapter_class": type(self).__name__,
            "preprocess": self._preprocess.to_dict(),
            "hidden_size": self._hidden_size,
            "depth": self._depth,
            "heads": self._heads,
            "feature_map_layers": list(self._feature_map_layers),
            "construction_seed": self._construction_seed,
            "max_full_volume_voxels": self._max_full_volume_voxels,
            "unsupported_xla_ops": list(self._unsupported_xla_ops),
            "custom_cuda_dependencies": list(self._custom_cuda_dependencies),
            "lora_state": self._lora_state,
        }

    def config_hash(self) -> str:
        return config_hash(self._config_dict())

    def save_checkpoint(self, directory: str | Path, *, include_backbone: bool = True) -> Path:
        from safetensors.torch import save_file

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        if include_backbone:
            tensors.update(
                {f"backbone.{k}": v.detach().cpu().contiguous() for k, v in self.backbone.state_dict().items()}
            )
        elif self._lora_state is not None:
            tensors.update(
                {
                    f"backbone.{k}": v.detach().cpu().contiguous()
                    for k, v in self.backbone.state_dict().items()
                    if "lora_" in k
                }
            )
        if not include_backbone and self._lora_state is None and self._head is None:
            raise Native3DCheckpointError("adapter-only export has no LoRA or attached head")
        if self._head is not None:
            tensors.update({f"head.{k}": v.detach().cpu().contiguous() for k, v in self._head.state_dict().items()})
        head_architecture: dict[str, int | str] | None = None
        if self._head is not None and hasattr(self._head, "linear"):
            linear = cast(nn.Linear, self._head.linear)
            head_architecture = {
                "type": "linear",
                "in_features": int(linear.in_features),
                "out_features": int(linear.out_features),
            }
        save_file(tensors, str(out_dir / "tensors.safetensors"))
        manifest = {
            "checkpoint_version": NATIVE_3D_CHECKPOINT_VERSION,
            "kind": "full" if include_backbone else "adapter_only",
            "model_id": self._model_id,
            "revision": self._revision,
            "config": self._config_dict(),
            "config_hash": self.config_hash(),
            "head_architecture": head_architecture,
            "tensor_file": "tensors.safetensors",
            "created_at": datetime.now(UTC).isoformat(),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return out_dir

    @classmethod
    def load_checkpoint(
        cls,
        directory: str | Path,
        *,
        rebuild: Callable[[dict[str, Any]], GenericMONAI3DAdapter],
        device: str | torch.device | None = None,
    ) -> GenericMONAI3DAdapter:
        from safetensors.torch import load_file

        from medfm.models.visual.base import LinearHead

        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("checkpoint_version") != NATIVE_3D_CHECKPOINT_VERSION:
            raise Native3DCheckpointError("unsupported native 3D checkpoint version")
        adapter = rebuild(dict(manifest["config"]))
        if adapter.model_id != manifest["model_id"]:
            raise Native3DCheckpointError("checkpoint model_id does not match rebuilt native 3D adapter")
        if manifest["kind"] == "adapter_only" and manifest["config"].get("lora_state") is not None:
            lora = manifest["config"]["lora_state"]
            adapter.inject_lora(
                rank=int(lora["rank"]),
                alpha=int(lora["alpha"]),
                dropout=float(lora["dropout"]),
                targets=tuple(str(target) for target in lora["targets"]),
            )
        if adapter.config_hash() != manifest["config_hash"]:
            raise Native3DCheckpointError("checkpoint config hash does not match rebuilt native 3D adapter")
        tensors = load_file(str(path / manifest["tensor_file"]))
        backbone = {k.removeprefix("backbone."): v for k, v in tensors.items() if k.startswith("backbone.")}
        if manifest["kind"] == "full":
            adapter.backbone.load_state_dict(backbone, strict=True)
        elif backbone:
            adapter.backbone.load_state_dict(backbone, strict=False)

        head_tensors = {k.removeprefix("head."): v for k, v in tensors.items() if k.startswith("head.")}
        architecture = manifest.get("head_architecture")
        if head_tensors and architecture and architecture.get("type") == "linear":
            head = LinearHead(int(architecture["in_features"]), int(architecture["out_features"]))
            head.load_state_dict(head_tensors, strict=True)
            adapter.attach_head(head)
        elif head_tensors:
            raise Native3DCheckpointError("native 3D checkpoint carries an unsupported head architecture")
        if device is not None:
            adapter.to(device)
        return adapter

    def export_adapter_checkpoint(self, directory: str | Path) -> Path:
        if not re.fullmatch(r"[0-9a-f]{40,64}", self._revision):
            raise Native3DCheckpointError(
                f"canonical native 3D export requires a pinned revision; got {self._revision!r}"
            )
        if self._lora_state is None and self._head is None:
            raise Native3DCheckpointError("nothing trained to export: attach a head or inject LoRA first")
        return self.save_checkpoint(directory, include_backbone=False)

    def tpu_smoke_config(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "batch_size": 2,
            "channels": self._preprocess.channels,
            "volume_shape": list(self._preprocess.spatial_shape),
            "patch_size": list(self._preprocess.patch_size),
            "dtype": "bfloat16",
            "static_batch": True,
            "attention": "sdpa",
            "unsupported_xla_ops": list(self._unsupported_xla_ops),
            "custom_cuda_dependencies": list(self._custom_cuda_dependencies),
            "window_batches_on_host": True,
        }

    def forward_cropped(
        self,
        volume: torch.Tensor,
        metadata: SpatialMetadata,
        *,
        output_spec: OutputSpec | None = None,
    ) -> EncoderOutput:
        pixels = volume.unsqueeze(0) if volume.ndim == 4 else volume
        if pixels.ndim != 5 or pixels.shape[0] != 1:
            raise ShapeContractError("forward_cropped accepts one [C,D,H,W] crop or [1,C,D,H,W]")
        batch = MedicalBatch(
            modality=self._capabilities.modalities[0],
            sample_ids=["crop"],
            pixel_values=pixels,
            spatial_metadata=cast(list[SpatialMetadata | None], [metadata]),
        )
        return self.encode(batch, output_spec=output_spec)

    def forward_full_volume(
        self,
        volume: torch.Tensor,
        metadata: Sequence[SpatialMetadata] | SpatialMetadata | None = None,
        *,
        output_spec: OutputSpec | None = None,
    ) -> EncoderOutput:
        if volume.ndim != 5:
            raise ShapeContractError("full-volume forward expects [B,C,D,H,W]")
        voxels = math.prod(int(v) for v in volume.shape[-3:])
        if voxels > self._max_full_volume_voxels:
            raise ShapeContractError(
                f"full-volume shape {tuple(volume.shape[-3:])} has {voxels:,} voxels, above configured limit "
                f"{self._max_full_volume_voxels:,}; use sliding_window_inference with fixed window batches"
            )
        if tuple(volume.shape[-3:]) != self._preprocess.spatial_shape:
            raise ShapeContractError(
                f"full-volume shape {tuple(volume.shape[-3:])} does not match fixed adapter shape "
                f"{self._preprocess.spatial_shape}; use sliding_window_inference"
            )
        metas: list[SpatialMetadata | None]
        if isinstance(metadata, SpatialMetadata):
            metas = [metadata] * int(volume.shape[0])
        elif metadata is None:
            raise ShapeContractError("full-volume forward requires one SpatialMetadata per sample")
        else:
            metas = cast(list[SpatialMetadata | None], list(metadata))
        if len(metas) != int(volume.shape[0]):
            raise ShapeContractError("full-volume metadata length must match batch size")
        batch = MedicalBatch(
            modality=self._capabilities.modalities[0],
            sample_ids=[f"full-{i}" for i in range(int(volume.shape[0]))],
            pixel_values=volume,
            spatial_metadata=metas,
        )
        return self.encode(batch, output_spec=output_spec)


def _window_starts(size: int, window: int, stride: int) -> list[int]:
    if window > size:
        raise ShapeContractError(f"window {window} exceeds volume dimension {size}")
    starts = list(range(0, max(size - window + 1, 1), stride))
    last = size - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_inference(
    volume: torch.Tensor,
    predictor: Callable[..., torch.Tensor],
    *,
    window_shape: tuple[int, int, int],
    overlap: float = 0.25,
    metadata: Sequence[SpatialMetadata] | None = None,
) -> torch.Tensor:
    """Host-side fixed-window reconstruction with overlap averaging.

    ``predictor`` receives a ``[B,C,d,h,w]`` crop and, when accepted, the
    matching metadata list as a second positional argument. The output must be
    ``[B,K,d,h,w]``; no unbounded window list is retained on device.
    """

    if volume.ndim != 5 or len(window_shape) != 3 or any(v <= 0 for v in window_shape):
        raise ShapeContractError("sliding_window_inference expects [B,C,D,H,W] and a positive 3D window")
    if not 0 <= overlap < 1:
        raise ShapeContractError("overlap must be in [0, 1)")
    b, _, d, h, w = volume.shape
    strides = tuple(max(1, int(window * (1.0 - overlap))) for window in window_shape)
    starts = tuple(
        _window_starts(size, window, stride)
        for size, window, stride in zip((d, h, w), window_shape, strides, strict=True)
    )
    output: torch.Tensor | None = None
    weights = torch.zeros((b, 1, d, h, w), device=volume.device, dtype=torch.float32)
    metas = list(metadata) if metadata is not None else [None] * b
    if len(metas) != b:
        raise ShapeContractError("metadata length must equal volume batch size")
    for zd in starts[0]:
        for yh in starts[1]:
            for xw in starts[2]:
                crop = volume[:, :, zd : zd + window_shape[0], yh : yh + window_shape[1], xw : xw + window_shape[2]]
                crop_metas = metas
                try:
                    prediction = predictor(crop, crop_metas)
                except TypeError:
                    prediction = predictor(crop)
                if prediction.ndim != 5 or tuple(prediction.shape[0:1] + prediction.shape[-3:]) != (b, *window_shape):
                    raise ShapeContractError("sliding-window predictor must return [B,K,*window_shape]")
                if output is None:
                    output = torch.zeros(
                        (b, int(prediction.shape[1]), d, h, w), device=prediction.device, dtype=prediction.dtype
                    )
                output[:, :, zd : zd + window_shape[0], yh : yh + window_shape[1], xw : xw + window_shape[2]] += (
                    prediction
                )
                weights[:, :, zd : zd + window_shape[0], yh : yh + window_shape[1], xw : xw + window_shape[2]] += 1
    assert output is not None
    return output / weights.to(dtype=output.dtype).clamp_min(1)


__all__ = [
    "GenericMONAI3DAdapter",
    "Native3DCheckpointError",
    "Native3DOptionalDependencyError",
    "Native3DPreprocess",
    "sliding_window_inference",
]
