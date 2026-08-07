"""Pathology tile encoders and the visual-token bridge used by Phase 09."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.core.enums import Modality
from medfm.core.serialization import config_hash


class PathologyTileEncoder:
    """Structural tile-encoder contract (CPU decode, batched accelerator inference)."""

    model_id: str
    revision: str
    preprocess_hash: str
    embedding_dim: int
    dtype: torch.dtype

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.encode_tiles(tiles)


class TorchPathologyTileEncoder(nn.Module, PathologyTileEncoder):
    """Wrap a frozen PyTorch image encoder without coupling WSI I/O to CUDA."""

    def __init__(
        self,
        module: nn.Module,
        *,
        model_id: str,
        revision: str,
        preprocess_hash: str,
        embedding_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.module = module
        self.model_id = model_id
        self.revision = revision
        self.preprocess_hash = preprocess_hash
        self.embedding_dim = int(embedding_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        if tiles.ndim != 4:
            raise ValueError(f"tiles must have shape [T,C,H,W]; got {tuple(tiles.shape)}")
        with torch.inference_mode():
            values = self.module(tiles.to(self.device, dtype=self.dtype))
        if isinstance(values, list | tuple):
            values = values[0]
        if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != self.embedding_dim:
            raise ValueError(
                f"tile encoder output must be [T,{self.embedding_dim}]; got {getattr(values, 'shape', None)}"
            )
        return values.detach().clone()

    forward = encode_tiles


class TinyPathologyTileEncoder(TorchPathologyTileEncoder):
    """Small deterministic local encoder for tests, smoke, and CPU development."""

    def __init__(self, *, embedding_dim: int = 32, image_size: int = 32, seed: int = 0) -> None:
        torch.manual_seed(seed)
        module = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, embedding_dim),
        )
        super().__init__(
            module,
            model_id="pathology-tiny",
            revision="local-tiny",
            preprocess_hash=config_hash({"image_size": image_size, "channels": 3, "seed": seed}),
            embedding_dim=embedding_dim,
        )
        self.image_size = int(image_size)

    @classmethod
    def build_tiny(cls, **kwargs: Any) -> TinyPathologyTileEncoder:
        return cls(**kwargs)


class HOptimusTileEncoder(PathologyTileEncoder):
    """Phase 06 H-Optimus adapter exposed through the Phase 08 tile contract."""

    def __init__(self, adapter: Any, *, embedding_dim: int | None = None) -> None:
        self.adapter = adapter
        self.model_id = str(adapter.model_id)
        self.revision = str(adapter.revision)
        backbone = getattr(adapter, "backbone", None)
        inferred_dim = getattr(backbone, "num_features", None)
        if inferred_dim is None:
            raise ValueError("H-Optimus adapter does not expose an embedding dimension")
        self.embedding_dim = int(embedding_dim or inferred_dim)
        self.dtype = adapter.compute_dtype()
        self.preprocess_hash = adapter.preprocess.registry_spec(self.model_id).spec_hash()

    @classmethod
    def build_tiny(cls, **kwargs: Any) -> HOptimusTileEncoder:
        from medfm.models.visual.hoptimus0 import HOptimus0Adapter

        return cls(HOptimus0Adapter.build_tiny(**kwargs))

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        from medfm.core.encoder import OutputSpec

        batch = MedicalBatch(
            modality=Modality.PATHOLOGY_TILE,
            sample_ids=[f"tile-{i}" for i in range(tiles.shape[0])],
            pixel_values=tiles,
        )
        result = self.adapter.extract(batch, output_spec=OutputSpec(pooled=True))
        if result.pooled_embedding is None:
            raise RuntimeError("H-Optimus adapter did not return pooled tile embeddings")
        return cast(torch.Tensor, result.pooled_embedding.detach())


class GigaPathTileEncoder(TinyPathologyTileEncoder):
    """GigaPath tile boundary with an offline fallback, never a fake download."""

    model_id = "gigapath-tile"

    def __init__(
        self, module: nn.Module | None = None, *, embedding_dim: int = 32, revision: str = "local-fallback"
    ) -> None:
        if module is None:
            super().__init__(embedding_dim=embedding_dim)
        else:
            TorchPathologyTileEncoder.__init__(
                self,
                module,
                model_id="gigapath-tile",
                revision=revision,
                preprocess_hash=config_hash({"model_id": "gigapath-tile", "revision": revision}),
                embedding_dim=embedding_dim,
            )


@dataclass(frozen=True)
class FixedTileBatch:
    """Static tile bucket with an explicit real-tile mask."""

    tiles: torch.Tensor
    mask: torch.Tensor


def make_fixed_tile_batch(tiles: torch.Tensor, max_tiles: int) -> FixedTileBatch:
    if tiles.ndim != 4:
        raise ValueError(f"tiles must be [T,C,H,W]; got {tuple(tiles.shape)}")
    if max_tiles <= 0 or tiles.shape[0] > max_tiles:
        raise ValueError(f"max_tiles must be positive and >= tile count; got {max_tiles} for {tiles.shape[0]}")
    padded = torch.zeros((max_tiles, *tiles.shape[1:]), dtype=tiles.dtype, device=tiles.device)
    padded[: tiles.shape[0]] = tiles
    mask = torch.zeros(max_tiles, dtype=torch.bool, device=tiles.device)
    mask[: tiles.shape[0]] = True
    return FixedTileBatch(padded, mask)


@dataclass(frozen=True)
class PathologyVLMOutput:
    visual_tokens: torch.Tensor
    visual_token_mask: torch.Tensor
    pooled_embedding: torch.Tensor
    text_alignment: torch.Tensor | None = None

    @property
    def token_mask(self) -> torch.Tensor:
        return self.visual_token_mask


class PathologyVLMAdapter(nn.Module):
    """Fixed-token visual bridge for pathology report/generation models."""

    def __init__(self, embedding_dim: int, *, visual_dim: int = 256, max_tokens: int = 64) -> None:
        super().__init__()
        if not 32 <= max_tokens <= 128:
            raise ValueError("max_tokens must be in [32, 128]")
        self.embedding_dim = int(embedding_dim)
        self.visual_dim = int(visual_dim)
        self.max_tokens = int(max_tokens)
        self.projector = nn.Linear(self.embedding_dim, self.visual_dim)
        self.norm = nn.LayerNorm(self.visual_dim)

    def encode(
        self,
        visual_tokens: torch.Tensor,
        visual_token_mask: torch.Tensor | None = None,
        *,
        text_embedding: torch.Tensor | None = None,
    ) -> PathologyVLMOutput:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.embedding_dim:
            raise ValueError(f"visual_tokens must be [B,N,{self.embedding_dim}]; got {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[1] != self.max_tokens:
            raise ValueError(f"visual_tokens must use fixed N={self.max_tokens}; got {visual_tokens.shape[1]}")
        mask = (
            torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
            if visual_token_mask is None
            else visual_token_mask.bool()
        )
        if mask.shape != visual_tokens.shape[:2]:
            raise ValueError("visual_token_mask must align with visual_tokens")
        projected = self.norm(self.projector(visual_tokens))
        weights = mask.unsqueeze(-1).to(projected.dtype)
        pooled = (projected * weights).sum(1) / weights.sum(1).clamp_min(1)
        alignment = None
        if text_embedding is not None:
            alignment = torch.nn.functional.cosine_similarity(pooled, text_embedding.to(pooled), dim=-1)
        return PathologyVLMOutput(projected, mask, pooled, alignment)

    forward = encode

    def to_medical_batch(self, output: PathologyVLMOutput, sample_ids: list[str]) -> MedicalBatch:
        if len(sample_ids) != output.visual_tokens.shape[0]:
            raise ValueError("sample_ids must match visual token batch size")
        return MedicalBatch(
            modality=Modality.PATHOLOGY_WSI,
            sample_ids=sample_ids,
            task_targets={"visual_tokens": output.visual_tokens, "visual_token_mask": output.visual_token_mask},
            bucket=BucketId(BucketKind.VISUAL_TOKENS, (self.max_tokens,)),
        )


class TITANAdapter(PathologyVLMAdapter):
    """TITAN image-text alignment boundary over fixed slide tokens."""


class GigaPathFlashAdapter(PathologyVLMAdapter):
    """GigaPath-Flash slide-level token boundary; weights remain registry-gated."""


__all__ = [
    "FixedTileBatch",
    "GigaPathFlashAdapter",
    "GigaPathTileEncoder",
    "HOptimusTileEncoder",
    "PathologyTileEncoder",
    "PathologyVLMAdapter",
    "PathologyVLMOutput",
    "TITANAdapter",
    "TinyPathologyTileEncoder",
    "TorchPathologyTileEncoder",
    "make_fixed_tile_batch",
]
