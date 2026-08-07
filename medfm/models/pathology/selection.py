"""Deterministic WSI tile sampling and fixed visual-token selection."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


def _xy(record: Any) -> tuple[int, int]:
    return int(getattr(record, "y", 0)), int(getattr(record, "x", 0))


def _quality(record: Any) -> float:
    quality = getattr(record, "quality", {}) or {}
    tissue = float(getattr(record, "tissue_fraction", 0.0))
    blur = float(quality.get("blur", 1.0))
    artifact = float(quality.get("artifact", 0.0))
    return max(1e-8, tissue * (1.0 + math.log1p(max(0.0, blur))) * max(0.0, 1.0 - artifact))


class TileSampler(Protocol):
    """Select at most a bounded number of tile records."""

    def select(
        self,
        records: Sequence[Any],
        budget: int,
        *,
        embeddings: torch.Tensor | None = None,
        attention: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        seed: int = 0,
    ) -> list[int]: ...

    def sample(self, records: Sequence[Any], max_tiles: int, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class TokenBudget:
    """Pre-compression and post-resampler limits for static-shape execution."""

    precompression: int = 256
    visual_tokens: int = 64

    def __post_init__(self) -> None:
        if not 128 <= self.precompression <= 1024:
            raise ValueError("precompression must be in [128, 1024]")
        if not 32 <= self.visual_tokens <= 128:
            raise ValueError("visual_tokens must be in [32, 128]")


class _SamplerMixin(TileSampler):
    def sample(self, records: Sequence[Any], max_tiles: int, **kwargs: Any) -> list[int]:
        return self.select(records, max_tiles, **kwargs)


class GridTileSampler(_SamplerMixin):
    """Row-major deterministic grid selection."""

    def select(self, records: Sequence[Any], budget: int, **_: Any) -> list[int]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        order = sorted(range(len(records)), key=lambda i: (*_xy(records[i]), str(getattr(records[i], "tile_id", i))))
        return order[:budget]


class RandomTileSampler(_SamplerMixin):
    """Seeded selection without replacement, stable across Python processes."""

    def select(self, records: Sequence[Any], budget: int, *, seed: int = 0, **_: Any) -> list[int]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        order = list(range(len(records)))
        random.Random(int(seed)).shuffle(order)
        return sorted(order[:budget])


class QualityWeightedTileSampler(_SamplerMixin):
    """Deterministic weighted sampling using seeded exponential keys."""

    def select(self, records: Sequence[Any], budget: int, *, seed: int = 0, **_: Any) -> list[int]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        rng = random.Random(int(seed))
        keyed = [(-math.log(max(rng.random(), 1e-12)) / _quality(record), i) for i, record in enumerate(records)]
        return [i for _, i in sorted(keyed)[:budget]]


class DiversityTileSampler(_SamplerMixin):
    """Greedy farthest-point selection over cached tile embeddings."""

    def select(
        self, records: Sequence[Any], budget: int, *, embeddings: torch.Tensor | None = None, **_: Any
    ) -> list[int]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            return GridTileSampler().select(records, budget)
        if budget == 0 or not records:
            return []
        x = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        first = max(range(len(records)), key=lambda i: (_quality(records[i]), -i))
        chosen = [first]
        distances = 1.0 - x @ x[first]
        while len(chosen) < min(budget, len(records)):
            distances[chosen] = -torch.inf
            chosen.append(int(torch.argmax(distances)))
            distances = torch.minimum(distances, 1.0 - x @ x[chosen[-1]])
        return chosen


class TopKAttentionTileSampler(_SamplerMixin):
    """Select highest-attention tiles, with coordinate tie-breaking."""

    def select(
        self, records: Sequence[Any], budget: int, *, attention: torch.Tensor | None = None, **_: Any
    ) -> list[int]:
        if attention is None or attention.numel() != len(records):
            return GridTileSampler().select(records, budget)
        values = attention.detach().float().reshape(-1).tolist()
        order = sorted(range(len(records)), key=lambda i: (-values[i], *_xy(records[i]), i))
        return order[:budget]


class MultiResolutionTileSampler(GridTileSampler):
    """Grid selector that keeps level diversity before filling the budget."""

    def select(self, records: Sequence[Any], budget: int, **kwargs: Any) -> list[int]:
        if budget <= 0:
            return []
        by_level: dict[int, list[int]] = {}
        for index in super().select(records, len(records), **kwargs):
            by_level.setdefault(int(getattr(records[index], "level", 0)), []).append(index)
        levels = sorted(by_level)
        chosen: list[int] = []
        cursor = 0
        while len(chosen) < min(budget, len(records)):
            added = False
            for level in levels:
                if cursor < len(by_level[level]):
                    chosen.append(by_level[level][cursor])
                    added = True
                    if len(chosen) >= budget:
                        break
            if not added:
                break
            cursor += 1
        return chosen


class TextConditionedTileSampler(TopKAttentionTileSampler):
    """Text-conditioned selection via cosine similarity to tile embeddings."""

    def select(
        self,
        records: Sequence[Any],
        budget: int,
        *,
        embeddings: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if embeddings is None or text_embedding is None:
            return super().select(records, budget, **kwargs)
        tile = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        text = torch.nn.functional.normalize(text_embedding.float().reshape(1, -1), dim=-1)
        return super().select(records, budget, attention=(tile @ text.T).reshape(-1))


@dataclass(frozen=True)
class SelectedTokens:
    """Fixed-width token tensor plus mask and source-tile evidence."""

    tokens: torch.Tensor
    mask: torch.Tensor
    indices: tuple[int, ...]
    records: tuple[Any, ...]

    @property
    def visual_tokens(self) -> torch.Tensor:
        return self.tokens

    @property
    def visual_token_mask(self) -> torch.Tensor:
        return self.mask


class WSITokenSelector:
    """Select and pad cached embeddings to a fixed visual-token count."""

    def __init__(self, sampler: TileSampler | None = None, budget: TokenBudget | None = None) -> None:
        self.sampler = sampler or GridTileSampler()
        self.budget = budget or TokenBudget()

    @property
    def visual_tokens(self) -> int:
        return self.budget.visual_tokens

    def select(
        self,
        embeddings: torch.Tensor,
        records: Sequence[Any],
        *,
        attention: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        seed: int = 0,
    ) -> SelectedTokens:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise ValueError(
                f"embeddings must be [N,D] aligned with {len(records)} records; got {tuple(embeddings.shape)}"
            )
        candidate_count = min(len(records), self.budget.precompression)
        indices = self.sampler.select(
            records,
            candidate_count,
            embeddings=embeddings,
            attention=attention,
            text_embedding=text_embedding,
            seed=seed,
        )[: self.budget.visual_tokens]
        tokens = torch.zeros(
            self.budget.visual_tokens, embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device
        )
        mask = torch.zeros(self.budget.visual_tokens, dtype=torch.bool, device=embeddings.device)
        if indices:
            count = len(indices)
            tokens[:count] = embeddings[indices]
            mask[:count] = True
        return SelectedTokens(
            tokens=tokens, mask=mask, indices=tuple(indices), records=tuple(records[i] for i in indices)
        )

    def __call__(self, embeddings: torch.Tensor, records: Sequence[Any], **kwargs: Any) -> SelectedTokens:
        return self.select(embeddings, records, **kwargs)


# Friendly aliases used in recipe configuration files.
GridSelector = GridTileSampler
SeededRandomSelector = RandomTileSampler
QualityWeightedSelector = QualityWeightedTileSampler
DiversitySelector = DiversityTileSampler
TopKAttentionSelector = TopKAttentionTileSampler
RandomTissueSampler = RandomTileSampler

__all__ = [
    "DiversitySelector",
    "DiversityTileSampler",
    "GridSelector",
    "GridTileSampler",
    "MultiResolutionTileSampler",
    "QualityWeightedSelector",
    "QualityWeightedTileSampler",
    "RandomTileSampler",
    "RandomTissueSampler",
    "SeededRandomSelector",
    "SelectedTokens",
    "TextConditionedTileSampler",
    "TileSampler",
    "TokenBudget",
    "TopKAttentionSelector",
    "TopKAttentionTileSampler",
    "WSITokenSelector",
]
