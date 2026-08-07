"""Deterministic WSI work assignment for GPU/TPU ranks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def shard_ids(ids: Sequence[str], *, rank: int, world_size: int) -> list[str]:
    """Assign each ID to exactly one rank using a stable hash, preserving sorted order."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}")
    return [value for value in sorted({str(item) for item in ids}) if _stable_hash(value) % world_size == rank]


@dataclass(frozen=True)
class DeterministicSlideSharder:
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    def slides(self, slide_ids: Sequence[str]) -> list[str]:
        return shard_ids(slide_ids, rank=self.rank, world_size=self.world_size)

    def chunks(self, chunk_ids: Sequence[str]) -> list[str]:
        return shard_ids(chunk_ids, rank=self.rank, world_size=self.world_size)


__all__ = ["DeterministicSlideSharder", "shard_ids"]
