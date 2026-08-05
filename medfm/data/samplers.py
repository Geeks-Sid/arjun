"""Group-aware distributed sampling with no leakage across ranks.

Contract (Phase 03, feeding Phase 12):

- **Group-intact shards**: all samples of a group (patient, study, slide, or
  ``group_id_hash`` override) go to ONE rank — studies/tiles/cases never
  straddle ranks, matching the split policy of ADR 0004 at the sampler level.
- **Exact coverage**: after removing padding, the per-rank shards are disjoint
  and their union is exactly the requested dataset (verified by tests).
- **Determinism**: shard contents are a pure function of
  ``(seed, epoch, group keys)``; :func:`worker_seed` derives per-worker seeds
  the same way, so every epoch is reproducible per rank and worker.
- **Padded final batches**: ranks are padded with :data:`PADDING_INDEX`
  sentinels to equal shard lengths so collective steps stay synchronized;
  metrics/dataset code must drop sentinels (they are never real samples).
- **Corrupt-sample resolution before collectives**:
  :func:`resolve_samples_before_collective` filters a shard rank-locally and
  deterministically BEFORE any cross-rank step, so one rank's corrupt sample
  can't hang the others.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

from medfm.data.errors import DataError

#: Sentinel marking padded positions; never a valid row index.
PADDING_INDEX = -1


def _order_key(seed: int, epoch: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).hexdigest()


def worker_seed(seed: int, epoch: int, rank: int, worker_id: int) -> int:
    """Deterministic per-worker seed (stable across hosts and restarts)."""
    digest = hashlib.sha256(f"worker:{seed}:{epoch}:{rank}:{worker_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def worker_init_fn(seed: int, epoch_getter: Callable[[], int], rank: int) -> Callable[[int], None]:
    """Build a DataLoader ``worker_init_fn`` seeding numpy/python/random per worker."""

    def _init(worker_id: int) -> None:
        import random

        import numpy as np

        value = worker_seed(seed, epoch_getter(), rank, worker_id)
        random.seed(value)
        np.random.seed(value % (2**32))

    return _init


@dataclass(frozen=True)
class SamplerShard:
    """One rank's sample indices plus its padding mask."""

    rank: int
    indices: tuple[int, ...]  # row indices; PADDING_INDEX entries are padding
    is_padding: tuple[bool, ...]

    @property
    def real_indices(self) -> tuple[int, ...]:
        return tuple(i for i, pad in zip(self.indices, self.is_padding, strict=True) if not pad)


class GroupAwareDistributedSampler:
    """Distributed sampler keeping patient/study/slide groups intact per rank.

    Parameters
    ----------
    df:
        Manifest frame (or any frame with a grouping column and sample_id).
    num_ranks / rank:
        World size and this rank's id.
    seed:
        Base seed; shards are deterministic in ``(seed, epoch)``.
    split:
        Optional split filter (e.g. ``"TRAIN"``); rows with other/null split
        values are excluded.
    group_column:
        Grouping column; ``group_id_hash`` wins per row when present (slide/
        case override), otherwise the column value.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        num_ranks: int,
        rank: int,
        seed: int,
        split: str | None = None,
        group_column: str = "patient_id_hash",
    ) -> None:
        if num_ranks < 1:
            raise DataError(f"num_ranks must be >= 1, got {num_ranks}")
        if not 0 <= rank < num_ranks:
            raise DataError(f"rank {rank} out of range for num_ranks={num_ranks}")
        if group_column not in df.columns:
            raise DataError(f"grouping column {group_column!r} not in manifest; cannot build group-aware shards")
        if "sample_id" not in df.columns:
            raise DataError("manifest needs a sample_id column for deterministic within-group ordering")
        self._num_ranks = num_ranks
        self._rank = rank
        self._seed = seed
        self._epoch = 0

        frame = df
        if split is not None:
            if "split" not in df.columns:
                raise DataError("split filter requested but manifest has no split column")
            frame = df[df["split"] == split]

        groups: dict[str, list[int]] = {}
        for position in frame.index:
            row = frame.loc[position]
            override = row.get("group_id_hash") if "group_id_hash" in frame.columns else None
            key = str(override) if override is not None and pd.notna(override) else str(row[group_column])
            groups.setdefault(key, []).append(int(position))
        for members in groups.values():
            members.sort(key=lambda i: str(frame.loc[i, "sample_id"]))
        self._groups = groups
        self._sample_count = sum(len(members) for members in groups.values())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise DataError(f"epoch must be >= 0, got {epoch}")
        self._epoch = epoch

    def _shards(self) -> list[list[int]]:
        """Greedy group assignment: shuffled group order, smallest shard wins."""
        ordered = sorted(self._groups, key=lambda key: _order_key(self._seed, self._epoch, key))
        shard_sizes = [0] * self._num_ranks
        shards: list[list[int]] = [[] for _ in range(self._num_ranks)]
        for key in ordered:
            target = min(range(self._num_ranks), key=lambda r: (shard_sizes[r], r))
            shards[target].extend(self._groups[key])
            shard_sizes[target] += len(self._groups[key])
        return shards

    def shard_for(self, rank: int) -> SamplerShard:
        """Return ``rank``'s shard with padding to the longest shard length."""
        if not 0 <= rank < self._num_ranks:
            raise DataError(f"rank {rank} out of range for num_ranks={self._num_ranks}")
        shards = self._shards()
        target = max(len(shard) for shard in shards)
        indices = list(shards[rank])
        padding = target - len(indices)
        is_padding = [False] * len(indices) + [True] * padding
        indices = indices + [PADDING_INDEX] * padding
        return SamplerShard(rank=rank, indices=tuple(indices), is_padding=tuple(is_padding))

    @property
    def total_samples(self) -> int:
        """Real (non-padding) sample count covered by all ranks together."""
        return self._sample_count

    def __iter__(self) -> Iterator[int]:
        return iter(self.shard_for(self._rank).indices)

    def __len__(self) -> int:
        return len(self.shard_for(self._rank).indices)


@dataclass(frozen=True)
class ResolvedSamples:
    """Outcome of pre-collective corrupt-sample resolution for one rank."""

    valid_indices: tuple[int, ...]
    quarantined: tuple[tuple[int, str], ...]  # (index, reason) — hashes/reasons only, never payloads


def resolve_samples_before_collective(
    indices: list[int],
    check: Callable[[int], str | None],
) -> ResolvedSamples:
    """Resolve corrupt samples rank-locally BEFORE any distributed collective.

    ``check(index)`` returns ``None`` for a healthy sample or a reason string
    for a corrupt one (e.g. a quarantined reader failure). Every rank runs
    this on its own shard, so all ranks reach subsequent collectives with
    their (possibly shorter) valid-index lists already finalized — no rank
    hangs waiting on another rank's corrupt sample. Padding sentinels pass
    through untouched (they are resolved collectively as padding, not samples).
    """
    valid: list[int] = []
    quarantined: list[tuple[int, str]] = []
    for index in indices:
        if index == PADDING_INDEX:
            valid.append(index)
            continue
        reason = check(index)
        if reason is None:
            valid.append(index)
        else:
            quarantined.append((index, reason))
    return ResolvedSamples(valid_indices=tuple(valid), quarantined=tuple(quarantined))


def combine_shards_for_metrics(shards: list[SamplerShard]) -> tuple[list[int], int]:
    """Union of all ranks' real indices (duplicates impossible by construction).

    Returns ``(sorted real indices, padding entries removed)`` — the exact
    sample set evaluation metrics must cover after dropping padding.
    """
    seen: set[int] = set()
    padding_count = 0
    for shard in shards:
        for index, pad in zip(shard.indices, shard.is_padding, strict=True):
            if pad:
                padding_count += 1
                continue
            if index in seen:
                raise DataError(f"sampler contract violated: index {index} appears on more than one rank")
            seen.add(index)
    return sorted(seen), padding_count
