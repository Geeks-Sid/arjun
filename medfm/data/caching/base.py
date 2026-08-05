"""Backend-neutral cache contracts: entries, statistics, and the cache protocol.

Tensors in a :class:`CacheEntry` are always CPU tensors with canonical
(accelerator-neutral) dtypes — devices are never part of the cache contract,
so entries are loadable identically by CPU, CUDA, and XLA consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from medfm.data.caching.keys import CacheKey


@dataclass(frozen=True)
class CacheEntry:
    """A cached artifact: CPU tensors plus JSON-serializable metadata."""

    tensors: dict[str, torch.Tensor]  # CPU only, canonical dtypes
    metadata: dict[str, Any]
    created_at: str  # ISO 8601 timestamp
    size_bytes: int


@dataclass(frozen=True)
class CacheStats:
    """Point-in-time cache statistics.

    ``entries``/``bytes`` describe current on-store contents;
    ``hits``/``misses``/``evictions``/``corrupt_quarantined`` are counters
    accumulated by the cache instance since construction.
    """

    entries: int = 0
    bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    corrupt_quarantined: int = 0


@runtime_checkable
class TensorCache(Protocol):
    """Storage contract for tensor caches (disk today; other backends later).

    ``get`` returns ``None`` both for a genuine miss and for a quarantined
    corrupt entry; corruption is additionally recorded in
    :meth:`stats` (``corrupt_quarantined``) so operators can distinguish the
    two.
    """

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Return the entry for ``key``, or ``None`` on miss or quarantined corruption."""
        ...

    def put(
        self,
        key: CacheKey,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store ``tensors`` (moved to CPU) and JSON-serializable ``metadata`` under ``key``."""
        ...

    def invalidate(self, key: CacheKey) -> bool:
        """Remove the entry for ``key``; return whether one existed."""
        ...

    def clear(self) -> None:
        """Remove every entry visible to this cache instance."""
        ...

    def stats(self) -> CacheStats:
        """Return current contents plus this instance's hit/miss/eviction counters."""
        ...
