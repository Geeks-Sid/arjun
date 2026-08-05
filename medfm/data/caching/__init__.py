"""Accelerator-neutral tensor caching: keys, contracts, and the disk backend.

See :mod:`medfm.data.caching.keys` for the invalidation contract and
:mod:`medfm.data.caching.disk` for the atomic, rank-safe on-disk store.
"""

from medfm.data.caching.base import CacheEntry, CacheStats, TensorCache
from medfm.data.caching.disk import DiskTensorCache
from medfm.data.caching.keys import CACHE_KEY_VERSION, CacheKey, CacheKind
from medfm.data.caching.typed import PreprocessingCache, TokenizationCache, VisualEmbeddingCache

__all__ = [
    "CACHE_KEY_VERSION",
    "CacheEntry",
    "CacheKey",
    "CacheKind",
    "CacheStats",
    "DiskTensorCache",
    "PreprocessingCache",
    "TensorCache",
    "TokenizationCache",
    "VisualEmbeddingCache",
]
