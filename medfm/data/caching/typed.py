"""The three Phase 03 cache types as thin, kind-safe wrappers.

- :class:`PreprocessingCache` — decoded + canonicalized/preprocessed tensors.
- :class:`VisualEmbeddingCache` — encoder outputs (model/adapter/layer/dtype aware).
- :class:`TokenizationCache` — tokenized restricted text (tokenizer identity in
  the model fields, tokenizer settings folded into ``preprocessing_hash``).

Each wrapper only constructs keys with the fields its kind is allowed (see
:mod:`medfm.data.caching.keys`) and delegates storage to any
:class:`TensorCache` (the disk backend by default). The invalidation contract
is the key's: changing normalization/resolution/crop policy, the model, the
adapter (via ``extra``), the tapped layer, or the dtype yields a new key and
therefore a miss — exactly the Phase 03 requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from medfm.data.caching.base import CacheEntry, CacheStats, TensorCache
from medfm.data.caching.disk import DiskTensorCache
from medfm.data.caching.keys import CacheKey, CacheKind


class _TypedCache:
    """Shared plumbing: delegate storage, expose kind-specific key builders."""

    kind: CacheKind

    def __init__(self, store: TensorCache) -> None:
        self._store = store

    @classmethod
    def on_disk(
        cls,
        root: str | Path,
        *,
        rank: int = 0,
        coordinator_only: bool = False,
        max_bytes: int | None = None,
    ) -> _TypedCache:
        return cls(DiskTensorCache(root, rank=rank, coordinator_only=coordinator_only, max_bytes=max_bytes))

    def get(self, key: CacheKey) -> CacheEntry | None:
        if key.kind is not self.kind:
            raise ValueError(f"{type(self).__name__} cannot read keys of kind {key.kind}; expected {self.kind}")
        return self._store.get(key)

    def put(
        self,
        key: CacheKey,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if key.kind is not self.kind:
            raise ValueError(f"{type(self).__name__} cannot write keys of kind {key.kind}; expected {self.kind}")
        self._store.put(key, tensors, metadata)

    def invalidate(self, key: CacheKey) -> bool:
        return self._store.invalidate(key)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> CacheStats:
        return self._store.stats()


class PreprocessingCache(_TypedCache):
    """Preprocessing outputs: keyed by source hash, reader version, preprocessing hash.

    Normalization and resolution/crop-policy changes are folded into
    ``preprocessing_hash`` by the caller (e.g. ``config_hash`` of the
    preprocessing config) — any such change invalidates exactly.
    """

    kind = CacheKind.PREPROCESSING

    @staticmethod
    def key(*, source_file_hash: str, reader_version: str, preprocessing_hash: str) -> CacheKey:
        return CacheKey(
            kind=CacheKind.PREPROCESSING,
            source_file_hash=source_file_hash,
            reader_version=reader_version,
            preprocessing_hash=preprocessing_hash,
        )


class VisualEmbeddingCache(_TypedCache):
    """Visual encoder outputs: model/revision/layer/dtype are all key components.

    Adapter identity (or any other producer discriminator) belongs in
    ``extra`` so swapping adapters invalidates the cache without touching
    the model fields.
    """

    kind = CacheKind.VISUAL_EMBEDDING

    @staticmethod
    def key(
        *,
        source_file_hash: str,
        reader_version: str,
        preprocessing_hash: str,
        model_id: str,
        model_revision: str,
        output_layer: str,
        dtype: str,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> CacheKey:
        return CacheKey(
            kind=CacheKind.VISUAL_EMBEDDING,
            source_file_hash=source_file_hash,
            reader_version=reader_version,
            preprocessing_hash=preprocessing_hash,
            model_id=model_id,
            model_revision=model_revision,
            output_layer=output_layer,
            dtype=dtype,
            extra=extra,
        )


class TokenizationCache(_TypedCache):
    """Tokenized text: tokenizer identity in model fields, settings in the hash.

    Restricted report text never appears in keys or metadata — only its
    content hash and tokenizer identity do (governance section 3).
    """

    kind = CacheKind.TOKENIZATION

    @staticmethod
    def key(
        *,
        source_file_hash: str,
        reader_version: str,
        tokenization_hash: str,
        tokenizer_id: str,
        tokenizer_revision: str,
    ) -> CacheKey:
        return CacheKey(
            kind=CacheKind.TOKENIZATION,
            source_file_hash=source_file_hash,
            reader_version=reader_version,
            preprocessing_hash=tokenization_hash,
            model_id=tokenizer_id,
            model_revision=tokenizer_revision,
        )
