"""Versioned, deterministic cache keys for the data layer.

A :class:`CacheKey` is the *complete* invalidation contract for a cached
artifact: any change to an input or a transformation that could change the
cached bytes must change at least one key component, and therefore the
derived :meth:`CacheKey.key_string`.

Which fields apply per :class:`CacheKind`:

- ``PREPROCESSING``: ``source_file_hash``, ``reader_version`` and
  ``preprocessing_hash`` only. No model fields — preprocessing output must
  not depend on any model. Normalization changes and resolution/crop-policy
  changes are folded into ``preprocessing_hash`` by the caller (e.g. via
  :func:`medfm.core.serialization.config_hash` over the preprocessing
  config).
- ``TOKENIZATION``: tokenizer identity goes in ``model_id`` /
  ``model_revision``; tokenizer settings (max length, truncation, special
  tokens) are folded into ``preprocessing_hash``.
- ``VISUAL_EMBEDDING``: all fields apply — ``model_id``, ``model_revision``,
  ``output_layer``, ``dtype`` (canonical dtype name), plus ``extra`` for
  adapter identity or other producer-specific discriminators.

``CACHE_KEY_VERSION`` is mixed into every key string. Bump it on *any*
storage-format or key-semantics change (even if the dataclass fields are
unchanged), and record the bump in the phase handoff.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from medfm.core.errors import SerializationError
from medfm.core.serialization import canonical_json, dtype_from_canonical
from medfm.data.errors import CacheError

#: Bumped on any storage-format or key-semantics change; mixed into every
#: derived key string. Record bumps in the phase handoff.
CACHE_KEY_VERSION = 1

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class CacheKind(StrEnum):
    """The kind of artifact a cache entry holds."""

    PREPROCESSING = "preprocessing"
    VISUAL_EMBEDDING = "visual_embedding"
    TOKENIZATION = "tokenization"


def _require_sha256_hex(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise CacheError(f"CacheKey.{name} must be a 64-character lowercase hex SHA-256 digest, got {value!r}")


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise CacheError(f"CacheKey.{name} must be a non-empty string")


@dataclass(frozen=True)
class CacheKey:
    """Complete, hashable identity of a cached artifact.

    Invalidation contract: :meth:`key_string` is a SHA-256 over the canonical
    JSON of :meth:`to_dict`, so changing *any* component — source content
    (``source_file_hash``), reader behavior (``reader_version``), preprocessing
    config including normalization and resolution/crop policy
    (``preprocessing_hash``), model or adapter identity (``model_id``),
    ``model_revision``, ``output_layer``, ``dtype``, ``extra``, or the key
    version itself — yields a different key string and therefore a cache miss
    against entries stored under the old key. Reconstructing an identical key
    yields the identical key string (a hit).
    """

    kind: CacheKind
    source_file_hash: str  # 64-hex SHA-256 of the source file content
    reader_version: str  # version of the reader that decoded the source
    preprocessing_hash: str  # 64-hex hash of the full preprocessing config
    model_id: str | None = None  # model or tokenizer identity (kind-dependent)
    model_revision: str | None = None  # pinned revision of model_id
    output_layer: str | None = None  # tapped layer for visual embeddings
    dtype: str | None = None  # canonical dtype name (medfm.core.serialization)
    extra: tuple[tuple[str, str], ...] = field(default=())  # sorted (name, value) discriminator pairs

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CacheKind):
            object.__setattr__(self, "kind", CacheKind(self.kind))
        _require_sha256_hex("source_file_hash", self.source_file_hash)
        _require_non_empty("reader_version", self.reader_version)
        _require_sha256_hex("preprocessing_hash", self.preprocessing_hash)
        for name in ("model_id", "model_revision", "output_layer"):
            value = getattr(self, name)
            if value is not None:
                _require_non_empty(name, value)
        if self.dtype is not None:
            # Validates the name is canonical (accelerator-neutral) or raises.
            try:
                dtype_from_canonical(self.dtype)
            except SerializationError as exc:
                raise CacheError(f"CacheKey.dtype must be a canonical dtype name, got {self.dtype!r}: {exc}") from exc
        extra = tuple(self.extra)
        for pair in extra:
            is_pair = isinstance(pair, tuple) and len(pair) == 2
            if not is_pair or not all(isinstance(part, str) and part for part in pair):
                raise CacheError(
                    f"CacheKey.extra entries must be (name, value) pairs of non-empty strings, got {pair!r}"
                )
        names = [name for name, _ in extra]
        if len(set(names)) != len(names):
            raise CacheError(f"CacheKey.extra names must be unique, got {names!r}")
        object.__setattr__(self, "extra", tuple(sorted(extra)))

    def to_dict(self) -> dict[str, Any]:
        """Deterministic mapping of every key component plus the key version."""
        return {
            "cache_key_version": CACHE_KEY_VERSION,
            "kind": str(self.kind),
            "source_file_hash": self.source_file_hash,
            "reader_version": self.reader_version,
            "preprocessing_hash": self.preprocessing_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "output_layer": self.output_layer,
            "dtype": self.dtype,
            "extra": [[name, value] for name, value in self.extra],
        }

    def key_string(self) -> str:
        """SHA-256 hex over ``canonical_json(to_dict())``; the storage-level key."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()
