"""Atomic, corruption-detecting, rank-safe on-disk tensor cache.

Layout on disk::

    <root>/<kind>/<partition>/<key_string>/payload.safetensors
    <root>/<kind>/<partition>/<key_string>/meta.json

``<partition>`` is empty for the shared layout (rank 0, or any rank when
``coordinator_only=True``) and ``rank-<n>`` otherwise.

Accelerator-neutrality: payloads are stored via safetensors after
:func:`medfm.core.serialization.materialize_cpu`, using canonical dtype names
only; device locations are never serialized. Loads always land on CPU, so an
entry written by a CUDA producer is loadable identically by CPU, CUDA, and
XLA consumers (they move tensors to their device themselves).

Atomicity: entries are assembled in a unique sibling tmp directory and
published with a single directory rename, so readers only ever observe a
complete entry or no entry. A present entry directory with a missing
``meta.json`` or a payload hash mismatch is treated as corrupt: it is
quarantined (deleted), counted in stats, and reported as a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save_file as st_save_file

from medfm.core.errors import SerializationError
from medfm.core.serialization import TensorMeta, canonical_dtype_name, canonical_json, materialize_cpu
from medfm.data.caching.base import CacheEntry, CacheStats, TensorCache
from medfm.data.caching.keys import CACHE_KEY_VERSION, CacheKey
from medfm.data.errors import CacheError, CorruptCacheEntryError

_PAYLOAD_NAME = "payload.safetensors"
_META_NAME = "meta.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskTensorCache(TensorCache):
    """On-disk :class:`TensorCache` with atomic writes, LRU eviction, and rank partitioning.

    Rank safety:

    - ``coordinator_only=True``: only rank 0 writes; ``put`` from any other
      rank is a documented no-op. All ranks read the shared (unpartitioned)
      layout, so non-coordinator ranks see what the coordinator wrote.
    - ``coordinator_only=False``: rank 0 reads/writes the shared layout;
      every other rank reads/writes its own ``rank-<n>`` partition, which is
      written atomically and never shares entry directories with other ranks.

    Eviction: when ``max_bytes`` is set, entries are evicted LRU by
    ``last_access`` after each ``put`` that pushes the cache over budget. The
    just-written entry is never evicted; if it alone exceeds ``max_bytes``,
    :class:`CacheError` is raised instead.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        rank: int = 0,
        coordinator_only: bool = False,
        max_bytes: int | None = None,
    ) -> None:
        if rank < 0:
            raise CacheError(f"rank must be >= 0, got {rank}")
        if max_bytes is not None and max_bytes <= 0:
            raise CacheError(f"max_bytes must be positive, got {max_bytes}")
        self._root = Path(root)
        self._rank = rank
        self._coordinator_only = coordinator_only
        self._max_bytes = max_bytes
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._corrupt_quarantined = 0

    # -- layout helpers ------------------------------------------------------

    @property
    def _partition(self) -> str:
        """Partition directory name for this instance (empty = shared layout)."""
        if self._coordinator_only or self._rank == 0:
            return ""
        return f"rank-{self._rank}"

    def _entry_dir(self, key: CacheKey) -> Path:
        entry_dir = self._root / str(key.kind)
        if self._partition:
            entry_dir = entry_dir / self._partition
        return entry_dir / key.key_string()

    def _visible_entry_dirs(self) -> list[Path]:
        """All entry directories in the kinds/partition this instance operates on.

        The shared layout (partition ``""``) and each ``rank-<n>`` partition are
        disjoint scopes: an instance only ever enumerates (for stats, eviction,
        and ``clear``) the entry directories inside its OWN partition, so one
        rank's maintenance never touches another rank's entries.
        """
        entries: list[Path] = []
        if not self._root.is_dir():
            return entries
        for kind_dir in sorted(self._root.iterdir()):
            if not kind_dir.is_dir():
                continue
            if self._partition:
                scope = kind_dir / self._partition
                if scope.is_dir():
                    entries.extend(entry for entry in sorted(scope.iterdir()) if entry.is_dir())
            else:
                # Shared layout: entry dirs are direct children, excluding rank partitions.
                for child in sorted(kind_dir.iterdir()):
                    if child.is_dir() and not child.name.startswith("rank-"):
                        entries.append(child)
        return entries

    # -- TensorCache interface -----------------------------------------------

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Return the entry for ``key``; miss or quarantined corruption both give ``None``."""
        entry_dir = self._entry_dir(key)
        if not entry_dir.is_dir():
            with self._lock:
                self._misses += 1
            return None
        try:
            entry = self._read_entry(entry_dir)
        except CorruptCacheEntryError:
            self._quarantine(entry_dir)
            with self._lock:
                self._misses += 1
            return None
        self._touch(entry_dir)
        with self._lock:
            self._hits += 1
        return entry

    def put(
        self,
        key: CacheKey,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store CPU-materialized ``tensors`` and JSON ``metadata`` under ``key``, atomically.

        No-op from non-zero ranks when ``coordinator_only=True``. Raises
        :class:`CacheError` for non-canonical dtypes, non-JSON-serializable
        metadata, or an entry that alone exceeds ``max_bytes``.
        """
        if self._coordinator_only and self._rank != 0:
            return  # documented no-op: only the coordinator rank writes
        metadata = {} if metadata is None else dict(metadata)
        if not tensors:
            raise CacheError("put requires at least one tensor")
        for name, tensor in tensors.items():
            if not isinstance(name, str) or not name:
                raise CacheError(f"tensor names must be non-empty strings, got {name!r}")
            if not isinstance(tensor, torch.Tensor):
                raise CacheError(f"cache tensor {name!r} must be a torch.Tensor, got {type(tensor).__name__}")
            try:
                canonical_dtype_name(tensor.dtype)
            except SerializationError as exc:
                raise CacheError(
                    f"cache tensor {name!r} has non-canonical dtype {tensor.dtype}; "
                    "cast to a canonical (accelerator-neutral) dtype before caching"
                ) from exc
        try:
            canonical_json(metadata)  # fail early with an actionable error; re-serialized inside meta below
        except (TypeError, ValueError) as exc:
            raise CacheError(f"cache metadata must be JSON-serializable: {exc}") from exc

        cpu_tensors = {name: materialize_cpu(tensor).contiguous() for name, tensor in tensors.items()}
        tensor_metas = [
            {"name": name, **TensorMeta.of(tensor).to_dict()} for name, tensor in sorted(cpu_tensors.items())
        ]

        entry_dir = self._entry_dir(key)
        parent = entry_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-{os.getpid()}-{threading.get_ident()}-", dir=parent))
        try:
            payload_path = tmp_dir / _PAYLOAD_NAME
            st_save_file(cpu_tensors, str(payload_path))
            payload_sha256 = _sha256_file(payload_path)
            size_bytes = payload_path.stat().st_size
            if self._max_bytes is not None and size_bytes > self._max_bytes:
                raise CacheError(
                    f"cache entry for key {key.key_string()} is {size_bytes} bytes, which alone exceeds "
                    f"max_bytes={self._max_bytes}; raise max_bytes, shard the entry, or bypass the cache "
                    "for this artifact"
                )
            now = _utc_now_iso()
            meta = {
                "cache_key_version": CACHE_KEY_VERSION,
                "key": key.to_dict(),
                "tensors": tensor_metas,
                "payload_sha256": payload_sha256,
                "created_at": now,
                "last_access": now,
                "size_bytes": size_bytes,
                "metadata": metadata,
            }
            (tmp_dir / _META_NAME).write_text(canonical_json(meta) + "\n", encoding="utf-8")
            # Publish atomically: the only states readers can observe are a
            # complete entry directory or none at all.
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
            os.replace(tmp_dir, entry_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        self._evict_if_over_budget(protect=entry_dir)

    def invalidate(self, key: CacheKey) -> bool:
        """Remove the entry for ``key``; return whether one existed."""
        entry_dir = self._entry_dir(key)
        if not entry_dir.is_dir():
            return False
        shutil.rmtree(entry_dir)
        return True

    def clear(self) -> None:
        """Remove every entry visible to this cache instance."""
        for entry_dir in self._visible_entry_dirs():
            shutil.rmtree(entry_dir, ignore_errors=True)

    def stats(self) -> CacheStats:
        """Current on-disk contents plus this instance's counters."""
        entries = 0
        total_bytes = 0
        for entry_dir in self._visible_entry_dirs():
            meta = self._read_meta(entry_dir)
            if meta is None:
                continue
            entries += 1
            total_bytes += int(meta.get("size_bytes", 0))
        with self._lock:
            return CacheStats(
                entries=entries,
                bytes=total_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                corrupt_quarantined=self._corrupt_quarantined,
            )

    # -- integrity -----------------------------------------------------------

    def verify_entry(self, key: CacheKey) -> None:
        """Raise :class:`CorruptCacheEntryError` unless the entry for ``key`` is intact.

        Read-only: unlike :meth:`get`, this never quarantines; callers decide
        what to do with a corrupt entry.
        """
        entry_dir = self._entry_dir(key)
        if not entry_dir.is_dir():
            raise CorruptCacheEntryError(f"no cache entry for key {key.key_string()} at {entry_dir}")
        self._read_entry(entry_dir)

    def _read_meta(self, entry_dir: Path) -> dict[str, Any] | None:
        meta_path = entry_dir / _META_NAME
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        return cast(dict[str, Any], meta)

    def _read_entry(self, entry_dir: Path) -> CacheEntry:
        meta = self._read_meta(entry_dir)
        if meta is None:
            raise CorruptCacheEntryError(f"cache entry at {entry_dir} has a missing or unreadable {_META_NAME}")
        payload_path = entry_dir / _PAYLOAD_NAME
        if not payload_path.is_file():
            raise CorruptCacheEntryError(f"cache entry at {entry_dir} is missing {_PAYLOAD_NAME}")
        actual_sha256 = _sha256_file(payload_path)
        expected_sha256 = meta.get("payload_sha256")
        if actual_sha256 != expected_sha256:
            raise CorruptCacheEntryError(
                f"cache entry at {entry_dir} failed integrity check: payload sha256 {actual_sha256} "
                f"!= recorded {expected_sha256}; the entry is partial or corrupt"
            )
        try:
            tensors = st_load(payload_path.read_bytes())
        except Exception as exc:
            raise CorruptCacheEntryError(f"cache entry at {entry_dir} has an undecodable payload: {exc}") from exc
        expected_tensors = {item["name"]: TensorMeta.from_dict(item) for item in meta.get("tensors", [])}
        if set(tensors) != set(expected_tensors):
            raise CorruptCacheEntryError(
                f"cache entry at {entry_dir} tensor set {sorted(tensors)} does not match "
                f"recorded {sorted(expected_tensors)}"
            )
        for name, tensor in tensors.items():
            meta_of = TensorMeta.of(tensor)
            if meta_of != expected_tensors[name]:
                raise CorruptCacheEntryError(
                    f"cache entry at {entry_dir} tensor {name!r} is {meta_of.to_dict()}, "
                    f"recorded {expected_tensors[name].to_dict()}"
                )
        return CacheEntry(
            tensors=tensors,
            metadata=dict(meta.get("metadata", {})),
            created_at=str(meta.get("created_at", "")),
            size_bytes=int(meta.get("size_bytes", 0)),
        )

    def _quarantine(self, entry_dir: Path) -> None:
        """Delete a corrupt entry and record it."""
        shutil.rmtree(entry_dir, ignore_errors=True)
        with self._lock:
            self._corrupt_quarantined += 1

    def _touch(self, entry_dir: Path) -> None:
        """Update ``last_access`` for LRU (atomic small-file rewrite)."""
        meta = self._read_meta(entry_dir)
        if meta is None:
            return
        meta["last_access"] = _utc_now_iso()
        meta_path = entry_dir / _META_NAME
        tmp_path = entry_dir / f".{_META_NAME}.{os.getpid()}.tmp"
        try:
            tmp_path.write_text(canonical_json(meta) + "\n", encoding="utf-8")
            os.replace(tmp_path, meta_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)

    def _evict_if_over_budget(self, protect: Path) -> None:
        """Evict LRU entries (by ``last_access``) until under ``max_bytes``."""
        if self._max_bytes is None:
            return
        candidates: list[tuple[str, Path, int]] = []
        total = 0
        for entry_dir in self._visible_entry_dirs():
            meta = self._read_meta(entry_dir)
            if meta is None:
                continue
            size = int(meta.get("size_bytes", 0))
            total += size
            candidates.append((str(meta.get("last_access", "")), entry_dir, size))
        candidates.sort(key=lambda item: item[0])  # oldest first; ISO timestamps sort chronologically
        evicted = 0
        for _, entry_dir, size in candidates:
            if total <= self._max_bytes:
                break
            if entry_dir == protect:
                continue  # never evict the entry just written
            shutil.rmtree(entry_dir, ignore_errors=True)
            total -= size
            evicted += 1
        if evicted:
            with self._lock:
                self._evictions += evicted
