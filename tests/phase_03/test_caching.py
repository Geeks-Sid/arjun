"""Cache keys, atomic disk store, invalidation, corrupt recovery, rank safety."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from medfm.core.serialization import config_hash
from medfm.data.caching import (
    CACHE_KEY_VERSION,
    CacheKey,
    CacheKind,
    DiskTensorCache,
    PreprocessingCache,
    TokenizationCache,
    VisualEmbeddingCache,
)
from medfm.data.errors import CacheError

SOURCE = "a" * 64
OTHER_SOURCE = "b" * 64
PRE = config_hash({"normalize": "ct_zscore", "resolution": 512})


def _preprocess_key(source: str = SOURCE, reader: str = "1.0.0", pre: str = PRE) -> CacheKey:
    return PreprocessingCache.key(source_file_hash=source, reader_version=reader, preprocessing_hash=pre)


def _embedding_key(**overrides: str) -> CacheKey:
    fields = {
        "source_file_hash": SOURCE,
        "reader_version": "1.0.0",
        "preprocessing_hash": PRE,
        "model_id": "opt-125m",
        "model_revision": "rev1",
        "output_layer": "encoder.layer.11",
        "dtype": "float32",
    }
    fields.update(overrides)
    return VisualEmbeddingCache.key(**fields)  # type: ignore[arg-type]


def test_key_requires_sha256_and_canonical_dtype() -> None:
    with pytest.raises(CacheError, match="source_file_hash"):
        CacheKey(kind=CacheKind.PREPROCESSING, source_file_hash="short", reader_version="1", preprocessing_hash=PRE)
    with pytest.raises(CacheError, match="dtype"):
        CacheKey(
            kind=CacheKind.VISUAL_EMBEDDING,
            source_file_hash=SOURCE,
            reader_version="1",
            preprocessing_hash=PRE,
            dtype="not-a-dtype",
        )


def test_key_string_is_deterministic_and_component_sensitive() -> None:
    a = _embedding_key()
    b = _embedding_key()
    assert a.key_string() == b.key_string()
    assert CACHE_KEY_VERSION == 1
    for field, value in {
        "source_file_hash": OTHER_SOURCE,
        "reader_version": "1.1.0",
        "model_id": "opt-350m",
        "model_revision": "rev2",
        "output_layer": "encoder.layer.7",
        "dtype": "float16",
    }.items():
        changed = _embedding_key(**{field: value})
        assert changed.key_string() != a.key_string(), f"key must change with {field}"


def test_put_get_round_trip_is_cpu_and_accelerator_neutral(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    key = _preprocess_key()
    cache.put(key, {"image": torch.randn(4, 8)}, metadata={"stage": "canonicalized"})
    entry = cache.get(key)
    assert entry is not None
    (tensor,) = entry.tensors.values()
    assert tensor.device.type == "cpu"
    assert entry.metadata == {"stage": "canonicalized"}
    stats = cache.stats()
    assert stats.hits == 1 and stats.misses == 0 and stats.entries == 1


def test_invalidation_on_normalization_resolution_and_reader_change(tmp_path: Path) -> None:
    cache = PreprocessingCache.on_disk(tmp_path)
    base = _preprocess_key()
    cache.put(base, {"image": torch.randn(2, 2)})
    assert cache.get(base) is not None

    # Changing normalization or resolution folds into preprocessing_hash -> miss.
    for altered_pre in (
        config_hash({"normalize": "percentile", "resolution": 512}),
        config_hash({"normalize": "ct_zscore", "resolution": 384}),
    ):
        assert cache.get(_preprocess_key(pre=altered_pre)) is None
    # Changing the reader version -> miss.
    assert cache.get(_preprocess_key(reader="2.0.0")) is None
    # Changing the source content hash -> miss.
    assert cache.get(_preprocess_key(source=OTHER_SOURCE)) is None


def test_invalidation_on_model_adapter_layer_dtype_change(tmp_path: Path) -> None:
    cache = VisualEmbeddingCache.on_disk(tmp_path)
    base = _embedding_key()
    cache.put(base, {"embedding": torch.randn(8)})
    assert cache.get(base) is not None
    for field, value in {
        "model_id": "other-model",
        "model_revision": "rev2",
        "output_layer": "encoder.layer.7",
        "dtype": "bfloat16",
    }.items():
        assert cache.get(_embedding_key(**{field: value})) is None, f"must miss when {field} changes"
    # Adapter identity rides in `extra`.
    adapted = VisualEmbeddingCache.key(
        source_file_hash=SOURCE,
        reader_version="1.0.0",
        preprocessing_hash=PRE,
        model_id="opt-125m",
        model_revision="rev1",
        output_layer="encoder.layer.11",
        dtype="float32",
        extra=(("adapter", "lora-r8"),),
    )
    assert cache.get(adapted) is None


def test_tokenization_cache_key_isolation(tmp_path: Path) -> None:
    cache = TokenizationCache.on_disk(tmp_path)
    key = TokenizationCache.key(
        source_file_hash=SOURCE,
        reader_version="1.0.0",
        tokenization_hash=config_hash({"max_length": 512}),
        tokenizer_id="gpt2-med",
        tokenizer_revision="v3",
    )
    assert key.kind is CacheKind.TOKENIZATION
    cache.put(key, {"input_ids": torch.zeros(16, dtype=torch.int64)})
    assert cache.get(key) is not None
    retokenized = TokenizationCache.key(
        source_file_hash=SOURCE,
        reader_version="1.0.0",
        tokenization_hash=config_hash({"max_length": 1024}),
        tokenizer_id="gpt2-med",
        tokenizer_revision="v3",
    )
    assert cache.get(retokenized) is None


def test_typed_cache_rejects_cross_kind_keys(tmp_path: Path) -> None:
    cache = PreprocessingCache.on_disk(tmp_path)
    foreign = _embedding_key()
    with pytest.raises(ValueError, match="kind"):
        cache.get(foreign)
    with pytest.raises(ValueError, match="kind"):
        cache.put(foreign, {"x": torch.randn(1)})


def test_rejects_non_canonical_dtype_and_empty_put(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    with pytest.raises(CacheError, match="canonical"):
        cache.put(_preprocess_key(), {"x": torch.randn(2).to(torch.complex64)})
    with pytest.raises(CacheError, match="at least one tensor"):
        cache.put(_preprocess_key(), {})
    with pytest.raises(CacheError, match="JSON-serializable"):
        cache.put(_preprocess_key(), {"x": torch.randn(1)}, metadata={"bad": object()})


def test_corrupt_entry_is_quarantined_and_reported_as_miss(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    key = _preprocess_key()
    cache.put(key, {"image": torch.randn(4)})
    # Tamper with the payload on disk -> integrity check must fail.
    entry_dir = tmp_path / str(key.kind) / key.key_string()
    payload = entry_dir / "payload.safetensors"
    payload.write_bytes(b"corrupted-bytes")
    assert cache.get(key) is None
    stats = cache.stats()
    assert stats.corrupt_quarantined == 1
    assert not entry_dir.exists()  # quarantined (deleted)
    # A subsequent put under the same key recovers cleanly.
    cache.put(key, {"image": torch.randn(4)})
    assert cache.get(key) is not None


def test_missing_meta_detected(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    key = _preprocess_key()
    cache.put(key, {"image": torch.randn(2)})
    entry_dir = tmp_path / str(key.kind) / key.key_string()
    (entry_dir / "meta.json").unlink()
    assert cache.get(key) is None
    assert cache.stats().corrupt_quarantined == 1


def test_partial_write_is_atomic(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    key = _preprocess_key()
    cache.put(key, {"image": torch.randn(1024)})
    # No stray tmp directories should remain after a successful put.
    leftovers = [p.name for p in (tmp_path / str(key.kind)).iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []
    assert cache.verify_entry(key) is None


def test_lru_eviction_respects_budget(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path, max_bytes=4 * 1024)
    keys = []
    for i in range(6):
        key = _preprocess_key(source=f"{i:064x}")
        cache.put(key, {"image": torch.randn(512)})  # ~2KB each
        keys.append(key)
    stats = cache.stats()
    assert stats.bytes <= 4 * 1024
    assert stats.evictions > 0
    # The most recently written entry survives.
    assert cache.get(keys[-1]) is not None


def test_entry_alone_larger_than_budget_raises(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path, max_bytes=16)
    with pytest.raises(CacheError, match="exceeds"):
        cache.put(_preprocess_key(), {"image": torch.randn(1024)})


def test_rank_safety_coordinator_only(tmp_path: Path) -> None:
    coordinator = DiskTensorCache(tmp_path, rank=0, coordinator_only=True)
    worker = DiskTensorCache(tmp_path, rank=3, coordinator_only=True)
    key = _preprocess_key()
    worker.put(key, {"image": torch.randn(2)})  # documented no-op from non-zero rank
    assert worker.stats().entries == 0
    coordinator.put(key, {"image": torch.randn(2)})
    assert worker.get(key) is not None  # all ranks read the shared layout


def test_rank_safety_per_rank_partitions(tmp_path: Path) -> None:
    rank0 = DiskTensorCache(tmp_path, rank=0)
    rank1 = DiskTensorCache(tmp_path, rank=1)
    key = _preprocess_key()
    rank0.put(key, {"image": torch.zeros(2)})
    rank1.put(key, {"image": torch.ones(2)})
    assert torch.equal(rank0.get(key).tensors["image"], torch.zeros(2))
    assert torch.equal(rank1.get(key).tensors["image"], torch.ones(2))
    rank1.clear()
    assert rank1.get(key) is None
    assert rank0.get(key) is not None  # rank 0's shared layout untouched


def test_invalidate_and_clear(tmp_path: Path) -> None:
    cache = DiskTensorCache(tmp_path)
    k1, k2 = _preprocess_key(source=SOURCE), _preprocess_key(source=OTHER_SOURCE)
    cache.put(k1, {"image": torch.randn(1)})
    cache.put(k2, {"image": torch.randn(1)})
    assert cache.invalidate(k1) is True
    assert cache.invalidate(k1) is False
    cache.clear()
    assert cache.stats().entries == 0
