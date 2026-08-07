from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from medfm.data.readers.pathology import EmbeddingStoreReader
from medfm.data.transforms.pathology import plan_tiles
from medfm.models.pathology import (
    AttentionMILAggregator,
    DeterministicSlideSharder,
    EmbeddingStore,
    GridTileSampler,
    MeanPoolingAggregator,
    PathologyVLMAdapter,
    RandomTileSampler,
    TinyPathologyTileEncoder,
    TokenBudget,
    WSITokenSelector,
    extract_slide_embeddings,
)


@dataclass(frozen=True)
class Record:
    tile_id: str
    x: int
    y: int
    width: int = 8
    height: int = 8
    level: int = 0
    mpp: float = 0.5
    quality: dict[str, float] = field(default_factory=dict)


class SyntheticReader:
    def read_tiles(self, locations, *, level, size, on_corrupt="skip"):
        tiles = torch.stack([torch.full((3, size[1], size[0]), float(x + y)) for x, y in locations])
        return SimpleNamespace(tiles=tiles, coords=torch.tensor(locations), errors=())


def _records(count: int = 5) -> list[Record]:
    return [Record(str(i), i * 8, 0, quality={"blur": float(i + 1)}) for i in range(count)]


def test_synthetic_pyramid_tile_plan_and_rank_sharding_are_stable() -> None:
    mask = np.ones((8, 8), dtype=np.bool_)
    first = plan_tiles((64, 64), 0.5, mask, tile_size=16, target_mpp=0.5, min_tissue_fraction=0.1, slide_key="slide")
    second = plan_tiles((64, 64), 0.5, mask, tile_size=16, target_mpp=0.5, min_tissue_fraction=0.1, slide_key="slide")
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert [(item.x, item.y) for item in first] == sorted(
        ((item.x, item.y) for item in first), key=lambda pair: (pair[1], pair[0])
    )
    assigned = [set(DeterministicSlideSharder(rank, 3).slides([f"slide-{i}" for i in range(17)])) for rank in range(3)]
    assert set.union(*assigned) == {f"slide-{i}" for i in range(17)}
    assert not (assigned[0] & assigned[1] or assigned[0] & assigned[2] or assigned[1] & assigned[2])


def test_selector_determinism_and_fixed_token_budget() -> None:
    records = _records()
    assert GridTileSampler().select(records, 3) == [0, 1, 2]
    assert RandomTileSampler().select(records, 3, seed=9) == RandomTileSampler().select(records, 3, seed=9)

    embeddings = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    selected = WSITokenSelector(budget=TokenBudget(precompression=128, visual_tokens=32)).select(embeddings, records)
    assert selected.tokens.shape == (32, 4)
    assert selected.mask.shape == (32,)
    assert int(selected.mask.sum()) == len(records)
    assert selected.records[0].tile_id == "0"


def test_mean_and_attention_mil_ignore_padding() -> None:
    embeddings = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    mean = MeanPoolingAggregator(2)(embeddings, mask)
    assert torch.allclose(mean, torch.tensor([[2.0, 0.0]]))
    attention = AttentionMILAggregator(2)
    result = attention.aggregate(embeddings, mask)
    assert result.attention is not None and result.attention[0, 2] == 0


def test_atomic_store_and_subset_reads(tmp_path) -> None:
    store = EmbeddingStore(tmp_path)
    encoder = TinyPathologyTileEncoder(embedding_dim=4)
    records = _records()
    stats = extract_slide_embeddings("slide/a", records, SyntheticReader(), encoder, store, chunk_size=2)
    assert stats.complete
    cached = store.read_slide("slide/a")
    assert cached.embeddings.shape == (5, 4)
    assert cached.tile_ids == tuple(str(i) for i in range(5))
    subset = store.read_subset("slide/a", [1, 3])
    assert subset.embeddings.shape == (2, 4)
    assert subset.coords.tolist() == [[8, 0, 8, 8], [24, 0, 8, 8]]
    assert store.validate("slide/a") == []


def test_embedding_store_reader_consumes_hdf5_store(tmp_path) -> None:
    store = EmbeddingStore(tmp_path)
    records = _records(2)
    store.write_slide(
        "slide-reader",
        torch.ones(2, 3),
        records,
        model_id="tiny",
        model_revision="rev",
        preprocess_hash="prep",
    )
    payload = EmbeddingStoreReader(tmp_path).read_slide("slide-reader")
    assert payload.tensors["image"].shape == (2, 3)
    assert payload.pathology is not None


def test_corrupt_tile_policy_continues_until_threshold(tmp_path) -> None:
    class OneBadReader(SyntheticReader):
        def read_tiles(self, locations, *, level, size, on_corrupt="skip"):
            kept = locations[:-1]
            result = super().read_tiles(kept, level=level, size=size, on_corrupt=on_corrupt)
            return SimpleNamespace(tiles=result.tiles, coords=result.coords, errors=("bad tile",))

    store = EmbeddingStore(tmp_path)
    stats = extract_slide_embeddings(
        "slide/b",
        _records(5),
        OneBadReader(),
        TinyPathologyTileEncoder(embedding_dim=4),
        store,
        chunk_size=5,
        failure_threshold=0.3,
    )
    assert not stats.complete and stats.skipped_tiles == 1
    with pytest.raises(RuntimeError, match="threshold"):
        extract_slide_embeddings(
            "slide/c",
            _records(5),
            OneBadReader(),
            TinyPathologyTileEncoder(embedding_dim=4),
            store,
            chunk_size=5,
            failure_threshold=0.1,
        )


def test_vlm_bridge_exposes_masked_fixed_tokens() -> None:
    adapter = PathologyVLMAdapter(embedding_dim=4, visual_dim=6, max_tokens=32)
    tokens = torch.randn(2, 32, 4)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[0, 4:] = False
    output = adapter.encode(tokens, mask)
    assert output.visual_tokens.shape == (2, 32, 6)
    assert output.pooled_embedding.shape == (2, 6)
    assert torch.allclose(output.pooled_embedding[0], output.visual_tokens[0, :4].mean(0), atol=1e-5)
