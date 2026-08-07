"""Two-stage WSI extraction: bounded tile reads, atomic HDF5 caches, and resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from medfm.core.serialization import canonical_dtype_name, config_hash

STORE_SCHEMA_VERSION = 1


class TileReader(Protocol):
    def read_tiles(
        self, locations: list[tuple[int, int]], *, level: int, size: tuple[int, int], on_corrupt: str = "raise"
    ) -> Any: ...


class TileEncoder(Protocol):
    model_id: str
    revision: str
    preprocess_hash: str

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class TileEmbeddingMetadata:
    slide_id: str
    tile_id: str
    x: int
    y: int
    width: int
    height: int
    level: int
    mpp: float
    quality: dict[str, float]

    @classmethod
    def from_record(cls, slide_id: str, record: Any) -> TileEmbeddingMetadata:
        return cls(
            slide_id=slide_id,
            tile_id=str(record.tile_id),
            x=int(record.x),
            y=int(record.y),
            width=int(record.width),
            height=int(record.height),
            level=int(record.level),
            mpp=float(record.mpp),
            quality={str(k): float(v) for k, v in sorted((getattr(record, "quality", {}) or {}).items())},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "tile_id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "mpp": self.mpp,
            "quality": dict(self.quality),
        }


@dataclass(frozen=True)
class StoredEmbeddings:
    embeddings: torch.Tensor
    tiles: tuple[TileEmbeddingMetadata, ...]
    metadata: dict[str, Any]

    @property
    def coords(self) -> torch.Tensor:
        return torch.tensor([[t.x, t.y, t.width, t.height] for t in self.tiles], dtype=torch.int64)

    @property
    def tile_ids(self) -> tuple[str, ...]:
        return tuple(t.tile_id for t in self.tiles)


@dataclass(frozen=True)
class ExtractionStats:
    slide_id: str
    requested_tiles: int
    encoded_tiles: int
    skipped_tiles: int
    resumed_tiles: int
    chunks: int
    complete: bool


class EmbeddingStore:
    """Concurrent-reader, atomic-writer HDF5 store selected for Phase 08.

    One HDF5 file is committed per slide. Embeddings and aligned tile columns
    are chunked and gzip-compressed; ``read_subset`` never materializes the
    complete slide. A marker is written last, so missing markers identify
    interrupted/incomplete stores. Chunk files under ``*.chunks`` provide
    tile/chunk-granularity resume after a process interruption.
    """

    schema_version = STORE_SCHEMA_VERSION

    def __init__(self, root: str | Path, *, compression: str = "gzip") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    @staticmethod
    def _safe(slide_id: str) -> str:
        encoded = slide_id.replace("/", "_").replace("\\", "_")
        return encoded if encoded and encoded not in (".", "..") else hashlib.sha256(slide_id.encode()).hexdigest()[:24]

    def _path(self, slide_id: str) -> Path:
        return self.root / f"{self._safe(slide_id)}.h5"

    def _metadata_path(self, slide_id: str) -> Path:
        return self.root / f"{self._safe(slide_id)}.json"

    def _complete_path(self, slide_id: str) -> Path:
        return self.root / f"{self._safe(slide_id)}.complete"

    def _chunks_dir(self, slide_id: str) -> Path:
        return self.root / f"{self._safe(slide_id)}.chunks"

    @staticmethod
    def _require_h5py() -> Any:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("EmbeddingStore requires h5py; install medfm[pathology]") from exc
        return h5py

    @staticmethod
    def _record_metadata(slide_id: str, records: Sequence[Any]) -> list[TileEmbeddingMetadata]:
        return [TileEmbeddingMetadata.from_record(slide_id, record) for record in records]

    def _write_h5(
        self, path: Path, embeddings: torch.Tensor, tiles: Sequence[TileEmbeddingMetadata], metadata: dict[str, Any]
    ) -> None:
        h5py = self._require_h5py()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(tiles):
            raise ValueError(f"embeddings [N,D] must align with {len(tiles)} tiles; got {tuple(embeddings.shape)}")
        if not tiles:
            raise ValueError("cannot commit an empty slide embedding store")
        array = embeddings.detach().cpu().contiguous().numpy()
        string_dtype = h5py.string_dtype(encoding="utf-8")
        coords = [[t.x, t.y, t.width, t.height] for t in tiles]
        with h5py.File(path, "w", libver="latest") as handle:
            handle.attrs["schema_version"] = STORE_SCHEMA_VERSION
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            handle.create_dataset(
                "embeddings", data=array, chunks=(min(len(tiles), 256), array.shape[1]), compression=self.compression
            )
            handle.create_dataset("tile_ids", data=[t.tile_id for t in tiles], dtype=string_dtype)
            handle.create_dataset("coords", data=coords, dtype="i8")
            handle.create_dataset("level", data=[t.level for t in tiles], dtype="i8")
            handle.create_dataset("mpp", data=[t.mpp for t in tiles], dtype="f8")
            handle.create_dataset(
                "quality_json", data=[json.dumps(t.quality, sort_keys=True) for t in tiles], dtype=string_dtype
            )
            handle.flush()

    def write_slide(
        self,
        slide_id: str,
        embeddings: torch.Tensor,
        records: Sequence[Any] | Sequence[TileEmbeddingMetadata],
        *,
        model_id: str,
        model_revision: str | None = None,
        revision: str | None = None,
        preprocess_hash: str,
        layer: str | int = "pooled",
        dtype: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically commit a complete slide and invalidate stale identities."""
        tiles = [
            r if isinstance(r, TileEmbeddingMetadata) else TileEmbeddingMetadata.from_record(slide_id, r)
            for r in records
        ]
        actual_dtype = dtype or canonical_dtype_name(embeddings.dtype)
        metadata: dict[str, Any] = {
            "schema_version": STORE_SCHEMA_VERSION,
            "slide_id": slide_id,
            "model_id": model_id,
            "model_revision": model_revision or revision or "unknown",
            "preprocess_hash": preprocess_hash,
            "layer": str(layer),
            "dtype": actual_dtype,
            "embedding_shape": list(embeddings.shape),
            "embedding_dim": int(embeddings.shape[-1]) if embeddings.ndim == 2 else None,
            "tile_count": len(tiles),
            "store_format": "hdf5",
        }
        metadata.update(extra_metadata or {})
        target = self._path(slide_id)
        self._complete_path(slide_id).unlink(missing_ok=True)
        self._metadata_path(slide_id).unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(prefix=target.name, suffix=".tmp", dir=self.root, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            self._write_h5(temp_path, embeddings, tiles, metadata)
            os.replace(temp_path, target)
            meta_tmp = self._metadata_path(slide_id).with_suffix(".json.tmp")
            meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(meta_tmp, self._metadata_path(slide_id))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self._complete_path(slide_id).write_text(digest, encoding="ascii")
        finally:
            temp_path.unlink(missing_ok=True)
        return target

    def write_chunk(
        self,
        slide_id: str,
        embeddings: torch.Tensor,
        records: Sequence[Any] | Sequence[TileEmbeddingMetadata],
        *,
        model_id: str,
        model_revision: str,
        preprocess_hash: str,
        layer: str | int = "pooled",
    ) -> Path:
        """Atomically persist one extraction chunk for resumable extraction."""
        tiles = [
            r if isinstance(r, TileEmbeddingMetadata) else TileEmbeddingMetadata.from_record(slide_id, r)
            for r in records
        ]
        if embeddings.shape[0] != len(tiles):
            raise ValueError("chunk embeddings and records are not aligned")
        chunk_dir = self._chunks_dir(slide_id)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        identity = config_hash(
            {"model_id": model_id, "revision": model_revision, "preprocess_hash": preprocess_hash, "layer": str(layer)}
        )
        name = hashlib.sha256((identity + ":" + ":".join(t.tile_id for t in tiles)).encode()).hexdigest()[:32]
        target = chunk_dir / f"{name}.pt"
        payload = {
            "embeddings": embeddings.detach().cpu(),
            "tiles": [t.to_dict() for t in tiles],
            "metadata": {
                "model_id": model_id,
                "model_revision": model_revision,
                "preprocess_hash": preprocess_hash,
                "layer": str(layer),
                "identity": identity,
            },
        }
        with tempfile.NamedTemporaryFile(prefix=target.name, suffix=".tmp", dir=chunk_dir, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)
        return target

    def _load_chunks(self, slide_id: str, identity: str) -> tuple[torch.Tensor, list[TileEmbeddingMetadata]]:
        directory = self._chunks_dir(slide_id)
        embeddings: list[torch.Tensor] = []
        tiles: list[TileEmbeddingMetadata] = []
        if not directory.is_dir():
            return torch.empty((0, 0)), []
        for path in sorted(directory.glob("*.pt")):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["metadata"]["identity"] != identity:
                    continue
                chunk_tiles = [TileEmbeddingMetadata(**item) for item in payload["tiles"]]
                chunk_embeddings = payload["embeddings"]
                if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[0] != len(chunk_tiles):
                    continue
            except Exception:
                continue
            embeddings.append(chunk_embeddings)
            tiles.extend(chunk_tiles)
        if not embeddings:
            return torch.empty((0, 0)), []
        order = sorted(range(len(tiles)), key=lambda i: (tiles[i].y, tiles[i].x, tiles[i].tile_id))
        return torch.cat(embeddings)[order], [tiles[i] for i in order]

    def finalize_chunks(
        self, slide_id: str, *, model_id: str, model_revision: str, preprocess_hash: str, layer: str | int = "pooled"
    ) -> Path:
        identity = config_hash(
            {"model_id": model_id, "revision": model_revision, "preprocess_hash": preprocess_hash, "layer": str(layer)}
        )
        embeddings, tiles = self._load_chunks(slide_id, identity)
        if not tiles:
            raise ValueError(f"no resumable chunks found for slide {slide_id!r}")
        path = self.write_slide(
            slide_id,
            embeddings,
            tiles,
            model_id=model_id,
            model_revision=model_revision,
            preprocess_hash=preprocess_hash,
            layer=layer,
        )
        for chunk in self._chunks_dir(slide_id).glob("*.pt"):
            chunk.unlink(missing_ok=True)
        self._chunks_dir(slide_id).rmdir()
        return path

    def is_complete(self, slide_id: str) -> bool:
        path, marker = self._path(slide_id), self._complete_path(slide_id)
        if not path.is_file() or not marker.is_file():
            return False
        try:
            return marker.read_text(encoding="ascii").strip() == hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False

    def metadata(self, slide_id: str) -> dict[str, Any]:
        if not self.is_complete(slide_id):
            raise RuntimeError(f"embedding store for slide {slide_id!r} is incomplete or corrupt")
        return json.loads(self._metadata_path(slide_id).read_text(encoding="utf-8"))

    def _read(self, slide_id: str, indices: Sequence[int] | None = None) -> StoredEmbeddings:
        if not self.is_complete(slide_id):
            raise RuntimeError(f"embedding store for slide {slide_id!r} is incomplete or corrupt")
        h5py = self._require_h5py()
        metadata = self.metadata(slide_id)
        with h5py.File(self._path(slide_id), "r", swmr=True) as handle:
            count = int(handle["embeddings"].shape[0])
            selected = list(range(count)) if indices is None else [int(i) for i in indices]
            if any(i < 0 or i >= count for i in selected):
                raise IndexError("embedding subset index out of range")
            order = sorted(range(len(selected)), key=lambda j: selected[j])
            sorted_indices = [selected[j] for j in order]
            inverse = [order.index(j) for j in range(len(order))]
            embeddings = torch.from_numpy(handle["embeddings"][sorted_indices])[inverse]
            ids = handle["tile_ids"][sorted_indices][inverse]
            coords = handle["coords"][sorted_indices][inverse]
            levels = handle["level"][sorted_indices][inverse]
            mpps = handle["mpp"][sorted_indices][inverse]
            qualities = handle["quality_json"][sorted_indices][inverse]
        tiles = []
        for tile_id, coord, level, mpp, quality in zip(ids, coords, levels, mpps, qualities, strict=True):
            tile_name = tile_id.decode() if isinstance(tile_id, bytes) else str(tile_id)
            quality_text = quality.decode() if isinstance(quality, bytes) else str(quality)
            tiles.append(
                TileEmbeddingMetadata(
                    slide_id,
                    tile_name,
                    int(coord[0]),
                    int(coord[1]),
                    int(coord[2]),
                    int(coord[3]),
                    int(level),
                    float(mpp),
                    json.loads(quality_text),
                )
            )
        if list(metadata.get("embedding_shape", []))[0:1] and len(selected) != int(metadata["embedding_shape"][0]):
            metadata = {**metadata, "subset": True, "subset_count": len(selected)}
        return StoredEmbeddings(embeddings=embeddings, tiles=tuple(tiles), metadata=metadata)

    def read_slide(self, slide_id: str) -> StoredEmbeddings:
        return self._read(slide_id)

    def read_subset(self, slide_id: str, indices: Sequence[int]) -> StoredEmbeddings:
        return self._read(slide_id, indices)

    def validate(self, slide_id: str) -> list[str]:
        errors: list[str] = []
        try:
            stored = self.read_slide(slide_id)
        except Exception as exc:
            return [str(exc)]
        if stored.embeddings.ndim != 2 or stored.embeddings.shape[0] != len(stored.tiles):
            errors.append("embedding rows and tile metadata are misaligned")
        if stored.metadata.get("schema_version") != STORE_SCHEMA_VERSION:
            errors.append("unsupported embedding-store schema version")
        ids = stored.tile_ids
        if len(ids) != len(set(ids)):
            errors.append("duplicate tile ids")
        return errors

    def iter_slide_ids(self) -> Iterable[str]:
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "slide_id" in data:
                    yield str(data["slide_id"])
            except Exception:
                continue

    def invalidate_if_changed(
        self, slide_id: str, *, model_revision: str, preprocess_hash: str, layer: str | int, dtype: str
    ) -> bool:
        if not self.is_complete(slide_id):
            return True
        data = self.metadata(slide_id)
        changed = any(
            data.get(k) != v
            for k, v in {
                "model_revision": model_revision,
                "preprocess_hash": preprocess_hash,
                "layer": str(layer),
                "dtype": dtype,
            }.items()
        )
        if changed:
            for path in (self._path(slide_id), self._metadata_path(slide_id), self._complete_path(slide_id)):
                path.unlink(missing_ok=True)
        return changed


def extract_slide_embeddings(
    slide_id: str,
    tile_records: Sequence[Any],
    reader: TileReader,
    encoder: TileEncoder,
    store: EmbeddingStore,
    *,
    tile_size: tuple[int, int] | None = None,
    chunk_size: int = 32,
    on_corrupt: str = "skip",
    failure_threshold: float = 0.1,
    level: int | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    preprocess_hash: str | None = None,
    layer: str | int = "pooled",
) -> ExtractionStats:
    """Encode a bounded tile chunk at a time and resume committed chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if on_corrupt not in ("raise", "skip"):
        raise ValueError("on_corrupt must be 'raise' or 'skip'")
    if not 0 <= failure_threshold < 1:
        raise ValueError("failure_threshold must be in [0, 1)")
    model_id = model_id or str(getattr(encoder, "model_id", "unknown"))
    model_revision = model_revision or str(getattr(encoder, "revision", "unknown"))
    preprocess_hash = preprocess_hash or str(getattr(encoder, "preprocess_hash", "unknown"))
    if store.is_complete(slide_id):
        data = store.read_slide(slide_id)
        return ExtractionStats(slide_id, len(tile_records), len(data.tiles), 0, len(data.tiles), 0, True)
    if not tile_records:
        raise ValueError("cannot extract an empty tile plan")
    complete_ids = {
        tile.tile_id for _, tile in _resume_records(store, slide_id, model_id, model_revision, preprocess_hash, layer)
    }
    skipped = 0
    encoded = 0
    resumed = len(complete_ids)
    chunks = 0
    all_failures: list[str] = []
    width_height = tile_size or (int(tile_records[0].width), int(tile_records[0].height))
    for start in range(0, len(tile_records), chunk_size):
        requested = [r for r in tile_records[start : start + chunk_size] if str(r.tile_id) not in complete_ids]
        if not requested:
            continue
        chunks += 1
        locations = [(int(r.x), int(r.y)) for r in requested]
        try:
            result = reader.read_tiles(
                locations,
                level=level if level is not None else int(requested[0].level),
                size=width_height,
                on_corrupt=on_corrupt,
            )
        except Exception as exc:
            if on_corrupt == "raise":
                raise
            all_failures.extend([str(exc)] * len(requested))
            skipped += len(requested)
            continue
        tiles = result.tiles
        coords = [tuple(int(v) for v in row) for row in result.coords.tolist()]
        by_coord = {(int(r.x), int(r.y)): r for r in requested}
        healthy = [by_coord[c] for c in coords if c in by_coord]
        skipped += len(requested) - len(healthy)
        all_failures.extend(list(getattr(result, "errors", ())))
        if not healthy:
            continue
        with torch.inference_mode():
            output = encoder.encode_tiles(tiles) if hasattr(encoder, "encode_tiles") else encoder(tiles)  # type: ignore[operator]
        if output.ndim != 2 or output.shape[0] != len(healthy):
            raise ValueError(f"tile encoder must return [N,D] aligned with healthy tiles; got {tuple(output.shape)}")
        store.write_chunk(
            slide_id,
            output,
            healthy,
            model_id=model_id,
            model_revision=model_revision,
            preprocess_hash=preprocess_hash,
            layer=layer,
        )
        complete_ids.update(str(r.tile_id) for r in healthy)
        encoded += len(healthy)
    total_requested = len(tile_records)
    if skipped / max(total_requested, 1) > failure_threshold:
        raise RuntimeError(f"slide {slide_id!r} exceeded corrupt-tile threshold: {skipped}/{total_requested}")
    if len(complete_ids) < total_requested:
        return ExtractionStats(slide_id, total_requested, encoded, skipped, resumed, chunks, False)
    store.finalize_chunks(
        slide_id, model_id=model_id, model_revision=model_revision, preprocess_hash=preprocess_hash, layer=layer
    )
    return ExtractionStats(slide_id, total_requested, encoded, skipped, resumed, chunks, True)


def _resume_records(
    store: EmbeddingStore, slide_id: str, model_id: str, revision: str, preprocess_hash: str, layer: str | int
) -> list[tuple[torch.Tensor, TileEmbeddingMetadata]]:
    identity = config_hash(
        {"model_id": model_id, "revision": revision, "preprocess_hash": preprocess_hash, "layer": str(layer)}
    )
    directory = store._chunks_dir(slide_id)
    found: list[tuple[torch.Tensor, TileEmbeddingMetadata]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["metadata"]["identity"] != identity:
                continue
            found.extend(
                (payload["embeddings"][i], TileEmbeddingMetadata(**item)) for i, item in enumerate(payload["tiles"])
            )
        except Exception:
            continue
    return found


__all__ = [
    "EmbeddingStore",
    "ExtractionStats",
    "STORE_SCHEMA_VERSION",
    "StoredEmbeddings",
    "TileEmbeddingMetadata",
    "extract_slide_embeddings",
]
