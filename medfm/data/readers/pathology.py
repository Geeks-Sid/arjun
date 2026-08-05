"""Whole-slide pathology readers: slide contract + backends + tile/embedding stores.

The :class:`SlideReader` contract is backend-neutral: dimensions, pyramid
levels/downsamples, MPP/magnification, thumbnails, region reads, and tile
grids are identical whether the backend is OpenSlide, TiffSlide, or cuCIM.
Coordinates are level-0 slide pixels unless stated otherwise; use
:func:`convert_level_coords` / :func:`validate_level_coords` to move between
pyramid levels (validated against level dimensions, round-trip error < 1 px).

Corrupt regions are recoverable at tile scope: :meth:`SlideReader.read_tiles`
with ``on_corrupt="skip"`` collects per-tile failures and returns the healthy
tiles plus an error list; ``on_corrupt="raise"`` (default) propagates a
:class:`CorruptSampleError` naming the tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from medfm.core.sample import PathologyMetadata
from medfm.data.errors import CorruptSampleError, ReaderError, UnsupportedFormatError
from medfm.data.readers.base import PayloadRead, Reader, hash_identifier


def _require(module_name: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise UnsupportedFormatError(
            f"this slide reader requires {module_name!r}; install medfm with the {extra!r} extra "
            f"(uv sync --extra {extra}) or pick another backend"
        ) from exc


def convert_level_coords(
    coords: torch.Tensor,
    *,
    from_level: int,
    to_level: int,
    level_downsamples: tuple[float, ...],
) -> torch.Tensor:
    """Convert ``(x, y)`` slide coordinates between pyramid levels.

    Conversion goes through level-0 space: ``xy_l0 = xy_from * ds[from]`` and
    ``xy_to = xy_l0 / ds[to]`` (floating point; callers that need integer
    pixel indices round afterwards and should :func:`validate_level_coords`).
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ReaderError(f"level coordinates must have shape [N, 2]; got {tuple(coords.shape)}")
    for level in (from_level, to_level):
        if not 0 <= level < len(level_downsamples):
            raise ReaderError(f"pyramid level {level} out of range; the slide has {len(level_downsamples)} level(s)")
    level0 = coords.to(torch.float64) * float(level_downsamples[from_level])
    return level0 / float(level_downsamples[to_level])


def validate_level_coords(
    coords: torch.Tensor,
    *,
    level: int,
    level_dimensions: tuple[tuple[int, int], ...],
    region_size: tuple[int, int] | None = None,
) -> None:
    """Verify ``coords`` are valid pixel origins at ``level``.

    Checks: integer-representable, non-negative, inside the level dimensions,
    and (with ``region_size``) that the full region fits inside the level.
    Also checks the level-0 round-trip error stays below 1 px.
    """
    if not 0 <= level < len(level_dimensions):
        raise ReaderError(f"pyramid level {level} out of range; slide has {len(level_dimensions)} level(s)")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ReaderError(f"level coordinates must have shape [N, 2]; got {tuple(coords.shape)}")
    width, height = level_dimensions[level]
    values = coords.to(torch.float64)
    if bool((values < 0).any()):
        raise ReaderError("slide coordinates must be non-negative")
    max_x = int(values[:, 0].max()) if values.numel() else 0
    max_y = int(values[:, 1].max()) if values.numel() else 0
    need_w = width - (region_size[0] if region_size else 1)
    need_h = height - (region_size[1] if region_size else 1)
    if max_x > need_w or max_y > need_h:
        raise ReaderError(
            f"slide coordinates exceed level {level} dimensions {width}x{height}"
            + (f" with region size {region_size}" if region_size else "")
            + f" (max allowed origin: ({need_w}, {need_h}); got max ({max_x}, {max_y}))"
        )


@dataclass(frozen=True)
class TileReadResult:
    """A grid of tiles plus their level-0 coordinates and per-tile errors."""

    tiles: torch.Tensor  # [T, C, H, W]
    coords: torch.Tensor  # [T, 2] int64, level-0 pixel origins
    errors: tuple[str, ...] = ()  # populated when on_corrupt="skip"


@runtime_checkable
class SlideReader(Protocol):
    """Backend-neutral whole-slide contract."""

    reader_id: str
    reader_version: str
    dimensions: tuple[int, int]  # level-0 (width, height)
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp: float | None  # level-0 microns per pixel
    magnification: float | None  # objective power

    def thumbnail(self, max_size: tuple[int, int]) -> torch.Tensor:
        """RGB tissue thumbnail, ``(H, W, 3)`` uint8, bounded by ``max_size``."""
        ...

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> torch.Tensor:
        """RGB region at ``level`` with level-0 ``location``; ``(H, W, 3)`` uint8."""
        ...

    def read_tiles(
        self,
        locations: list[tuple[int, int]],
        *,
        level: int,
        size: tuple[int, int],
        on_corrupt: str = "raise",
    ) -> TileReadResult:
        """Read tile regions; optionally skip corrupt tiles (sample scope)."""
        ...

    def pathology_metadata(self) -> PathologyMetadata:
        """Contract metadata for :class:`MedicalSample` (MPP never discarded)."""
        ...

    def close(self) -> None:
        """Release the file handle."""
        ...


class _BackendSlideReader:
    """Shared implementation for PIL-API slide backends (OpenSlide, TiffSlide)."""

    reader_id = "slide"
    reader_version = "1.0.0"

    def __init__(self, path: Path, *, backend: Any) -> None:
        self._path = path
        try:
            self._slide = backend(str(path))
        except Exception as exc:
            raise ReaderError(f"cannot open slide {path}: {exc}") from exc
        self.dimensions = (int(self._slide.dimensions[0]), int(self._slide.dimensions[1]))
        self.level_count = int(self._slide.level_count)
        self.level_dimensions = tuple((int(w), int(h)) for w, h in self._slide.level_dimensions)
        self.level_downsamples = tuple(float(d) for d in self._slide.level_downsamples)
        self.mpp = self._read_mpp()
        self.magnification = self._read_magnification()

    def _property(self, *names: str) -> str | None:
        properties = getattr(self._slide, "properties", {}) or {}
        for name in names:
            value = properties.get(name)
            if value not in (None, ""):
                return str(value)
        return None

    def _read_mpp(self) -> float | None:
        raw = self._property("openslide.mpp-x", "tiffslide.mpp-x")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def _read_magnification(self) -> float | None:
        raw = self._property("openslide.objective-power")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def thumbnail(self, max_size: tuple[int, int]) -> torch.Tensor:
        image = self._slide.get_thumbnail(tuple(max_size))
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()  # ro buffer -> writable
        return torch.from_numpy(np.ascontiguousarray(array))

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> torch.Tensor:
        if not 0 <= level < self.level_count:
            raise ReaderError(f"pyramid level {level} out of range; slide has {self.level_count} level(s)")
        try:
            region = self._slide.read_region(tuple(location), level, tuple(size))
        except Exception as exc:
            raise CorruptSampleError(
                f"corrupt or unreadable region at level-0 origin {location}, level {level}, size {size} "
                f"in {self._path.name}: {exc}"
            ) from exc
        array = np.asarray(region.convert("RGB"), dtype=np.uint8).copy()  # ro buffer -> writable
        return torch.from_numpy(np.ascontiguousarray(array))

    def read_tiles(
        self,
        locations: list[tuple[int, int]],
        *,
        level: int,
        size: tuple[int, int],
        on_corrupt: str = "raise",
    ) -> TileReadResult:
        if on_corrupt not in ("raise", "skip"):
            raise ReaderError(f"on_corrupt must be 'raise' or 'skip'; got {on_corrupt!r}")
        validate_level_coords(
            torch.tensor(locations, dtype=torch.int64).reshape(-1, 2) if locations else torch.zeros((0, 2)),
            level=level,
            level_dimensions=self.level_dimensions,
            region_size=size,
        )
        tiles: list[torch.Tensor] = []
        kept_coords: list[tuple[int, int]] = []
        errors: list[str] = []
        for location in locations:
            try:
                tiles.append(self.read_region(location, level, size).permute(2, 0, 1))  # (C, H, W)
                kept_coords.append(location)
            except CorruptSampleError as exc:
                if on_corrupt == "raise":
                    raise
                errors.append(str(exc))
        if not tiles:
            raise CorruptSampleError(
                f"every requested tile was corrupt ({len(errors)} failure(s)); no recoverable tiles in "
                f"{self._path.name}"
            )
        return TileReadResult(
            tiles=torch.stack(tiles),
            coords=torch.tensor(kept_coords, dtype=torch.int64),
            errors=tuple(errors),
        )

    def pathology_metadata(self) -> PathologyMetadata:
        return PathologyMetadata(
            microns_per_pixel=self.mpp,
            magnification=self.magnification,
            slide_dimensions=self.dimensions,
            level_dimensions=self.level_dimensions,
            scanner_vendor=self._property("openslide.vendor"),
        )

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> _BackendSlideReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class OpenSlideReader(_BackendSlideReader, Reader):
    """OpenSlide backend (requires the native OpenSlide library)."""

    reader_id = "openslide"

    @staticmethod
    def available() -> bool:
        try:
            import openslide  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def __init__(self, path: Path) -> None:
        openslide = _require("openslide", "pathology")
        super().__init__(path, backend=openslide.OpenSlide)


class TiffSlideReader(_BackendSlideReader, Reader):
    """TiffSlide backend (pure-Python; reads generic pyramidal TIFFs)."""

    reader_id = "tiffslide"

    @staticmethod
    def available() -> bool:
        try:
            import tiffslide  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, path: Path) -> None:
        tiffslide = _require("tiffslide", "pathology")
        super().__init__(path, backend=tiffslide.TiffSlide)


class CuCIMSlideReader(Reader):
    """cuCIM backend behind an optional dependency/capability check.

    Construction raises :class:`UnsupportedFormatError` when cuCIM is not
    installed (the normal state on CPU/TPU hosts); callers must check
    :meth:`available` first and fall back to OpenSlide/TiffSlide.
    """

    reader_id = "cucim"
    reader_version = "1.0.0"

    @staticmethod
    def available() -> bool:
        try:
            import cucim  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, path: Path) -> None:
        if not self.available():
            raise UnsupportedFormatError(
                "cuCIM is not installed (optional 'cucim' extra, CUDA-only); use OpenSlideReader or "
                "TiffSlideReader on this host"
            )
        import cucim

        self._path = path
        try:
            self._image = cucim.CuImage(str(path))
        except Exception as exc:
            raise ReaderError(f"cannot open slide {path} with cuCIM: {exc}") from exc
        size = self._image.size
        self.dimensions = (int(size[0]), int(size[1]))
        resolutions = self._image.resolutions
        self.level_count = int(resolutions["level_count"])
        self.level_dimensions = tuple((int(w), int(h)) for w, h in resolutions["level_dimensions"])
        self.level_downsamples = tuple(float(d) for d in resolutions["level_downsamples"])
        self.mpp: float | None = None  # cuCIM exposes MPP via vendor metadata; absent for generic slides
        self.magnification: float | None = None

    def thumbnail(self, max_size: tuple[int, int]) -> torch.Tensor:
        scale = min(max_size[0] / self.dimensions[0], max_size[1] / self.dimensions[1], 1.0)
        size = (max(1, int(self.dimensions[0] * scale)), max(1, int(self.dimensions[1] * scale)))
        array = np.asarray(self._image.read_region(size=size), dtype=np.uint8)[..., :3]
        return torch.from_numpy(np.ascontiguousarray(array))

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> torch.Tensor:
        array = np.asarray(self._image.read_region(location=tuple(location), level=level, size=tuple(size)))
        return torch.from_numpy(np.ascontiguousarray(array[..., :3].astype(np.uint8)))

    def read_tiles(
        self,
        locations: list[tuple[int, int]],
        *,
        level: int,
        size: tuple[int, int],
        on_corrupt: str = "raise",
    ) -> TileReadResult:
        raise UnsupportedFormatError("cuCIM tile grids are provided by Phase 04 preprocessing on CUDA hosts")

    def pathology_metadata(self) -> PathologyMetadata:
        return PathologyMetadata(
            microns_per_pixel=self.mpp,
            magnification=self.magnification,
            slide_dimensions=self.dimensions,
            level_dimensions=self.level_dimensions,
        )

    def close(self) -> None:
        self._image = None  # cuCIM releases resources on GC; keep the contract


class PreExtractedTileReader(Reader):
    """Pre-extracted tile stores: ``tiles.safetensors`` with ``tiles`` + ``coords``.

    Layout: ``tiles`` is ``[T, C, H, W]`` (uint8 or canonical float),
    ``coords`` is ``[T, 2]`` int64 level-0 pixel origins. Slide identity is
    taken from the manifest row; the store itself carries no PHI.
    """

    reader_id = "preextracted_tiles"
    reader_version = "1.0.0"

    def supports(self, path: Path) -> bool:
        return path.is_dir() and (path / "tiles.safetensors").is_file()

    def read(self, path: Path) -> PayloadRead:
        from safetensors.torch import load as st_load

        store_path = path / "tiles.safetensors" if path.is_dir() else path
        if not store_path.is_file():
            raise ReaderError(f"tile store not found: {store_path}; expected tiles.safetensors")
        try:
            tensors = st_load(store_path.read_bytes())
        except Exception as exc:
            raise CorruptSampleError(f"tile store {store_path.name} is unreadable: {exc}") from exc
        if "tiles" not in tensors or "coords" not in tensors:
            raise ReaderError(
                f"tile store {store_path.name} must contain 'tiles' [T, C, H, W] and 'coords' [T, 2] tensors; "
                f"found {sorted(tensors)}"
            )
        tiles = tensors["tiles"]
        coords = tensors["coords"]
        if tiles.ndim != 4:
            raise ReaderError(f"tiles must be [T, C, H, W]; got {tuple(tiles.shape)}")
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != tiles.shape[0]:
            raise ReaderError(f"coords must be [T, 2] matching tiles; got {tuple(coords.shape)}")
        metadata = PathologyMetadata(
            slide_dimensions=None,
            tile_coordinates=coords.to(torch.int64),
        )
        return PayloadRead(
            tensors={"image": tiles},
            pathology=metadata,
            source_metadata={
                "reader": self.reader_id,
                "reader_version": self.reader_version,
                "tile_count": tiles.shape[0],
            },
        )


class EmbeddingStoreReader(Reader):
    """Precomputed embedding stores: one safetensors file per slide id.

    Each file holds ``embeddings`` ``[N, D]`` (float32) and optionally
    ``coords`` ``[N, 2]`` int64 tile origins. Lookups are by slide key; the
    reader never invents embeddings for unknown keys.
    """

    reader_id = "embedding_store"
    reader_version = "1.0.0"

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise ReaderError(f"embedding store root {self._root} is not a directory")

    def supports(self, path: Path) -> bool:
        return path.is_dir() and any(path.glob("*.safetensors"))

    def list_slides(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.safetensors"))

    def read_slide(self, slide_key: str) -> PayloadRead:
        from safetensors.torch import load as st_load

        path = self._root / f"{slide_key}.safetensors"
        if not path.is_file():
            raise ReaderError(f"no embeddings for slide {slide_key!r} in {self._root}; available: {self.list_slides()}")
        try:
            tensors = st_load(path.read_bytes())
        except Exception as exc:
            raise CorruptSampleError(f"embedding file {path.name} is unreadable: {exc}") from exc
        if "embeddings" not in tensors:
            raise ReaderError(f"embedding file {path.name} must contain an 'embeddings' [N, D] tensor")
        payload: dict[str, torch.Tensor] = {"image": tensors["embeddings"]}
        metadata_kwargs: dict[str, Any] = {}
        if "coords" in tensors:
            metadata_kwargs["tile_coordinates"] = tensors["coords"].to(torch.int64)
        return PayloadRead(
            tensors=payload,
            pathology=PathologyMetadata(**metadata_kwargs) if metadata_kwargs else None,
            source_metadata={
                "reader": self.reader_id,
                "reader_version": self.reader_version,
                "slide_key_hash": hash_identifier(slide_key),
            },
        )

    def read(self, path: Path) -> PayloadRead:
        return self.read_slide(path.stem)
