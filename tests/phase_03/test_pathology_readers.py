"""Slide-reader contract: levels, MPP, regions, tiles, coordinate conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from synthetic import write_embedding_store, write_pyramid_tiff, write_tile_store

from medfm.data.errors import ReaderError, UnsupportedFormatError
from medfm.data.readers.pathology import (
    CuCIMSlideReader,
    EmbeddingStoreReader,
    OpenSlideReader,
    PreExtractedTileReader,
    TiffSlideReader,
    convert_level_coords,
    validate_level_coords,
)


@pytest.fixture()
def slide(tmp_path: Path) -> tuple[Path, np.ndarray]:
    path = tmp_path / "slide.tif"
    base = write_pyramid_tiff(path, size=(512, 512), levels=3, mpp=0.5, seed=11)
    return path, base


def test_tiffslide_exposes_contract_geometry(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = TiffSlideReader(path)
    assert reader.dimensions == (512, 512)
    assert reader.level_count == 3
    assert reader.level_dimensions == ((512, 512), (256, 256), (128, 128))
    assert reader.level_downsamples == (1.0, 2.0, 4.0)
    assert reader.mpp == pytest.approx(0.5)
    metadata = reader.pathology_metadata()
    assert metadata.microns_per_pixel == pytest.approx(0.5)
    assert metadata.slide_dimensions == (512, 512)
    reader.close()


def test_read_region_matches_level0_pixels(slide: tuple[Path, np.ndarray]) -> None:
    path, base = slide
    reader = TiffSlideReader(path)
    region = reader.read_region((10, 20), 0, (32, 48)).numpy()
    assert np.array_equal(region, base[20:68, 10:42, :])  # location is (x, y)
    reader.close()


def test_thumbnail_is_bounded(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = TiffSlideReader(path)
    thumb = reader.thumbnail((64, 64))
    assert thumb.ndim == 3 and thumb.shape[2] == 3
    assert thumb.shape[0] <= 64 and thumb.shape[1] <= 64
    reader.close()


def test_read_tiles_returns_chw_with_level0_coords(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = TiffSlideReader(path)
    result = reader.read_tiles([(0, 0), (64, 64)], level=0, size=(32, 32))
    assert tuple(result.tiles.shape) == (2, 3, 32, 32)
    assert result.coords.tolist() == [[0, 0], [64, 64]]
    assert result.errors == ()
    reader.close()


def test_openslide_backend_opens_same_file(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = OpenSlideReader(path)
    assert reader.dimensions == (512, 512)
    assert reader.level_count >= 1
    region = reader.read_region((0, 0), 0, (16, 16))
    assert tuple(region.shape) == (16, 16, 3)
    reader.close()


def test_cucim_reader_is_capability_gated() -> None:
    if CuCIMSlideReader.available():
        pytest.skip("cuCIM installed; capability-gated absence test does not apply")
    with pytest.raises(UnsupportedFormatError, match="cuCIM"):
        CuCIMSlideReader(Path("does-not-matter.svs"))


def test_convert_level_coords_round_trips() -> None:
    downsamples = (1.0, 2.0, 4.0)
    coords = torch.tensor([[100.0, 200.0], [40.0, 80.0]], dtype=torch.float64)
    level2 = convert_level_coords(coords, from_level=0, to_level=2, level_downsamples=downsamples)
    assert torch.allclose(level2, coords / 4.0)
    back = convert_level_coords(level2, from_level=2, to_level=0, level_downsamples=downsamples)
    assert torch.allclose(back, coords)


def test_validate_level_coords_rejects_out_of_bounds() -> None:
    dims = ((512, 512), (256, 256))
    good = torch.tensor([[0, 0], [480, 480]])
    validate_level_coords(good, level=0, level_dimensions=dims, region_size=(32, 32))
    bad = torch.tensor([[500, 0]])  # 500 + 32 > 512
    with pytest.raises(ReaderError, match="exceed level 0 dimensions"):
        validate_level_coords(bad, level=0, level_dimensions=dims, region_size=(32, 32))
    bad_level = torch.tensor([[0, 0]])
    with pytest.raises(ReaderError, match="out of range"):
        validate_level_coords(bad_level, level=5, level_dimensions=dims)


def test_invalid_level_is_actionable(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = TiffSlideReader(path)
    with pytest.raises(ReaderError, match="out of range"):
        reader.read_region((0, 0), 9, (16, 16))
    reader.close()


def test_preextracted_tile_reader(tmp_path: Path) -> None:
    store = tmp_path / "tiles"
    tiles, coords = write_tile_store(store, tile_count=5, seed=2)
    reader = PreExtractedTileReader()
    assert reader.supports(store)
    read = reader.read(store)
    assert torch.equal(read.image, tiles)
    assert read.pathology.tile_coordinates.shape == (5, 2)
    with pytest.raises(ReaderError, match="tiles.safetensors"):
        reader.read(tmp_path / "missing")


def test_embedding_store_reader(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    write_embedding_store(root, "slideA", embeddings=4, dim=8, seed=3)
    reader = EmbeddingStoreReader(root)
    assert reader.list_slides() == ["slideA"]
    read = reader.read_slide("slideA")
    assert tuple(read.image.shape) == (4, 8)
    assert read.pathology.tile_coordinates.shape == (4, 2)
    with pytest.raises(ReaderError, match="no embeddings for slide"):
        reader.read_slide("slideZ")


def test_corrupt_region_is_recoverable_at_tile_scope(slide: tuple[Path, np.ndarray]) -> None:
    path, _ = slide
    reader = TiffSlideReader(path)
    original = reader.read_region

    def flaky(location: tuple[int, int], level: int, size: tuple[int, int]) -> torch.Tensor:
        if location == (64, 64):
            from medfm.data.errors import CorruptSampleError

            raise CorruptSampleError("simulated corrupt tile")
        return original(location, level, size)

    reader.read_region = flaky  # type: ignore[method-assign]
    result = reader.read_tiles([(0, 0), (64, 64), (128, 128)], level=0, size=(32, 32), on_corrupt="skip")
    assert tuple(result.tiles.shape) == (2, 3, 32, 32)  # corrupt tile dropped
    assert len(result.errors) == 1 and "simulated corrupt tile" in result.errors[0]
    assert result.coords.tolist() == [[0, 0], [128, 128]]
    reader.close()
