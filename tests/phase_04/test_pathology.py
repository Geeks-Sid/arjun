"""Tests for medfm.data.transforms.pathology: masks, filters, tile planning, stain ops.

Synthetic slides are numpy arrays only — a pinkish disk on white for tissue,
an all-white slide for the zero-tissue case, gaussian-blurred vs sharp tiles
for focus scoring, and a dark saturated "pen mark" for artifact scoring. No
slide backends (OpenSlide/TiffSlide) are involved anywhere.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from medfm.data.errors import TransformError
from medfm.data.transforms.base import TransformContext, TransformData
from medfm.data.transforms.pathology import (
    ReinhardStainNormalize,
    StainAugment,
    artifact_score,
    blur_score,
    compute_tissue_mask,
    make_thumbnail,
    make_tile_id,
    plan_tiles,
    tissue_fraction,
)
from medfm.data.transforms.pipeline import TransformPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_STATS = {"mean": [60.0, 18.0, -4.0], "std": [12.0, 6.0, 4.0]}
OTHER_REFERENCE_STATS = {"mean": [55.0, 22.0, -8.0], "std": [9.0, 4.0, 3.0]}


def _disk_slide(size: int = 128, radius: int = 30) -> np.ndarray:
    """White slide with a centered pinkish tissue disk."""
    slide = np.full((size, size, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    disk = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    slide[disk] = (230, 150, 170)
    return slide


def _white_slide(size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _sharp_tile(size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _blurry_tile(size: int = 64, seed: int = 0) -> np.ndarray:
    sharp = _sharp_tile(size, seed).astype(np.float64)
    return gaussian_filter(sharp, sigma=(4.0, 4.0, 0.0)).astype(np.uint8)


def _pen_mark_tile(size: int = 64) -> np.ndarray:
    tile = np.full((size, size, 3), (230, 150, 170), dtype=np.uint8)
    tile[8:24, 8:24] = (10, 10, 120)  # dark saturated ink-like block
    return tile


def _tile_data(tile: np.ndarray) -> TransformData:
    tensor = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return TransformData(image=tensor)


# --- thumbnails and tissue masks -------------------------------------------------


def test_thumbnail_downscales_deterministically() -> None:
    slide = _disk_slide(size=128)
    first = make_thumbnail(slide, 32)
    second = make_thumbnail(slide, 32)
    assert first.shape == (32, 32, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_thumbnail_identity_when_within_bounds() -> None:
    slide = _disk_slide(size=64)
    thumb = make_thumbnail(slide, 64)
    assert thumb.shape == slide.shape
    assert np.array_equal(thumb, slide)
    assert thumb is not slide  # defensive copy, not an alias


def test_thumbnail_rejects_invalid_input() -> None:
    with pytest.raises(TransformError, match="max_size"):
        make_thumbnail(_disk_slide(), 0)
    with pytest.raises(TransformError, match="uint8"):
        make_thumbnail(_disk_slide().astype(np.float32), 32)


def test_tissue_mask_detects_blob_not_background() -> None:
    slide = _disk_slide(size=128, radius=30)
    mask = compute_tissue_mask(slide)
    assert mask.dtype == np.bool_
    assert mask.shape == slide.shape[:2]
    assert mask[64, 64]  # disk center is tissue
    assert not mask[2, 2]  # white corner is background
    center_fraction = mask[34:95, 34:95].mean()
    assert center_fraction > 0.5


def test_tissue_mask_all_white_slide_is_all_false() -> None:
    mask = compute_tissue_mask(_white_slide())
    assert not mask.any()


def test_tissue_mask_works_on_thumbnail() -> None:
    thumb = make_thumbnail(_disk_slide(size=128), 32)
    mask = compute_tissue_mask(thumb)
    assert mask.shape == (32, 32)
    assert 0.0 < mask.mean() < 1.0


# --- quality filters --------------------------------------------------------------


def test_tissue_fraction_scores() -> None:
    slide = _disk_slide(size=128)
    mask = compute_tissue_mask(slide)
    tissue_tile = slide[48:80, 48:80]
    tissue_tile_mask = mask[48:80, 48:80]
    assert tissue_fraction(tissue_tile, tissue_tile_mask) == pytest.approx(1.0)
    background_tile = slide[0:32, 0:32]
    background_mask = mask[0:32, 0:32]
    assert tissue_fraction(background_tile, background_mask) == 0.0
    # A background-only tile fails any positive minimum tissue fraction.
    assert tissue_fraction(background_tile, background_mask) < 0.5


def test_tissue_fraction_validates_mask() -> None:
    tile = _sharp_tile(16)
    with pytest.raises(TransformError, match="boolean"):
        tissue_fraction(tile, np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(TransformError, match="does not match"):
        tissue_fraction(tile, np.zeros((8, 8), dtype=bool))


def test_blur_score_sharp_beats_blurry() -> None:
    sharp = blur_score(_sharp_tile())
    blurry = blur_score(_blurry_tile())
    assert sharp > 0.0
    assert blurry < sharp


def test_blur_score_constant_tile_is_zero() -> None:
    assert blur_score(np.full((32, 32, 3), 200, dtype=np.uint8)) == pytest.approx(0.0)


def test_artifact_score_pen_mark() -> None:
    clean = artifact_score(np.full((64, 64, 3), (230, 150, 170), dtype=np.uint8))
    marked = artifact_score(_pen_mark_tile())
    assert marked > clean
    assert marked == pytest.approx(16 * 16 / (64 * 64), abs=0.02)


def test_scores_are_deterministic() -> None:
    tile = _pen_mark_tile()
    mask = compute_tissue_mask(tile)
    assert blur_score(tile) == blur_score(tile)
    assert artifact_score(tile) == artifact_score(tile)
    assert tissue_fraction(tile, mask) == tissue_fraction(tile, mask)


# --- tile planning ------------------------------------------------------------------


def test_mpp_normalization_same_physical_tile_size() -> None:
    mask = np.ones((64, 64), dtype=bool)
    slide_shape = (256, 256)
    coarse = plan_tiles(slide_shape, 0.5, mask, tile_size=64, target_mpp=0.5, min_tissue_fraction=0.5)
    fine = plan_tiles(slide_shape, 0.25, mask, tile_size=64, target_mpp=0.5, min_tissue_fraction=0.5)
    assert coarse and fine
    # Same physical size: 64 px * 0.5 mpp = 32 microns on both slides.
    assert coarse[0].width * 0.5 == pytest.approx(fine[0].width * 0.25)
    # Level-0 pixel sizes differ by the MPP ratio.
    assert fine[0].width == 2 * coarse[0].width
    assert {r.mpp for r in coarse + fine} == {0.5}


def test_plan_tiles_deterministic_twice() -> None:
    slide = _disk_slide(size=128)
    mask = compute_tissue_mask(slide)
    kwargs = dict(tile_size=32, target_mpp=0.5, min_tissue_fraction=0.5, slide_key="slide-a")
    first = plan_tiles((128, 128), 0.5, mask, **kwargs)  # type: ignore[arg-type]
    second = plan_tiles((128, 128), 0.5, mask, **kwargs)  # type: ignore[arg-type]
    assert first == second
    assert [r.tile_id for r in first] == [r.tile_id for r in second]


def test_plan_tiles_deterministic_across_processes() -> None:
    mask = np.ones((64, 64), dtype=bool)
    local = plan_tiles((128, 128), 0.5, mask, tile_size=32, target_mpp=0.5, min_tissue_fraction=0.5, slide_key="s")
    script = (
        "import json; import numpy as np; "
        "from medfm.data.transforms.pathology import plan_tiles; "
        "mask = np.ones((64, 64), dtype=bool); "
        "recs = plan_tiles((128, 128), 0.5, mask, tile_size=32, target_mpp=0.5, "
        "min_tissue_fraction=0.5, slide_key='s'); "
        "print(json.dumps([r.to_dict() for r in recs]))"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    remote = json.loads(result.stdout)
    assert [r.to_dict() for r in local] == remote


def test_plan_tiles_coordinates_within_bounds_property() -> None:
    rng = np.random.default_rng(7)
    for slide_shape in [(97, 131), (64, 64), (33, 200), (128, 41)]:
        mask = rng.random(slide_shape) > 0.4
        records = plan_tiles(slide_shape, 0.33, mask, tile_size=24, target_mpp=0.5, min_tissue_fraction=0.2)
        slide_h, slide_w = slide_shape
        for record in records:
            assert 0 <= record.x
            assert 0 <= record.y
            assert record.x + record.width <= slide_w
            assert record.y + record.height <= slide_h
            assert record.tissue_fraction >= 0.2


def test_plan_tiles_edge_tiles_shifted_in_bounds() -> None:
    # 100 is not a multiple of the 32-px tile side; the last column/row must be
    # clamped back inside the slide instead of escaping it.
    mask = np.ones((100, 100), dtype=bool)
    records = plan_tiles((100, 100), 0.5, mask, tile_size=32, target_mpp=0.5, min_tissue_fraction=0.5)
    assert records
    for record in records:
        assert record.x + record.width <= 100
        assert record.y + record.height <= 100


def test_zero_tissue_slide_yields_empty_plan() -> None:
    mask = compute_tissue_mask(_white_slide(128))
    assert not mask.any()
    records = plan_tiles((128, 128), 0.5, mask, tile_size=32, target_mpp=0.5, min_tissue_fraction=0.2)
    assert records == []


def test_background_only_tiles_fail_min_tissue_fraction() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True  # single tissue block, rest background
    records = plan_tiles((128, 128), 0.5, mask, tile_size=32, target_mpp=0.5, min_tissue_fraction=0.9)
    assert records
    assert all(r.tissue_fraction >= 0.9 for r in records)
    # Every kept tile overlaps the tissue block; corner background tiles are gone.
    assert all(r.x >= 8 and r.y >= 8 for r in records)


def test_max_tiles_caps_in_scan_order() -> None:
    mask = np.ones((64, 64), dtype=bool)
    kwargs = dict(tile_size=16, target_mpp=0.5, min_tissue_fraction=0.5)
    full = plan_tiles((128, 128), 0.5, mask, **kwargs)  # type: ignore[arg-type]
    capped = plan_tiles((128, 128), 0.5, mask, max_tiles=5, **kwargs)  # type: ignore[arg-type]
    assert len(capped) == 5
    assert capped == full[:5]


def test_plan_tiles_rejects_invalid_config() -> None:
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(TransformError, match="slide_mpp"):
        plan_tiles((32, 32), 0.0, mask, tile_size=8, target_mpp=0.5, min_tissue_fraction=0.5)
    with pytest.raises(TransformError, match="min_tissue_fraction"):
        plan_tiles((32, 32), 0.5, mask, tile_size=8, target_mpp=0.5, min_tissue_fraction=1.5)
    with pytest.raises(TransformError, match="tissue_mask"):
        plan_tiles((32, 32), 0.5, np.zeros((16, 16)), tile_size=8, target_mpp=0.5, min_tissue_fraction=0.5)
    with pytest.raises(TransformError, match="max_tiles"):
        plan_tiles((32, 32), 0.5, mask, tile_size=8, target_mpp=0.5, min_tissue_fraction=0.5, max_tiles=-1)


def test_tile_id_deterministic_and_input_sensitive() -> None:
    base = make_tile_id("slide", 0, 0, 64, 64, 0)
    assert base == make_tile_id("slide", 0, 0, 64, 64, 0)
    assert len(base) == 16
    assert base != make_tile_id("slide", 32, 0, 64, 64, 0)
    assert base != make_tile_id("other-slide", 0, 0, 64, 64, 0)
    assert base != make_tile_id("slide", 0, 0, 64, 64, 1)


# --- stain normalization / augmentation (opt-in) ------------------------------------


def test_stain_normalize_config_hash_tracks_reference() -> None:
    first = ReinhardStainNormalize(REFERENCE_STATS)
    second = ReinhardStainNormalize(REFERENCE_STATS)
    other = ReinhardStainNormalize(OTHER_REFERENCE_STATS)
    assert first.config_hash() == second.config_hash()
    assert first.config_hash() != other.config_hash()
    assert first.stage == "deterministic"


def test_stain_normalize_rejects_bad_reference() -> None:
    with pytest.raises(TransformError, match="reference_stats"):
        ReinhardStainNormalize({"mean": [1.0, 2.0], "std": [1.0, 1.0, 1.0]})
    with pytest.raises(TransformError, match="positive"):
        ReinhardStainNormalize({"mean": [1.0, 2.0, 3.0], "std": [1.0, 0.0, 1.0]})


def test_stain_normalize_moves_lab_stats_toward_reference() -> None:
    from skimage.color import rgb2lab

    tile = _disk_slide(size=64, radius=24)[16:48, 16:48]
    before = rgb2lab(tile.astype(np.float64) / 255.0).reshape(-1, 3).mean(axis=0)
    transform = ReinhardStainNormalize(REFERENCE_STATS)
    data = _tile_data(tile)
    result = transform(data, None)
    after = rgb2lab(result.image.permute(1, 2, 0).numpy()).reshape(-1, 3).mean(axis=0)
    reference = np.array(REFERENCE_STATS["mean"])
    assert np.linalg.norm(after - reference) < np.linalg.norm(before - reference)
    # Deterministic stage: ctx=None is fine, output repeatable, history recorded.
    repeat = transform(_tile_data(tile), None)
    assert torch.equal(result.image, repeat.image)
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0
    assert result.history[-1].name == "reinhard_stain_normalize"
    assert result.history[-1].stage == "deterministic"


def test_stain_augment_requires_seeded_context() -> None:
    with pytest.raises(TransformError, match="TransformContext"):
        StainAugment()(_tile_data(_sharp_tile(32)), None)


def test_stain_augment_reproducible_under_fixed_seed() -> None:
    augment = StainAugment(alpha=0.2, beta=0.05)
    first = augment(_tile_data(_sharp_tile(32)), TransformContext.for_sample(123, 0, 0, "tile-1"))
    second = augment(_tile_data(_sharp_tile(32)), TransformContext.for_sample(123, 0, 0, "tile-1"))
    other_key = augment(_tile_data(_sharp_tile(32)), TransformContext.for_sample(123, 0, 0, "tile-2"))
    assert torch.equal(first.image, second.image)
    assert not torch.equal(first.image, other_key.image)
    assert first.image.min() >= 0.0 and first.image.max() <= 1.0
    assert first.history[-1].name == "stain_augment"
    assert first.history[-1].stage == "stochastic"


def test_stain_augment_config_hash_tracks_params() -> None:
    assert StainAugment(alpha=0.1).config_hash() != StainAugment(alpha=0.2).config_hash()
    assert StainAugment(alpha=0.1, beta=0.1).config_hash() == StainAugment(alpha=0.1, beta=0.1).config_hash()


def test_stain_ops_are_opt_in_pipeline_steps() -> None:
    # Stain transforms are never in a default pipeline: they must be passed
    # explicitly, and pipeline stage validation keeps the stochastic one behind
    # the cache boundary.
    normalize = ReinhardStainNormalize(REFERENCE_STATS)
    augment = StainAugment()
    pipeline = TransformPipeline(deterministic=[normalize], stochastic=[augment])
    assert pipeline.deterministic_transforms == (normalize,)
    assert pipeline.stochastic_transforms == (augment,)
    with pytest.raises(TransformError, match="deterministic"):
        TransformPipeline(deterministic=[augment])
    with pytest.raises(TransformError, match="stochastic"):
        TransformPipeline(deterministic=[], stochastic=[normalize])
