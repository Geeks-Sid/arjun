"""3D patch samplers: determinism, positivity, padding, coverage (Phase 04)."""

from __future__ import annotations

import pytest
import torch

from medfm.data.errors import PatchSamplingError
from medfm.data.samplers import (
    PADDING_INDEX,
    BoxPatchSampler,
    ClassBalancedPatchSampler,
    ForegroundPatchSampler,
    GridPatchSampler,
    GroupAwareDistributedSampler,
    LesionCenteredPatchSampler,
    PatchInfo,
    RandomPatchSampler,
    extract_patch,
)

PATCH = (8, 8, 8)
VOL = (32, 32, 32)


def _volume(shape: tuple[int, ...] = (1, *VOL)) -> torch.Tensor:
    image = torch.zeros(shape)
    for i in range(shape[-1]):
        image[..., i] = float(i)  # non-uniform content so extraction offsets are observable
    return image


def _block_mask(block: slice = slice(14, 18), shape: tuple[int, int, int] = VOL) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.int64)
    mask[block, block, block] = 1
    return mask


def _origins(sampler, image: torch.Tensor, draws: int, mask: torch.Tensor | None = None) -> list[tuple[int, int, int]]:
    return [sampler.sample(image, mask=mask).info.origin for _ in range(draws)]


def test_random_sampler_deterministic_same_seed() -> None:
    image = _volume()
    first = RandomPatchSampler(PATCH, seed=123)
    second = RandomPatchSampler(PATCH, seed=123)
    origins_a = _origins(first, image, 16)
    origins_b = _origins(second, image, 16)
    assert origins_a == origins_b
    patch_a = RandomPatchSampler(PATCH, seed=123).sample(image)
    patch_b = RandomPatchSampler(PATCH, seed=123).sample(image)
    assert torch.equal(patch_a.image, patch_b.image)


def test_random_sampler_different_seeds_differ() -> None:
    image = _volume()
    origins_a = _origins(RandomPatchSampler(PATCH, seed=1), image, 16)
    origins_b = _origins(RandomPatchSampler(PATCH, seed=2), image, 16)
    assert origins_a != origins_b


def test_random_sampler_accepts_caller_generator() -> None:
    image = _volume()
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    origins_a = _origins(RandomPatchSampler(PATCH, seed=generator_a), image, 8)
    origins_b = _origins(RandomPatchSampler(PATCH, seed=generator_b), image, 8)
    assert origins_a == origins_b


def test_foreground_positive_ratio_is_measurable() -> None:
    image = _volume()
    mask = _block_mask()
    sampler = ForegroundPatchSampler(PATCH, positive_ratio=0.7, seed=11)
    trials = 600
    patches = [sampler.sample(image, mask=mask) for _ in range(trials)]
    # target_positive is the realized overlap, checked against the mask patch.
    for patch in patches:
        assert patch.mask is not None
        assert patch.info.target_positive == bool((patch.mask > 0).any().item())
        assert patch.info.sampling_probability in (pytest.approx(0.7), pytest.approx(0.3))
    measured = sum(p.info.target_positive for p in patches) / trials
    # Expect 0.7 (foreground draws always overlap) + 0.3 * P(uniform overlap).
    assert 0.67 <= measured <= 0.78


def test_foreground_centered_draws_hit_the_block() -> None:
    image = _volume()
    mask = _block_mask()
    sampler = ForegroundPatchSampler(PATCH, positive_ratio=1.0, seed=5)
    for _ in range(50):
        patch = sampler.sample(image, mask=mask)
        assert patch.info.target_positive
        assert patch.info.sampling_probability == 1.0
        # Center of the patch lands on a foreground voxel.
        center = tuple(patch.info.origin[a] + PATCH[a] // 2 for a in range(3))
        assert mask[center] == 1


def test_foreground_without_foreground_raises() -> None:
    image = _volume()
    mask = torch.zeros(VOL, dtype=torch.int64)
    sampler = ForegroundPatchSampler(PATCH, positive_ratio=1.0, seed=3)
    with pytest.raises(PatchSamplingError, match="no foreground"):
        sampler.sample(image, mask=mask)


def test_foreground_without_mask_degrades_to_uniform() -> None:
    image = _volume()
    sampler = ForegroundPatchSampler(PATCH, positive_ratio=0.9, seed=3)
    patch = sampler.sample(image)
    assert patch.mask is None
    assert not patch.info.target_positive
    assert patch.info.sampling_probability == 1.0


def test_invalid_configs_raise() -> None:
    with pytest.raises(PatchSamplingError, match="patch_shape"):
        RandomPatchSampler((8, -1, 8))
    with pytest.raises(PatchSamplingError, match="patch_shape"):
        RandomPatchSampler((8, 8))  # type: ignore[arg-type]
    with pytest.raises(PatchSamplingError, match="positive_ratio"):
        ForegroundPatchSampler(PATCH, positive_ratio=1.5)
    with pytest.raises(PatchSamplingError, match="positive_ratio"):
        ForegroundPatchSampler(PATCH, positive_ratio=-0.1)
    with pytest.raises(PatchSamplingError, match="box"):
        BoxPatchSampler(PATCH, box=(4, 4, 4, 8, 8, 2))  # max < min on W
    with pytest.raises(PatchSamplingError, match="overlap"):
        GridPatchSampler(PATCH, overlap=1.0)
    with pytest.raises(PatchSamplingError, match="classes"):
        ClassBalancedPatchSampler(PATCH, classes=())
    with pytest.raises(PatchSamplingError, match="jitter"):
        LesionCenteredPatchSampler(PATCH, jitter_voxels=-1)


def test_smaller_than_patch_volume_has_explicit_padding() -> None:
    image = _volume((1, 4, 5, 6))
    mask = torch.zeros((4, 5, 6), dtype=torch.int64)
    mask[1:3, 1:3, 1:3] = 1
    sampler = RandomPatchSampler(PATCH, seed=9)
    patch = sampler.sample(image, mask=mask, pad_value=-1.0)
    info = patch.info
    assert info.origin == (0, 0, 0)
    assert info.original_shape == (4, 5, 6)
    assert info.padded
    assert info.padding == (4, 3, 2)
    assert tuple(patch.image.shape) == (1, *PATCH)
    assert patch.mask is not None and tuple(patch.mask.shape) == PATCH
    # Real voxels land in the leading corner; pad_value fills the rest.
    assert torch.equal(patch.image[:, :4, :5, :6], image)
    assert bool((patch.image[:, 4:, :, :] == -1.0).all())
    assert bool((patch.image[:, :, 5:, :] == -1.0).all())
    assert bool((patch.image[:, :, :, 6:] == -1.0).all())
    # Mask padding is zeros; positivity reflects only real voxels.
    assert bool((patch.mask[4:, :, :] == 0).all())
    assert info.target_positive


def test_origins_clamped_so_patch_stays_full_shape() -> None:
    image = _volume()
    sampler = RandomPatchSampler((16, 16, 16), seed=21)
    for _ in range(20):
        patch = sampler.sample(image)
        assert not patch.info.padded
        assert tuple(patch.image.shape) == (1, 16, 16, 16)
        for axis in range(3):
            assert 0 <= patch.info.origin[axis] <= VOL[axis] - 16


def test_grid_sampler_covers_volume_exactly() -> None:
    shape = (17, 19, 23)
    image = _volume((1, *shape))
    sampler = GridPatchSampler(PATCH, overlap=0.25)
    patches = list(sampler.iter_patches(image))
    covered = torch.zeros(shape, dtype=torch.bool)
    for patch in patches:
        assert tuple(patch.image.shape) == (1, *PATCH)
        assert not patch.info.padded
        origin = patch.info.origin
        for axis in range(3):
            assert origin[axis] + PATCH[axis] <= shape[axis]
        covered[
            origin[0] : origin[0] + PATCH[0],
            origin[1] : origin[1] + PATCH[1],
            origin[2] : origin[2] + PATCH[2],
        ] = True
    assert bool(covered.all()), "grid must cover every voxel"


def test_grid_sampler_exact_tiling_without_overlap() -> None:
    sampler = GridPatchSampler(PATCH)
    origins = sampler.origins_for((16, 16, 16))
    assert len(origins) == 8
    assert len(set(origins)) == 8  # no duplicates on an exact multiple


def test_grid_sampler_deterministic_without_rng() -> None:
    shape = (17, 19, 23)
    first = GridPatchSampler(PATCH, overlap=0.5).origins_for(shape)
    second = GridPatchSampler(PATCH, overlap=0.5).origins_for(shape)
    assert first == second
    image = _volume((1, *shape))
    walker = GridPatchSampler(PATCH, overlap=0.5)
    walked = [walker.sample(image).info.origin for _ in range(len(first))]
    assert walked == list(first)


def test_lesion_sampler_centers_on_components() -> None:
    image = _volume()
    mask = torch.zeros(VOL, dtype=torch.int64)
    mask[4:8, 4:8, 4:8] = 1
    mask[20:26, 20:26, 20:26] = 1
    sampler = LesionCenteredPatchSampler(PATCH, seed=13)
    centers = set()
    for _ in range(40):
        patch = sampler.sample(image, mask=mask)
        center = tuple(patch.info.origin[a] + PATCH[a] // 2 for a in range(3))
        centers.add(center)
        assert patch.info.target_positive
        assert patch.info.sampling_probability == 0.5
    # Centroids are (5.5, ...) and (22.5, ...) -> rounded to 6 and 22.
    assert centers == {(6, 6, 6), (22, 22, 22)}


def test_lesion_sampler_jitter_is_bounded_and_deterministic() -> None:
    image = _volume()
    mask = _block_mask()
    origins_a = _origins(LesionCenteredPatchSampler(PATCH, jitter_voxels=3, seed=17), image, 16, mask)
    origins_b = _origins(LesionCenteredPatchSampler(PATCH, jitter_voxels=3, seed=17), image, 16, mask)
    assert origins_a == origins_b
    for origin in origins_a:
        center = tuple(origin[a] + PATCH[a] // 2 for a in range(3))
        for axis in range(3):
            assert abs(center[axis] - 16) <= 3  # block 14:18 -> centroid 15.5 -> rounds to 16


def test_lesion_sampler_requires_lesions() -> None:
    image = _volume()
    mask = torch.zeros(VOL, dtype=torch.int64)
    sampler = LesionCenteredPatchSampler(PATCH, seed=3)
    with pytest.raises(PatchSamplingError, match="no connected components"):
        sampler.sample(image, mask=mask)


def test_box_sampler_stays_inside_box() -> None:
    image = _volume()
    box = (4, 8, 12, 20, 24, 28)
    sampler = BoxPatchSampler(PATCH, box=box, seed=19)
    for _ in range(50):
        origin = sampler.sample(image).info.origin
        for axis in range(3):
            assert box[axis] <= origin[axis]
            assert origin[axis] + PATCH[axis] <= box[axis + 3]


def test_box_sampler_outside_volume_raises() -> None:
    image = _volume()
    sampler = BoxPatchSampler(PATCH, box=(0, 0, 0, 40, 40, 40), seed=3)
    with pytest.raises(PatchSamplingError, match="exceeds volume"):
        sampler.sample(image)


def test_class_balanced_sampler_visits_each_class() -> None:
    image = _volume()
    mask = torch.zeros(VOL, dtype=torch.int64)
    mask[2:6, 2:6, 2:6] = 1
    mask[24:30, 24:30, 24:30] = 2
    sampler = ClassBalancedPatchSampler(PATCH, classes=(1, 2), seed=23)
    visited: set[int] = set()
    for draw in range(10):
        patch = sampler.sample(image, mask=mask)
        center = tuple(patch.info.origin[a] + PATCH[a] // 2 for a in range(3))
        cls = int(mask[center])
        visited.add(cls)
        assert cls == (1 if draw % 2 == 0 else 2)  # round-robin cursor
        assert patch.info.sampling_probability == 0.5
    assert visited == {1, 2}


def test_class_balanced_missing_class_raises() -> None:
    image = _volume()
    mask = _block_mask()
    sampler = ClassBalancedPatchSampler(PATCH, classes=(1, 2), seed=3)
    sampler.sample(image, mask=mask)  # class 1 present
    with pytest.raises(PatchSamplingError, match="class 2"):
        sampler.sample(image, mask=mask)


def test_physical_bounding_box_from_spacing() -> None:
    image = _volume()
    # Box forces the origin exactly: box_max - box_min == patch on every axis.
    box = (4, 2, 10, 12, 10, 18)
    sampler = BoxPatchSampler(PATCH, box=box, seed=3)
    spacing = (2.0, 1.5, 0.5)
    patch = sampler.sample(image, spacing_mm=spacing)
    assert patch.info.origin == (4, 2, 10)
    assert patch.info.physical_min_mm == (8.0, 3.0, 5.0)
    assert patch.info.physical_max_mm == (24.0, 15.0, 9.0)


def test_physical_bounding_box_none_without_spacing() -> None:
    image = _volume()
    patch = RandomPatchSampler(PATCH, seed=3).sample(image)
    assert patch.info.physical_min_mm is None
    assert patch.info.physical_max_mm is None


def test_extract_patch_pads_via_info_metadata() -> None:
    image = _volume((1, 4, 4, 4))
    info = PatchInfo(
        origin=(0, 0, 0),
        patch_shape=PATCH,
        original_shape=(4, 4, 4),
        target_positive=False,
        sampling_probability=1.0,
        padded=True,
        padding=(4, 4, 4),
    )
    patch = extract_patch(image, info, pad_value=5.0)
    assert tuple(patch.shape) == (1, *PATCH)
    assert torch.equal(patch[:, :4, :4, :4], image)
    assert bool((patch[:, 4:, :, :] == 5.0).all())


def test_phase03_names_still_reexported() -> None:
    # tests/phase_03/test_samplers.py exercises the behavior; here we pin the
    # package-level re-export surface the Phase 03 imports rely on.
    import medfm.data.samplers as samplers

    assert samplers.PADDING_INDEX == PADDING_INDEX == -1
    assert samplers.GroupAwareDistributedSampler is GroupAwareDistributedSampler
    for name in (
        "SamplerShard",
        "ResolvedSamples",
        "combine_shards_for_metrics",
        "resolve_samples_before_collective",
        "worker_seed",
        "worker_init_fn",
    ):
        assert hasattr(samplers, name)
