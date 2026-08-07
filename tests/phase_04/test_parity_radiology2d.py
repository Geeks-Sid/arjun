"""Parity checks for torchvision-backed radiology transform kernels."""

from __future__ import annotations

import math

import torch
from torchvision.transforms import functional as TVF

from medfm.data.transforms.base import TransformContext, TransformData
from medfm.data.transforms.radiology2d import RandomFlip2D, _affine_resample


def _ctx(seed: int) -> TransformContext:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return TransformContext(rng=rng, seed=seed)


def test_torchvision_flip_kernels_preserve_float32_values() -> None:
    image = torch.rand((3, 11, 17), generator=torch.Generator().manual_seed(101))

    assert torch.equal(TVF.hflip(image), torch.flip(image, dims=(-1,)))
    assert torch.equal(TVF.vflip(image), torch.flip(image, dims=(-2,)))
    assert TVF.hflip(image).dtype == image.dtype
    assert TVF.vflip(image).dtype == image.dtype


def test_random_flip_uses_context_rng_and_torchvision_kernels() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)
    seed = 23
    probability = 0.5

    expected_rng = torch.Generator(device="cpu")
    expected_rng.manual_seed(seed)
    expected_h = bool(torch.rand((), generator=expected_rng) < probability)
    expected_v = bool(torch.rand((), generator=expected_rng) < probability)
    expected = image
    if expected_h:
        expected = TVF.hflip(expected)
    if expected_v:
        expected = TVF.vflip(expected)

    data = RandomFlip2D(True, allow_vertical=True, p=probability)(
        TransformData(image=image.clone(), metadata={}), _ctx(seed)
    )

    assert torch.equal(data.image, expected)
    assert data.history[-1].params["flipped_horizontal"] is expected_h
    assert data.history[-1].params["flipped_vertical"] is expected_v


def test_torchvision_affine_candidate_does_not_match_contract_kernel() -> None:
    """The torchvision affine parameterization has measurable contract drift."""
    image = torch.arange(1, 26, dtype=torch.float32).reshape(1, 5, 5)
    angle = 17.0
    radians = math.radians(angle)
    theta = torch.tensor(
        [[math.cos(radians), -math.sin(radians), 0.0], [math.sin(radians), math.cos(radians), 0.0]],
        dtype=torch.float32,
    )

    contract = _affine_resample(image, theta)
    candidate = TVF.affine(
        image,
        angle,
        [0, 0],
        1.0,
        [0.0, 0.0],
        interpolation=TVF.InterpolationMode.BILINEAR,
        fill=0.0,
    )

    drift = float((contract - candidate).abs().max())
    assert drift > 1e-3
