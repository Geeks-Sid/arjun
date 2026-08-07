"""Parity checks for sliding-window inference against MONAI candidates."""

from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference as monai_sliding_window_inference
from monai.inferers.utils import compute_importance_map

from medfm.inference.sliding_window import gaussian_importance_map, sliding_window_inference


def test_monai_constant_blending_does_not_match_gaussian_contract() -> None:
    """The repository's Gaussian overlap weighting is observably non-constant."""
    volume = torch.arange(1 * 1 * 8 * 9 * 10, dtype=torch.float32).reshape(1, 1, 8, 9, 10)

    def window_statistic(crop: torch.Tensor) -> torch.Tensor:
        return crop.new_zeros(crop.shape) + crop.mean(dim=(-3, -2, -1), keepdim=True)

    ours = sliding_window_inference(volume, window_statistic, window_shape=(4, 5, 4), overlap=0.5, sw_batch_size=2)
    candidate = monai_sliding_window_inference(
        volume,
        roi_size=(4, 5, 4),
        sw_batch_size=2,
        predictor=window_statistic,
        overlap=0.5,
        mode="constant",
    )

    drift = float((ours - candidate).abs().max())
    assert drift > 1.0
    assert not torch.allclose(ours, candidate, atol=1e-4, rtol=1e-5)


def test_monai_gaussian_importance_map_does_not_match_repo_clamp() -> None:
    """MONAI's minimum-weight clamp changes the map at window edges."""
    ours = gaussian_importance_map((3, 4, 3))
    candidate = compute_importance_map((3, 4, 3), mode="gaussian").unsqueeze(0).unsqueeze(0)

    drift = float((ours - candidate).abs().max())
    assert drift == 0.0009909352520480752
    assert not torch.equal(ours, candidate)


def test_monai_identity_is_exact_with_constant_blending_but_not_contract_proof() -> None:
    """Identity reconstruction alone cannot establish parity for weighted outputs."""
    volume = torch.arange(1 * 1 * 20 * 18 * 16, dtype=torch.float32).reshape(1, 1, 20, 18, 16)
    candidate = monai_sliding_window_inference(
        volume,
        roi_size=(8, 8, 8),
        sw_batch_size=1,
        predictor=lambda crop: crop,
        overlap=0.5,
        mode="constant",
    )

    assert torch.equal(candidate, volume)
