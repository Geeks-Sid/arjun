"""Parity checks for pathology kernels against candidate library implementations."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import laplace
from skimage.filters import threshold_otsu

from medfm.data.errors import TransformError
from medfm.data.transforms.pathology import _grayscale, _otsu_threshold, blur_score


def test_otsu_fixed_histogram_matches_skimage_on_discrete_fixture() -> None:
    """The fixed 256-bin implementation agrees when skimage uses the same bins."""
    values = np.repeat(np.arange(256, dtype=np.float64) / 255.0, 3)

    ours = _otsu_threshold(values)
    candidate = float(threshold_otsu(values))

    assert ours == candidate == 0.498046875


def test_otsu_constant_fixture_exposes_skimage_histogram_drift() -> None:
    """Constant-channel behavior is part of the tissue-mask contract."""
    values = np.full(64, 0.5, dtype=np.float64)

    ours = _otsu_threshold(values)
    candidate = float(threshold_otsu(values))

    assert ours == 0.001953125
    assert candidate == 0.5
    assert ours != candidate


def test_otsu_random_fixture_exposes_skimage_histogram_drift() -> None:
    """A deterministic non-discrete fixture also differs in the candidate kernel."""
    values = np.random.default_rng(0).random(1000)

    ours = _otsu_threshold(values)
    candidate = float(threshold_otsu(values))

    assert ours == pytest.approx(0.501953125)
    assert candidate == pytest.approx(0.5057010168773981)
    assert ours != candidate


def test_blur_score_contract_and_edge_handling_parity() -> None:
    """The custom interior stencil preserves contracts that scipy's edge policy changes."""
    with pytest.raises(TransformError, match="at least 3x3"):
        blur_score(np.zeros((2, 3, 3), dtype=np.uint8))

    constant = np.full((5, 5, 3), 128, dtype=np.uint8)
    assert blur_score(constant) == 0.0

    tile = np.zeros((5, 5, 3), dtype=np.uint8)
    tile[0, 0, :] = 255
    custom = blur_score(tile)
    candidate = float(laplace(_grayscale(tile)).var())

    assert custom == 0.0
    assert candidate > custom
