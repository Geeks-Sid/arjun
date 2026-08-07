"""Synthetic asymmetric CT/MRI fixtures for native 3D contract tests."""

from __future__ import annotations

import pytest
import torch

from medfm.core.sample import SpatialMetadata


@pytest.fixture
def ct_metadata() -> SpatialMetadata:
    return SpatialMetadata(
        original_shape=(8, 12, 16),
        current_shape=(8, 12, 16),
        affine=torch.tensor(
            [[0.0, 0.0, 2.0, 10.0], [0.0, 3.0, 0.0, 20.0], [4.0, 0.0, 0.0, 30.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        spacing_mm=(2.0, 3.0, 4.0),
        orientation="RAS",
    )


@pytest.fixture
def asymmetric_ct() -> torch.Tensor:
    values = torch.arange(8 * 12 * 16, dtype=torch.float32).reshape(1, 1, 8, 12, 16)
    return values


@pytest.fixture
def asymmetric_mri() -> torch.Tensor:
    d = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1, 1)
    h = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 16, 1)
    return torch.cat((d.expand(1, 1, 16, 16, 16), (h + 10).expand(1, 1, 16, 16, 16)), dim=1)
