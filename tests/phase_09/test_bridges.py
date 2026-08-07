from __future__ import annotations

import pytest
import torch

from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError
from medfm.models.bridges import (
    CoordinateAwareBridge,
    LinearBridge,
    MLPBridge,
    PerceiverBridge,
    ThreeDCoordinateEncoder,
    TwoDCoordinateEncoder,
    WSICoordinateEncoder,
)


def test_linear_and_mlp_bridges_validate_dimensions_and_preserve_masks() -> None:
    tokens = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    for bridge in (
        LinearBridge(
            source_dim=8,
            target_dim=12,
            output_tokens=4,
            max_input_tokens=4,
            source_modality=Modality.XRAY_2D,
        ),
        MLPBridge(
            source_dim=8,
            target_dim=12,
            output_tokens=4,
            max_input_tokens=4,
            source_modality=Modality.PATHOLOGY_WSI,
        ),
    ):
        result = bridge(tokens, mask)
        assert result.tokens.shape == (2, 4, 12)
        assert torch.equal(result.token_mask, mask)
        assert torch.equal(result.tokens[~mask], torch.zeros_like(result.tokens[~mask]))
        with pytest.raises(ShapeContractError):
            bridge(torch.randn(2, 3, 8), mask[:, :3])


def test_perceiver_uses_fixed_queries_and_ignores_padded_visual_values() -> None:
    bridge = PerceiverBridge(
        source_dim=8,
        target_dim=16,
        output_tokens=4,
        max_input_tokens=5,
        source_modality=Modality.CT_3D,
    ).eval()
    tokens = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, False, False], [False, False, False, False, False]])
    first = bridge(tokens, mask)
    changed = tokens.clone()
    changed[:, 2:] = 1000.0
    second = bridge(changed, mask)
    assert first.tokens.shape == (2, 4, 16)
    assert torch.isfinite(first.tokens).all()
    assert torch.allclose(first.tokens, second.tokens, atol=1e-6)
    assert first.token_mask is not None
    assert first.token_mask[0].all() and not first.token_mask[1].any()


def test_coordinate_encoders_cover_2d_3d_and_wsi_semantics() -> None:
    positions_2d = torch.tensor([[[0.1, 0.2], [0.8, 0.9]]])
    encoder_2d = TwoDCoordinateEncoder(output_dim=8)
    baseline = encoder_2d(positions_2d, {"image_index": 0, "view": 0, "timepoint": 0, "slice_index": 0})
    changed = encoder_2d(positions_2d, {"image_index": 1, "view": 0, "timepoint": 0, "slice_index": 0})
    assert baseline.shape == (1, 2, 8)
    assert not torch.allclose(baseline, changed)

    encoder_3d = ThreeDCoordinateEncoder(output_dim=8)
    result_3d = encoder_3d(
        torch.rand(1, 2, 3),
        {"physical_position": torch.ones(3), "spacing": torch.tensor([1.0, 2.0, 3.0]), "series_index": 2},
    )
    assert result_3d.shape == (1, 2, 8)

    encoder_wsi = WSICoordinateEncoder(output_dim=8)
    result_wsi = encoder_wsi(
        torch.rand(1, 2, 2),
        {"mpp": 0.25, "pyramid_level": 2, "slide_index": 3, "slide_x": 10, "slide_y": 20},
    )
    assert result_wsi.shape == (1, 2, 8)
    assert encoder_wsi.coordinate_system is CoordinateSystem.MICRONS


def test_coordinate_aware_bridge_adds_features_without_changing_budget() -> None:
    base = MLPBridge(
        source_dim=8,
        target_dim=12,
        output_tokens=4,
        max_input_tokens=4,
        source_modality=Modality.MULTI_IMAGE_2D,
    )
    bridge = CoordinateAwareBridge(base, TwoDCoordinateEncoder(output_dim=8))
    tokens = torch.zeros(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    coordinates = torch.rand(1, 4, 2)
    first = bridge(tokens, mask, coordinates=coordinates, coordinate_metadata={"image_index": 0})
    second = bridge(tokens, mask, coordinates=coordinates, coordinate_metadata={"image_index": 1})
    assert first.tokens.shape == (1, 4, 12)
    assert not torch.allclose(first.tokens, second.tokens)
