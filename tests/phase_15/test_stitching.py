from __future__ import annotations

import json

import pytest
import torch

from medfm.recipes.pathology_stitching import (
    COORDINATE_SYSTEM,
    TilePrediction,
    evidence_payload,
    evidence_tiles_from_scores,
    map_normalized_coordinates_to_wsi,
    normalized_to_level0_geometry,
    serialize_evidence_json,
    stitch_tile_predictions,
    validate_evidence_json,
)


def test_stitching_blends_overlap_and_reports_missing_tiles() -> None:
    predictions = [
        TilePrediction("slide-1", "tile-a", torch.ones(4, 4), 0, 0, 4, 4, score=0.2),
        TilePrediction("slide-1", "tile-b", torch.full((4, 4), 3.0), 2, 0, 4, 4, score=0.9),
        {"slide_id": "slide-1", "tile_id": "missing", "logits": None, "x": 0, "y": 0, "width": 4, "height": 4},
    ]
    stitched = stitch_tile_predictions(predictions, (4, 6), blend_mode="gaussian")
    assert stitched.logits.shape == (1, 4, 6)
    assert stitched.coverage_mask[:, 0].all()
    assert stitched.missing_tile_ids == ("missing",)
    assert torch.allclose(stitched.logits[0, :, 0], torch.ones(4), atol=1e-5)
    assert torch.allclose(stitched.logits[0, :, -1], torch.full((4,), 3.0), atol=1e-5)
    assert 0.0 < stitched.covered_fraction <= 1.0


def test_stitching_maps_pyramid_level_to_level0() -> None:
    stitched = stitch_tile_predictions(
        [TilePrediction("slide", "l1", torch.ones(2, 2), 2, 3, 2, 2, level=1)],
        (10, 12),
        level_downsamples={1: 2.0},
    )
    assert stitched.coverage_mask[6:10, 4:8].all()
    assert not stitched.coverage_mask[:6].any()


def test_evidence_coordinates_round_trip_and_rank_deterministically() -> None:
    records = [
        {"slide_id": "slide", "tile_id": "b", "x": 2, "y": 1, "width": 4, "height": 4, "level": 1, "mpp": 0.5},
        {"slide_id": "slide", "tile_id": "a", "x": 0, "y": 0, "width": 4, "height": 4, "level": 0, "mpp": 0.5},
    ]
    rows = evidence_tiles_from_scores(records, [0.4, 0.9], top_k=2, slide_shape=(16, 20), level_downsamples={1: 2.0})
    assert [row["tile_id"] for row in rows] == ["a", "b"]
    assert rows[0]["rank"] == 1
    assert rows[1]["x"] == 4 and rows[1]["y"] == 2
    assert rows[1]["coordinate_system"] == COORDINATE_SYSTEM
    assert normalized_to_level0_geometry(rows[1]["normalized"], (16, 20)) == (4, 2, 8, 8)
    assert map_normalized_coordinates_to_wsi(rows[0]["normalized"], (16, 20))["coordinate_system"] == COORDINATE_SYSTEM


def test_evidence_json_is_phi_safe_and_rejects_invalid_geometry() -> None:
    payload = evidence_payload(
        [{"tile_id": "tile-1", "x": 0, "y": 0, "width": 4, "height": 4, "coordinate_system": COORDINATE_SYSTEM}],
        slide_id="deidentified-slide-1",
        slide_shape=(8, 8),
        recipe_id="phase15-test",
    )
    serialized = serialize_evidence_json(payload)
    decoded = json.loads(serialized)
    assert "patient" not in serialized.lower()
    assert validate_evidence_json(decoded) == []
    invalid = {**decoded, "tiles": [{"tile_id": "bad", "x": 7, "y": 0, "width": 4, "height": 4}]}
    assert validate_evidence_json(invalid) == ["evidence tile 'bad' escapes slide bounds"]
    assert validate_evidence_json({"schema_version": "bad", "tiles": None})


def test_invalid_stitching_geometry_fails_fast() -> None:
    with pytest.raises(ValueError, match="positive"):
        stitch_tile_predictions([], (0, 4))
    with pytest.raises(ValueError, match="level downsample"):
        stitch_tile_predictions(
            [TilePrediction("slide", "tile", torch.ones(2, 2), 0, 0, 2, 2, level=1)],
            (4, 4),
            level_downsamples={1: 0},
        )
