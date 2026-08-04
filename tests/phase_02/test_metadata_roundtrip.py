"""Spatial/pathology metadata serialization round-trips (must be lossless)."""

import torch
from contract_fixtures import make_pathology, make_spatial

from medfm.core import PathologyMetadata, SpatialMetadata, canonical_json, config_hash
from medfm.core.serialization import canonical_yaml


def test_spatial_metadata_roundtrip_lossless():
    original = make_spatial((16, 32, 32))
    restored = SpatialMetadata.from_dict(original.to_dict())
    assert restored.original_shape == original.original_shape
    assert restored.current_shape == original.current_shape
    assert torch.equal(restored.affine, original.affine)
    assert torch.equal(restored.original_affine, original.original_affine)
    assert restored.spacing_mm == original.spacing_mm
    assert restored.orientation == original.orientation
    assert restored.anatomical_axes == original.anatomical_axes
    assert torch.equal(restored.slice_positions_mm, original.slice_positions_mm)
    assert restored.frame_of_reference_hash == original.frame_of_reference_hash


def test_spatial_metadata_roundtrip_through_json_text():
    original = make_spatial((8, 16, 16))
    import json

    payload = json.loads(canonical_json(original.to_dict()))
    restored = SpatialMetadata.from_dict(payload)
    assert restored.to_dict() == original.to_dict()


def test_spatial_metadata_float32_precision_preserved():
    affine = torch.eye(4, dtype=torch.float32) * 1.3333333
    original = SpatialMetadata(
        original_shape=(4, 4, 4), current_shape=(4, 4, 4), affine=affine, spacing_mm=(1.5, 1.5, 4.9999995)
    )
    restored = SpatialMetadata.from_dict(original.to_dict())
    assert restored.affine.dtype == torch.float32
    assert torch.equal(restored.affine, affine)  # bitwise, not allclose
    assert restored.spacing_mm == original.spacing_mm


def test_pathology_metadata_roundtrip_lossless():
    original = make_pathology(num_tiles=6)
    restored = PathologyMetadata.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert torch.equal(restored.tile_coordinates, original.tile_coordinates)
    assert restored.microns_per_pixel == original.microns_per_pixel


def test_pathology_tile_coordinates_exact_integers():
    coords = torch.tensor([[0, 0], [123456789, 987654321], [256, 512]], dtype=torch.int64)
    original = PathologyMetadata(microns_per_pixel=0.2425, tile_coordinates=coords)
    restored = PathologyMetadata.from_dict(original.to_dict())
    assert torch.equal(restored.tile_coordinates, coords)
    assert restored.microns_per_pixel == 0.2425


def test_canonical_serialization_is_deterministic():
    metadata = make_spatial((8, 8, 8)).to_dict()
    assert canonical_json(metadata) == canonical_json(dict(reversed(list(metadata.items()))))
    assert canonical_yaml(metadata) == canonical_yaml(dict(reversed(list(metadata.items()))))
    assert config_hash(metadata) == config_hash(SpatialMetadata.from_dict(metadata).to_dict())
