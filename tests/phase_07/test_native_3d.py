"""Phase 07 generic native-volume contract tests."""

from __future__ import annotations

import tempfile

import pytest
import torch

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec
from medfm.core.enums import Modality
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.core.sample import SpatialMetadata
from medfm.models.visual import CTFMAdapter, GenericMONAI3DAdapter, LinearHead, MedSAM2Adapter, TriadAdapter
from medfm.models.visual.native_3d import sliding_window_inference


def _metadata(shape: tuple[int, int, int], spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> SpatialMetadata:
    return SpatialMetadata(
        original_shape=shape,
        current_shape=shape,
        affine=torch.diag(torch.tensor((*spacing, 1.0), dtype=torch.float64)),
        spacing_mm=spacing,
        orientation="RAS",
    )


def _batch(volume: torch.Tensor, modality: Modality = Modality.CT_3D) -> MedicalBatch:
    return MedicalBatch(
        modality=modality,
        sample_ids=[f"sample-{i}" for i in range(volume.shape[0])],
        pixel_values=volume,
        spatial_metadata=[_metadata(tuple(volume.shape[-3:])) for _ in range(volume.shape[0])],
    )


def test_generic_returns_volume_tokens_maps_and_physical_coordinates() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.eval()
    output = adapter.encode(
        _batch(torch.randn(2, 1, 16, 16, 16)),
        output_spec=OutputSpec(pooled=True, spatial_tokens=True, feature_maps=True, token_coordinates=True),
    )
    assert output.pooled_embedding is not None and output.pooled_embedding.shape == (2, 32)
    assert output.spatial_tokens is not None and output.spatial_tokens.shape == (2, 64, 32)
    assert output.feature_maps is not None and output.feature_maps[-1].shape[-3:] == (4, 4, 4)
    assert output.token_coordinates is not None and output.token_coordinates.shape == (2, 64, 3)
    assert output.auxiliary["flatten_order"].startswith("depth,height,width")


def test_coordinate_grid_follows_flattened_token_order() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.eval()
    output = adapter.encode(
        _batch(torch.zeros(1, 1, 16, 16, 16)),
        output_spec=OutputSpec(spatial_tokens=True, token_coordinates=True),
    )
    assert output.token_coordinates is not None
    coords = output.token_coordinates[0]
    assert torch.equal(coords[0], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.equal(coords[1], torch.tensor([6.0, 2.0, 2.0]))
    assert torch.equal(coords[4], torch.tensor([2.0, 6.0, 2.0]))
    assert torch.equal(coords[16], torch.tensor([2.0, 2.0, 6.0]))


def test_missing_physical_metadata_is_an_explicit_limitation() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.eval()
    batch = MedicalBatch(modality=Modality.CT_3D, sample_ids=["x"], pixel_values=torch.zeros(1, 1, 16, 16, 16))
    with pytest.raises(ShapeContractError, match="MILLIMETERS"):
        adapter.encode(batch, output_spec=OutputSpec(spatial_tokens=True, token_coordinates=True))


def test_axis_and_shape_validation_rejects_transposition() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.eval()
    with pytest.raises(ShapeContractError, match="preprocess mismatch"):
        adapter.encode(_batch(torch.zeros(1, 1, 8, 16, 16)))


def test_sliding_window_reconstructs_asymmetric_identity() -> None:
    volume = torch.arange(1 * 1 * 20 * 18 * 16, dtype=torch.float32).reshape(1, 1, 20, 18, 16)
    output = sliding_window_inference(volume, lambda crop: crop, window_shape=(8, 8, 8), overlap=0.5)
    assert torch.equal(output, volume)


def test_lora_targets_exclude_patch_convolution() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    assert all("patch_embed" not in pattern for pattern in adapter.lora_target_patterns())


def test_head_backward_and_checkpoint_roundtrip() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.freeze_backbone()
    adapter.attach_head(LinearHead(in_features=32, out_features=2))
    adapter.eval()
    batch = _batch(torch.randn(2, 1, 16, 16, 16))
    output = adapter.encode(batch)
    assert output.pooled_embedding is not None
    logits = adapter.head_logits(output.pooled_embedding)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in adapter.attached_head.parameters())
    with tempfile.TemporaryDirectory() as directory:
        adapter.save_checkpoint(directory)
        restored = GenericMONAI3DAdapter.load_checkpoint(
            directory,
            rebuild=lambda config: GenericMONAI3DAdapter(
                model_id=config["model_id"],
                revision=config["revision"],
                preprocess=adapter.preprocess.from_dict(config["preprocess"]),
                hidden_size=config["hidden_size"],
                depth=config["depth"],
                heads=config["heads"],
                feature_map_layers=tuple(config["feature_map_layers"]),
                construction_seed=config["construction_seed"],
                max_full_volume_voxels=config["max_full_volume_voxels"],
            ),
        )
        restored.eval()
        assert torch.allclose(
            adapter.encode(batch).pooled_embedding,
            restored.encode(batch).pooled_embedding,
            atol=2e-6,
        )


def test_medsam2_lifecycle_keeps_sequential_memory_separate() -> None:
    adapter = MedSAM2Adapter.build_tiny()
    adapter.eval()

    batch = _batch(torch.randn(1, 1, 16, 16, 16))
    with pytest.raises(ShapeContractError):
        adapter.encode_image(batch)
    adapter.initialize()
    adapter.encode_image(batch)
    adapter.prompt({"point": [2, 2, 2], "label": 1})
    mask = adapter.decode()
    assert mask.shape == (1, 1, 64)
    assert adapter.memory_state == {"frame_index": 0, "batch": 1, "tokens": 64, "prompt_count": 1}


def test_ct_and_mri_adapters_support_inference_and_backward() -> None:
    for adapter, modality, channels in (
        (CTFMAdapter.build_tiny(), Modality.CT_3D, 1),
        (TriadAdapter.build_tiny(), Modality.MRI_3D, 2),
    ):
        adapter.train()
        adapter.freeze_backbone()
        adapter.attach_head(LinearHead(in_features=32, out_features=2))
        volume = torch.randn(2, channels, 16, 16, 16)
        output = adapter.encode(_batch(volume, modality))
        assert output.pooled_embedding is not None
        adapter.head_logits(output.pooled_embedding).sum().backward()
        assert any(parameter.grad is not None for parameter in adapter.attached_head.parameters())


def test_unsupported_requested_outputs_fail_before_fabrication() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter._capabilities = adapter.capabilities.__class__(
        model_id=adapter.model_id,
        modalities=adapter.capabilities.modalities,
        supports_pooled=True,
        supports_spatial_tokens=True,
        supports_feature_maps=False,
        supports_token_coordinates=False,
    )
    with pytest.raises(UnsupportedCapabilityError):
        adapter.encode(_batch(torch.zeros(1, 1, 16, 16, 16)), output_spec=OutputSpec(feature_maps=True))
