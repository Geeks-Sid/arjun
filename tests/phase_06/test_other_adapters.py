"""H-Optimus-0 and MedGemma vision pathway specific tests."""

import json

import pytest
import torch

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec
from medfm.core.enums import Modality
from medfm.core.errors import UnsupportedCapabilityError
from medfm.models.visual.base import LoRAGateError
from medfm.models.visual.hoptimus0 import HOPTIMUS_PREPROCESS, HOptimus0Adapter
from medfm.models.visual.medgemma_vision import MEDGEMMA_PREPROCESS, MedGemmaVisionAdapter


# ---- H-Optimus ----
def test_hoptimus_frozen_default():
    ho = HOptimus0Adapter.build_tiny()
    assert ho.trainable_backbone_parameters() == 0


def test_hoptimus_lora_gate_blocks():
    ho = HOptimus0Adapter.build_tiny()
    with pytest.raises(LoRAGateError):
        ho.inject_lora()


def test_hoptimus_embedding_cache(tmp_path):
    ho = HOptimus0Adapter.build_tiny()
    ho.set_deterministic_eval()
    pp = ho.preprocess
    batches = [
        MedicalBatch(
            modality=Modality.PATHOLOGY_TILE,
            sample_ids=["a", "b"],
            pixel_values=torch.randn(2, pp.channels, pp.image_size[0], pp.image_size[1]),
        ),
        MedicalBatch(
            modality=Modality.PATHOLOGY_TILE,
            sample_ids=["c"],
            pixel_values=torch.randn(1, pp.channels, pp.image_size[0], pp.image_size[1]),
        ),
    ]
    out_dir = ho.generate_embedding_cache(batches, tmp_path / "cache")
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "embeddings.safetensors").exists()
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["model_id"] == ho.model_id
    assert meta["num_samples"] == 3
    assert meta["sample_ids"] == ["a", "b", "c"]
    from safetensors.torch import load_file

    emb = load_file(str(out_dir / "embeddings.safetensors"))
    assert tuple(emb["pooled_embedding"].shape) == (3, 192)


def test_hoptimus_preprocess_defaults():
    assert HOPTIMUS_PREPROCESS.image_size == (224, 224)
    assert HOPTIMUS_PREPROCESS.patch_size == 14


# ---- MedGemma ----
def test_medgemma_native_connector():
    mg = MedGemmaVisionAdapter.build_tiny()
    assert mg.capabilities.native_visual_connector is True


def test_medgemma_projected_tokens():
    mg = MedGemmaVisionAdapter.build_tiny()
    mg.eval()
    out = mg.encode(
        MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 32, 32)),
        output_spec=OutputSpec(spatial_tokens=True, pooled=True),
    )
    assert out.spatial_tokens.shape[1] == 4  # mm_tokens_per_image=4
    assert out.pooled_embedding.shape == (1, 48)


def test_medgemma_pooled_is_mean():
    mg = MedGemmaVisionAdapter.build_tiny()
    mg.eval()
    out = mg.encode(
        MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 32, 32)),
        output_spec=OutputSpec(spatial_tokens=True, pooled=True),
    )
    manual_mean = out.spatial_tokens.mean(dim=1)
    assert torch.allclose(out.pooled_embedding, manual_mean, atol=1e-6)


def test_medgemma_feature_maps_unsupported():
    mg = MedGemmaVisionAdapter.build_tiny()
    mg.eval()
    with pytest.raises(UnsupportedCapabilityError):
        mg.encode(
            MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 32, 32)),
            output_spec=OutputSpec(feature_maps=True),
        )


def test_medgemma_preprocess_defaults():
    assert MEDGEMMA_PREPROCESS.image_size == (896, 896)
    assert MEDGEMMA_PREPROCESS.patch_size == 14
