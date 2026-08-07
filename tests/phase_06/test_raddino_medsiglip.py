"""RAD-DINO and MedSigLIP specific tests."""

import torch
import pytest
from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec
from medfm.core.enums import Modality
from medfm.models.visual.raddino import RADDINOAdapter, RADDINO_PREPROCESS
from medfm.models.visual.medsiglip import MedSigLIPAdapter, MEDSIGLIP_PREPROCESS, MEDSIGLIP_LORA_VISION


# ---- RAD-DINO ----
def test_raddino_pooled_is_cls():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    bh = MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 32, 32))
    out = rd.encode(bh, output_spec=OutputSpec(pooled=True, spatial_tokens=True))
    # pooled should equal the CLS token (prefix token 0)
    result = rd._forward_backbone(bh.pixel_values, False)
    cls_from_raw = result.last_hidden_state[:, 0, :]
    assert torch.equal(out.pooled_embedding, cls_from_raw)


def test_raddino_patch_token_count():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    out = rd.encode(
        MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0", "s1"], pixel_values=torch.randn(2, 3, 32, 32)),
        output_spec=OutputSpec(spatial_tokens=True),
    )
    assert out.spatial_tokens.shape[1] == 4  # (32/16)^2


def test_raddino_patches_exclude_cls():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    x = torch.randn(1, 3, 32, 32)
    out = rd.encode(
        MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=x),
        output_spec=OutputSpec(spatial_tokens=True),
    )
    result = rd._forward_backbone(x, False)
    manual_patches = result.last_hidden_state[:, 1:, :]
    assert torch.equal(out.spatial_tokens, manual_patches)


def test_raddino_feature_map_shapes():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    out = rd.encode(
        MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 32, 32)),
        output_spec=OutputSpec(feature_maps=True),
        output_hidden_states=True,
    )
    assert len(out.feature_maps) == 4
    for fm in out.feature_maps:
        assert tuple(fm.shape) == (1, 64, 2, 2)  # B, D, h, w


def test_raddino_preprocess_defaults():
    assert RADDINO_PREPROCESS.image_size == (518, 518)
    assert RADDINO_PREPROCESS.patch_size == 14
    assert RADDINO_PREPROCESS.mean[0] == pytest.approx(0.5307, abs=1e-4)
    assert RADDINO_PREPROCESS.std[1] == pytest.approx(0.2583, abs=1e-4)


# ---- MedSigLIP ----
def test_medsiglip_image_text_similarity():
    ms = MedSigLIPAdapter.build_tiny()
    ms.eval()
    img = ms.encode_image_normalized(torch.randn(3, 3, 32, 32))
    txt = ms.encode_text(torch.randint(0, 128, (2, 8)))
    sim = ms.image_text_similarity(img, txt)
    assert tuple(sim.shape) == (3, 2)


def test_medsiglip_text_embeddings_normalized():
    ms = MedSigLIPAdapter.build_tiny()
    ms.eval()
    emb = ms.encode_text(torch.randint(0, 128, (4, 8)))
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_medsiglip_preprocess_defaults():
    assert MEDSIGLIP_PREPROCESS.image_size == (448, 448)
    assert MEDSIGLIP_PREPROCESS.patch_size == 14
    assert MEDSIGLIP_PREPROCESS.mean == (0.5, 0.5, 0.5)


def test_medsiglip_multi_image_fold():
    ms = MedSigLIPAdapter.build_tiny()
    ms.eval()
    bh = MedicalBatch(modality=Modality.MULTI_IMAGE_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 2, 3, 32, 32))
    out = ms.encode(bh, output_spec=OutputSpec(spatial_tokens=True))
    assert out.spatial_tokens.shape[0] == 2


def test_medsiglip_vision_only_lora_scope():
    ms = MedSigLIPAdapter.build_tiny()
    matched = ms.inject_lora(targets=(MEDSIGLIP_LORA_VISION.pattern,))
    assert len(matched) > 0
    assert all("vision_model" in n for n in matched)
    assert not any("text_model" in n for n in matched)
