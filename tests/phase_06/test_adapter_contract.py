"""Contract tests: every 2D adapter satisfies the VisualEncoder protocol."""

import torch
import pytest

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec, VisualEncoder
from medfm.core.enums import Modality
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.models.visual.base import LinearHead
from medfm.models.visual.medgemma_vision import MedGemmaVisionAdapter
from medfm.models.visual.medsiglip import MedSigLIPAdapter
from medfm.models.visual.raddino import RADDINOAdapter
from medfm.models.visual.hoptimus0 import HOptimus0Adapter


TINY_ADAPTERS = [
    lambda: MedSigLIPAdapter.build_tiny(),
    lambda: RADDINOAdapter.build_tiny(),
    lambda: HOptimus0Adapter.build_tiny(),
    lambda: MedGemmaVisionAdapter.build_tiny(),
]

ADAPTER_NAMES = ["MedSigLIP", "RADDINO", "HOptimus0", "MedGemma"]


def _batch(n=2, c=3, h=32, w=32, mod=Modality.XRAY_2D):
    return MedicalBatch(modality=mod, sample_ids=[f"s{i}" for i in range(n)], pixel_values=torch.randn(n, c, h, w))


def _batch_for(adapter, n=2):
    mod = adapter.capabilities.modalities[0]
    pp = adapter.preprocess
    return MedicalBatch(
        modality=mod,
        sample_ids=[f"s{i}" for i in range(n)],
        pixel_values=torch.randn(n, pp.channels, pp.image_size[0], pp.image_size[1]),
    )


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("build, name", list(zip(TINY_ADAPTERS, ADAPTER_NAMES, strict=True)))
def test_satisfies_visual_encoder_protocol(build, name):
    adapter = build()
    assert isinstance(adapter, VisualEncoder), f"{name} is not a VisualEncoder"


@pytest.mark.parametrize("build, name", list(zip(TINY_ADAPTERS, ADAPTER_NAMES, strict=True)))
def test_preprocess_spec_matches_registry_spec(build, name):
    adapter = build()
    core = adapter.preprocess_spec()
    reg = adapter.registry_preprocess_spec()
    assert tuple(core.image_size) == tuple(reg.spatial_shape)
    assert core.channels == reg.channels


# --------------------------------------------------------------------------- #
# Default encode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("build, name", list(zip(TINY_ADAPTERS, ADAPTER_NAMES, strict=True)))
def test_encode_default_pooled_only(build, name):
    adapter = build()
    adapter.eval()
    out = adapter.encode(_batch_for(adapter))
    assert out.pooled_embedding is not None
    assert out.pooled_embedding.ndim == 2
    assert out.spatial_tokens is None
    assert out.feature_maps is None


# --------------------------------------------------------------------------- #
# Spatial tokens + coordinates + mask
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("build, name", [*list(zip(TINY_ADAPTERS, ADAPTER_NAMES, strict=True))])
def test_spatial_tokens_and_coordinates(build, name):
    adapter = build()
    adapter.eval()
    out = adapter.encode(_batch_for(adapter), output_spec=OutputSpec(spatial_tokens=True, token_coordinates=True))
    assert out.spatial_tokens is not None
    assert out.spatial_tokens.ndim == 3
    assert out.token_mask is not None
    assert out.token_mask.shape == out.spatial_tokens.shape[:2]
    assert torch.all(out.token_mask > 0)
    assert out.token_coordinates is not None
    assert out.token_coordinates.shape[0] == out.spatial_tokens.shape[0]
    assert out.token_coordinates.shape[2] == 2
    coord = out.token_coordinates[0]
    assert torch.all((coord >= 0) & (coord <= 1))


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #


def test_feature_maps_raddino():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    out = rd.encode(_batch(), output_spec=OutputSpec(spatial_tokens=True, feature_maps=True))
    assert out.feature_maps is not None
    assert isinstance(out.feature_maps, tuple)
    assert len(out.feature_maps) == 4
    for fm in out.feature_maps:
        assert fm.ndim == 4  # [B, D, H, W]


def test_feature_maps_medsiglip():
    ms = MedSigLIPAdapter.build_tiny()
    # Override to enable feature maps for the tiny config
    ms._feature_map_layers = (1, 2)
    ms.eval()
    out = ms.encode(_batch(h=32, w=32), output_spec=OutputSpec(feature_maps=True), output_hidden_states=True)
    assert out.feature_maps is not None
    assert isinstance(out.feature_maps, tuple)


def test_feature_maps_unsupported_on_medgemma():
    mg = MedGemmaVisionAdapter.build_tiny()
    mg.eval()
    with pytest.raises(UnsupportedCapabilityError):
        mg.encode(_batch(), output_spec=OutputSpec(feature_maps=True))


# --------------------------------------------------------------------------- #
# Preprocess mismatch
# --------------------------------------------------------------------------- #


def test_preprocess_mismatch_shape():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    bad = MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 3, 64, 64))
    with pytest.raises(ShapeContractError, match="preprocess mismatch"):
        rd.encode(bad)


def test_preprocess_mismatch_channels():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    bad = MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=torch.randn(1, 1, 32, 32))
    with pytest.raises(ShapeContractError, match="preprocess mismatch"):
        rd.encode(bad)


# --------------------------------------------------------------------------- #
# Native outputs
# --------------------------------------------------------------------------- #


def test_native_outputs_on_hidden_states():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    out = rd.encode(_batch(), output_hidden_states=True)
    assert out.native_outputs is not None
    assert "native_outputs_kind" in out.auxiliary


# --------------------------------------------------------------------------- #
# Frozen mode
# --------------------------------------------------------------------------- #


def test_frozen_zero_trainable():
    rd = RADDINOAdapter.build_tiny()
    rd.freeze_backbone()
    assert rd.trainable_backbone_parameters() == 0


# --------------------------------------------------------------------------- #
# Task-head attachment
# --------------------------------------------------------------------------- #


def test_attach_head_and_backward():
    rd = RADDINOAdapter.build_tiny()
    rd.freeze_backbone()
    head = LinearHead(in_features=64, out_features=3)
    rd.attach_head(head)
    out = rd.encode(_batch(), output_spec=OutputSpec(pooled=True))
    logits = rd.head_logits(out.pooled_embedding)
    assert tuple(logits.shape) == (2, 3)
    loss = logits.sum()
    loss.backward()
    head_grads = sum(1 for p in rd.attached_head.parameters() if p.grad is not None)
    assert head_grads > 0


# --------------------------------------------------------------------------- #
# Smoke shortcut
# --------------------------------------------------------------------------- #


def test_forward_smoke_works_in_eval():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    out = rd(torch.randn(1, 3, 32, 32))
    assert out.ndim == 2  # pooled [1, D]


def test_forward_smoke_rejects_training():
    rd = RADDINOAdapter.build_tiny()
    rd.train()
    with pytest.raises(ShapeContractError):
        rd(torch.randn(1, 3, 32, 32))


# --------------------------------------------------------------------------- #
# Deterministic eval
# --------------------------------------------------------------------------- #


def test_deterministic_eval():
    rd = RADDINOAdapter.build_tiny()
    rd.set_deterministic_eval()
    x = torch.randn(1, 3, 32, 32)
    out1 = rd.encode(MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=x.clone()))
    out2 = rd.encode(MedicalBatch(modality=Modality.XRAY_2D, sample_ids=["s0"], pixel_values=x.clone()))
    assert torch.equal(out1.pooled_embedding, out2.pooled_embedding)


# --------------------------------------------------------------------------- #
# TPU smoke config
# --------------------------------------------------------------------------- #


def test_tpu_smoke_config_static():
    rd = RADDINOAdapter.build_tiny()
    cfg = rd.tpu_smoke_config()
    assert cfg["model_id"]
    assert cfg["batch_size"] == 2
    assert cfg["static_batch"] is True
    assert cfg["dtype"] == "bfloat16"
    assert "image_size" in cfg
