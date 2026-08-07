"""Checkpoint save/load round-trip tests (full and adapter-only, ADR 0006)."""

import torch
import pytest

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec
from medfm.core.enums import Modality
from medfm.models.visual.base import AdapterCheckpointError, BaseVisualAdapter2D, LinearHead
from medfm.models.visual.raddino import RADDINOAdapter


def _batch(rd):
    pp = rd.preprocess
    return MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["s0", "s1"],
        pixel_values=torch.randn(2, pp.channels, pp.image_size[0], pp.image_size[1]),
    )


def _pooled(rd, batch=None):
    rd.eval()
    if batch is None:
        batch = _batch(rd)
    return rd.encode(batch, output_spec=OutputSpec(pooled=True)).pooled_embedding


# --------------------------------------------------------------------------- #
# Full round-trip
# --------------------------------------------------------------------------- #


def test_full_round_trip(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42)
    out_before = _pooled(rd)
    rd.save_checkpoint(tmp_path / "ckpt", include_backbone=True)

    # Rebuild from the manifest config
    from medfm.models.visual.base import BaseVisualAdapter2D

    def rebuild(config):
        return RADDINOAdapter.from_config_dict(config)

    rd2 = BaseVisualAdapter2D.load_checkpoint(tmp_path / "ckpt", rebuild=rebuild, device="cpu")
    out_after = _pooled(rd2)
    assert torch.equal(out_before, out_after)


# --------------------------------------------------------------------------- #
# Head round-trip
# --------------------------------------------------------------------------- #


def test_head_round_trip(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42)
    head = LinearHead(64, 3)
    rd.attach_head(head)
    out_before = _pooled(rd)
    logits_before = rd.head_logits(out_before)
    rd.save_checkpoint(tmp_path / "ckpt-head", include_backbone=True)

    def rebuild(config):
        return RADDINOAdapter.from_config_dict(config)

    rd2 = BaseVisualAdapter2D.load_checkpoint(tmp_path / "ckpt-head", rebuild=rebuild, device="cpu")
    out_after = _pooled(rd2)
    assert torch.equal(out_before, out_after)
    logits_after = rd2.head_logits(out_after)
    assert torch.equal(logits_before, logits_after)


# --------------------------------------------------------------------------- #
# Adapter-only export
# --------------------------------------------------------------------------- #


def test_adapter_only_export_round_trip(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42, revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    rd.eval()
    rd.inject_lora(rank=4)
    head = LinearHead(64, 3)
    rd.attach_head(head)
    # "Train" one step
    out = rd.encode(_batch(rd), output_spec=OutputSpec(pooled=True))
    logits = rd.head_logits(out.pooled_embedding)
    loss = logits.sum()
    loss.backward()
    # Save lora + head only
    rd.export_adapter_checkpoint(tmp_path / "adapter-only")

    def rebuild(config):
        return RADDINOAdapter.from_config_dict(config)

    rd2 = BaseVisualAdapter2D.load_checkpoint(tmp_path / "adapter-only", rebuild=rebuild, device="cpu")
    # Check output after restore is different from fresh (trained weights loaded)
    out2 = _pooled(rd2)
    assert not torch.equal(_pooled(RADDINOAdapter.build_tiny(construction_seed=42)), out2)


# --------------------------------------------------------------------------- #
# Export guards
# --------------------------------------------------------------------------- #


def test_adapter_only_export_needs_pinned_revision(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42)  # revision="local-tiny"
    rd.inject_lora(rank=4)
    with pytest.raises(AdapterCheckpointError, match="pinned"):
        rd.export_adapter_checkpoint(tmp_path / "bad")


def test_adapter_only_export_needs_trained_params(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42, revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(AdapterCheckpointError, match="nothing trained"):
        rd.export_adapter_checkpoint(tmp_path / "bad")


# --------------------------------------------------------------------------- #
# Mismatched model_id
# --------------------------------------------------------------------------- #


def test_load_mismatched_model_id(tmp_path):
    rd = RADDINOAdapter.build_tiny(construction_seed=42, model_id="model-a")
    rd.save_checkpoint(tmp_path / "a", include_backbone=True)

    def rebuild(config):
        cfg = dict(config)
        cfg["model_id"] = "model-b"
        return RADDINOAdapter.from_config_dict(cfg)

    with pytest.raises(AdapterCheckpointError, match="belongs to 'model-a'"):
        BaseVisualAdapter2D.load_checkpoint(tmp_path / "a", rebuild=rebuild, device="cpu")
