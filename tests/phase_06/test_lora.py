"""LoRA injection and gradient-scoping tests across all 2D adapter families."""

import torch
import pytest

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import OutputSpec
from medfm.core.errors import UnsupportedCapabilityError
from medfm.models.visual.base import LinearHead, LoRAGateError
from medfm.models.visual.medgemma_vision import MedGemmaVisionAdapter
from medfm.models.visual.medsiglip import MEDSIGLIP_LORA_VISION, MedSigLIPAdapter
from medfm.models.visual.raddino import RADDINOAdapter
from medfm.models.visual.hoptimus0 import HOptimus0Adapter


def _batch(adapter, n=2):
    pp = adapter.preprocess
    mod = adapter.capabilities.modalities[0]
    return MedicalBatch(
        modality=mod,
        sample_ids=[f"s{i}" for i in range(n)],
        pixel_values=torch.randn(n, pp.channels, pp.image_size[0], pp.image_size[1]),
    )


# --------------------------------------------------------------------------- #
# RAD-DINO: basic injection + gradient scoping
# --------------------------------------------------------------------------- #


def test_raddino_inject_lora_returns_matched_names():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    matched = rd.inject_lora()
    assert len(matched) > 0
    assert rd.lora_active


def test_raddino_lora_gradients_lora_only():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    rd.inject_lora()
    head = LinearHead(64, 3)
    rd.attach_head(head)
    out = rd.encode(_batch(rd), output_spec=OutputSpec(pooled=True))
    logits = rd.head_logits(out.pooled_embedding)
    logits.sum().backward()
    grads = {n for n, p in rd.named_parameters() if p.grad is not None}
    assert all("lora_" in g or "head." in g for g in grads), f"unexpected grads: {sorted(grads)}"


def test_raddino_undeclared_target_raises():
    rd = RADDINOAdapter.build_tiny()
    with pytest.raises(UnsupportedCapabilityError, match="not declared"):
        rd.inject_lora(targets=("nonexistent_pattern",))


def test_raddino_double_inject_raises():
    rd = RADDINOAdapter.build_tiny()
    rd.inject_lora()
    with pytest.raises(LoRAGateError, match="already has LoRA"):
        rd.inject_lora()


def test_raddino_lora_state_populated():
    rd = RADDINOAdapter.build_tiny()
    rd.inject_lora(rank=4, alpha=16, dropout=0.1)
    state = rd.lora_state
    assert state is not None
    assert state["rank"] == 4
    assert state["alpha"] == 16


# --------------------------------------------------------------------------- #
# MedSigLIP: vision-only LoRA
# --------------------------------------------------------------------------- #


def test_medsiglip_vision_only_lora():
    ms = MedSigLIPAdapter.build_tiny()
    ms.eval()
    matched = ms.inject_lora(targets=(MEDSIGLIP_LORA_VISION.pattern,))
    assert len(matched) > 0
    assert all("vision_model" in n for n in matched), f"text model leaked: {matched}"


# --------------------------------------------------------------------------- #
# H-Optimus: LoRA gate
# --------------------------------------------------------------------------- #


def test_hoptimus_lora_gate():
    ho = HOptimus0Adapter.build_tiny()
    with pytest.raises(LoRAGateError, match="frozen baseline"):
        ho.inject_lora()
    ho.accept_frozen_baseline(measured_peak_bytes=1_000_000)
    matched = ho.inject_lora()
    assert len(matched) > 0


# --------------------------------------------------------------------------- #
# MedGemma: vision tower LoRA
# --------------------------------------------------------------------------- #


def test_medgemma_lora():
    mg = MedGemmaVisionAdapter.build_tiny()
    mg.eval()
    matched = mg.inject_lora()
    assert len(matched) > 0
    assert all("vision_tower" in n for n in matched)


# --------------------------------------------------------------------------- #
# LoRA + head combined gradients
# --------------------------------------------------------------------------- #


def test_lora_head_combined_backward():
    rd = RADDINOAdapter.build_tiny()
    rd.eval()
    rd.inject_lora()
    head = LinearHead(64, 3)
    rd.attach_head(head)
    out = rd.encode(_batch(rd), output_spec=OutputSpec(pooled=True))
    logits = rd.head_logits(out.pooled_embedding)
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 3, (out.pooled_embedding.shape[0],)))
    loss.backward()
    lora_grads = sum(1 for n, p in rd.named_parameters() if p.grad is not None and "lora_" in n)
    head_grads = sum(1 for p in head.parameters() if p.grad is not None)
    assert lora_grads > 0
    assert head_grads > 0
    # Base backbone parameters should have zero grads
    base_grads = sum(
        1 for n, p in rd.named_parameters() if p.grad is not None and "lora_" not in n and not n.startswith("_head")
    )
    assert base_grads == 0, f"base parameters received gradients: base_grads={base_grads}"
