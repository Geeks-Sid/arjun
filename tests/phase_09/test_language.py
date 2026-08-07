from __future__ import annotations

import pytest
import torch
from torch import nn

from medfm.core.enums import Modality
from medfm.core.language import GenerationConfig, ProjectedVisualTokens
from medfm.models.bridges import MLPBridge, TokenPlacementConfig
from medfm.models.language import (
    ArchitectureMismatchError,
    GemmaCausalLMAdapter,
    GenericHFCausalLMAdapter,
    M3DLaMedAdapter,
    MedGemmaAdapter,
)


def _external() -> tuple[GenericHFCausalLMAdapter, ProjectedVisualTokens]:
    model = GenericHFCausalLMAdapter.build_tiny(
        hidden_size=16,
        vocab_size=48,
        max_text_tokens=64,
        visual_token_buckets=(4,),
        text_token_buckets=(32, 64),
        construction_seed=5,
    )
    bridge = MLPBridge(
        source_dim=8,
        target_dim=16,
        output_tokens=4,
        max_input_tokens=4,
        source_modality=Modality.XRAY_2D,
    )
    return model, bridge(torch.randn(1, 4, 8), torch.tensor([[True, True, True, False]]))


def test_external_language_loss_masks_prompt_and_visual_tokens() -> None:
    model, visual = _external()
    text = model.tokenize(["prompt answer"])
    labels = torch.full_like(text.input_ids, 7)
    prompt_mask = torch.ones_like(text.input_ids, dtype=torch.bool)
    prompt_mask[:, -1] = False
    text = type(text)(text.input_ids, text.attention_mask, {"prompt_token_mask": prompt_mask})
    result = model.forward_with_visual_tokens(text, visual, labels)
    assert result.loss is not None and torch.isfinite(result.loss)
    assert result.auxiliary["supervised_token_count"] == 1
    assert result.auxiliary["placement"]["visual_tokens"] == 4
    placed_attention = result.auxiliary["attention_mask"]
    assert not placed_attention[:, 4].item()  # one padded visual token
    assert placed_attention[:, 3].item()  # preceding visual token is real


def test_external_loss_has_bridge_gradients_and_frozen_stage_one_modules() -> None:
    model, visual = _external()
    visual.tokens.retain_grad()
    text = model.tokenize(["prompt answer"])
    labels = torch.full_like(text.input_ids, -100)
    labels[:, -1] = 4
    bridge = nn.Linear(8, 8)
    # Stage 1 declarations are explicit and keep LM parameters frozen; the
    # standalone bridge remains trainable for the gradient audit.
    from medfm.models.bridges import TrainingStage, apply_stage_freeze, stage_config

    apply_stage_freeze(
        {"vision": bridge, "language": model.model, "bridge": bridge, "boundary": model.boundary_embeddings},
        stage_config(TrainingStage.BRIDGE_ONLY),
    )
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
    assert any(parameter.requires_grad for parameter in model.boundary_embeddings.parameters())
    result = model.forward_with_visual_tokens(text, visual, labels)
    assert result.loss is not None
    result.loss.backward()
    assert visual.tokens.grad is not None
    assert any(parameter.grad is not None for parameter in model.boundary_embeddings.parameters())


def test_native_medgemma_has_separate_connector_capability() -> None:
    model, visual = _external()
    native = MedGemmaAdapter.build_tiny(hidden_size=16, vocab_size=48, visual_token_buckets=(4,))
    text = native.tokenize(["prompt answer"])
    labels = torch.full_like(text.input_ids, -100)
    labels[:, -1] = 3
    result = native.forward_with_visual_tokens(text, visual, labels)
    assert result.loss is not None and torch.isfinite(result.loss)
    assert not native.capabilities.accepts_inputs_embeds
    assert native.capabilities.native_visual_connector
    assert native.verify_tied_weights()
    generated_a = native.generate(text, visual, GenerationConfig(max_new_tokens=3))
    generated_b = native.generate(text, visual, GenerationConfig(max_new_tokens=3))
    assert generated_a.token_ids is not None and generated_a.token_ids.shape[1] <= 3
    assert torch.equal(generated_a.token_ids, generated_b.token_ids)
    stopped = native.generate(
        text,
        visual,
        GenerationConfig(max_new_tokens=3, stop_strings=("<tok:",)),
    )
    assert stopped.texts == ("",)

    assert generated_a.token_ids is not None
    stopped_by_id = native.generate(
        text,
        visual,
        GenerationConfig(max_new_tokens=3, stop_token_ids=(int(generated_a.token_ids[0, 0]),)),
    )
    assert stopped_by_id.token_ids is not None and stopped_by_id.token_ids.shape[1] == 1


def test_architecture_checks_and_research_gate() -> None:
    class EncoderDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"model_type": "t5", "is_encoder_decoder": True})()
            self.embedding = nn.Embedding(16, 8)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embedding

        def forward(self, **kwargs):
            return kwargs

    with pytest.raises(ArchitectureMismatchError):
        GenericHFCausalLMAdapter(model=EncoderDecoder())
    with pytest.raises(Exception, match="research/license gated"):
        M3DLaMedAdapter(model=GenericHFCausalLMAdapter.build_tiny().model)
    assert M3DLaMedAdapter.build_tiny(hidden_size=16, vocab_size=32).integration_status()["license_accepted"]


def test_visual_bucket_is_fixed() -> None:
    model, _ = _external()
    text = model.tokenize(["prompt"])
    labels = torch.full_like(text.input_ids, -100)
    with pytest.raises(Exception, match="configured fixed bucket"):
        wrong = ProjectedVisualTokens(torch.randn(1, 3, 16), Modality.XRAY_2D)
        model.forward_with_visual_tokens(text, wrong, labels)


def test_gemma_adapter_and_placement_config_are_versioned() -> None:
    model = GemmaCausalLMAdapter.build_tiny(hidden_size=16, vocab_size=32, visual_token_buckets=(4,))
    assert model.config.chat_template.name
    assert model.placement_config.config_name == "external-prefix-v1"
    suffix = TokenPlacementConfig(mode="suffix", use_boundary_embeddings=False, config_name="suffix-v1")
    assert suffix.mode == "suffix"
