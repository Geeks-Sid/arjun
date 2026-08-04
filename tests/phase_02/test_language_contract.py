"""Language adapter contract: conformance, visual-token entry requirements."""

import pytest
import torch
from contract_fixtures import (
    DummyLanguageModelAdapter,
    TextOnlyLanguageModelAdapter,
)

from medfm.core import (
    GeneratedText,
    GenerationConfig,
    LanguageModelAdapter,
    LanguageOutput,
    Modality,
    ProjectedVisualTokens,
    ShapeContractError,
    TokenizedText,
    UnsupportedCapabilityError,
)


def test_dummy_lm_conforms_to_protocol():
    assert isinstance(DummyLanguageModelAdapter(), LanguageModelAdapter)
    assert isinstance(TextOnlyLanguageModelAdapter(), LanguageModelAdapter)


def test_adapters_must_declare_visual_entry_path():
    caps = DummyLanguageModelAdapter().capabilities
    assert caps.accepts_visual_tokens  # inputs_embeds declared
    text_only = TextOnlyLanguageModelAdapter().capabilities
    assert not text_only.accepts_visual_tokens  # neither path declared


def test_text_only_adapter_rejects_visual_tokens():
    adapter = TextOnlyLanguageModelAdapter()
    text = adapter.tokenize(["hello"])
    visual = ProjectedVisualTokens(tokens=torch.randn(1, 4, 8), source_modality=Modality.XRAY_2D)
    with pytest.raises(UnsupportedCapabilityError, match="visual"):
        adapter.forward_with_visual_tokens(text, visual, None)


def test_forward_with_visual_tokens_output_shapes():
    adapter = DummyLanguageModelAdapter()
    text = adapter.tokenize(["a", "b"])
    visual = ProjectedVisualTokens(
        tokens=torch.randn(2, 4, 8),
        source_modality=Modality.XRAY_2D,
        token_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    output = adapter.forward_with_visual_tokens(text, visual, labels=text.input_ids)
    assert output.logits.shape == (2, 6 + 4, 32)
    assert output.loss.ndim == 0


def test_tokenized_text_shape_validation():
    with pytest.raises(ShapeContractError):
        TokenizedText(input_ids=torch.zeros(2, 4, 4, dtype=torch.int64), attention_mask=torch.ones(2, 4, 4))
    with pytest.raises(ShapeContractError, match="attention_mask"):
        TokenizedText(input_ids=torch.zeros(2, 4, dtype=torch.int64), attention_mask=torch.ones(2, 3))


def test_projected_visual_tokens_mask_shape():
    with pytest.raises(ShapeContractError, match="token_mask"):
        ProjectedVisualTokens(
            tokens=torch.randn(2, 4, 8),
            source_modality=Modality.XRAY_2D,
            token_mask=torch.ones(2, 5, dtype=torch.bool),
        )


def test_generation_config_and_output():
    adapter = DummyLanguageModelAdapter()
    text = adapter.tokenize(["a", "b"])
    generated = adapter.generate(text, None, GenerationConfig(max_new_tokens=5))
    assert isinstance(generated, GeneratedText)
    assert generated.token_ids.shape == (2, 5)
    assert len(generated.texts) == 2
    with pytest.raises(ShapeContractError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ShapeContractError, match="token_ids"):
        GeneratedText(texts=("one",), token_ids=torch.zeros(2, 3, dtype=torch.int64))


def test_language_output_scalar_loss():
    LanguageOutput(logits=torch.randn(1, 3, 32), loss=torch.tensor(0.5))
    with pytest.raises(ShapeContractError, match="scalar"):
        LanguageOutput(loss=torch.tensor([0.5, 0.6]))
