"""Supervised-example construction and VLM loss masking tests."""

import logging

import pytest
import torch

from medfm.core.sample import ConversationTurn
from medfm.data.errors import TextPreprocessError
from medfm.data.textprep import (
    SimpleWhitespaceTokenizer,
    SupervisedExample,
    build_supervised_example,
    validate_supervised_batch,
)

VOCAB = {
    "You": 10,
    "are": 11,
    "a": 12,
    "radiologist": 13,
    "<image>": 32000,
    "What": 20,
    "does": 21,
    "the": 22,
    "chest": 23,
    "xray": 24,
    "show": 25,
    "?": 26,
    "No": 30,
    "acute": 31,
    "abnormality": 32,
    ".": 40,
}
PLACEHOLDER_ID = 32000
BOS, EOS = 1, 2

SYSTEM_IDS = [10, 11, 12, 13, 40]
USER_IDS = [32000, 20, 21, 22, 23, 24, 25, 26]
ASSISTANT_IDS = [30, 31, 32, 40]

FULL_IDS = [BOS] + SYSTEM_IDS + [EOS] + USER_IDS + [EOS] + ASSISTANT_IDS + [EOS]
ASSISTANT_SPAN_START = len(FULL_IDS) - len(ASSISTANT_IDS) - 1  # assistant content + trailing EOS
FULL_LABELS = [-100] * ASSISTANT_SPAN_START + ASSISTANT_IDS + [EOS]


@pytest.fixture
def tokenizer() -> SimpleWhitespaceTokenizer:
    return SimpleWhitespaceTokenizer(
        VOCAB,
        pad_token_id=0,
        bos_token_id=BOS,
        eos_token_id=EOS,
        visual_placeholder_token_ids=(PLACEHOLDER_ID,),
    )


@pytest.fixture
def turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(role="system", content="You are a radiologist ."),
        ConversationTurn(role="user", content="<image> What does the chest xray show ?"),
        ConversationTurn(role="assistant", content="No acute abnormality ."),
    ]


def _supervised_positions(example: SupervisedExample) -> list[int]:
    return [i for i, label in enumerate(example.labels.tolist()) if label != -100]


def test_labels_supervise_exactly_assistant_span(tokenizer, turns):
    example = build_supervised_example(turns, tokenizer, max_length=64)
    assert example.input_ids.tolist() == FULL_IDS
    assert example.attention_mask.dtype == torch.bool
    assert bool(example.attention_mask.all())
    assert _supervised_positions(example) == list(range(ASSISTANT_SPAN_START, len(FULL_IDS)))
    assert example.labels.tolist() == FULL_LABELS
    assert example.supervised_token_count == len(ASSISTANT_IDS) + 1


def test_system_user_and_placeholder_tokens_are_masked(tokenizer, turns):
    example = build_supervised_example(turns, tokenizer, max_length=64)
    labels = example.labels.tolist()
    placeholder_index = 1 + len(SYSTEM_IDS) + 1  # BOS + system + EOS, then first user token
    assert example.input_ids[placeholder_index].item() == PLACEHOLDER_ID
    assert labels[placeholder_index] == -100
    for i in range(ASSISTANT_SPAN_START):
        assert labels[i] == -100


def test_zero_supervision_conversation_rejected(tokenizer):
    turns = [
        ConversationTurn(role="system", content="You are a radiologist ."),
        ConversationTurn(role="user", content="What does the chest xray show ?"),
    ]
    with pytest.raises(TextPreprocessError, match="zero supervised"):
        build_supervised_example(turns, tokenizer, max_length=64)


def test_truncation_drops_from_left_and_logs_counts_only(tokenizer, turns, caplog):
    max_length = len(FULL_IDS) - 4  # drops BOS + the first three system tokens
    with caplog.at_level(logging.INFO, logger="medfm.data.textprep.tokenize"):
        example = build_supervised_example(turns, tokenizer, max_length=max_length)
    assert example.truncated is True
    assert example.token_count_before_truncation == len(FULL_IDS)
    assert len(example.input_ids) == max_length
    # Left-side drop: the retained sequence is the exact right tail.
    assert example.input_ids.tolist() == FULL_IDS[4:]
    assert example.labels.tolist() == FULL_LABELS[4:]  # labels track the same left-side drop
    assert example.labels[0].item() == -100  # oldest surviving tokens stay masked
    # The supervised span is untouched and the count reflects the surviving labels.
    assert example.supervised_token_count == int((example.labels != -100).sum().item())
    assert caplog.records
    message = caplog.records[0].getMessage()
    assert str(len(FULL_IDS)) in message and str(max_length) in message
    for fragment in ("radiologist", "acute", "xray", "<image>"):
        assert fragment not in caplog.text


def test_not_truncated_when_within_limit(tokenizer, turns):
    example = build_supervised_example(turns, tokenizer, max_length=len(FULL_IDS))
    assert example.truncated is False
    assert example.token_count_before_truncation == len(FULL_IDS)


def test_mask_boilerplate_toggle(tokenizer, turns):
    boilerplate_ids = (VOCAB["."],)  # "." appears in both system and assistant turns
    masked = build_supervised_example(turns, tokenizer, max_length=64, boilerplate_token_ids=boilerplate_ids)
    unmasked = build_supervised_example(
        turns, tokenizer, max_length=64, mask_boilerplate=False, boilerplate_token_ids=boilerplate_ids
    )
    assert masked.supervised_token_count == len(ASSISTANT_IDS)  # assistant "." no longer supervised
    assert unmasked.supervised_token_count == len(ASSISTANT_IDS) + 1
    assert masked.labels.tolist().count(VOCAB["."]) == 0
    assert VOCAB["."] in unmasked.labels.tolist()


def test_unknown_token_raises_without_echoing_text(tokenizer):
    turns = [ConversationTurn(role="assistant", content="No acute frobnicate .")]
    with pytest.raises(TextPreprocessError) as excinfo:
        build_supervised_example(turns, tokenizer, max_length=64)
    assert "frobnicate" not in str(excinfo.value)


def test_tokenizer_rejects_invalid_vocab():
    with pytest.raises(TextPreprocessError):
        SimpleWhitespaceTokenizer({}, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def test_invalid_max_length_rejected(tokenizer, turns):
    with pytest.raises(TextPreprocessError):
        build_supervised_example(turns, tokenizer, max_length=0)


def _zero_example() -> SupervisedExample:
    return SupervisedExample(
        input_ids=torch.tensor([BOS, 10, EOS], dtype=torch.long),
        attention_mask=torch.ones(3, dtype=torch.bool),
        labels=torch.tensor([-100, -100, -100], dtype=torch.long),
        supervised_token_count=0,
        truncated=False,
        token_count_before_truncation=3,
    )


def test_validate_supervised_batch_all_zero_raises():
    with pytest.raises(TextPreprocessError, match="zero supervised"):
        validate_supervised_batch([_zero_example(), _zero_example()])


def test_validate_supervised_batch_empty_raises():
    with pytest.raises(TextPreprocessError):
        validate_supervised_batch([])


def test_validate_supervised_batch_passes_with_supervision(tokenizer, turns):
    example = build_supervised_example(turns, tokenizer, max_length=64)
    validate_supervised_batch([_zero_example(), example])


def test_supervised_example_validates_count_consistency():
    with pytest.raises(TextPreprocessError):
        SupervisedExample(
            input_ids=torch.tensor([BOS, EOS], dtype=torch.long),
            attention_mask=torch.ones(2, dtype=torch.bool),
            labels=torch.tensor([-100, -100], dtype=torch.long),
            supervised_token_count=3,  # inconsistent with labels
            truncated=False,
            token_count_before_truncation=2,
        )


def test_supervised_example_round_trip(tokenizer, turns):
    example = build_supervised_example(turns, tokenizer, max_length=64)
    restored = SupervisedExample.from_dict(example.to_dict())
    assert torch.equal(restored.input_ids, example.input_ids)
    assert torch.equal(restored.attention_mask, example.attention_mask)
    assert torch.equal(restored.labels, example.labels)
    assert restored.input_ids.dtype == torch.long
    assert restored.attention_mask.dtype == torch.bool
    assert restored.supervised_token_count == example.supervised_token_count
    assert restored.truncated == example.truncated
    assert restored.token_count_before_truncation == example.token_count_before_truncation
