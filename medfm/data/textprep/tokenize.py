"""Tokenization and supervised-loss masking for VLM instruction tuning.

:func:`build_supervised_example` flattens a
:class:`~medfm.core.sample.ConversationTurn` list into a single token
sequence with BOS prepended and EOS after every turn, and builds the label
tensor alongside: ONLY assistant-turn content tokens and their trailing EOS
are supervised — system tokens, user tokens, visual placeholder tokens, and
(optionally) configured boilerplate tokens are masked with ``-100``.

Truncation drops tokens from the LEFT (system/oldest context first) so the
most recent context — including the supervised assistant span — survives as
long as possible. An example with zero supervised tokens is rejected, and
:func:`validate_supervised_batch` rejects whole batches with zero supervised
tokens.

Privacy rule (docs/data_governance.md): logging in this module emits token
COUNTS only, via the standard ``logging`` module — never any text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from medfm.core.sample import ConversationTurn
from medfm.data.errors import TextPreprocessError

_logger = logging.getLogger(__name__)

#: Label value for masked (non-supervised) positions, per HF convention.
IGNORE_INDEX = -100


class TokenizerProtocol(Protocol):
    """Structural contract for tokenizers used by this package.

    Any object with an ``encode`` method and the four token-id attributes
    qualifies — Hugging Face fast tokenizers and the local
    :class:`SimpleWhitespaceTokenizer` both satisfy it.
    """

    @property
    def pad_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def visual_placeholder_token_ids(self) -> tuple[int, ...]: ...

    def encode(self, text: str) -> list[int]: ...


class SimpleWhitespaceTokenizer:
    """Deterministic whitespace tokenizer for tests and local smoke runs.

    The vocabulary is a fixed ``token -> id`` map supplied by the caller;
    nothing is ever downloaded. Unknown tokens raise
    :class:`TextPreprocessError` reporting only the token POSITION (never
    the token text, per the privacy rule).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        visual_placeholder_token_ids: tuple[int, ...] = (),
    ) -> None:
        if not vocab:
            raise TextPreprocessError("SimpleWhitespaceTokenizer vocab must be non-empty")
        if any(v < 0 for v in vocab.values()):
            raise TextPreprocessError("SimpleWhitespaceTokenizer vocab ids must be non-negative")
        for name, value in (("pad", pad_token_id), ("bos", bos_token_id), ("eos", eos_token_id)):
            if value < 0:
                raise TextPreprocessError(f"SimpleWhitespaceTokenizer {name}_token_id must be non-negative")
        self._vocab = dict(vocab)
        self._pad_token_id = pad_token_id
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._visual_placeholder_token_ids = tuple(visual_placeholder_token_ids)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def visual_placeholder_token_ids(self) -> tuple[int, ...]:
        return self._visual_placeholder_token_ids

    def encode(self, text: str) -> list[int]:
        """Encode whitespace-delimited tokens; raise on unknown tokens."""
        ids: list[int] = []
        for position, token in enumerate(text.split()):
            token_id = self._vocab.get(token)
            if token_id is None:
                raise TextPreprocessError(f"SimpleWhitespaceTokenizer: unknown token at position {position}")
            ids.append(token_id)
        return ids


@dataclass
class SupervisedExample:
    """A flattened, label-masked token sequence ready for collation.

    ``input_ids``/``labels`` are ``torch.long`` ``[L]``; ``attention_mask``
    is ``torch.bool`` ``[L]``. ``labels`` is ``-100`` at every
    non-supervised position; ``supervised_token_count`` is the number of
    supervised positions and is validated against ``labels``.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    supervised_token_count: int
    truncated: bool
    token_count_before_truncation: int

    def __post_init__(self) -> None:
        if self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise TextPreprocessError("SupervisedExample input_ids/labels must be torch.long")
        if self.attention_mask.dtype != torch.bool:
            raise TextPreprocessError("SupervisedExample attention_mask must be torch.bool")
        if not (
            self.input_ids.ndim == 1
            and self.attention_mask.shape == self.input_ids.shape
            and self.labels.shape == self.input_ids.shape
        ):
            raise TextPreprocessError("SupervisedExample tensors must be 1D with equal shapes")
        actual = int((self.labels != IGNORE_INDEX).sum().item())
        if actual != self.supervised_token_count:
            raise TextPreprocessError(
                f"SupervisedExample supervised_token_count {self.supervised_token_count} "
                f"does not match labels ({actual} supervised)"
            )
        if self.token_count_before_truncation < len(self.input_ids):
            raise TextPreprocessError("token_count_before_truncation must be >= current length")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain Python types (CPU, nested lists) for handoff."""
        return {
            "input_ids": self.input_ids.tolist(),
            "attention_mask": self.attention_mask.tolist(),
            "labels": self.labels.tolist(),
            "supervised_token_count": self.supervised_token_count,
            "truncated": self.truncated,
            "token_count_before_truncation": self.token_count_before_truncation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupervisedExample:
        """Rebuild from :meth:`to_dict` output; round-trips are lossless."""
        return cls(
            input_ids=torch.tensor([int(v) for v in data["input_ids"]], dtype=torch.long),
            attention_mask=torch.tensor([bool(v) for v in data["attention_mask"]], dtype=torch.bool),
            labels=torch.tensor([int(v) for v in data["labels"]], dtype=torch.long),
            supervised_token_count=int(data["supervised_token_count"]),
            truncated=bool(data["truncated"]),
            token_count_before_truncation=int(data["token_count_before_truncation"]),
        )


def build_supervised_example(
    turns: list[ConversationTurn],
    tokenizer: TokenizerProtocol,
    *,
    max_length: int,
    mask_boilerplate: bool = True,
    boilerplate_token_ids: tuple[int, ...] = (),
    logger: logging.Logger | None = None,
) -> SupervisedExample:
    """Flatten ``turns`` into a :class:`SupervisedExample` with loss masking.

    Sequence layout: ``BOS``, then per turn ``content tokens + EOS``. Labels
    supervise ONLY assistant content tokens and their trailing EOS; system
    and user content, all other BOS/EOS markers, visual placeholder tokens,
    and (when ``mask_boilerplate``) ``boilerplate_token_ids`` are ``-100``.

    Overlong sequences are truncated from the LEFT (oldest context first).
    Raises :class:`TextPreprocessError` when the result has zero supervised
    tokens. Logging emits counts only.
    """
    log = logger if logger is not None else _logger
    if max_length <= 0:
        raise TextPreprocessError(f"max_length must be positive; got {max_length}")
    if not turns:
        raise TextPreprocessError("turns must be non-empty")

    visual_ids = frozenset(tokenizer.visual_placeholder_token_ids)
    boilerplate_ids = frozenset(boilerplate_token_ids) if mask_boilerplate else frozenset()

    ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [IGNORE_INDEX]
    for turn in turns:
        content_ids = tokenizer.encode(turn.content)
        supervised_turn = turn.role == "assistant"
        for token_id in content_ids:
            ids.append(token_id)
            if supervised_turn and token_id not in visual_ids and token_id not in boilerplate_ids:
                labels.append(token_id)
            else:
                labels.append(IGNORE_INDEX)
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id if supervised_turn else IGNORE_INDEX)

    token_count_before = len(ids)
    truncated = token_count_before > max_length
    if truncated:
        dropped = token_count_before - max_length
        ids = ids[dropped:]
        labels = labels[dropped:]
        log.info(
            "truncated conversation from the left: %d -> %d tokens (%d dropped)",
            token_count_before,
            max_length,
            dropped,
        )

    supervised_count = sum(1 for label in labels if label != IGNORE_INDEX)
    if supervised_count == 0:
        raise TextPreprocessError(
            f"conversation produced zero supervised tokens (length={len(ids)}); zero-supervision examples are rejected"
        )
    log.debug("built supervised example: tokens=%d supervised=%d truncated=%s", len(ids), supervised_count, truncated)

    return SupervisedExample(
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.ones(len(ids), dtype=torch.bool),
        labels=torch.tensor(labels, dtype=torch.long),
        supervised_token_count=supervised_count,
        truncated=truncated,
        token_count_before_truncation=token_count_before,
    )


def validate_supervised_batch(examples: Sequence[SupervisedExample]) -> None:
    """Reject a batch whose total supervised token count is zero.

    A whole batch with no supervision contributes no gradient and usually
    signals a masking/mapping bug, so it is a hard error rather than a
    silent no-op. An empty batch is rejected too.
    """
    if not examples:
        raise TextPreprocessError("validate_supervised_batch received an empty batch")
    total = sum(example.supervised_token_count for example in examples)
    if total == 0:
        raise TextPreprocessError(
            f"batch of {len(examples)} example(s) has zero supervised tokens; zero-supervision batches are rejected"
        )
