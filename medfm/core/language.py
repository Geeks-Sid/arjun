"""Language-model adapter contract.

Every language adapter (Phase 09) must declare how visual information enters
the model: either it accepts ``inputs_embeds`` (the bridge splices projected
visual tokens into the embedding sequence) or it ships a native visual
connector (e.g. MedGemma's visual pathway). Adapters that can do neither are
text-only and must declare that; a VLM batch must fail loudly against a
text-only adapter instead of silently dropping the images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from medfm.core.enums import CoordinateSystem, Modality
from medfm.core.errors import ShapeContractError


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class TokenizedText:
    """Tokenized conversations/prompts: ``input_ids [B, L]`` + mask."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. visual-token span indices

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ShapeContractError(f"input_ids must be [B, L]; got {tuple(self.input_ids.shape)}")
        if self.input_ids.dtype not in (torch.int32, torch.int64):
            raise ShapeContractError(f"input_ids must be int32/int64; got {self.input_ids.dtype}")
        if tuple(self.attention_mask.shape) != tuple(self.input_ids.shape):
            raise ShapeContractError(
                f"attention_mask shape {tuple(self.attention_mask.shape)} must equal input_ids shape "
                f"{tuple(self.input_ids.shape)}"
            )


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class ProjectedVisualTokens:
    """Visual tokens projected into the language embedding space.

    ``tokens [B, N, Dlm]`` — Dlm is the language model's hidden size.
    ``token_mask [B, N]`` distinguishes real tokens from bucket padding.
    """

    tokens: torch.Tensor
    source_modality: Modality
    token_mask: torch.Tensor | None = None
    coordinate_system: CoordinateSystem | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ShapeContractError(f"projected visual tokens must be [B, N, Dlm]; got {tuple(self.tokens.shape)}")
        if self.token_mask is not None and tuple(self.token_mask.shape) != tuple(self.tokens.shape[:2]):
            raise ShapeContractError(
                f"token_mask shape {tuple(self.token_mask.shape)} must equal tokens [B, N] = "
                f"{tuple(self.tokens.shape[:2])}"
            )

    @property
    def hidden_size(self) -> int:
        return int(self.tokens.shape[2])


@dataclass(frozen=True)
class LanguageModelCapabilities:
    """Declared language-adapter capabilities."""

    model_id: str
    accepts_inputs_embeds: bool
    native_visual_connector: bool = False
    supports_generation: bool = True
    max_text_tokens: int | None = None
    max_visual_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ShapeContractError("LanguageModelCapabilities.model_id must be non-empty")

    @property
    def accepts_visual_tokens(self) -> bool:
        """Whether visual tokens can enter the model at all."""
        return self.accepts_inputs_embeds or self.native_visual_connector


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class LanguageOutput:
    """Forward-pass output: per-token logits [B, L, V] plus optional loss."""

    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None  # scalar
    hidden_states: tuple[torch.Tensor, ...] | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logits is not None and self.logits.ndim != 3:
            raise ShapeContractError(f"logits must be [B, L, V]; got {tuple(self.logits.shape)}")
        if self.loss is not None and self.loss.ndim != 0:
            raise ShapeContractError(f"loss must be a scalar; got shape {tuple(self.loss.shape)}")


@dataclass(frozen=True)
class GenerationConfig:
    """Generation hyperparameters (decoded by the adapter's own tokenizer)."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    do_sample: bool = False
    stop_strings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ShapeContractError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ShapeContractError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ShapeContractError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ShapeContractError("top_k must be >= 0 (0 disables it)")


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class GeneratedText:
    """Generation output: decoded strings plus the raw token ids [B, G]."""

    texts: tuple[str, ...]
    token_ids: torch.Tensor | None = None
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.token_ids is not None and self.token_ids.ndim != 2:
            raise ShapeContractError(f"token_ids must be [B, G]; got {tuple(self.token_ids.shape)}")
        if self.token_ids is not None and int(self.token_ids.shape[0]) != len(self.texts):
            raise ShapeContractError(
                f"token_ids batch dim {int(self.token_ids.shape[0])} != number of texts {len(self.texts)}"
            )
        if self.scores is not None and len(self.scores) != len(self.texts):
            raise ShapeContractError("scores must align with texts")


@runtime_checkable
class LanguageModelAdapter(Protocol):
    """Contract every language adapter implements (Phase 09).

    ``capabilities`` must declare ``accepts_inputs_embeds`` and/or
    ``native_visual_connector``; passing visual tokens to an adapter that
    accepts neither must raise :class:`UnsupportedCapabilityError`.
    """

    @property
    def capabilities(self) -> LanguageModelCapabilities: ...

    def tokenize(self, conversations: list[Any]) -> TokenizedText: ...

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def forward_with_visual_tokens(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        labels: torch.Tensor | None,
    ) -> LanguageOutput: ...

    def generate(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        generation_config: GenerationConfig,
    ) -> GeneratedText: ...
