"""Language-model adapters for external and native vision-language modes."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.core.language import (
    GeneratedText,
    GenerationConfig,
    LanguageModelCapabilities,
    LanguageOutput,
    ProjectedVisualTokens,
    TokenizedText,
)
from medfm.data.textprep.tokenize import TokenizerProtocol
from medfm.models.bridges.placement import (
    IGNORE_INDEX,
    TokenPlacementConfig,
    VisualBoundaryEmbeddings,
    mask_causal_labels,
    place_visual_tokens,
)

logger = logging.getLogger(__name__)


class LanguageAdapterError(ShapeContractError):
    """Malformed language adapter configuration or input."""


class ArchitectureMismatchError(LanguageAdapterError):
    """The supplied checkpoint is not a supported causal architecture."""


class LanguageDependencyError(LanguageAdapterError):
    """An optional language-model dependency is unavailable."""


@dataclass(frozen=True)
class ChatTemplateConfig:
    """Versioned chat/stop-token declaration; no raw prompts are logged."""

    name: str = "framework-role-v1"
    revision: str = "1"
    stop_tokens: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    assistant_role: str = "assistant"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise LanguageAdapterError("chat template name and revision must be non-empty")
        if not self.assistant_role:
            raise LanguageAdapterError("assistant_role must be non-empty")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise LanguageAdapterError("chat template stop_token_ids must be non-negative")


@dataclass(frozen=True)
class LanguageAdapterConfig:
    """Static language configuration used by CPU/CUDA/XLA paths."""

    model_id: str
    architecture: str
    mode: str = "external"
    hidden_size: int = 0
    vocab_size: int = 0
    max_text_tokens: int = 1024
    max_visual_tokens: int = 128
    text_token_buckets: tuple[int, ...] = (256, 512, 1024)
    visual_token_buckets: tuple[int, ...] = (32, 64, 128)
    attention_backend: str = "sdpa"
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)

    def __post_init__(self) -> None:
        if not self.model_id or not self.architecture:
            raise LanguageAdapterError("model_id and architecture must be non-empty")
        if self.mode not in ("external", "native"):
            raise LanguageAdapterError("language adapter mode must be 'external' or 'native'")
        if self.hidden_size < 0 or self.vocab_size < 0:
            raise LanguageAdapterError("hidden_size and vocab_size cannot be negative")
        if self.max_text_tokens <= 0 or self.max_visual_tokens <= 0:
            raise LanguageAdapterError("language token limits must be positive")
        if not self.text_token_buckets or not self.visual_token_buckets:
            raise LanguageAdapterError("text and visual token buckets cannot be empty")
        if tuple(sorted(set(self.text_token_buckets))) != self.text_token_buckets:
            raise LanguageAdapterError("text_token_buckets must be sorted and unique")
        if tuple(sorted(set(self.visual_token_buckets))) != self.visual_token_buckets:
            raise LanguageAdapterError("visual_token_buckets must be sorted and unique")
        if any(v > self.max_text_tokens for v in self.text_token_buckets):
            raise LanguageAdapterError("text bucket exceeds max_text_tokens")
        if any(v > self.max_visual_tokens for v in self.visual_token_buckets):
            raise LanguageAdapterError("visual bucket exceeds max_visual_tokens")
        if self.attention_backend not in {"eager", "sdpa", "flash_attention_2", "xla"}:
            raise LanguageAdapterError("attention_backend must be eager, sdpa, flash_attention_2, or xla")


class _FallbackTokenizer:
    """Deterministic local tokenizer for offline tiny adapters."""

    def __init__(self, vocab_size: int, *, pad_token_id: int = 0, bos_token_id: int = 1, eos_token_id: int = 2) -> None:
        if vocab_size < 8:
            raise LanguageAdapterError("tiny tokenizer vocab_size must be >= 8")
        self._vocab_size = int(vocab_size)
        self._pad_token_id = int(pad_token_id)
        self._bos_token_id = int(bos_token_id)
        self._eos_token_id = int(eos_token_id)

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
        return ()

    def encode(self, text: str) -> list[int]:
        # Hashing text pieces avoids retaining or logging clinical strings.
        return [3 + (sum(bytearray(piece.encode("utf-8"))) % (self._vocab_size - 3)) for piece in text.split()]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(f"<tok:{int(token_id)}>" for token_id in ids)


class _TinyCausalLM(nn.Module):
    """Small causal LM using standard PyTorch attention only."""

    def __init__(
        self, vocab_size: int, hidden_size: int, *, depth: int = 2, heads: int = 4, max_positions: int = 2048
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise LanguageAdapterError("tiny hidden_size must be divisible by heads")
        self.config = type(
            "TinyConfig",
            (),
            {
                "model_type": "tiny_causal",
                "is_encoder_decoder": False,
                "is_decoder": True,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "max_position_embeddings": max_positions,
            },
        )()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> Any:
        length = int(inputs_embeds.shape[1])
        positions = torch.arange(length, device=inputs_embeds.device).clamp_max(self.position.num_embeddings - 1)
        hidden = inputs_embeds + self.position(positions).unsqueeze(0)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool, device=inputs_embeds.device), diagonal=1)
        padding = None if attention_mask is None else ~attention_mask.bool()
        encoded = self.layers(hidden, mask=causal, src_key_padding_mask=padding)
        encoded = self.norm(encoded)
        logits = self.lm_head(encoded)
        loss = None
        if labels is not None:
            loss = _causal_cross_entropy(logits, labels)
        return type(
            "TinyOutput",
            (),
            {"logits": logits, "loss": loss, "hidden_states": (encoded,) if output_hidden_states else None},
        )()


def _causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise LanguageAdapterError("logits and labels sequence lengths must match")
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(IGNORE_INDEX)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]), shifted_labels.view(-1), ignore_index=IGNORE_INDEX
    )


class GenericHFCausalLMAdapter(nn.Module):
    """Architecture-checked adapter for causal Hugging Face language models."""

    supported_architectures = frozenset(
        {"gpt2", "llama", "mistral", "gemma", "gemma2", "gemma3", "qwen2", "phi3", "tiny_causal"}
    )

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        tokenizer: TokenizerProtocol | Any | None = None,
        model_id: str = "generic-hf-causal-lm",
        architecture: str | None = None,
        mode: str = "external",
        hidden_size: int | None = None,
        vocab_size: int | None = None,
        max_text_tokens: int = 1024,
        max_visual_tokens: int = 128,
        text_token_buckets: tuple[int, ...] = (256, 512, 1024),
        visual_token_buckets: tuple[int, ...] = (32, 64, 128),
        attention_backend: str = "sdpa",
        chat_template: ChatTemplateConfig | None = None,
        placement: TokenPlacementConfig | None = None,
        boundary_embeddings: VisualBoundaryEmbeddings | None = None,
        accepts_inputs_embeds: bool | None = None,
        native_visual_connector: bool = False,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else _TinyCausalLM(vocab_size or 128, hidden_size or 64)
        self._check_architecture(self.model, architecture)
        self._input_embeddings = self._get_input_embeddings(self.model)
        inferred_hidden = int(self._input_embeddings.embedding_dim)
        inferred_vocab = int(self._input_embeddings.num_embeddings)
        self.hidden_size = int(hidden_size or inferred_hidden)
        self.vocab_size = int(vocab_size or inferred_vocab)
        if self.hidden_size != inferred_hidden or self.vocab_size != inferred_vocab:
            raise ArchitectureMismatchError("declared language dimensions do not match input embeddings")
        supports_inputs = self._supports_inputs_embeds(self.model)
        if accepts_inputs_embeds is None:
            accepts_inputs_embeds = supports_inputs
        if accepts_inputs_embeds and not supports_inputs:
            raise ArchitectureMismatchError(
                "model declares inputs_embeds support but forward has no inputs_embeds parameter"
            )
        self._accepts_inputs_embeds = bool(accepts_inputs_embeds)
        self._native_visual_connector = bool(native_visual_connector)
        if not self._accepts_inputs_embeds and not self._native_visual_connector:
            raise ArchitectureMismatchError(
                "language model accepts neither inputs_embeds nor an official visual connector"
            )
        self._tokenizer = tokenizer or _FallbackTokenizer(self.vocab_size)
        self._placement = placement or TokenPlacementConfig()
        self.boundary_embeddings = boundary_embeddings or VisualBoundaryEmbeddings(self.hidden_size)
        template = chat_template or ChatTemplateConfig()
        self.config = LanguageAdapterConfig(
            model_id=model_id,
            architecture=architecture or str(getattr(getattr(self.model, "config", None), "model_type", "unknown")),
            mode=mode,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_text_tokens=max_text_tokens,
            max_visual_tokens=max_visual_tokens,
            text_token_buckets=text_token_buckets,
            visual_token_buckets=visual_token_buckets,
            attention_backend=attention_backend,
            chat_template=template,
        )
        self.generation_operations: list[dict[str, Any]] = []
        self._configure_attention(attention_backend)

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "generic-causal-tiny",
        hidden_size: int = 32,
        vocab_size: int = 64,
        depth: int = 1,
        heads: int = 4,
        max_text_tokens: int = 128,
        visual_token_buckets: tuple[int, ...] = (32, 64, 128),
        construction_seed: int = 0,
        **kwargs: Any,
    ) -> GenericHFCausalLMAdapter:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(construction_seed)
            model = _TinyCausalLM(
                vocab_size,
                hidden_size,
                depth=depth,
                heads=heads,
                max_positions=max_text_tokens + max(visual_token_buckets) + 8,
            )
        defaults: dict[str, Any] = {
            "model_id": model_id,
            "architecture": "tiny_causal",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "max_text_tokens": max_text_tokens,
            "max_visual_tokens": visual_token_buckets[-1],
            "text_token_buckets": tuple(v for v in (32, 64, 128) if v <= max_text_tokens) or (max_text_tokens,),
            "visual_token_buckets": visual_token_buckets,
        }
        defaults.update(kwargs)
        return cls(model=model, **defaults)

    @staticmethod
    def _get_input_embeddings(model: nn.Module) -> nn.Embedding:
        getter = getattr(model, "get_input_embeddings", None)
        embeddings = getter() if callable(getter) else getattr(model, "embed_tokens", None)
        if not isinstance(embeddings, nn.Embedding) and not (
            hasattr(embeddings, "weight") and embeddings.weight.ndim == 2
        ):
            raise ArchitectureMismatchError("causal model must expose get_input_embeddings() with a rank-2 weight")
        return cast(nn.Embedding, embeddings)

    @classmethod
    def _check_architecture(cls, model: nn.Module, architecture: str | None) -> None:
        config = getattr(model, "config", None)
        if config is None:
            raise ArchitectureMismatchError("causal model must expose a config")
        if bool(getattr(config, "is_encoder_decoder", False)):
            raise ArchitectureMismatchError("encoder-decoder models are not causal language adapters")
        model_type = str(architecture or getattr(config, "model_type", "")).lower()
        is_decoder = bool(getattr(config, "is_decoder", False))
        if model_type not in cls.supported_architectures and not is_decoder:
            raise ArchitectureMismatchError(f"unsupported causal architecture {model_type!r}")

    @staticmethod
    def _supports_inputs_embeds(model: nn.Module) -> bool:
        try:
            return "inputs_embeds" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _configure_attention(self, attention_backend: str) -> None:
        model_config = getattr(self.model, "config", None)
        if model_config is not None and attention_backend in {"eager", "sdpa", "flash_attention_2"}:
            # Transformers reads this field when present.  No FlashAttention
            # import is performed; SDPA/eager remain safe fallback paths.
            if hasattr(model_config, "_attn_implementation"):
                model_config._attn_implementation = attention_backend
        if attention_backend == "flash_attention_2":
            self.flash_attention_optional = True
        else:
            self.flash_attention_optional = False

    @property
    def capabilities(self) -> LanguageModelCapabilities:
        return LanguageModelCapabilities(
            model_id=self.config.model_id,
            accepts_inputs_embeds=self._accepts_inputs_embeds,
            native_visual_connector=self._native_visual_connector,
            supports_generation=True,
            max_text_tokens=self.config.max_text_tokens,
            max_visual_tokens=self.config.max_visual_tokens,
        )

    def _validate_visual_bucket(self, visual_tokens: ProjectedVisualTokens | None) -> None:
        if visual_tokens is None:
            return
        count = int(visual_tokens.tokens.shape[1])
        if count not in self.config.visual_token_buckets:
            raise LanguageAdapterError(
                f"visual token count {count} is not a configured fixed bucket {self.config.visual_token_buckets}"
            )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def placement_config(self) -> TokenPlacementConfig:
        return self._placement

    def _apply_model_chat_template(self, item: Any) -> list[int] | None:
        """Use an official tokenizer template when one is available."""
        apply_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_template) or not isinstance(item, list | tuple):
            return None
        try:
            result = apply_template(item, tokenize=True, add_generation_prompt=False)
        except TypeError:
            try:
                result = apply_template(item, tokenize=True)
            except (TypeError, ValueError):
                return None
        except (ValueError, RuntimeError):
            return None
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().reshape(-1).tolist()
        if isinstance(result, list) and all(isinstance(token_id, int) for token_id in result):
            return [int(token_id) for token_id in result]
        return None

    def tokenize(self, conversations: list[Any]) -> TokenizedText:
        """Tokenize roles using the adapter's versioned template, not visuals."""
        if not conversations:
            raise LanguageAdapterError("conversations cannot be empty")
        encoded: list[list[int]] = []
        for item in conversations:
            template_ids = self._apply_model_chat_template(item)
            if template_ids is not None:
                ids = template_ids
            elif isinstance(item, str):
                ids = [self._tokenizer.bos_token_id, *self._tokenizer.encode(item), self._tokenizer.eos_token_id]
            elif isinstance(item, Mapping):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                ids = [
                    self._tokenizer.bos_token_id,
                    *self._tokenizer.encode(f"{role}: {content}"),
                    self._tokenizer.eos_token_id,
                ]
            elif isinstance(item, list | tuple):
                ids = [self._tokenizer.bos_token_id]
                for turn in item:
                    role = str(getattr(turn, "role", turn.get("role", "user") if isinstance(turn, Mapping) else "user"))
                    content = str(
                        getattr(turn, "content", turn.get("content", "") if isinstance(turn, Mapping) else turn)
                    )
                    ids.extend(self._tokenizer.encode(f"{role}: {content}"))
                    ids.append(self._tokenizer.eos_token_id)
            else:
                raise LanguageAdapterError(f"unsupported conversation item type {type(item).__name__}")
            if len(ids) > self.config.max_text_tokens:
                ids = ids[-self.config.max_text_tokens :]
            encoded.append(ids)
        length = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), length), int(self._tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = True
        return TokenizedText(
            input_ids=input_ids, attention_mask=attention, metadata={"chat_template": self.config.chat_template.name}
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise LanguageAdapterError("input_ids must be integer [B,L]")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size:
            raise LanguageAdapterError("input_ids contain an out-of-range token id")
        return cast(torch.Tensor, self._input_embeddings(input_ids.to(device=self._input_embeddings.weight.device)))

    def _forward_model(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None
    ) -> Any:
        if not self._accepts_inputs_embeds:
            raise UnsupportedCapabilityError("this adapter requires its native visual connector path")
        kwargs: dict[str, Any] = {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
        }
        if labels is not None:
            kwargs["labels"] = labels
        try:
            return self.model(**kwargs)
        except TypeError:
            kwargs.pop("output_hidden_states", None)
            try:
                return self.model(**kwargs)
            except TypeError as second:
                raise ArchitectureMismatchError(
                    "causal model failed its official inputs_embeds forward contract"
                ) from second

    @staticmethod
    def _extract_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, tuple | list):
            logits = output[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise LanguageAdapterError("causal model output must expose logits [B,L,V]")
        loss = getattr(output, "loss", None)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None:
            hidden_states = tuple(hidden_states)
        return logits, loss, hidden_states

    def forward_with_visual_tokens(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        labels: torch.Tensor | None,
    ) -> LanguageOutput:
        if text.input_ids.shape != text.attention_mask.shape:
            raise LanguageAdapterError("TokenizedText attention mask is misaligned")
        self._validate_visual_bucket(visual_tokens)
        if visual_tokens is not None and not self.capabilities.accepts_visual_tokens:
            raise UnsupportedCapabilityError(f"{self.config.model_id} does not accept visual tokens")
        text_embeddings = self.embed_tokens(text.input_ids)
        if visual_tokens is not None:
            visual_tokens = ProjectedVisualTokens(
                tokens=visual_tokens.tokens.to(device=text_embeddings.device, dtype=text_embeddings.dtype),
                source_modality=visual_tokens.source_modality,
                token_mask=None
                if visual_tokens.token_mask is None
                else visual_tokens.token_mask.to(text_embeddings.device),
                coordinate_system=visual_tokens.coordinate_system,
            )
        if labels is not None:
            if tuple(labels.shape) != tuple(text.input_ids.shape):
                raise LanguageAdapterError("labels must align with unplaced text tokens")
            prompt_mask = text.metadata.get("prompt_token_mask")
            placeholder_mask = text.metadata.get("visual_placeholder_mask")
            masks = [
                torch.as_tensor(value, device=text.input_ids.device, dtype=torch.bool)
                for value in (prompt_mask, placeholder_mask)
                if value is not None
            ]
            combined_prompt_mask = None
            if masks:
                combined_prompt_mask = masks[0]
                for current_mask in masks[1:]:
                    combined_prompt_mask = combined_prompt_mask | current_mask
            labels = mask_causal_labels(
                labels,
                attention_mask=text.attention_mask,
                prompt_token_mask=combined_prompt_mask,
            )
        embeddings, attention, placed_labels, placement = place_visual_tokens(
            text_embeddings,
            text.attention_mask.to(device=text_embeddings.device),
            visual_tokens,
            labels=labels,
            config=self._placement,
            boundary_embeddings=self.boundary_embeddings,
        )
        output = self._forward_model(embeddings, attention, placed_labels)
        logits, model_loss, hidden_states = self._extract_output(output)
        loss = model_loss
        if placed_labels is not None:
            # Always calculate with the framework mask.  This protects models
            # whose native loss implementation ignores visual/prompt masks.
            loss = _causal_cross_entropy(logits, placed_labels)
        supervised = 0 if placed_labels is None else int(placed_labels[:, 1:].ne(IGNORE_INDEX).sum())
        return LanguageOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            auxiliary={"placement": placement, "supervised_token_count": supervised, "attention_mask": attention},
        )

    def _generation_inputs(
        self, text: TokenizedText, visual_tokens: ProjectedVisualTokens | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_visual_bucket(visual_tokens)
        embeddings = self.embed_tokens(text.input_ids)
        if visual_tokens is not None:
            visual_tokens = ProjectedVisualTokens(
                tokens=visual_tokens.tokens.to(device=embeddings.device, dtype=embeddings.dtype),
                source_modality=visual_tokens.source_modality,
                token_mask=None if visual_tokens.token_mask is None else visual_tokens.token_mask.to(embeddings.device),
                coordinate_system=visual_tokens.coordinate_system,
            )
        placed, attention, _, _ = place_visual_tokens(
            embeddings,
            text.attention_mask.to(embeddings.device),
            visual_tokens,
            config=self._placement,
            boundary_embeddings=self.boundary_embeddings,
        )
        return placed, attention

    def generate(
        self,
        text: TokenizedText,
        visual_tokens: ProjectedVisualTokens | None,
        generation_config: GenerationConfig,
    ) -> GeneratedText:
        if not self.capabilities.supports_generation:
            raise UnsupportedCapabilityError(f"{self.config.model_id} does not support generation")
        embeddings, attention = self._generation_inputs(text, visual_tokens)
        generated: list[torch.Tensor] = []
        current_embeddings = embeddings
        current_attention = attention
        stop_ids = generation_config.stop_token_ids or self.config.chat_template.stop_token_ids
        stop_id_tensor = (
            torch.tensor(stop_ids, dtype=torch.long, device=embeddings.device)
            if stop_ids
            else torch.empty((0,), dtype=torch.long, device=embeddings.device)
        )
        for _ in range(generation_config.max_new_tokens):
            output = self._forward_model(current_embeddings, current_attention, None)
            logits, _, _ = self._extract_output(output)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token)
            next_embedding = self.embed_tokens(next_token[:, None]).to(dtype=current_embeddings.dtype)
            current_embeddings = torch.cat((current_embeddings, next_embedding), dim=1)
            current_attention = torch.cat(
                (
                    current_attention,
                    torch.ones((current_attention.shape[0], 1), dtype=torch.bool, device=current_attention.device),
                ),
                dim=1,
            )
            reached_stop = next_token.eq(int(self._tokenizer.eos_token_id))
            if stop_ids:
                reached_stop = reached_stop | torch.isin(next_token, stop_id_tensor)
            if bool(torch.all(reached_stop)):
                break
        token_ids = (
            torch.stack(generated, dim=1) if generated else torch.empty((text.input_ids.shape[0], 0), dtype=torch.long)
        )
        texts = tuple(self._decode_with_stops(row.tolist(), generation_config.stop_strings) for row in token_ids)
        self.record_generation_operation(
            "greedy_generation", causes_xla_recompilation=False, host_synchronization=False
        )
        return GeneratedText(texts=texts, token_ids=token_ids)

    def _decode_with_stops(self, ids: list[int], stop_strings: tuple[str, ...]) -> str:
        decoder = getattr(self._tokenizer, "decode", None)
        text = str(decoder(ids) if callable(decoder) else " ".join(str(value) for value in ids))
        stops = stop_strings or self.config.chat_template.stop_tokens
        for stop in stops:
            index = text.find(stop)
            if index >= 0:
                text = text[:index]
                break
        return text

    def retie_shared_embeddings(self) -> bool:
        """Retie input/output weights after accelerator placement when supported."""
        tie = getattr(self.model, "tie_weights", None)
        if callable(tie):
            tie()
        output_getter = getattr(self.model, "get_output_embeddings", None)
        output = output_getter() if callable(output_getter) else getattr(self.model, "lm_head", None)
        return bool(output is not None and getattr(output, "weight", None) is self._input_embeddings.weight)

    def verify_tied_weights(self) -> bool:
        output_getter = getattr(self.model, "get_output_embeddings", None)
        output = output_getter() if callable(output_getter) else getattr(self.model, "lm_head", None)
        return bool(output is not None and getattr(output, "weight", None) is self._input_embeddings.weight)

    def record_generation_operation(
        self, operation: str, *, causes_xla_recompilation: bool, host_synchronization: bool
    ) -> None:
        self.generation_operations.append(
            {
                "operation": str(operation),
                "causes_xla_recompilation": bool(causes_xla_recompilation),
                "host_synchronization": bool(host_synchronization),
            }
        )

    def xla_generation_report(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self.generation_operations)

    def trainable_module_declarations(self) -> dict[str, tuple[str, ...]]:
        language_names = tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        lora_names = tuple(name for name in language_names if "lora_" in name or "dora_magnitude_" in name)
        return {
            "language": language_names,
            "language_lora": lora_names,
        }


__all__ = [
    "ArchitectureMismatchError",
    "ChatTemplateConfig",
    "GenericHFCausalLMAdapter",
    "LanguageAdapterConfig",
    "LanguageAdapterError",
    "LanguageDependencyError",
]
