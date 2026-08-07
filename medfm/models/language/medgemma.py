"""Native MedGemma language/vision wrapper with an explicit connector gate."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from medfm.core.errors import UnsupportedCapabilityError
from medfm.core.language import LanguageModelCapabilities, ProjectedVisualTokens
from medfm.models.language.base import (
    ArchitectureMismatchError,
    _TinyCausalLM,
)
from medfm.models.language.gemma import GemmaCausalLMAdapter


class MedGemmaAdapter(GemmaCausalLMAdapter):
    """Native-VLM mode for MedGemma's processor/tower/connector/LM stack.

    Real checkpoints must expose an upstream visual connector (typically
    ``multi_modal_projector``) or provide one explicitly.  The offline tiny
    builder supplies a local connector solely for contract tests; it does not
    claim compatibility with a released checkpoint.
    """

    native_model_ids = frozenset({"medgemma", "medgemma-1.5-4b", "medgemma-1.5-4b-it", "tiny_causal"})

    def __init__(
        self,
        *,
        native_connector: nn.Module | None = None,
        processor: Any | None = None,
        architecture: str | None = None,
        **kwargs: Any,
    ) -> None:
        provided_model = kwargs.get("model")
        resolved = architecture or str(getattr(getattr(provided_model, "config", None), "model_type", "gemma3"))
        if resolved.lower() not in self.native_model_ids and not resolved.lower().startswith("gemma"):
            raise ArchitectureMismatchError(f"MedGemma requires Gemma/Gemma3 architecture, got {resolved!r}")
        official_connector = None
        if provided_model is not None:
            for name in ("multi_modal_projector", "visual_projector", "mm_projector"):
                candidate = getattr(provided_model, name, None)
                if isinstance(candidate, nn.Module):
                    official_connector = candidate
                    break
        if provided_model is not None and native_connector is None and official_connector is None:
            raise ArchitectureMismatchError(
                "MedGemma checkpoint exposes no official visual connector; refusing to accept visual tokens"
            )
        self.processor = processor
        self.connector_source = (
            "official" if official_connector is not None and native_connector is None else "local_tiny"
        )
        kwargs["accepts_inputs_embeds"] = False
        kwargs["native_visual_connector"] = True
        kwargs["mode"] = "native"
        kwargs["architecture"] = resolved
        super().__init__(**kwargs)
        self.native_connector = native_connector if native_connector is not None else official_connector
        if self.native_connector is None:
            self.native_connector = nn.Linear(self.hidden_size, self.hidden_size)
            self.connector_source = "local_tiny"

    @property
    def vision_tower(self) -> nn.Module | None:
        """Expose the upstream vision tower when the native model provides it."""
        for name in ("vision_tower", "vision_model", "visual"):
            candidate = getattr(self.model, name, None)
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    @property
    def language_model(self) -> nn.Module:
        """Expose the native language component without requiring a HF class."""
        candidate = getattr(self.model, "language_model", None)
        return candidate if isinstance(candidate, nn.Module) else self.model

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "medgemma-native-tiny",
        hidden_size: int = 32,
        vocab_size: int = 64,
        construction_seed: int = 0,
        processor: Any | None = None,
        **kwargs: Any,
    ) -> MedGemmaAdapter:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(construction_seed)
            model = _TinyCausalLM(vocab_size, hidden_size, depth=1, heads=4, max_positions=2048)
            connector = nn.Linear(hidden_size, hidden_size)
        defaults: dict[str, Any] = {
            "model_id": model_id,
            "architecture": "tiny_causal",
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "max_text_tokens": 128,
            "max_visual_tokens": 128,
            "text_token_buckets": (32, 64, 128),
            "visual_token_buckets": (32, 64, 128),
        }
        defaults.update(kwargs)
        return cls(
            model=model,
            native_connector=connector,
            processor=processor,
            **defaults,
        )

    @property
    def capabilities(self) -> LanguageModelCapabilities:
        base = super().capabilities
        return LanguageModelCapabilities(
            model_id=base.model_id,
            accepts_inputs_embeds=False,
            native_visual_connector=True,
            supports_generation=True,
            max_text_tokens=base.max_text_tokens,
            max_visual_tokens=base.max_visual_tokens,
        )

    def _forward_model(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None
    ) -> Any:
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
                    "native MedGemma LM lacks its official embedding forward path"
                ) from second

    def _native_project(self, visual_tokens: ProjectedVisualTokens) -> ProjectedVisualTokens:
        if self.native_connector is None:
            raise UnsupportedCapabilityError("MedGemma visual connector is unavailable")
        projected = self.native_connector(visual_tokens.tokens)
        if projected.shape[-1] != self.hidden_size:
            raise ArchitectureMismatchError("MedGemma visual connector output must equal LM hidden size")
        return ProjectedVisualTokens(
            tokens=projected,
            source_modality=visual_tokens.source_modality,
            token_mask=visual_tokens.token_mask,
            coordinate_system=visual_tokens.coordinate_system,
        )

    def forward_with_visual_tokens(
        self,
        text: Any,
        visual_tokens: ProjectedVisualTokens | None,
        labels: torch.Tensor | None,
    ) -> Any:
        if visual_tokens is not None:
            visual_tokens = self._native_project(visual_tokens)
        return super().forward_with_visual_tokens(text, visual_tokens, labels)


__all__ = ["MedGemmaAdapter"]
