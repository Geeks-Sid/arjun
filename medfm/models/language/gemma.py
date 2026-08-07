"""Gemma-family causal language adapter."""

from __future__ import annotations

from typing import Any

from medfm.models.language.base import GenericHFCausalLMAdapter, LanguageAdapterError


class GemmaCausalLMAdapter(GenericHFCausalLMAdapter):
    """External-token Gemma adapter with explicit causal architecture checks."""

    supported_gemma_architectures = frozenset({"gemma", "gemma2", "gemma3", "tiny_causal"})

    def __init__(self, *, architecture: str | None = None, **kwargs: Any) -> None:
        resolved = architecture or str(getattr(getattr(kwargs.get("model"), "config", None), "model_type", "gemma3"))
        if resolved.lower() not in self.supported_gemma_architectures:
            raise LanguageAdapterError(f"GemmaCausalLMAdapter requires a Gemma architecture, got {resolved!r}")
        super().__init__(architecture=resolved, **kwargs)

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "gemma-causal-tiny",
        hidden_size: int = 32,
        vocab_size: int = 64,
        construction_seed: int = 0,
        **kwargs: Any,
    ) -> GemmaCausalLMAdapter:
        return super().build_tiny(
            model_id=model_id,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            construction_seed=construction_seed,
            **kwargs,
        )  # type: ignore[return-value]


__all__ = ["GemmaCausalLMAdapter"]
