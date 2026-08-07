"""Research/license-gated M3D-LaMed language integration."""

from __future__ import annotations

from typing import Any

from medfm.core.language import LanguageModelCapabilities
from medfm.models.language.base import GenericHFCausalLMAdapter, LanguageAdapterError


class M3DLaMedAdapter(GenericHFCausalLMAdapter):
    """M3D-LaMed boundary; upstream weights remain explicitly gated."""

    research_gate = "M3D-LaMed checkpoint, license, and QLoRA memory review"

    def __init__(
        self,
        *,
        license_accepted: bool = False,
        research_acknowledged: bool = False,
        **kwargs: Any,
    ) -> None:
        provided_model = kwargs.get("model")
        if provided_model is not None and not (license_accepted and research_acknowledged):
            raise LanguageAdapterError(
                "M3D-LaMed integration is research/license gated; pass license_accepted and research_acknowledged"
            )
        kwargs.setdefault("model_id", "m3d-lamed")
        kwargs.setdefault("architecture", "tiny_causal" if provided_model is None else None)
        kwargs.setdefault("mode", "external")
        super().__init__(**kwargs)
        self.license_accepted = bool(license_accepted)
        self.research_acknowledged = bool(research_acknowledged)

    @classmethod
    def build_tiny(
        cls,
        *,
        model_id: str = "m3d-lamed-tiny",
        hidden_size: int = 32,
        vocab_size: int = 64,
        construction_seed: int = 0,
        **kwargs: Any,
    ) -> M3DLaMedAdapter:
        return super().build_tiny(
            model_id=model_id,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            construction_seed=construction_seed,
            license_accepted=True,
            research_acknowledged=True,
            **kwargs,
        )  # type: ignore[return-value]

    @property
    def capabilities(self) -> LanguageModelCapabilities:
        base = super().capabilities
        return LanguageModelCapabilities(
            model_id=base.model_id,
            accepts_inputs_embeds=base.accepts_inputs_embeds,
            native_visual_connector=False,
            supports_generation=base.supports_generation,
            max_text_tokens=base.max_text_tokens,
            max_visual_tokens=base.max_visual_tokens,
        )

    def integration_status(self) -> dict[str, Any]:
        return {
            "research_gate": self.research_gate,
            "license_accepted": self.license_accepted,
            "research_acknowledged": self.research_acknowledged,
            "generation_enabled": self.capabilities.supports_generation,
        }


__all__ = ["M3DLaMedAdapter"]
