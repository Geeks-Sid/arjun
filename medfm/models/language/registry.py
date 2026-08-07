"""Small explicit registry for native versus external language pathways."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from medfm.core.errors import ShapeContractError
from medfm.core.language import LanguageModelCapabilities
from medfm.models.language.base import GenericHFCausalLMAdapter
from medfm.models.language.gemma import GemmaCausalLMAdapter
from medfm.models.language.m3d_lamed import M3DLaMedAdapter
from medfm.models.language.medgemma import MedGemmaAdapter


class LanguageAdapterMode(StrEnum):
    EXTERNAL = "external"
    NATIVE = "native"


@dataclass(frozen=True)
class LanguageAdapterDescriptor:
    name: str
    mode: LanguageAdapterMode
    builder: Callable[..., Any]
    capability: LanguageModelCapabilities
    research_gate: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ShapeContractError("language adapter descriptor name must be non-empty")
        if self.mode is LanguageAdapterMode.NATIVE and not self.capability.native_visual_connector:
            raise ShapeContractError("native language descriptors require native_visual_connector")


_LANGUAGE_DESCRIPTORS: dict[str, LanguageAdapterDescriptor] = {
    "generic_hf_causal": LanguageAdapterDescriptor(
        "generic_hf_causal",
        LanguageAdapterMode.EXTERNAL,
        GenericHFCausalLMAdapter,
        LanguageModelCapabilities("generic-hf-causal-lm", True, False, True, 1024, 128),
    ),
    "gemma_causal": LanguageAdapterDescriptor(
        "gemma_causal",
        LanguageAdapterMode.EXTERNAL,
        GemmaCausalLMAdapter,
        LanguageModelCapabilities("gemma-causal", True, False, True, 1024, 128),
    ),
    "medgemma_native": LanguageAdapterDescriptor(
        "medgemma_native",
        LanguageAdapterMode.NATIVE,
        MedGemmaAdapter,
        LanguageModelCapabilities("medgemma", False, True, True, 1024, 128),
    ),
    "m3d_lamed_external": LanguageAdapterDescriptor(
        "m3d_lamed_external",
        LanguageAdapterMode.EXTERNAL,
        M3DLaMedAdapter,
        LanguageModelCapabilities("m3d-lamed", True, False, True, 1024, 128),
        research_gate=M3DLaMedAdapter.research_gate,
    ),
}


def language_descriptors() -> dict[str, LanguageAdapterDescriptor]:
    return dict(_LANGUAGE_DESCRIPTORS)


def get_language_descriptor(name: str) -> LanguageAdapterDescriptor:
    try:
        return _LANGUAGE_DESCRIPTORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown language adapter {name!r}; choose from {sorted(_LANGUAGE_DESCRIPTORS)}") from exc


def build_language_adapter(name: str, *, tiny: bool = False, **kwargs: Any) -> Any:
    descriptor = get_language_descriptor(name)
    if tiny and hasattr(descriptor.builder, "build_tiny"):
        return descriptor.builder.build_tiny(**kwargs)
    return descriptor.builder(**kwargs)


__all__ = [
    "LanguageAdapterDescriptor",
    "LanguageAdapterMode",
    "build_language_adapter",
    "get_language_descriptor",
    "language_descriptors",
]
