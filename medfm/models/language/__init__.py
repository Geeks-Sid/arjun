"""Causal language adapters and native VLM wrappers (Phase 09)."""

from medfm.core.language import (
    GeneratedText,
    GenerationConfig,
    LanguageModelAdapter,
    LanguageModelCapabilities,
    LanguageOutput,
    ProjectedVisualTokens,
    TokenizedText,
)
from medfm.models.language.base import (
    ArchitectureMismatchError,
    ChatTemplateConfig,
    GenericHFCausalLMAdapter,
    LanguageAdapterConfig,
    LanguageAdapterError,
    LanguageDependencyError,
)
from medfm.models.language.configs import (
    DEFAULT_BUCKETS,
    EXTERNAL_2D_RECIPE,
    EXTERNAL_3D_RECIPE,
    EXTERNAL_WSI_RECIPE,
    NATIVE_MEDGEMMA_RECIPE,
    LanguageBridgeRecipe,
    Phase09ShapeBuckets,
)
from medfm.models.language.gemma import GemmaCausalLMAdapter
from medfm.models.language.m3d_lamed import M3DLaMedAdapter
from medfm.models.language.medgemma import MedGemmaAdapter
from medfm.models.language.registry import (
    LanguageAdapterDescriptor,
    LanguageAdapterMode,
    build_language_adapter,
    get_language_descriptor,
    language_descriptors,
)

__all__ = [
    "ArchitectureMismatchError",
    "ChatTemplateConfig",
    "DEFAULT_BUCKETS",
    "EXTERNAL_2D_RECIPE",
    "EXTERNAL_3D_RECIPE",
    "EXTERNAL_WSI_RECIPE",
    "GeneratedText",
    "GenerationConfig",
    "GenericHFCausalLMAdapter",
    "GemmaCausalLMAdapter",
    "LanguageAdapterConfig",
    "LanguageAdapterDescriptor",
    "LanguageAdapterError",
    "LanguageAdapterMode",
    "LanguageBridgeRecipe",
    "LanguageDependencyError",
    "LanguageModelAdapter",
    "LanguageModelCapabilities",
    "LanguageOutput",
    "M3DLaMedAdapter",
    "MedGemmaAdapter",
    "NATIVE_MEDGEMMA_RECIPE",
    "Phase09ShapeBuckets",
    "ProjectedVisualTokens",
    "TokenizedText",
    "build_language_adapter",
    "get_language_descriptor",
    "language_descriptors",
]
