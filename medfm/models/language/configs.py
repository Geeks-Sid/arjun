"""Versioned Phase 09 language/bridge configuration declarations."""

from __future__ import annotations

from dataclasses import dataclass

from medfm.core.errors import ShapeContractError


@dataclass(frozen=True)
class Phase09ShapeBuckets:
    """Static token buckets shared by CUDA parity and XLA compilation."""

    visual_tokens: tuple[int, ...] = (32, 64, 128)
    text_tokens: tuple[int, ...] = (256, 512, 1024)
    attention: str = "sdpa"

    def __post_init__(self) -> None:
        for name, values in (("visual_tokens", self.visual_tokens), ("text_tokens", self.text_tokens)):
            if not values or tuple(sorted(set(values))) != values or any(value <= 0 for value in values):
                raise ShapeContractError(f"{name} must be sorted, unique, and positive")
        if self.attention not in {"eager", "sdpa", "flash_attention_2", "xla"}:
            raise ShapeContractError("attention must be eager, sdpa, flash_attention_2, or xla")

    def bucket_for_visual(self, count: int) -> int:
        for bucket in self.visual_tokens:
            if count <= bucket:
                return bucket
        raise ShapeContractError(f"visual token count {count} exceeds configured buckets {self.visual_tokens}")

    def bucket_for_text(self, count: int) -> int:
        for bucket in self.text_tokens:
            if count <= bucket:
                return bucket
        raise ShapeContractError(f"text token count {count} exceeds configured buckets {self.text_tokens}")


@dataclass(frozen=True)
class LanguageBridgeRecipe:
    """Separate native and external mode names for registry/config consumers."""

    name: str
    mode: str
    language_adapter: str
    bridge: str | None
    buckets: Phase09ShapeBuckets = Phase09ShapeBuckets()

    def __post_init__(self) -> None:
        if not self.name or not self.language_adapter:
            raise ShapeContractError("recipe name and language_adapter must be non-empty")
        if self.mode not in {"native", "external"}:
            raise ShapeContractError("recipe mode must be native or external")
        if self.mode == "native" and self.bridge is not None:
            raise ShapeContractError("native recipes do not declare an external bridge")
        if self.mode == "external" and not self.bridge:
            raise ShapeContractError("external recipes require a bridge")


DEFAULT_BUCKETS = Phase09ShapeBuckets()
EXTERNAL_2D_RECIPE = LanguageBridgeRecipe(
    name="external-2d-gemma-v1",
    mode="external",
    language_adapter="gemma_causal",
    bridge="mlp",
)
EXTERNAL_3D_RECIPE = LanguageBridgeRecipe(
    name="external-3d-gemma-v1",
    mode="external",
    language_adapter="gemma_causal",
    bridge="perceiver_resampler",
)
EXTERNAL_WSI_RECIPE = LanguageBridgeRecipe(
    name="external-wsi-gemma-v1",
    mode="external",
    language_adapter="gemma_causal",
    bridge="mlp",
)
NATIVE_MEDGEMMA_RECIPE = LanguageBridgeRecipe(
    name="native-medgemma-v1",
    mode="native",
    language_adapter="medgemma_native",
    bridge=None,
)


__all__ = [
    "DEFAULT_BUCKETS",
    "EXTERNAL_2D_RECIPE",
    "EXTERNAL_3D_RECIPE",
    "EXTERNAL_WSI_RECIPE",
    "LanguageBridgeRecipe",
    "NATIVE_MEDGEMMA_RECIPE",
    "Phase09ShapeBuckets",
]
