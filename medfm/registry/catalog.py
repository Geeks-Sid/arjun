"""v1 model catalog: real roster from model_registry/ YAMLs, fail-closed.

Every v1 model is registered as READY or BLOCKED with a structured reason.
License records that are unresolved or pending review are BLOCKING (phase-00
policy: terms are never guessed), so models without an approved license load
as BLOCKED with the review owner named. Revisions stay as placeholders until
the pinned SHA is recorded after license acceptance.

Adapter phases (06-08) replace placeholder capabilities with verified ones
and attach real PreprocessSpecs; the registry internals do not change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medfm.core.enums import LoadingMode, Modality, TaskType
from medfm.registry.core import ModelRegistry
from medfm.registry.schema import (
    BACKEND_KEYS,
    BackendStatus,
    BackendSupport,
    FeatureStatus,
    LicenseClass,
    LicenseSpec,
    LicenseStatus,
    MemoryEstimate,
    MemoryProfile,
    ModelCapability,
    ModelSpec,
    ModelStatus,
    OutputCapability,
    PeftCapability,
    WeightFormat,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LICENSES_PATH = REPO_ROOT / "model_registry" / "licenses.yaml"
SCOPE_PATH = REPO_ROOT / "model_registry" / "v1_scope.yaml"

#: Rough parameter counts (billions) for conservative pre-adapter estimates.
_APPROX_PARAMS_B: dict[str, float] = {
    "medsiglip": 0.4,
    "rad-dino": 0.09,
    "medgemma-1.5-4b": 4.0,
    "h-optimus-0": 1.1,
    "conch": 0.1,
    "ct-fm": 0.1,
    "flexict-3d": 0.1,
    "merlin": 0.2,
    "m3d-lamed": 8.0,
    "triad": 0.1,
    "nv-segment-ctmr": 0.5,
    "brainiac": 0.1,
    "medsam2": 0.2,
    "gigapath-flash": 1.1,
    "titan": 0.7,
    "gemma-generic": 4.0,
    "qwen-generic": 7.0,
}

#: Declared output capabilities per model; must agree with declared tasks.
_OUTPUTS: dict[str, tuple[OutputCapability, ...]] = {
    "medsiglip": (
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.SPATIAL_TOKENS,
        OutputCapability.FEATURE_MAPS,
        OutputCapability.HIDDEN_STATES,
    ),
    "rad-dino": (
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.SPATIAL_TOKENS,
        OutputCapability.FEATURE_MAPS,
        OutputCapability.HIDDEN_STATES,
    ),
    "medgemma-1.5-4b": (OutputCapability.NATIVE_TEXT, OutputCapability.HIDDEN_STATES),
    "h-optimus-0": (
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.SPATIAL_TOKENS,
        OutputCapability.FEATURE_MAPS,
    ),
    "conch": (OutputCapability.POOLED_EMBEDDINGS, OutputCapability.SPATIAL_TOKENS),
    "ct-fm": (
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.FEATURE_MAPS,
        OutputCapability.HIDDEN_STATES,
    ),
    "flexict-3d": (
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.SPATIAL_TOKENS,
        OutputCapability.FEATURE_MAPS,
    ),
    "merlin": (OutputCapability.NATIVE_TEXT, OutputCapability.POOLED_EMBEDDINGS),
    "m3d-lamed": (OutputCapability.NATIVE_TEXT, OutputCapability.POOLED_EMBEDDINGS),
    "triad": (OutputCapability.POOLED_EMBEDDINGS, OutputCapability.FEATURE_MAPS),
    "nv-segment-ctmr": (OutputCapability.FEATURE_MAPS, OutputCapability.SPATIAL_TOKENS),
    "brainiac": (OutputCapability.POOLED_EMBEDDINGS,),
    "medsam2": (OutputCapability.FEATURE_MAPS, OutputCapability.SPATIAL_TOKENS),
    "gigapath-flash": (OutputCapability.POOLED_EMBEDDINGS, OutputCapability.SPATIAL_TOKENS),
    "titan": (
        OutputCapability.NATIVE_TEXT,
        OutputCapability.POOLED_EMBEDDINGS,
        OutputCapability.SPATIAL_TOKENS,
    ),
    "gemma-generic": (OutputCapability.NATIVE_TEXT,),
    "qwen-generic": (OutputCapability.NATIVE_TEXT,),
}

_GENERATIVE = {"medgemma-1.5-4b", "merlin", "m3d-lamed", "titan", "gemma-generic", "qwen-generic"}


def _license_spec(record: dict[str, Any]) -> tuple[LicenseSpec, str | None]:
    """Map a phase-00 license record to a runtime LicenseSpec + block reason."""
    status_raw = record["status"]
    commercial = record["commercial_use"]
    # Deployment class requires explicitly permitted commercial use; anything
    # less (conditional/unresolved/prohibited) stays out of that catalog.
    class_type = LicenseClass.DEPLOYMENT if commercial == "permitted" else LicenseClass.RESEARCH
    gated = bool(record["gated_access"])

    if status_raw in ("approved_research", "approved_commercial"):
        return (
            LicenseSpec(
                name=record["weights_license"],
                status=LicenseStatus.APPROVED,
                class_type=class_type,
                approved_use_cases=tuple(record["approved_use_cases"]),
                prohibited_use_cases=tuple(record["prohibited_use_cases"]),
                terms_url=record["repository"] if gated else None,
                gated=gated,
                acceptance_required=gated,
            ),
            None,
        )

    reason = (
        f"license status '{status_raw}' (owner: {record['review_owner']}, "
        f"review due {record['review_date']}): unresolved terms are blocking"
    )
    return (
        LicenseSpec(
            name=record["weights_license"],
            status=LicenseStatus.UNKNOWN,
            class_type=class_type,
            approved_use_cases=tuple(record["approved_use_cases"]),
            prohibited_use_cases=tuple(record["prohibited_use_cases"]),
            terms_url=record["repository"] if gated else None,
            gated=gated,
            acceptance_required=gated,
        ),
        reason,
    )


def _memory_profile(model_id: str, override: Any | None = None) -> MemoryProfile:
    params = override.parameters_b if override is not None else _APPROX_PARAMS_B[model_id]
    bf16_bytes = int(params * 2 * 1e9)
    nf4_bytes = int(params * 0.55 * 1e9)
    note = (
        "adapter-verified parameter count"
        if override is not None
        else "approximate parameter count; pre-adapter conservative estimate"
    )
    modes = {
        LoadingMode.FULL: MemoryEstimate(
            host_bytes=bf16_bytes,
            device_bytes=bf16_bytes,
            weight_format=WeightFormat.BF16,
            uncertainty_note=note,
        ),
        LoadingMode.LORA: MemoryEstimate(
            host_bytes=bf16_bytes,
            device_bytes=int(bf16_bytes * 1.3),
            weight_format=WeightFormat.BF16,
            uncertainty_note="BF16 weights + adapter/optimizer headroom; pre-adapter estimate",
        ),
        LoadingMode.FROZEN: MemoryEstimate(
            host_bytes=bf16_bytes,
            device_bytes=bf16_bytes,
            weight_format=WeightFormat.BF16,
            frozen_cache=True,
            uncertainty_note="frozen BF16 extraction; no optimizer state",
        ),
    }
    if model_id in _GENERATIVE:
        modes[LoadingMode.QLORA_NF4] = MemoryEstimate(
            host_bytes=nf4_bytes,
            device_bytes=int(nf4_bytes * 1.5),
            weight_format=WeightFormat.NF4,
            uncertainty_note="bitsandbytes NF4, CUDA-only; rejected on xla_tpu",
        )
    return MemoryProfile(
        parameters_b=params,
        max_seq_len=None,
        loading_modes=modes,
        compile_risk_note=(
            override.compile_risk_note
            if override is not None
            else "unverified until adapter smoke; dynamic shapes possible"
        ),
    )


def ensure_v1_catalog() -> None:
    """Idempotent catalog load: safe to call from CLI entry points and tests."""
    try:
        ModelRegistry.get("rad-dino")
        return  # already loaded
    except KeyError:
        pass
    load_v1_catalog()


def load_v1_catalog() -> list[ModelSpec]:
    """Register every model from the phase-00 roster; returns the specs."""
    import yaml

    licenses = yaml.safe_load(LICENSES_PATH.read_text())
    scope = yaml.safe_load(SCOPE_PATH.read_text())
    scope_models = {m["model_id"]: m for m in scope["models"]}
    from medfm.models.visual.specs import adapter_spec_overrides, register_2d_plugins

    overrides = adapter_spec_overrides()

    specs: list[ModelSpec] = []
    for model_id, record in sorted(licenses.items()):
        scope_entry = scope_models[model_id]
        license_spec, block_reason = _license_spec(record)

        backend_support = {
            key: BackendSupport(status=BackendStatus(s)) for key, s in scope_entry["accelerator_support"].items()
        }
        assert set(backend_support) == set(BACKEND_KEYS)

        repository = record["repository"]
        override = overrides.get(model_id)
        if override is not None:
            # Adapter phases replace placeholder capabilities with verified ones
            # (pinned revision, real preprocess, declared LoRA targets).
            peft = PeftCapability(supported=True, known_target_modules=override.peft_known_target_modules)
            preprocess = override.preprocess
            spatial_tokens_status = override.spatial_tokens_status
            custom_operators = override.custom_operators
            pure_pytorch_fallback = override.pure_pytorch_fallback
            aliases = override.aliases
            revision = override.revision
            memory = _memory_profile(model_id, override)
        else:
            peft = PeftCapability(supported=True, unknown_family_confirmation_required=True)
            preprocess = None
            spatial_tokens_status = FeatureStatus.UNAVAILABLE
            custom_operators = ()
            pure_pytorch_fallback = True
            aliases = ()
            revision = "unresolved-pending-license-acceptance"
            memory = _memory_profile(model_id)
        spec = ModelSpec(
            model_id=model_id,
            repository=repository if repository != "unresolved" else f"unresolved:{model_id}",
            # Adapter-covered models carry their pinned SHA (public metadata);
            # other roster models keep the placeholder until license acceptance.
            revision=revision,
            license=license_spec,
            capabilities=ModelCapability(
                modalities=tuple(Modality(m) for m in scope_entry["modalities"]),
                tasks=tuple(TaskType(t) for t in scope_entry["tasks"]),
                output_types=_OUTPUTS[model_id],
                spatial_tokens_status=spatial_tokens_status,
                peft=peft,
                custom_operators=custom_operators,
                pure_pytorch_fallback=pure_pytorch_fallback,
            ),
            memory=memory,
            status=ModelStatus.BLOCKED if block_reason else ModelStatus.READY,
            blocked_reason=block_reason,
            backend_support=backend_support,
            aliases=aliases,
            preprocess=preprocess,
        )
        ModelRegistry.register(spec)
        specs.append(spec)
    register_2d_plugins()
    return specs
