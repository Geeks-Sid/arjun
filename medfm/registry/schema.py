"""Registry schemas for model capabilities, licenses, and weight metadata.

The registry schema is versioned (``REGISTRY_SCHEMA_VERSION``) and frozen for
the duration of the adapter phases (06-08). Adapters register ModelSpecs
against this schema; changes require a schema version bump and migration notes
in the phase handoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medfm.core.enums import LoadingMode, Modality, PrecisionMode, StrictStrEnum, TaskType
from medfm.data.transforms.specs import PreprocessSpec

#: Frozen registry schema version for adapter phases 06-08.
REGISTRY_SCHEMA_VERSION = 1

#: Per-backend keys are mandatory for every model; no blanket cross-backend key
#: is allowed (matches model_registry/v1_scope.yaml ``backend_keys``).
BACKEND_KEYS: tuple[str, ...] = (
    "cpu",
    "cuda_single",
    "cuda_distributed",
    "tpu_single_host",
    "tpu_multi_host",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")


class LicenseStatus(StrictStrEnum):
    """Runtime load-gate status; distinct from the phase-00 review workflow."""

    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"
    EXPIRED = "EXPIRED"
    PROHIBITED = "PROHIBITED"


class LicenseClass(StrictStrEnum):
    """Catalog separation: research-only models never appear in deployment."""

    RESEARCH = "RESEARCH"
    DEPLOYMENT = "DEPLOYMENT"


class FeatureStatus(StrictStrEnum):
    """How a feature (e.g. spatial tokens) is exposed by an architecture."""

    NATIVE = "NATIVE"
    HOOKED = "HOOKED"
    UNAVAILABLE = "UNAVAILABLE"


class ModelStatus(StrictStrEnum):
    """Top-level model availability in the registry."""

    READY = "READY"
    BLOCKED = "BLOCKED"
    UNTESTED = "UNTESTED"


class BackendStatus(StrictStrEnum):
    """Accelerator capability status vocabulary (v1_scope backend_status_enum).

    Source inspection alone never promotes a status past UNTESTED: any
    SUPPORTED_* status requires recorded smoke evidence.
    """

    UNTESTED = "UNTESTED"
    CPU_CONTRACT_ONLY = "CPU_CONTRACT_ONLY"
    SUPPORTED_SINGLE_DEVICE = "SUPPORTED_SINGLE_DEVICE"
    SUPPORTED_REPLICATED = "SUPPORTED_REPLICATED"
    SUPPORTED_SHARDED = "SUPPORTED_SHARDED"
    BLOCKED_CUSTOM_OP = "BLOCKED_CUSTOM_OP"
    BLOCKED_MEMORY = "BLOCKED_MEMORY"
    BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
    NOT_APPLICABLE = "NOT_APPLICABLE"


SUPPORTED_BACKEND_STATUSES: frozenset[BackendStatus] = frozenset(
    {
        BackendStatus.SUPPORTED_SINGLE_DEVICE,
        BackendStatus.SUPPORTED_REPLICATED,
        BackendStatus.SUPPORTED_SHARDED,
        BackendStatus.CPU_CONTRACT_ONLY,
    }
)


class WeightFormat(StrictStrEnum):
    """Weight precision/quantization of a loading configuration."""

    FP32 = "FP32"
    BF16 = "BF16"
    FP16 = "FP16"
    INT8 = "INT8"
    NF4 = "NF4"


class Topology(StrictStrEnum):
    """Execution topology for a loading configuration."""

    SINGLE_DEVICE = "SINGLE_DEVICE"
    REPLICATED = "REPLICATED"
    SPMD_FSDP = "SPMD_FSDP"


class AttentionBackend(StrictStrEnum):
    EAGER = "EAGER"
    SDPA = "SDPA"
    FLASH_ATTENTION_2 = "FLASH_ATTENTION_2"
    XFORMERS = "XFORMERS"


class OutputCapability(StrictStrEnum):
    """Structured output types a model can expose."""

    NATIVE_TEXT = "NATIVE_TEXT"
    POOLED_EMBEDDINGS = "POOLED_EMBEDDINGS"
    SPATIAL_TOKENS = "SPATIAL_TOKENS"
    FEATURE_MAPS = "FEATURE_MAPS"
    HIDDEN_STATES = "HIDDEN_STATES"


#: Outputs each task family requires; validated against declared capabilities.
TASK_REQUIRED_OUTPUTS: dict[TaskType, frozenset[OutputCapability]] = {
    TaskType.BINARY_CLASSIFICATION: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.MULTICLASS_CLASSIFICATION: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.MULTILABEL_CLASSIFICATION: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.ORDINAL_CLASSIFICATION: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.IMAGE_TEXT_RETRIEVAL: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.TEXT_IMAGE_RETRIEVAL: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.CONTRASTIVE_ALIGNMENT: frozenset({OutputCapability.POOLED_EMBEDDINGS}),
    TaskType.SEMANTIC_SEGMENTATION: frozenset({OutputCapability.FEATURE_MAPS, OutputCapability.SPATIAL_TOKENS}),
    TaskType.PROMPTABLE_SEGMENTATION: frozenset({OutputCapability.FEATURE_MAPS, OutputCapability.SPATIAL_TOKENS}),
    TaskType.VISUAL_QUESTION_ANSWERING: frozenset({OutputCapability.NATIVE_TEXT}),
    TaskType.REPORT_GENERATION: frozenset({OutputCapability.NATIVE_TEXT}),
    TaskType.STRUCTURED_FINDING_GENERATION: frozenset({OutputCapability.NATIVE_TEXT}),
}


@dataclass(frozen=True)
class LicenseSpec:
    """Runtime license record. Approved/prohibited use cases are mandatory so
    inspection output can state them explicitly."""

    name: str
    status: LicenseStatus
    class_type: LicenseClass
    approved_use_cases: tuple[str, ...] = ()
    prohibited_use_cases: tuple[str, ...] = ()
    terms_url: str | None = None
    gated: bool = False
    acceptance_required: bool = False

    def __post_init__(self) -> None:
        if self.status == LicenseStatus.APPROVED and not self.approved_use_cases:
            raise ValueError("APPROVED license must declare approved_use_cases")
        if self.gated and not self.acceptance_required:
            raise ValueError("gated license must set acceptance_required=True")
        if self.acceptance_required and not self.terms_url:
            raise ValueError("acceptance_required license must provide terms_url")


@dataclass(frozen=True)
class MemoryEstimate:
    """Conservative estimate for one loading configuration.

    ``measured_peak_bytes`` is filled in later from real runs to refine the
    estimate; estimates are inputs, measurements are evidence.
    """

    host_bytes: int
    device_bytes: int
    weight_format: WeightFormat = WeightFormat.BF16
    topology: Topology = Topology.SINGLE_DEVICE
    cpu_offload: bool = False
    frozen_cache: bool = False
    uncertainty_note: str | None = None
    measured_peak_bytes: int | None = None


@dataclass(frozen=True)
class MemoryProfile:
    parameters_b: float
    max_seq_len: int | None
    loading_modes: dict[LoadingMode, MemoryEstimate]
    # Compile risk (dynamic shapes, custom operators) is recorded separately
    # from memory estimates.
    compile_risk_note: str | None = None

    def __post_init__(self) -> None:
        if not self.loading_modes:
            raise ValueError("MemoryProfile must declare at least one loading mode")


@dataclass(frozen=True)
class PeftCapability:
    supported: bool
    known_target_modules: tuple[str, ...] = ()
    unknown_family_confirmation_required: bool = False

    def __post_init__(self) -> None:
        if self.supported and not self.known_target_modules and not self.unknown_family_confirmation_required:
            raise ValueError(
                "PEFT-supported models must list known_target_modules or set unknown_family_confirmation_required=True"
            )


@dataclass(frozen=True)
class BackendSupport:
    """Per-backend support status plus the evidence backing it.

    A SUPPORTED_* status without smoke evidence is invalid: source inspection
    alone cannot promote accelerator support.
    """

    status: BackendStatus = BackendStatus.UNTESTED
    smoke_revision: str | None = None
    smoke_date: str | None = None
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status in SUPPORTED_BACKEND_STATUSES and self.status != BackendStatus.CPU_CONTRACT_ONLY:
            if not self.smoke_revision or not self.smoke_date:
                raise ValueError(f"{self.status.value} requires smoke evidence (smoke_revision + smoke_date)")
        if (
            self.status
            in (
                BackendStatus.BLOCKED_CUSTOM_OP,
                BackendStatus.BLOCKED_MEMORY,
                BackendStatus.BLOCKED_UPSTREAM,
            )
            and not self.blocked_reason
        ):
            raise ValueError(f"{self.status.value} requires a blocked_reason")


@dataclass(frozen=True)
class ModelCapability:
    modalities: tuple[Modality, ...]
    tasks: tuple[TaskType, ...]
    output_types: tuple[OutputCapability, ...] = ()
    spatial_tokens_status: FeatureStatus = FeatureStatus.UNAVAILABLE
    peft: PeftCapability = field(default_factory=lambda: PeftCapability(supported=False))
    cuda_only_extensions: bool = False
    pure_pytorch_fallback: bool = True
    custom_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("At least one modality must be specified")
        if not self.tasks:
            raise ValueError("At least one task must be specified")
        if self.cuda_only_extensions and self.pure_pytorch_fallback:
            raise ValueError("cuda_only_extensions=True contradicts pure_pytorch_fallback=True")
        # Output capabilities must agree with task declarations.
        for task in self.tasks:
            required = TASK_REQUIRED_OUTPUTS.get(task)
            if required is None:
                continue  # task with no fixed output requirement (e.g. MULTITASK)
            if task in (
                TaskType.SEMANTIC_SEGMENTATION,
                TaskType.PROMPTABLE_SEGMENTATION,
            ):
                # any-of: feature maps OR spatial tokens
                if not required.intersection(self.output_types):
                    raise ValueError(f"{task.value} requires FEATURE_MAPS or SPATIAL_TOKENS outputs")
            elif not required.issubset(self.output_types):
                missing = sorted(o.value for o in required.difference(self.output_types))
                raise ValueError(f"{task.value} requires output capabilities {missing}")


@dataclass(frozen=True)
class ModelSpec:
    """One registry record. Frozen: per-backend evidence updates go through
    ModelRegistry.record_backend_result, which replaces the record atomically."""

    model_id: str
    repository: str
    revision: str  # pinned commit SHA; placeholder allowed only when BLOCKED
    license: LicenseSpec
    capabilities: ModelCapability
    memory: MemoryProfile
    preprocess: PreprocessSpec | None = None
    trust_remote_code_allowed: bool = False
    status: ModelStatus = ModelStatus.READY
    blocked_reason: str | None = None
    aliases: tuple[str, ...] = ()
    backend_support: dict[str, BackendSupport] = field(default_factory=dict)
    tested_precisions: tuple[PrecisionMode, ...] = ()
    tested_attention_backends: tuple[AttentionBackend, ...] = ()
    last_smoke_revision: str | None = None
    schema_version: int = REGISTRY_SCHEMA_VERSION
    deprecated: bool = False
    replaced_by: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} != frozen registry schema {REGISTRY_SCHEMA_VERSION}"
            )
        if self.status == ModelStatus.READY:
            # Production-loadable records must pin an exact commit SHA.
            if not _SHA_RE.match(self.revision):
                raise ValueError("READY models must pin revision to a commit SHA")
        elif not self.revision:
            raise ValueError("revision must be provided (or a placeholder when BLOCKED)")

        if self.status == ModelStatus.BLOCKED and not self.blocked_reason:
            raise ValueError("blocked_reason must be provided if status is BLOCKED")

        if self.license.status != LicenseStatus.APPROVED and self.status == ModelStatus.READY:
            raise ValueError("Models with non-APPROVED licenses must have status=BLOCKED")

        if self.license.class_type == LicenseClass.RESEARCH and self.status == ModelStatus.READY:
            # Research-only models can be READY for research loads but must never
            # surface in the deployment catalog; enforced at query time.
            pass

        unknown_backends = set(self.backend_support) - set(BACKEND_KEYS)
        if unknown_backends:
            raise ValueError(f"unknown backend keys: {sorted(unknown_backends)}")
        if len(self.backend_support) != len(BACKEND_KEYS):
            missing = sorted(set(BACKEND_KEYS) - set(self.backend_support))
            raise ValueError(f"backend_support must declare every backend; missing {missing}")

        if self.deprecated and not self.replaced_by:
            raise ValueError("deprecated models must name a replaced_by successor")

        for alias in self.aliases:
            if not alias or any(c.isspace() for c in alias):
                raise ValueError(f"unsafe alias: {alias!r}")

        if self.preprocess is not None:
            self._validate_preprocess_completeness()

    def _validate_preprocess_completeness(self) -> None:
        """Pixel-input models need a complete normalization declaration."""
        spec = self.preprocess
        assert spec is not None
        norm = spec.normalization
        if norm is None:
            raise ValueError("preprocess spec is missing normalization")
        if len(norm.mean) != spec.channels or len(norm.std) != spec.channels:
            raise ValueError(
                "normalization mean/std must have one entry per channel "
                f"({spec.channels}); got {len(norm.mean)}/{len(norm.std)}"
            )
