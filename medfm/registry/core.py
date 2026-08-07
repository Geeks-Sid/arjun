"""Core model registry and capability discovery."""

from __future__ import annotations

import dataclasses
import logging

from medfm.core.enums import LoadingMode, Modality, TaskType
from medfm.registry.schema import (
    BACKEND_KEYS,
    SUPPORTED_BACKEND_STATUSES,
    BackendStatus,
    BackendSupport,
    LicenseClass,
    ModelSpec,
    ModelStatus,
    WeightFormat,
)

logger = logging.getLogger(__name__)

#: Backends that execute via PyTorch/XLA.
_TPU_BACKENDS = frozenset({"tpu_single_host", "tpu_multi_host"})
#: bitsandbytes-backed weight formats (CUDA-only upstream).
_BNB_FORMATS = frozenset({WeightFormat.INT8, WeightFormat.NF4})


class ModelRegistry:
    """Central registry for model capabilities and metadata.

    Registration is in-memory and explicit: adapters call ``register`` at
    import time; the v1 catalog is loaded via ``medfm.registry.catalog``.
    """

    _models: dict[str, ModelSpec] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        """Register a ModelSpec; duplicate IDs and unsafe aliases are rejected."""
        if spec.model_id in cls._models:
            raise ValueError(f"Duplicate model ID: {spec.model_id}")
        if spec.model_id in cls._aliases:
            raise ValueError(f"Model ID {spec.model_id} is already used as an alias.")
        for alias in spec.aliases:
            if alias in cls._aliases:
                raise ValueError(f"Duplicate alias: {alias}")
            if alias in cls._models:
                raise ValueError(f"Alias {alias} conflicts with an existing model ID.")

        cls._models[spec.model_id] = spec
        for alias in spec.aliases:
            cls._aliases[alias] = spec.model_id

    @classmethod
    def get(cls, model_id: str) -> ModelSpec:
        """Get a ModelSpec by ID or alias."""
        target_id = cls._aliases.get(model_id, model_id)
        if target_id not in cls._models:
            raise KeyError(f"Unknown model: {model_id}")
        return cls._models[target_id]

    @classmethod
    def list_models(
        cls,
        modality: Modality | None = None,
        task: TaskType | None = None,
        loading_mode: LoadingMode | None = None,
        license_class: LicenseClass | None = None,
        backend: str | None = None,
        include_blocked: bool = False,
        include_deprecated: bool = False,
    ) -> list[ModelSpec]:
        """Discover models matching capability criteria.

        ``backend`` filters to models with a SUPPORTED_* status (with smoke
        evidence) on that backend key; UNTESTED/BLOCKED do not match.
        """
        if backend is not None and backend not in BACKEND_KEYS:
            raise ValueError(f"unknown backend key: {backend}")

        results = []
        for spec in cls._models.values():
            if not include_blocked and spec.status == ModelStatus.BLOCKED:
                continue
            if not include_deprecated and spec.deprecated:
                continue
            if modality and modality not in spec.capabilities.modalities:
                continue
            if task and task not in spec.capabilities.tasks:
                continue
            if loading_mode and loading_mode not in spec.memory.loading_modes:
                continue
            if license_class and spec.license.class_type != license_class:
                continue
            if backend:
                support = spec.backend_support.get(backend)
                if support is None or support.status not in SUPPORTED_BACKEND_STATUSES:
                    continue
            results.append(spec)
        return results

    @classmethod
    def record_backend_result(
        cls,
        model_id: str,
        backend: str,
        revision: str,
        success: bool,
        date: str,
    ) -> ModelSpec:
        """Record a smoke result for exactly one backend, atomically.

        The revision must match the registered revision: evidence gathered on
        any other revision does not apply. A CUDA result never mutates TPU (or
        any other backend) status — each backend key is updated independently
        and the frozen ModelSpec is replaced in one operation.
        """
        if backend not in BACKEND_KEYS:
            raise ValueError(f"unknown backend key: {backend}")
        spec = cls.get(model_id)
        if revision != spec.revision:
            raise ValueError(
                f"smoke revision {revision} does not match registered revision {spec.revision} for {model_id}"
            )

        new_support = dict(spec.backend_support)
        if success:
            promoted = {
                "cpu": BackendStatus.CPU_CONTRACT_ONLY,
                "cuda_single": BackendStatus.SUPPORTED_SINGLE_DEVICE,
                "cuda_distributed": BackendStatus.SUPPORTED_REPLICATED,
                "tpu_single_host": BackendStatus.SUPPORTED_SINGLE_DEVICE,
                "tpu_multi_host": BackendStatus.SUPPORTED_SHARDED,
            }[backend]
            new_support[backend] = BackendSupport(status=promoted, smoke_revision=revision, smoke_date=date)
        else:
            # Failure keeps the previous status unless it was UNTESTED, in
            # which case it stays UNTESTED with no evidence recorded.
            new_support[backend] = spec.backend_support.get(backend, BackendSupport())

        updated = dataclasses.replace(spec, backend_support=new_support, last_smoke_revision=revision)
        cls._models[spec.model_id] = updated
        return updated

    @classmethod
    def validate_backend(cls, spec: ModelSpec, backend: str, loading_mode: LoadingMode) -> None:
        """Reject unsupported model/mode/backend combinations before any weight
        allocation or model construction happens."""
        if backend not in BACKEND_KEYS:
            raise ValueError(f"unknown backend key: {backend}")
        if loading_mode not in spec.memory.loading_modes:
            raise ValueError(f"Loading mode {loading_mode} not supported by {spec.model_id}")

        estimate = spec.memory.loading_modes[loading_mode]
        if backend in _TPU_BACKENDS:
            if estimate.weight_format in _BNB_FORMATS:
                raise ValueError(
                    f"bitsandbytes {estimate.weight_format.value} is not supported on "
                    f"{backend} (xla_tpu); rejected for {spec.model_id}. Use BF16 LoRA."
                )
            if spec.capabilities.cuda_only_extensions and not spec.capabilities.pure_pytorch_fallback:
                raise ValueError(
                    f"Model {spec.model_id} has CUDA-only extensions without a tested "
                    f"pure-PyTorch/SDPA path; not eligible for {backend}."
                )

    @classmethod
    def accelerator_report(cls) -> dict[str, dict[str, str]]:
        """Per-model, per-backend compatibility report.

        Values are BackendStatus strings; SUPPORTED_* always carries recorded
        smoke evidence (enforced by BackendSupport validation).
        """
        return {
            model_id: {
                backend: spec.backend_support.get(backend, BackendSupport()).status.value for backend in BACKEND_KEYS
            }
            for model_id, spec in sorted(cls._models.items())
        }

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (for testing)."""
        cls._models.clear()
        cls._aliases.clear()
