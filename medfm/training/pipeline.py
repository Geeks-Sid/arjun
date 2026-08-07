"""Pre-allocation capability gates and explicit training-pipeline stages."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from medfm.core.enums import LoadingMode, TaskType
from medfm.registry import ModelRegistry, catalog
from medfm.registry.schema import BackendStatus, ModelSpec
from medfm.training.backend import AcceleratorBackend, create_backend
from medfm.training.config import RunConfig
from medfm.training.optimizer import OptimizerBundle, build_optimizer


class PipelineBuildError(RuntimeError):
    """A named pipeline stage could not be constructed."""


@dataclass(frozen=True)
class CapabilityReport:
    model_id: str
    backend: str
    loading_mode: str
    precision: str
    distribution: str
    registry_status: str | None
    backend_status: str | None
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "loading_mode": self.loading_mode,
            "precision": self.precision,
            "distribution": self.distribution,
            "registry_status": self.registry_status,
            "backend_status": self.backend_status,
            "issues": list(self.issues),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class ModelSummary:
    model_id: str
    allocated: bool
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    parameter_dtypes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "allocated": self.allocated,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "parameter_dtypes": list(self.parameter_dtypes),
            "notes": list(self.notes),
        }


@dataclass
class ComponentBuilders:
    """Recipe-supplied builders; the engine does not encode model choices."""

    registry: Callable[..., Any] | None = None
    dataset: Callable[..., Any] | None = None
    model: Callable[..., Any] | None = None
    peft: Callable[..., Any] | None = None
    optimizer: Callable[..., Any] | None = None
    task: Callable[..., Any] | None = None
    trainer: Callable[..., Any] | None = None
    evaluator: Callable[..., Any] | None = None
    checkpoint: Callable[..., Any] | None = None


@dataclass
class BuildResult:
    config: RunConfig
    backend: AcceleratorBackend | None = None
    registry: Any | None = None
    dataset: Any | None = None
    model: Any | None = None
    peft_model: Any | None = None
    optimizer: OptimizerBundle | Any | None = None
    task: Any | None = None
    trainer: Any | None = None
    evaluator: Any | None = None
    checkpoint: Any | None = None
    capability_report: CapabilityReport | None = None
    model_summary: ModelSummary | None = None
    stages: list[str] = field(default_factory=list)


_BACKEND_REGISTRY_KEY = {
    "cpu": "cpu",
}


def _registry_backend_key(config: RunConfig) -> str:
    if config.accelerator.backend == "cuda":
        return "cuda_distributed" if config.accelerator.distribution != "single" else "cuda_single"
    if config.accelerator.backend == "xla_tpu":
        return "tpu_multi_host" if config.accelerator.distribution == "spmd_fsdp" else "tpu_single_host"
    return _BACKEND_REGISTRY_KEY[config.accelerator.backend]


def _loading_mode(config: RunConfig) -> LoadingMode:
    if config.quantization.is_qlora:
        return LoadingMode.QLORA_NF4
    if config.peft.enabled:
        return LoadingMode.LORA
    if config.memory.frozen_embedding_mode or config.memory.token_cache_mode:
        return LoadingMode.FROZEN
    return LoadingMode.FULL


def validate_capabilities(config: RunConfig, *, spec: ModelSpec | None = None) -> CapabilityReport:
    """Validate all cheap compatibility rules before a model builder runs."""
    issues: list[str] = []
    loading = _loading_mode(config)
    registry_status: str | None = None
    backend_status: str | None = None
    if spec is not None:
        registry_status = spec.status.value
        backend_key = _registry_backend_key(config)
        support = spec.backend_support.get(backend_key)
        backend_status = support.status.value if support is not None else None
        try:
            ModelRegistry.validate_backend(spec, backend_key, loading)
        except ValueError as exc:
            issues.append(str(exc))
        if spec.status.value != "READY":
            issues.append(f"model {spec.model_id} is not ready: {spec.status.value}")
        if support is None:
            issues.append(f"model {spec.model_id} has no backend capability record for {backend_key}")
        elif support.status in {
            BackendStatus.UNTESTED,
            BackendStatus.BLOCKED_CUSTOM_OP,
            BackendStatus.BLOCKED_MEMORY,
            BackendStatus.BLOCKED_UPSTREAM,
            BackendStatus.NOT_APPLICABLE,
        } or (support.status == BackendStatus.CPU_CONTRACT_ONLY and config.accelerator.backend != "cpu"):
            issues.append(
                f"model {spec.model_id} is not accepted on {config.accelerator.backend}: {support.status.value}"
            )
        estimate = spec.memory.loading_modes.get(loading)
        if estimate is not None and config.accelerator.backend == "cuda":
            budget = int(config.memory.max_gpu_memory_gb * 1024**3)
            reserve = int(config.memory.reserve_gpu_memory_gb * 1024**3)
            if estimate.device_bytes + reserve > budget:
                issues.append(
                    f"estimated device memory {estimate.device_bytes} bytes plus reserve "
                    f"exceeds configured CUDA budget {budget} bytes"
                )
        if config.accelerator.precision.upper() not in {precision.value for precision in spec.tested_precisions}:
            # An empty tested_precisions record is common for contract-only
            # local models; only reject a non-empty declared matrix.
            if spec.tested_precisions:
                issues.append(f"precision {config.accelerator.precision} has no registry evidence for {spec.model_id}")
        if config.task:
            task_name = str(config.task.get("type", config.task.get("name", ""))).upper()
            try:
                task_type = TaskType(task_name)
            except ValueError:
                task_type = None
            if task_type is not None and task_type not in spec.capabilities.tasks:
                issues.append(f"model {spec.model_id} does not declare task capability {task_type.value}")
    return CapabilityReport(
        model_id=config.model_id,
        backend=config.accelerator.backend,
        loading_mode=loading.value,
        precision=config.accelerator.precision,
        distribution=config.accelerator.distribution,
        registry_status=registry_status,
        backend_status=backend_status,
        issues=tuple(issues),
    )


def _invoke(builder: Callable[..., Any], candidates: tuple[tuple[Any, ...], ...]) -> Any:
    """Call a recipe builder using its declared arity, not TypeError retries."""
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return builder(*candidates[0])
    for args in candidates:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return builder(*args)
    raise PipelineBuildError(f"builder {builder!r} does not match any supported stage signature")


def _prepare_distributed_model(
    model: Any,
    *,
    config: RunConfig,
    backend: AcceleratorBackend,
) -> Any:
    distribution = config.accelerator.distribution
    if distribution in {"single", "replicated"} and config.accelerator.backend != "xla_tpu":
        return model
    if backend.uses_accelerate:
        # Accelerate owns process-group setup and wrapping when explicitly
        # selected; preparing the same module with DDP/FSDP twice is unsafe.
        return model
    from medfm.training.distributed import (
        initialize_cuda_process_group,
        initialize_tpu_process_group,
        wrap_model,
    )

    if config.accelerator.backend == "cuda":
        initialize_cuda_process_group(config.accelerator)
    elif config.accelerator.backend == "xla_tpu":
        initialize_tpu_process_group(config.accelerator)
    if not isinstance(model, nn.Module):
        raise PipelineBuildError("distributed training requires an nn.Module model")
    model = backend.configure_model_for_training(
        model,
        use_cache=config.memory.use_cache_during_training,
        gradient_checkpointing=any(config.memory.gradient_checkpointing.values()),
    )
    model = model.to(backend.device)
    return wrap_model(
        model,
        backend,
        replicated_tpu_accepted=bool(config.extensions.get("tpu_replicated_accepted", False)),
    )


class TrainingPipeline:
    """Construct a run in the fixed registry→dataset→model→... order."""

    def __init__(
        self,
        config: RunConfig,
        *,
        builders: ComponentBuilders | None = None,
        backend: AcceleratorBackend | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.builders = builders or ComponentBuilders()
        self.backend = backend
        self.dry_run = dry_run

    def preflight(self) -> CapabilityReport:
        try:
            catalog.ensure_v1_catalog()
        except (OSError, KeyError, ValueError) as exc:
            raise PipelineBuildError(f"registry catalog preflight failed: {exc}") from exc
        spec: ModelSpec | None = None
        try:
            spec = ModelRegistry.get(self.config.model_id)
        except KeyError:
            # Recipe-local tiny models do not need a catalog entry. A caller
            # that names a registry model still receives the fail-closed report.
            if self.config.model_id not in {"tiny", "tiny_multitask", "local"}:
                reason = f"unknown model {self.config.model_id!r}; register it before loading weights"
                raise PipelineBuildError(reason) from None
        report = validate_capabilities(self.config, spec=spec)
        if not report.valid:
            raise PipelineBuildError("capability preflight failed: " + "; ".join(report.issues))
        return report

    def dry_run_summary(self) -> BuildResult:
        report = self.preflight()
        backend = self.backend
        summary = ModelSummary(
            model_id=self.config.model_id,
            allocated=False,
            notes=("dry-run: model weights were not allocated", "capabilities validated before model construction"),
        )
        return BuildResult(
            config=self.config,
            backend=backend,
            capability_report=report,
            model_summary=summary,
            stages=["preflight"],
        )

    def build(self) -> BuildResult:
        if self.dry_run:
            return self.dry_run_summary()
        report = self.preflight()
        result = BuildResult(config=self.config, capability_report=report, stages=["preflight"])
        result.backend = self.backend or create_backend(
            self.config.accelerator,
            gradient_accumulation_steps=self.config.batch.gradient_accumulation_steps,
        )
        result.stages.append("backend")

        registry = self.builders.registry
        result.registry = _invoke(registry, ((self.config,), ())) if registry is not None else ModelRegistry
        result.stages.append("registry")

        if self.builders.dataset is not None:
            result.dataset = _invoke(self.builders.dataset, ((self.config, result.registry), (self.config,), ()))
        result.stages.append("dataset")

        if self.builders.model is None:
            raise PipelineBuildError("model builder is required for a non-dry run")
        result.model = _invoke(
            self.builders.model,
            ((self.config, result.registry, result.dataset), (self.config, result.registry), (self.config,)),
        )
        result.stages.append("model")
        result.model_summary = summarize_model(self.config.model_id, result.model)

        result.peft_model = result.model
        if self.builders.peft is not None:
            result.peft_model = _invoke(
                self.builders.peft, ((result.model, self.config), (result.model,), (self.config, result.model))
            )
        result.stages.append("peft")
        result.peft_model = _prepare_distributed_model(
            result.peft_model,
            config=self.config,
            backend=result.backend,
        )
        result.model = result.peft_model

        if self.builders.task is not None:
            result.task = _invoke(
                self.builders.task,
                ((self.config, result.peft_model), (self.config,), (result.peft_model,)),
            )
        result.stages.append("task")

        task_components: dict[str, nn.Module] | None = (
            {"task": result.task} if isinstance(result.task, nn.Module) else None
        )
        if self.builders.optimizer is not None:
            result.optimizer = _invoke(
                self.builders.optimizer,
                (
                    (result.peft_model, result.task, self.config, result.backend),
                    (result.peft_model, self.config, result.backend),
                    (result.peft_model, self.config),
                    (result.peft_model,),
                ),
            )
        else:
            result.optimizer = build_optimizer(
                result.peft_model,
                self.config.optimizer,
                backend=result.backend.name,
                components=task_components,
            )
        result.stages.append("optimizer")

        if self.builders.evaluator is not None:
            result.evaluator = _invoke(
                self.builders.evaluator,
                ((self.config, result.peft_model, result.task), (result.peft_model, result.task), (self.config,)),
            )
        result.stages.append("evaluator")

        if self.builders.trainer is not None:
            result.trainer = _invoke(
                self.builders.trainer,
                (
                    (self.config, result.backend, result.peft_model, result.optimizer, result.task, result.dataset),
                    (result.peft_model, result.optimizer, result.task, result.dataset),
                    (self.config,),
                ),
            )
        result.stages.append("trainer")

        if self.builders.checkpoint is not None:
            result.checkpoint = _invoke(self.builders.checkpoint, ((self.config,), (result.trainer,), ()))
        result.stages.append("checkpoint")
        return result


def summarize_model(model_id: str, model: Any) -> ModelSummary:
    try:
        parameters = list(model.parameters())
    except AttributeError:
        return ModelSummary(model_id=model_id, allocated=True, notes=("model is not an nn.Module",))
    return ModelSummary(
        model_id=model_id,
        allocated=True,
        total_parameters=sum(int(parameter.numel()) for parameter in parameters),
        trainable_parameters=sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad),
        parameter_dtypes=tuple(sorted({str(parameter.dtype) for parameter in parameters})),
    )


__all__ = [
    "BuildResult",
    "CapabilityReport",
    "ComponentBuilders",
    "ModelSummary",
    "PipelineBuildError",
    "TrainingPipeline",
    "summarize_model",
    "validate_capabilities",
]
