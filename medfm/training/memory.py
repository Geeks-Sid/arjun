"""Resource planning and actionable OOM/XLA compilation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from medfm.training.backend import AcceleratorBackend, MemorySnapshot
from medfm.training.config import MemoryConfig, RunConfig


CUDA_OOM_SUGGESTIONS: tuple[str, ...] = (
    "reduce microbatch to 1",
    "enable activation checkpointing",
    "disable language-model KV cache",
    "reduce maximum text length",
    "reduce visual-token budget",
    "reduce the number of 2D images or slices",
    "reduce 3D patch dimensions",
    "freeze the visual encoder",
    "cache visual embeddings",
    "enable optimizer-state reduction",
    "use CPU offload only as an explicit last resort",
)
TPU_RECOMMENDATIONS: tuple[str, ...] = (
    "use a fixed smaller shape bucket",
    "reduce visual and/or text token budgets",
    "enable activation checkpointing",
    "freeze or cache the encoder",
    "transition from replicated execution to SPMD/FSDP",
)


class MemoryPlanningError(RuntimeError):
    """A requested memory budget or performance gate cannot be satisfied."""


class MemoryBudgetError(MemoryPlanningError):
    """The estimated run exceeds the explicit configured budget."""


class TpuPerformanceGateError(MemoryPlanningError):
    """Steady-state XLA recompilation or fallback exceeded the configured gate."""


@dataclass(frozen=True)
class MemoryEstimate:
    model_bytes: int
    optimizer_bytes: int
    gradient_bytes: int
    activation_bytes: int
    input_bytes: int
    total_bytes: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_bytes": self.model_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "gradient_bytes": self.gradient_bytes,
            "activation_bytes": self.activation_bytes,
            "input_bytes": self.input_bytes,
            "total_bytes": self.total_bytes,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MemoryPlan:
    backend: str
    budget_bytes: int | None
    reserve_bytes: int
    estimate: MemoryEstimate
    fits: bool
    headroom_bytes: int | None
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "budget_bytes": self.budget_bytes,
            "reserve_bytes": self.reserve_bytes,
            "estimate": self.estimate.to_dict(),
            "fits": self.fits,
            "headroom_bytes": self.headroom_bytes,
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True)
class OOMDiagnostic:
    backend: str
    message: str
    suggestions: tuple[str, ...]
    configuration: dict[str, Any]
    estimate: MemoryEstimate | None = None
    snapshot: MemorySnapshot | None = None
    scientific_configuration_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "message": self.message,
            "suggestions": list(self.suggestions),
            "configuration": dict(self.configuration),
            "estimate": self.estimate.to_dict() if self.estimate is not None else None,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "scientific_configuration_mutated": self.scientific_configuration_mutated,
        }

    def render(self) -> str:
        lines = [f"{self.backend} out-of-memory diagnostic: {self.message}", "ordered suggestions:"]
        lines.extend(f"  {index}. {suggestion}" for index, suggestion in enumerate(self.suggestions, start=1))
        lines.append("scientific_configuration_mutated: false")
        return "\n".join(lines)


@dataclass(frozen=True)
class CompilationEvent:
    step: int
    bucket: str | None
    sample: str | None
    compilation_count: int
    graph_count: int
    host_device_transfers: int
    unsupported_op_count: int
    steady_state: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "bucket": self.bucket,
            "sample": self.sample,
            "compilation_count": self.compilation_count,
            "graph_count": self.graph_count,
            "host_device_transfers": self.host_device_transfers,
            "unsupported_op_count": self.unsupported_op_count,
            "steady_state": self.steady_state,
        }


class CompilationMonitor:
    """Name the bucket/sample responsible for XLA shape recompilation."""

    def __init__(self, *, warmup_steps: int = 0, max_steady_state_compilations: int = 0, fail: bool = False) -> None:
        if warmup_steps < 0 or max_steady_state_compilations < 0:
            raise ValueError("compilation monitor thresholds must be >= 0")
        self.warmup_steps = warmup_steps
        self.max_steady_state_compilations = max_steady_state_compilations
        self.fail = fail
        self._previous_compilations: int | None = None
        self._steady_state_compilations = 0
        self.events: list[CompilationEvent] = []

    @property
    def steady_state_compilations(self) -> int:
        return self._steady_state_compilations

    def observe(
        self,
        *,
        step: int,
        metrics: dict[str, Any],
        bucket: str | None = None,
        sample: str | None = None,
    ) -> CompilationEvent:
        compilation_count = int(metrics.get("compilation_count", metrics.get("compile_count", 0)))
        graph_count = int(metrics.get("graph_count", 0))
        transfers = int(metrics.get("host_device_transfers", metrics.get("transfers", 0)))
        unsupported = int(metrics.get("unsupported_op_count", metrics.get("fallback_count", 0)))
        previous = self._previous_compilations
        delta = max(0, compilation_count - previous) if previous is not None else compilation_count
        steady_state = step >= self.warmup_steps
        if steady_state:
            self._steady_state_compilations += delta
        event = CompilationEvent(
            step=step,
            bucket=bucket,
            sample=sample,
            compilation_count=compilation_count,
            graph_count=graph_count,
            host_device_transfers=transfers,
            unsupported_op_count=unsupported,
            steady_state=steady_state,
        )
        self.events.append(event)
        self._previous_compilations = compilation_count
        if self.fail and steady_state and self._steady_state_compilations > self.max_steady_state_compilations:
            label = f"bucket={bucket!r}, sample={sample!r}"
            raise TpuPerformanceGateError(
                "steady-state XLA recompilation threshold exceeded at "
                f"{label}: {self._steady_state_compilations} > {self.max_steady_state_compilations}"
            )
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "max_steady_state_compilations": self.max_steady_state_compilations,
            "steady_state_compilations": self.steady_state_compilations,
            "events": [event.to_dict() for event in self.events],
        }


class MemoryPlanner:
    """Backend-neutral estimate; concrete planners own measured snapshots."""

    backend_name = "cpu"

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def estimate(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        sample_batch: Any | None = None,
        activation_multiplier: float = 2.0,
    ) -> MemoryEstimate:
        model_bytes = 0
        trainable_bytes = 0
        input_bytes = 0
        for parameter in model.parameters():
            bytes_for_parameter = int(parameter.numel() * parameter.element_size())
            model_bytes += bytes_for_parameter
            if parameter.requires_grad:
                trainable_bytes += bytes_for_parameter
        if sample_batch is not None:
            input_bytes = _tensor_bytes(sample_batch)
        activation_bytes = int(max(0.0, float(input_bytes) * activation_multiplier))
        # AdamW keeps two FP32 moment tensors.  The estimate intentionally stays
        # conservative when the optimizer has not initialized its state yet.
        optimizer_bytes = trainable_bytes * 2 if optimizer is not None else trainable_bytes * 2
        total = model_bytes + optimizer_bytes + trainable_bytes + activation_bytes + input_bytes
        notes: list[str] = []
        if self.config.activation_checkpointing:
            activation_bytes //= 2
            total -= activation_bytes
            notes.append("activation checkpointing estimate applied")
        if self.config.frozen_embedding_mode or self.config.token_cache_mode:
            notes.append("frozen/cached encoder mode is explicit")
        return MemoryEstimate(
            model_bytes=model_bytes,
            optimizer_bytes=optimizer_bytes,
            gradient_bytes=trainable_bytes,
            activation_bytes=activation_bytes,
            input_bytes=input_bytes,
            total_bytes=total,
            notes=tuple(notes),
        )

    def plan(
        self,
        model: nn.Module,
        *,
        backend: AcceleratorBackend | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        sample_batch: Any | None = None,
    ) -> MemoryPlan:
        estimate = self.estimate(model, optimizer=optimizer, sample_batch=sample_batch)
        budget = self.budget_bytes(backend)
        reserve = int(self.config.reserve_gpu_memory_gb * 1024**3)
        fits = budget is None or estimate.total_bytes + reserve <= budget
        headroom = None if budget is None else budget - reserve - estimate.total_bytes
        return MemoryPlan(
            backend=self.backend_name,
            budget_bytes=budget,
            reserve_bytes=reserve,
            estimate=estimate,
            fits=fits,
            headroom_bytes=headroom,
            configuration=self.config.to_dict(),
        )

    def budget_bytes(self, backend: AcceleratorBackend | None) -> int | None:
        return None

    def snapshot(self, backend: AcceleratorBackend) -> MemorySnapshot:
        if backend.name != self.backend_name:
            raise MemoryPlanningError(f"{self.backend_name} planner cannot measure backend {backend.name}")
        return backend.memory_snapshot()

    def oom_diagnostic(
        self,
        error: BaseException,
        *,
        run_config: RunConfig | None = None,
        estimate: MemoryEstimate | None = None,
        snapshot: MemorySnapshot | None = None,
    ) -> OOMDiagnostic:
        configuration = run_config.to_dict() if run_config is not None else self.config.to_dict()
        suggestions = CUDA_OOM_SUGGESTIONS if self.backend_name == "cuda" else TPU_RECOMMENDATIONS
        return OOMDiagnostic(
            backend=self.backend_name,
            message=str(error),
            suggestions=suggestions,
            configuration=configuration,
            estimate=estimate,
            snapshot=snapshot,
        )


class CpuMemoryPlanner(MemoryPlanner):
    backend_name = "cpu"


class CudaMemoryPlanner(MemoryPlanner):
    backend_name = "cuda"

    def budget_bytes(self, backend: AcceleratorBackend | None) -> int | None:
        return int(self.config.max_gpu_memory_gb * 1024**3)


class TpuMemoryPlanner(MemoryPlanner):
    backend_name = "xla_tpu"

    def budget_bytes(self, backend: AcceleratorBackend | None) -> int | None:
        # XLA HBM varies by chip and runtime.  Do not invent a CUDA-style
        # budget; a caller can supply a measured profile in the config.
        return None

    def oom_diagnostic(
        self,
        error: BaseException,
        *,
        run_config: RunConfig | None = None,
        estimate: MemoryEstimate | None = None,
        snapshot: MemorySnapshot | None = None,
    ) -> OOMDiagnostic:
        config = run_config.to_dict() if run_config is not None else self.config.to_dict()
        return OOMDiagnostic(
            backend=self.backend_name,
            message=str(error),
            suggestions=TPU_RECOMMENDATIONS,
            configuration=config,
            estimate=estimate,
            snapshot=snapshot,
        )


def planner_for_backend(backend: str, config: MemoryConfig | None = None) -> MemoryPlanner:
    normalized = str(backend).lower()
    if normalized == "cuda":
        return CudaMemoryPlanner(config)
    if normalized in {"xla", "xla_tpu", "tpu"}:
        return TpuMemoryPlanner(config)
    if normalized == "cpu":
        return CpuMemoryPlanner(config)
    raise MemoryPlanningError(f"unknown memory-planner backend {backend!r}")


def diagnose_oom(
    error: BaseException,
    *,
    backend: str,
    run_config: RunConfig | None = None,
    estimate: MemoryEstimate | None = None,
    snapshot: MemorySnapshot | None = None,
) -> OOMDiagnostic:
    return planner_for_backend(backend, run_config.memory if run_config is not None else None).oom_diagnostic(
        error, run_config=run_config, estimate=estimate, snapshot=snapshot
    )


def enforce_memory_plan(plan: MemoryPlan) -> None:
    if not plan.fits:
        raise MemoryBudgetError(
            f"estimated {plan.estimate.total_bytes} bytes exceeds {plan.backend} budget "
            f"with reserved headroom ({plan.budget_bytes} bytes)"
        )


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    # MedicalBatch fields are explicit and metadata is intentionally ignored.
    total = 0
    for name in ("pixel_values", "image_mask", "tile_coordinates", "input_ids", "attention_mask", "labels"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, torch.Tensor):
            total += int(candidate.numel() * candidate.element_size())
    targets = getattr(value, "task_targets", {})
    if isinstance(targets, Mapping):
        total += _tensor_bytes(targets)
    return total


__all__ = [
    "CUDA_OOM_SUGGESTIONS",
    "CompilationEvent",
    "CompilationMonitor",
    "CpuMemoryPlanner",
    "CudaMemoryPlanner",
    "MemoryBudgetError",
    "MemoryEstimate",
    "MemoryPlan",
    "MemoryPlanner",
    "MemoryPlanningError",
    "OOMDiagnostic",
    "TPU_RECOMMENDATIONS",
    "TpuMemoryPlanner",
    "TpuPerformanceGateError",
    "diagnose_oom",
    "enforce_memory_plan",
    "planner_for_backend",
]
