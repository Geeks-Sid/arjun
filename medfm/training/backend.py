"""Accelerator abstraction for CPU, CUDA, and PyTorch/XLA training.

Models, task modules, and losses receive ordinary PyTorch objects.  CUDA and
XLA imports are kept behind their respective backend constructors/methods so a
CPU contract test can import the training package without initializing either
runtime.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from medfm.training.config import AcceleratorConfig


class BackendError(RuntimeError):
    """Base class for backend setup or execution failures."""


class BackendUnavailableError(BackendError):
    """The requested accelerator runtime is not available."""


class BackendCapabilityError(BackendError):
    """A requested precision/distribution feature is unsupported."""


class NonFiniteError(BackendError):
    """Loss or gradients contain NaN/Inf before a collective is attempted."""


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    device_count: int
    supports_bf16: bool
    supports_fp16_scaler: bool
    supports_accumulation: bool = True
    supports_ddp: bool = False
    supports_fsdp: bool = False
    supports_xla: bool = False
    supports_pinned_memory: bool = False
    supports_nonblocking_transfer: bool = False
    uses_accelerate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device_count": self.device_count,
            "supports_bf16": self.supports_bf16,
            "supports_fp16_scaler": self.supports_fp16_scaler,
            "supports_accumulation": self.supports_accumulation,
            "supports_ddp": self.supports_ddp,
            "supports_fsdp": self.supports_fsdp,
            "supports_xla": self.supports_xla,
            "supports_pinned_memory": self.supports_pinned_memory,
            "supports_nonblocking_transfer": self.supports_nonblocking_transfer,
            "uses_accelerate": self.uses_accelerate,
        }


@dataclass(frozen=True)
class BackendTopology:
    """Rank/topology facts recorded in run metadata and checkpoints."""

    backend: str
    rank: int
    local_rank: int
    host_index: int
    world_size: int
    device_count: int
    topology: str | None = None
    sharding_mesh: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "host_index": self.host_index,
            "world_size": self.world_size,
            "device_count": self.device_count,
            "topology": self.topology,
            "sharding_mesh": dict(self.sharding_mesh),
        }


@dataclass(frozen=True)
class AttentionResolution:
    requested: str
    selected: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"requested": self.requested, "selected": self.selected, "reason": self.reason}


@dataclass(frozen=True)
class MemorySnapshot:
    backend: str
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    total_bytes: int | None = None
    free_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
        }


def _distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def resolve_attention(requested: str, *, backend: str) -> AttentionResolution:
    """Select SDPA first, then FlashAttention, then explicit eager fallback."""
    raw = str(requested).lower()
    if backend == "xla_tpu":
        if raw not in {"auto", "sdpa", "xla", "eager"}:
            return AttentionResolution(raw, "eager", "requested attention is not XLA-compatible")
        return AttentionResolution(raw, "xla" if raw == "xla" else "sdpa", "XLA lowering/SDPA path")
    if raw in {"auto", "sdpa"} and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return AttentionResolution(raw, "sdpa", "PyTorch scaled-dot-product attention is the portable first choice")
    if raw in {"auto", "flash", "flash_attention", "flashattention"}:
        if backend == "cuda" and importlib.util.find_spec("flash_attn") is not None:
            return AttentionResolution(raw, "flash_attention", "compatible FlashAttention package is installed")
        if raw != "auto":
            return AttentionResolution(raw, "eager", "FlashAttention is unavailable; eager fallback is explicit")
    if raw in {"eager", "auto", "sdpa"}:
        return AttentionResolution(raw, "eager", "portable eager attention fallback")
    raise BackendCapabilityError(f"unknown attention implementation {requested!r}")


class AcceleratorBackend(ABC):
    """Backend-owned operations shared by the trainer."""

    name: str
    device: torch.device
    precision: str

    def __init__(
        self,
        config: AcceleratorConfig,
        *,
        use_accelerate: bool | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.config = config
        self.precision = config.precision
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self._accelerator: Any | None = None
        if use_accelerate is None:
            use_accelerate = importlib.util.find_spec("accelerate") is not None
        if use_accelerate:
            self._init_accelerate()

    def _init_accelerate(self) -> None:
        """Create Accelerate lazily; subclasses may disable it for XLA hooks.

        The Accelerator must be pinned to the backend's declared device rather
        than letting Accelerate auto-detect hardware, otherwise a CPU backend on
        a CUDA-capable host silently moves models to ``cuda:0`` (the reported
        ``cpu tensor vs cuda weight`` mismatches).
        """
        try:
            from accelerate import Accelerator  # noqa: PLC0415

            mixed_precision = {"fp32": "no", "bf16": "bf16", "fp16": "fp16"}[self.precision]
            self._accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                cpu=(self.config.backend == "cpu"),
            )
        except (ImportError, RuntimeError) as exc:
            raise BackendUnavailableError("Accelerate was requested but could not be initialized") from exc

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @property
    @abstractmethod
    def topology(self) -> BackendTopology: ...

    @property
    def uses_accelerate(self) -> bool:
        return self._accelerator is not None

    def validate(self) -> None:
        if self.precision == "bf16" and not self.capabilities.supports_bf16:
            raise BackendCapabilityError(f"{self.name} does not support BF16")
        if self.precision == "fp16" and not self.capabilities.supports_fp16_scaler:
            raise BackendCapabilityError(f"{self.name} does not support FP16 gradient scaling")
        if self.config.distribution == "ddp" and not self.capabilities.supports_ddp:
            raise BackendCapabilityError(f"{self.name} does not support DDP")
        if self.config.distribution in {"fsdp", "spmd_fsdp"} and not self.capabilities.supports_fsdp:
            raise BackendCapabilityError(f"{self.name} does not support {self.config.distribution}")

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if self.precision == "fp32":
            yield
            return
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        device_type = self.device.type
        if device_type == "xla":
            device_type = "xla"
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield

    def prepare_model(self, model: nn.Module) -> nn.Module:
        if self._accelerator is not None:
            return self._accelerator.prepare(model)
        return model.to(self.device)

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        """Prepare model/optimizer/loader together when Accelerate is present."""
        if self._accelerator is not None:
            return tuple(self._accelerator.prepare(*objects))
        prepared: list[Any] = []
        for value in objects:
            prepared.append(value.to(self.device) if isinstance(value, nn.Module) else value)
        return tuple(prepared)

    def prepare_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: Any,
        *,
        scheduler: Any | None = None,
        validation_dataloader: Any | None = None,
    ) -> tuple[nn.Module, torch.optim.Optimizer, Any, Any | None, Any | None]:
        """Prepare the coupled model/optimizer/loaders exactly once."""
        if self._accelerator is None:
            return (
                self.prepare_model(model),
                optimizer,
                self.prepare_dataloader(train_dataloader),
                self.prepare_dataloader(validation_dataloader) if validation_dataloader is not None else None,
                scheduler,
            )
        objects: list[Any] = [model, optimizer, train_dataloader]
        scheduler_index: int | None = None
        validation_index: int | None = None
        if scheduler is not None:
            scheduler_index = len(objects)
            objects.append(scheduler)
        if validation_dataloader is not None:
            validation_index = len(objects)
            objects.append(validation_dataloader)
        prepared = list(self._accelerator.prepare(*objects))
        prepared_model = prepared[0]
        prepared_optimizer = prepared[1]
        prepared_train = prepared[2]
        prepared_scheduler = prepared[scheduler_index] if scheduler_index is not None else scheduler
        prepared_validation = prepared[validation_index] if validation_index is not None else validation_dataloader
        return (
            prepared_model,
            prepared_optimizer,
            prepared_train,
            prepared_validation,
            prepared_scheduler,
        )

    def prepare_optimizer_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
    ) -> tuple[torch.optim.Optimizer, Any | None]:
        """Prepare a newly rebuilt optimizer without re-preparing the model."""
        if self._accelerator is None:
            return optimizer, scheduler
        objects: list[Any] = [optimizer]
        if scheduler is not None:
            objects.append(scheduler)
        prepared = tuple(self._accelerator.prepare(*objects))
        return prepared[0], prepared[1] if scheduler is not None else None

    def prepare_batch(self, batch: Any) -> Any:
        if hasattr(batch, "to") and not isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {key: self.prepare_batch(value) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self.prepare_batch(value) for value in batch)
        if isinstance(batch, list):
            return [self.prepare_batch(value) for value in batch]
        return batch

    def prepare_dataloader(self, dataloader: Any) -> Any:
        return dataloader

    def backward(self, loss: torch.Tensor) -> None:
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

    def unscale_optimizer(self, optimizer: torch.optim.Optimizer) -> None:  # noqa: B027 (default no-op hook)
        """Unscale FP16 gradients before clipping; BF16/XLA are no-op.

        Default hook: backends without a scaler (or that rely on Accelerate's
        own unscaling) inherit this no-op rather than overriding it, so it is
        intentionally not abstract.
        """

    def clip_grad_norm(
        self,
        parameters: Any,
        max_norm: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.Tensor:
        if self._accelerator is not None:
            value = self._accelerator.clip_grad_norm_(parameters, max_norm)
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if optimizer is not None:
            self.unscale_optimizer(optimizer)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._accelerator is not None:
            optimizer.step()
        else:
            optimizer.step()

    def zero_grad(self, optimizer: torch.optim.Optimizer, *, set_to_none: bool = True) -> None:
        optimizer.zero_grad(set_to_none=set_to_none)

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if self.name == "xla_tpu":
            return self._xla_reduce_sum(value)
        if not _distributed_initialized():
            return value
        result = value.clone()
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result

    def reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        if self.name == "xla_tpu":
            return self._xla_reduce_max(value)
        if not _distributed_initialized():
            return value
        result = value.clone()
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
        return result

    def synchronize(self) -> None:
        if _distributed_initialized():
            torch.distributed.barrier()

    def mark_step(self) -> None:  # noqa: B027 (default no-op hook; only XLA overrides)
        """Commit deferred device work; XLA overrides this hook.

        CPU/CUDA have no deferred-work commit step, so they inherit the no-op
        default; only XLA overrides. Intentionally not abstract.
        """

    def memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(backend=self.name)

    def runtime_metrics(self) -> dict[str, Any]:
        return {}

    def attention(self) -> AttentionResolution:
        return resolve_attention(self.config.attention, backend=self.name)

    def enable_activation_checkpointing(self, model: nn.Module) -> None:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()  # type: ignore[attr-defined]

    def retie_weights(self, model: nn.Module) -> None:
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()

    def configure_model_for_training(
        self,
        model: nn.Module,
        *,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
    ) -> nn.Module:
        """Apply non-scientific runtime toggles before optimizer construction."""
        model_config = getattr(model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = bool(use_cache)
        if gradient_checkpointing:
            self.enable_activation_checkpointing(model)
        return model

    def _xla_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def _xla_reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        return value


class CpuBackend(AcceleratorBackend):
    name = "cpu"

    def __init__(
        self,
        config: AcceleratorConfig | None = None,
        *,
        use_accelerate: bool | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            config or AcceleratorConfig(backend="cpu"),
            use_accelerate=use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.device = torch.device("cpu")
        self.validate()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            device_count=1,
            supports_bf16=True,
            supports_fp16_scaler=False,
            supports_ddp=False,
            supports_fsdp=False,
            supports_pinned_memory=torch.cuda.is_available(),
            supports_nonblocking_transfer=False,
            uses_accelerate=self.uses_accelerate,
        )

    @property
    def topology(self) -> BackendTopology:
        return BackendTopology(
            backend=self.name,
            rank=self.config.rank,
            local_rank=self.config.local_rank,
            host_index=self.config.host_index,
            world_size=self.config.world_size,
            device_count=1,
            topology=self.config.topology or "cpu",
            sharding_mesh=self.config.sharding_mesh,
        )


class CudaBackend(AcceleratorBackend):
    name = "cuda"

    def __init__(
        self,
        config: AcceleratorConfig | None = None,
        *,
        use_accelerate: bool | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        selected = config or AcceleratorConfig(backend="cuda", distribution="single", precision="bf16")
        if not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA backend requested but torch.cuda.is_available() is false")
        super().__init__(
            selected,
            use_accelerate=use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        index = selected.local_rank if selected.local_rank < torch.cuda.device_count() else 0
        self.device = torch.device("cuda", index)
        torch.cuda.set_device(self.device)
        self._scaler: torch.amp.GradScaler | None = None
        if self.precision == "fp16":
            self._scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.validate()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            device_count=torch.cuda.device_count(),
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            supports_fp16_scaler=True,
            supports_ddp=bool(torch.distributed.is_available()),
            supports_fsdp=bool(torch.distributed.is_available()),
            supports_pinned_memory=True,
            supports_nonblocking_transfer=True,
            uses_accelerate=self.uses_accelerate,
        )

    @property
    def topology(self) -> BackendTopology:
        world = self.config.world_size
        if _distributed_initialized():
            world = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = self.config.rank
        return BackendTopology(
            backend=self.name,
            rank=rank,
            local_rank=self.device.index or 0,
            host_index=self.config.host_index,
            world_size=world,
            device_count=torch.cuda.device_count(),
            topology=self.config.topology or "cuda",
            sharding_mesh=self.config.sharding_mesh,
        )

    def prepare_batch(self, batch: Any) -> Any:
        # MedicalBatch owns its tensor transfer contract.  Generic mappings use
        if hasattr(batch, "to") and not isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=bool(getattr(batch, "pinned", False)))
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, Mapping):
            return {key: self.prepare_batch(value) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self.prepare_batch(value) for value in batch)
        if isinstance(batch, list):
            return [self.prepare_batch(value) for value in batch]
        return batch

    def unscale_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def backward(self, loss: torch.Tensor) -> None:
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        elif self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._accelerator is not None:
            optimizer.step()
        elif self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def memory_snapshot(self) -> MemorySnapshot:
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        free, total = torch.cuda.mem_get_info(self.device)
        return MemorySnapshot(
            backend=self.name,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            total_bytes=int(total),
            free_bytes=int(free),
        )

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)


class XlaTpuBackend(AcceleratorBackend):
    name = "xla_tpu"

    def __init__(
        self,
        config: AcceleratorConfig | None = None,
        *,
        use_accelerate: bool | None = False,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        selected = config or AcceleratorConfig(
            backend="xla_tpu",
            distribution="replicated",
            precision="bf16",
            static_shapes=True,
        )
        if importlib.util.find_spec("torch_xla") is None:
            raise BackendUnavailableError("xla_tpu backend requested but torch_xla is not installed")
        super().__init__(
            selected,
            use_accelerate=use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        try:
            import torch_xla.core.xla_model as xm  # noqa: PLC0415
            import torch_xla.runtime as xr  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError("xla_tpu backend requires the PyTorch/XLA runtime") from exc
        self._xm = xm
        self._xr = xr
        try:
            self.device = xm.xla_device()
        except Exception as exc:
            raise BackendUnavailableError(f"could not acquire an XLA device: {exc}") from exc
        self.validate()

    @property
    def capabilities(self) -> BackendCapabilities:
        try:
            count = int(self._xr.global_runtime_device_count())
        except Exception:
            count = self.config.world_size
        return BackendCapabilities(
            name=self.name,
            device_count=max(1, count),
            supports_bf16=True,
            supports_fp16_scaler=False,
            supports_ddp=False,
            supports_fsdp=self.config.distribution == "spmd_fsdp",
            supports_xla=True,
            supports_pinned_memory=False,
            supports_nonblocking_transfer=False,
            uses_accelerate=False,
        )

    @property
    def topology(self) -> BackendTopology:
        try:
            rank = int(self._xr.global_ordinal())
        except Exception:
            rank = self.config.rank
        try:
            count = int(self._xr.global_runtime_device_count())
        except Exception:
            count = self.config.world_size
        return BackendTopology(
            backend=self.name,
            rank=rank,
            local_rank=rank,
            host_index=_env_int("TPU_WORKER_ID", self.config.host_index),
            world_size=count,
            device_count=count,
            topology=self.config.topology or str(self._xr.device_type()),
            sharding_mesh=self.config.sharding_mesh,
        )

    def prepare_dataloader(self, dataloader: Any) -> Any:
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError("XLA device loader is unavailable in this torch_xla build") from exc
        return MpDeviceLoader(dataloader, self.device)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        prepared = super().prepare_model(model)
        self.retie_weights(prepared)
        return prepared

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        self._xm.optimizer_step(optimizer, barrier=False)

    def mark_step(self) -> None:
        self._xm.mark_step()

    def synchronize(self) -> None:
        self._xm.mark_step()
        rendezvous = getattr(self._xm, "rendezvous", None)
        if callable(rendezvous):
            rendezvous("medfm-training")

    def _xla_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        try:
            return self._xm.all_reduce(self._xm.REDUCE_SUM, value)
        except Exception:
            return value

    def _xla_reduce_max(self, value: torch.Tensor) -> torch.Tensor:
        try:
            reduce_max = getattr(self._xm, "REDUCE_MAX", None)
            if reduce_max is None:
                return value
            return self._xm.all_reduce(reduce_max, value)
        except Exception:
            return value

    def memory_snapshot(self) -> MemorySnapshot:
        # XLA HBM is intentionally not queried through torch.cuda.  The XLA
        # runtime metrics/profiler owns TPU memory accounting.
        return MemorySnapshot(backend=self.name)

    def runtime_metrics(self) -> dict[str, Any]:
        try:
            import torch_xla.debug.metrics as met  # noqa: PLC0415

            report = met.short_metrics_report()
        except Exception:
            report = ""
        return parse_xla_metrics(report)


def parse_xla_metrics(report: str) -> dict[str, Any]:
    """Extract stable counters from a PyTorch/XLA metrics report."""
    values: dict[str, Any] = {"raw_report": report}
    patterns = {
        "compilation_count": r"(?:CompileTime|CompileCount|CompilationCount)[^0-9]*(\d+)",
        "graph_count": r"(?:GraphCount|Graph)[^0-9]*(\d+)",
        "host_device_transfers": r"(?:TransferToDevice|TransferFromDevice|Transfer)[^0-9]*(\d+)",
        "unsupported_op_count": r"(?:Unsupported|Fallback)[^0-9]*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report, flags=re.IGNORECASE)
        values[key] = int(match.group(1)) if match else 0
    return values


def create_backend(
    config: AcceleratorConfig | str,
    *,
    use_accelerate: bool | None = None,
    gradient_accumulation_steps: int = 1,
) -> AcceleratorBackend:
    """Create and validate one backend without importing other accelerator runtimes."""
    if isinstance(config, str):
        config = AcceleratorConfig(backend=config)
    if config.backend == "cpu":
        return CpuBackend(
            config,
            use_accelerate=use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    if config.backend == "cuda":
        return CudaBackend(
            config,
            use_accelerate=use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    if config.backend == "xla_tpu":
        return XlaTpuBackend(
            config,
            use_accelerate=False if use_accelerate is None else use_accelerate,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    raise BackendError(f"unknown backend {config.backend!r}")


__all__ = [
    "AcceleratorBackend",
    "AttentionResolution",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendError",
    "BackendTopology",
    "BackendUnavailableError",
    "CpuBackend",
    "CudaBackend",
    "MemorySnapshot",
    "NonFiniteError",
    "XlaTpuBackend",
    "create_backend",
    "parse_xla_metrics",
    "resolve_attention",
]
