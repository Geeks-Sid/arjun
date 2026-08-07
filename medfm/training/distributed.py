"""Distributed launch/wrapping helpers with fail-closed topology semantics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.training.backend import AcceleratorBackend, BackendCapabilityError
from medfm.training.config import AcceleratorConfig, BatchConfig
from medfm.training.data import DeterministicDistributedSampler


class DistributedLaunchError(RuntimeError):
    """Distributed runtime cannot satisfy the requested topology."""


@dataclass(frozen=True)
class DistributedContext:
    backend: str
    distribution: str
    rank: int
    local_rank: int
    host_index: int
    world_size: int
    topology: str | None
    sharding_mesh: dict[str, Any]

    @classmethod
    def from_backend(cls, backend: AcceleratorBackend) -> DistributedContext:
        topology = backend.topology
        return cls(
            backend=backend.name,
            distribution=backend.config.distribution,
            rank=topology.rank,
            local_rank=topology.local_rank,
            host_index=topology.host_index,
            world_size=topology.world_size,
            topology=topology.topology,
            sharding_mesh=dict(topology.sharding_mesh),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "distribution": self.distribution,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "host_index": self.host_index,
            "world_size": self.world_size,
            "topology": self.topology,
            "sharding_mesh": dict(self.sharding_mesh),
        }


def global_batch_size(batch: BatchConfig, world_size: int) -> int:
    """Compute the only accepted global-batch formula."""
    return batch.resolved_global_batch(world_size)


def initialize_cuda_process_group(config: AcceleratorConfig) -> DistributedContext:
    if config.backend != "cuda":
        raise DistributedLaunchError("CUDA process groups require accelerator.backend=cuda")
    if config.distribution == "single":
        return DistributedContext(
            backend="cuda",
            distribution="single",
            rank=config.rank,
            local_rank=config.local_rank,
            host_index=config.host_index,
            world_size=1,
            topology=config.topology or "cuda",
            sharding_mesh=dict(config.sharding_mesh),
        )
    if not torch.distributed.is_available():
        raise DistributedLaunchError("this PyTorch build has no distributed support")
    if not torch.distributed.is_initialized():
        backend = "nccl"
        if not torch.cuda.is_available():
            raise DistributedLaunchError("CUDA distributed launch requires a CUDA runtime")
        try:
            torch.distributed.init_process_group(backend=backend, init_method="env://")
        except (RuntimeError, ValueError) as exc:
            raise DistributedLaunchError(f"could not initialize CUDA process group: {exc}") from exc
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", config.local_rank))
    return DistributedContext(
        backend="cuda",
        distribution=config.distribution,
        rank=rank,
        local_rank=local_rank,
        host_index=config.host_index,
        world_size=world,
        topology=config.topology or "cuda",
        sharding_mesh=dict(config.sharding_mesh),
    )


def initialize_tpu_process_group(config: AcceleratorConfig) -> DistributedContext:
    if config.backend != "xla_tpu":
        raise DistributedLaunchError("TPU process groups require accelerator.backend=xla_tpu")
    if not torch.distributed.is_available():
        raise DistributedLaunchError("this PyTorch build has no distributed support")
    try:
        import torch_xla.distributed.xla_backend  # noqa: F401, PLC0415
        import torch_xla.runtime as xr  # noqa: PLC0415
    except ImportError as exc:
        raise DistributedLaunchError("TPU distributed launch requires PyTorch/XLA") from exc
    if config.distribution == "spmd_fsdp":
        use_spmd = getattr(xr, "use_spmd", None)
        if callable(use_spmd):
            use_spmd()
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(backend="gloo", init_method="xla://")
        except (RuntimeError, ValueError) as exc:
            raise DistributedLaunchError(f"could not initialize TPU process group: {exc}") from exc
    return DistributedContext(
        backend="xla_tpu",
        distribution=config.distribution,
        rank=torch.distributed.get_rank(),
        local_rank=torch.distributed.get_rank(),
        host_index=int(os.environ.get("TPU_WORKER_ID", config.host_index)),
        world_size=torch.distributed.get_world_size(),
        topology=config.topology or "tpu",
        sharding_mesh=dict(config.sharding_mesh),
    )


def wrap_model(
    model: nn.Module,
    backend: AcceleratorBackend,
    *,
    transformer_block_classes: tuple[type[nn.Module], ...] = (),
    replicated_tpu_accepted: bool = False,
) -> nn.Module:
    """Wrap only after backend capability validation and model placement."""
    distribution = backend.config.distribution
    if distribution == "single" or distribution == "replicated":
        if backend.name == "xla_tpu" and distribution == "replicated":
            if not replicated_tpu_accepted:
                raise DistributedLaunchError("TPU replicated wrapping requires a passing single-host acceptance marker")
        return model
    if backend.name == "cuda" and distribution == "ddp":
        if not torch.distributed.is_initialized():
            raise DistributedLaunchError("CUDA DDP requires an initialized process group")
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[backend.device.index] if backend.device.index is not None else None,
            output_device=backend.device.index,
        )
    if backend.name == "cuda" and distribution == "fsdp":
        if not torch.distributed.is_initialized():
            raise DistributedLaunchError("CUDA FSDP requires an initialized process group")
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: PLC0415
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: PLC0415
        except ImportError as exc:
            raise DistributedLaunchError("this PyTorch build lacks FSDP") from exc
        auto_wrap_policy = None
        if transformer_block_classes:

            def policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
                return transformer_auto_wrap_policy(
                    module=module,
                    recurse=recurse,
                    nonwrapped_numel=nonwrapped_numel,
                    transformer_layer_cls=set(transformer_block_classes),
                )

            auto_wrap_policy = policy
        return FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=backend.device)
    if backend.name == "xla_tpu" and distribution == "spmd_fsdp":
        if not replicated_tpu_accepted:
            raise DistributedLaunchError("TPU SPMD/FSDP is gated until replicated TPU acceptance passes")
        try:
            from torch_xla.distributed import spmd  # noqa: PLC0415
        except ImportError as exc:
            raise DistributedLaunchError("PyTorch/XLA SPMD is unavailable") from exc
        # Importing the module is the capability gate.  Sharding specs are
        # recipe-owned; this function intentionally does not invent a mesh.
        del spmd
        mesh = backend.topology.sharding_mesh
        if not mesh:
            raise DistributedLaunchError("SPMD/FSDP requires an explicit sharding_mesh in RunConfig")
        return model
    raise BackendCapabilityError(f"unsupported distribution {distribution!r} for backend {backend.name}")


def synchronize_failure(backend: AcceleratorBackend, failed: bool) -> bool:
    """Propagate failure status without waiting on a later collective."""
    value = torch.tensor(1 if failed else 0, dtype=torch.int32, device=backend.device)
    reduced = backend.reduce_max(value)
    return bool(reduced.detach().cpu())


__all__ = [
    "DeterministicDistributedSampler",
    "DistributedContext",
    "DistributedLaunchError",
    "global_batch_size",
    "initialize_cuda_process_group",
    "initialize_tpu_process_group",
    "synchronize_failure",
    "wrap_model",
]
