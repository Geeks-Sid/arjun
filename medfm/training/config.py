"""Typed, reproducible configuration for the unified training engine.

The training layer owns the run schema.  Recipe modules may add values under
``recipe``/``extensions`` but the engine never silently changes scientific
configuration in response to a resource failure.  A :class:`RunConfig` can be
loaded from YAML or JSON, validated before model construction, and hashed from
its canonical resolved representation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import yaml

from medfm.core.enums import PrecisionMode
from medfm.core.serialization import canonical_json, config_hash
from medfm.peft.config import LoRAConfig, QuantizationConfig, validate_backend_combination

SCHEMA_VERSION = 1


class RunConfigError(ValueError):
    """Raised when a run configuration is malformed or internally inconsistent."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunConfigError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RunConfigError(f"{name} must be an integer") from exc
    if result < 1:
        raise RunConfigError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RunConfigError(f"{name} must be an integer") from exc
    if result < 0:
        raise RunConfigError(f"{name} must be >= 0")
    return result


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RunConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RunConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise RunConfigError(f"{name} must be >= {minimum}")
    return result


def _normalize_precision(value: str | PrecisionMode) -> str:
    raw = str(value).strip().lower()
    aliases = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    raw = aliases.get(raw, raw)
    if raw not in {"fp32", "fp16", "bf16"}:
        raise RunConfigError("precision must be one of fp32, fp16, or bf16")
    return raw


def _normalize_backend(value: str) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "cpu": "cpu",
        "cuda": "cuda",
        "cuda_single": "cuda",
        "cuda_distributed": "cuda",
        "xla": "xla_tpu",
        "tpu": "xla_tpu",
        "xla_tpu": "xla_tpu",
        "tpu_single_host": "xla_tpu",
        "tpu_multi_host": "xla_tpu",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise RunConfigError(f"unknown accelerator backend {value!r}") from exc


def _normalize_distribution(value: str, backend: str) -> str:
    raw = str(value).strip().lower()
    if backend == "cpu":
        allowed = {"single"}
    elif backend == "cuda":
        allowed = {"single", "ddp", "fsdp"}
    else:
        allowed = {"replicated", "spmd_fsdp"}
    if raw not in allowed:
        raise RunConfigError(f"distribution {value!r} is not valid for backend {backend}; expected {sorted(allowed)}")
    return raw


@dataclass(frozen=True)
class AcceleratorConfig:
    """Resolved backend, distribution, precision, and static-shape policy."""

    backend: str = "cpu"
    distribution: str = "single"
    precision: str = "fp32"
    world_size: int = 1
    compile: bool = False
    attention: str = "sdpa"
    static_shapes: bool = False
    fail_on_recompilation_after_warmup: bool = False
    max_steady_state_compilations: int = 0
    warmup_steps: int = 0
    rank: int = 0
    local_rank: int = 0
    host_index: int = 0
    topology: str | None = None
    sharding_mesh: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = _normalize_backend(self.backend)
        precision = _normalize_precision(self.precision)
        attention = str(self.attention).strip().lower()
        if attention not in {"auto", "sdpa", "flash", "flash_attention", "flashattention", "eager", "xla"}:
            raise RunConfigError("accelerator.attention must be auto, sdpa, flash, or eager")
        requested_distribution = self.distribution
        distribution = _normalize_distribution(requested_distribution, backend)
        world_size = _positive_int(self.world_size, "accelerator.world_size")
        rank = _nonnegative_int(self.rank, "accelerator.rank")
        local_rank = _nonnegative_int(self.local_rank, "accelerator.local_rank")
        host_index = _nonnegative_int(self.host_index, "accelerator.host_index")
        if rank >= world_size:
            raise RunConfigError("accelerator.rank must be smaller than world_size")
        if backend == "xla_tpu" and precision == "fp16":
            raise RunConfigError("xla_tpu uses BF16 or FP32; FP16-style scaling is not supported")
        if backend == "xla_tpu" and not self.static_shapes:
            # Dynamic shapes are legal for a diagnostic run, but never for the
            # accepted TPU training distributions.
            if distribution in {"replicated", "spmd_fsdp"}:
                raise RunConfigError("TPU replicated/SPMD training requires static_shapes=true")
        if self.max_steady_state_compilations < 0:
            raise RunConfigError("max_steady_state_compilations must be >= 0")
        if self.warmup_steps < 0:
            raise RunConfigError("warmup_steps must be >= 0")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "attention", attention)
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "local_rank", local_rank)
        object.__setattr__(self, "host_index", host_index)
        object.__setattr__(self, "sharding_mesh", dict(self.sharding_mesh))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        raw = _mapping(data, "accelerator")
        return cls(
            backend=str(raw.get("backend", "cpu")),
            distribution=str(raw.get("distribution", "single")),
            precision=str(raw.get("precision", "fp32")),
            world_size=int(raw.get("world_size", 1)),
            compile=bool(raw.get("compile", False)),
            attention=str(raw.get("attention", "sdpa")).lower(),
            static_shapes=bool(raw.get("static_shapes", False)),
            fail_on_recompilation_after_warmup=bool(raw.get("fail_on_recompilation_after_warmup", False)),
            max_steady_state_compilations=int(raw.get("max_steady_state_compilations", 0)),
            warmup_steps=int(raw.get("warmup_steps", 0)),
            rank=int(raw.get("rank", 0)),
            local_rank=int(raw.get("local_rank", 0)),
            host_index=int(raw.get("host_index", 0)),
            topology=raw.get("topology"),
            sharding_mesh=_mapping(raw.get("sharding_mesh"), "accelerator.sharding_mesh"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "distribution": self.distribution,
            "precision": self.precision,
            "world_size": self.world_size,
            "compile": self.compile,
            "attention": self.attention,
            "static_shapes": self.static_shapes,
            "fail_on_recompilation_after_warmup": self.fail_on_recompilation_after_warmup,
            "max_steady_state_compilations": self.max_steady_state_compilations,
            "warmup_steps": self.warmup_steps,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "host_index": self.host_index,
            "topology": self.topology,
            "sharding_mesh": dict(self.sharding_mesh),
        }


@dataclass(frozen=True)
class BatchConfig:
    """Microbatch geometry and its explicit global-batch invariant."""

    microbatch_per_device: int = 1
    gradient_accumulation_steps: int = 1
    global_batch_size: int | None = None
    drop_last: bool = False
    pad_final_batch: bool = False

    def __post_init__(self) -> None:
        micro = _positive_int(self.microbatch_per_device, "batch.microbatch_per_device")
        accumulation = _positive_int(self.gradient_accumulation_steps, "batch.gradient_accumulation_steps")
        if self.global_batch_size is not None:
            global_batch = _positive_int(self.global_batch_size, "batch.global_batch_size")
        else:
            global_batch = None
        object.__setattr__(self, "microbatch_per_device", micro)
        object.__setattr__(self, "gradient_accumulation_steps", accumulation)
        object.__setattr__(self, "global_batch_size", global_batch)

    @property
    def micro_batch_size(self) -> int:
        """Compatibility spelling used by memory recipes."""
        return self.microbatch_per_device

    def resolved_global_batch(self, world_size: int) -> int:
        world = _positive_int(world_size, "world_size")
        computed = self.microbatch_per_device * world * self.gradient_accumulation_steps
        if self.global_batch_size is not None and self.global_batch_size != computed:
            raise RunConfigError(
                "batch.global_batch_size must equal "
                "microbatch_per_device * world_size * gradient_accumulation_steps "
                f"({self.microbatch_per_device} * {world} * {self.gradient_accumulation_steps} = {computed}), "
                f"got {self.global_batch_size}"
            )
        return computed

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        raw = _mapping(data, "batch")
        micro = raw.get("microbatch_per_device", raw.get("micro_batch_size", raw.get("microbatch", 1)))
        accumulation = raw.get("gradient_accumulation_steps", raw.get("accumulation_steps", 1))
        return cls(
            microbatch_per_device=int(micro),
            gradient_accumulation_steps=int(accumulation),
            global_batch_size=(int(raw["global_batch_size"]) if raw.get("global_batch_size") is not None else None),
            drop_last=bool(raw.get("drop_last", False)),
            pad_final_batch=bool(raw.get("pad_final_batch", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "microbatch_per_device": self.microbatch_per_device,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_batch_size": self.global_batch_size,
            "drop_last": self.drop_last,
            "pad_final_batch": self.pad_final_batch,
        }


@dataclass(frozen=True)
class MemoryConfig:
    """Explicit resource controls; none are changed automatically on OOM."""

    max_gpu_memory_gb: float = 46.0
    reserve_gpu_memory_gb: float = 2.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False
    gradient_checkpointing: dict[str, bool] = field(default_factory=dict)
    use_cache_during_training: bool = False
    visual_token_budget: int | None = 64
    max_text_tokens: int | None = 1024
    patch_size: tuple[int, ...] | None = None
    max_3d_patch: tuple[int, ...] | None = None
    empty_cache_on_validation: bool = False
    log_peak_memory: bool = True
    frozen_embedding_mode: bool = False
    token_cache_mode: bool = False
    cpu_offload: bool = False

    def __post_init__(self) -> None:
        max_memory = _finite_float(self.max_gpu_memory_gb, "memory.max_gpu_memory_gb", minimum=0.0)
        reserve = _finite_float(self.reserve_gpu_memory_gb, "memory.reserve_gpu_memory_gb", minimum=0.0)
        micro = _positive_int(self.micro_batch_size, "memory.micro_batch_size")
        accumulation = _positive_int(self.gradient_accumulation_steps, "memory.gradient_accumulation_steps")
        if max_memory == 0.0 and reserve == 0.0:
            raise RunConfigError("memory must reserve a non-zero device budget")
        object.__setattr__(self, "max_gpu_memory_gb", max_memory)
        object.__setattr__(self, "reserve_gpu_memory_gb", reserve)
        object.__setattr__(self, "micro_batch_size", micro)
        object.__setattr__(self, "gradient_accumulation_steps", accumulation)
        for name in ("visual_token_budget", "max_text_tokens"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_int(value, f"memory.{name}"))
        for name in ("patch_size", "max_3d_patch"):
            value = getattr(self, name)
            if value is not None:
                shape = tuple(_positive_int(v, f"memory.{name}") for v in value)
                object.__setattr__(self, name, shape)
        object.__setattr__(self, "gradient_checkpointing", dict(self.gradient_checkpointing))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        raw = _mapping(data, "memory")
        patch = raw.get("patch_size", raw.get("patch_dimensions"))
        max_patch = raw.get("max_3d_patch", raw.get("volume_patch_size"))
        return cls(
            max_gpu_memory_gb=float(raw.get("max_gpu_memory_gb", raw.get("max_device_memory_gb", 46.0))),
            reserve_gpu_memory_gb=float(raw.get("reserve_gpu_memory_gb", raw.get("reserve_headroom_gb", 2.0))),
            micro_batch_size=int(raw.get("micro_batch_size", raw.get("microbatch_per_device", 1))),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            activation_checkpointing=bool(raw.get("activation_checkpointing", False)),
            gradient_checkpointing=_mapping(raw.get("gradient_checkpointing"), "memory.gradient_checkpointing"),
            use_cache_during_training=bool(raw.get("use_cache_during_training", False)),
            visual_token_budget=(
                int(raw["visual_token_budget"]) if raw.get("visual_token_budget") is not None else None
            ),
            max_text_tokens=(int(raw["max_text_tokens"]) if raw.get("max_text_tokens") is not None else None),
            patch_size=tuple(int(v) for v in patch) if patch is not None else None,
            max_3d_patch=tuple(int(v) for v in max_patch) if max_patch is not None else None,
            empty_cache_on_validation=bool(raw.get("empty_cache_on_validation", False)),
            log_peak_memory=bool(raw.get("log_peak_memory", True)),
            frozen_embedding_mode=bool(raw.get("frozen_embedding_mode", raw.get("frozen_embeddings", False))),
            token_cache_mode=bool(raw.get("token_cache_mode", raw.get("token_cache", False))),
            cpu_offload=bool(raw.get("cpu_offload", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_gpu_memory_gb": self.max_gpu_memory_gb,
            "reserve_gpu_memory_gb": self.reserve_gpu_memory_gb,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "activation_checkpointing": self.activation_checkpointing,
            "gradient_checkpointing": dict(self.gradient_checkpointing),
            "use_cache_during_training": self.use_cache_during_training,
            "visual_token_budget": self.visual_token_budget,
            "max_text_tokens": self.max_text_tokens,
            "patch_size": list(self.patch_size) if self.patch_size is not None else None,
            "max_3d_patch": list(self.max_3d_patch) if self.max_3d_patch is not None else None,
            "empty_cache_on_validation": self.empty_cache_on_validation,
            "log_peak_memory": self.log_peak_memory,
            "frozen_embedding_mode": self.frozen_embedding_mode,
            "token_cache_mode": self.token_cache_mode,
            "cpu_offload": self.cpu_offload,
        }


_DEFAULT_GROUPS: dict[str, dict[str, float]] = {
    "bridge": {"lr": 1.0e-4},
    "task_head": {"lr": 1.0e-4},
    "decoder": {"lr": 1.0e-4},
    "vision_lora": {"lr": 2.0e-5},
    "language_lora": {"lr": 1.0e-5},
}


@dataclass(frozen=True)
class OptimizerConfig:
    """Portable optimizer policy, measured in optimizer steps."""

    name: str = "adamw"
    groups: dict[str, dict[str, float]] = field(default_factory=lambda: dict(_DEFAULT_GROUPS))
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    warmup_steps: int = 0
    total_steps: int | None = None
    max_grad_norm: float | None = None
    fused: bool = False
    use_8bit: bool = False
    sync_free: bool = False

    def __post_init__(self) -> None:
        if str(self.name).lower() not in {"adamw", "adam"}:
            raise RunConfigError("optimizer.name must be adamw or adam")
        object.__setattr__(self, "name", str(self.name).lower())
        object.__setattr__(self, "lr", _finite_float(self.lr, "optimizer.lr", minimum=0.0))
        object.__setattr__(
            self, "weight_decay", _finite_float(self.weight_decay, "optimizer.weight_decay", minimum=0.0)
        )
        if len(self.betas) != 2 or not all(0.0 <= float(v) < 1.0 for v in self.betas):
            raise RunConfigError("optimizer.betas must contain two values in [0, 1)")
        object.__setattr__(self, "betas", (float(self.betas[0]), float(self.betas[1])))
        object.__setattr__(self, "eps", _finite_float(self.eps, "optimizer.eps", minimum=0.0))
        object.__setattr__(self, "warmup_steps", _nonnegative_int(self.warmup_steps, "optimizer.warmup_steps"))
        if self.total_steps is not None:
            object.__setattr__(self, "total_steps", _positive_int(self.total_steps, "optimizer.total_steps"))
        if self.max_grad_norm is not None:
            object.__setattr__(
                self, "max_grad_norm", _finite_float(self.max_grad_norm, "optimizer.max_grad_norm", minimum=0.0)
            )
        normalized: dict[str, dict[str, float]] = {}
        for name, value in _mapping(self.groups, "optimizer.groups").items():
            group = _mapping(value, f"optimizer.groups.{name}")
            lr = _finite_float(group.get("lr", self.lr), f"optimizer.groups.{name}.lr", minimum=0.0)
            decay = _finite_float(
                group.get("weight_decay", self.weight_decay), f"optimizer.groups.{name}.weight_decay", minimum=0.0
            )
            normalized[str(name)] = {"lr": lr, "weight_decay": decay}
        object.__setattr__(self, "groups", normalized)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Self:
        raw = _mapping(data, "optimizer")
        groups = raw.get("groups", raw.get("optimizer_groups", _DEFAULT_GROUPS))
        return cls(
            name=str(raw.get("name", "adamw")),
            groups=_mapping(groups, "optimizer.groups"),
            lr=float(raw.get("lr", 1.0e-4)),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            betas=tuple(float(v) for v in raw.get("betas", (0.9, 0.999))),
            eps=float(raw.get("eps", 1.0e-8)),
            warmup_steps=int(raw.get("warmup_steps", 0)),
            total_steps=(int(raw["total_steps"]) if raw.get("total_steps") is not None else None),
            max_grad_norm=(float(raw["max_grad_norm"]) if raw.get("max_grad_norm") is not None else None),
            fused=bool(raw.get("fused", False)),
            use_8bit=bool(raw.get("use_8bit", False)),
            sync_free=bool(raw.get("sync_free", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "groups": {name: dict(value) for name, value in sorted(self.groups.items())},
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "max_grad_norm": self.max_grad_norm,
            "fused": self.fused,
            "use_8bit": self.use_8bit,
            "sync_free": self.sync_free,
        }


@dataclass(frozen=True)
class FreezeStageConfig:
    """Trainable module roles from a boundary step onward."""

    until_step: int | None
    train: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.until_step is not None and self.until_step < 0:
            raise RunConfigError("freeze_schedule.until_step must be >= 0 or null")
        names = tuple(str(name) for name in self.train)
        if not names or any(not name for name in names):
            raise RunConfigError("each freeze stage must train at least one named module")
        if len(set(names)) != len(names):
            raise RunConfigError("freeze stage train names must be unique")
        object.__setattr__(self, "train", names)

    def to_dict(self) -> dict[str, Any]:
        return {"until_step": self.until_step, "train": list(self.train)}


def _parse_freeze_schedule(value: Any) -> tuple[FreezeStageConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RunConfigError("freeze_schedule must be a list")
    stages: list[FreezeStageConfig] = []
    previous: int | None = None
    for index, item in enumerate(value):
        raw = _mapping(item, f"freeze_schedule[{index}]")
        until = raw.get("until_step")
        until_value = int(until) if until is not None else None
        if previous is not None and until_value is not None and until_value <= previous:
            raise RunConfigError("freeze_schedule boundaries must be strictly increasing")
        if until_value is not None:
            previous = until_value
        train = raw.get("train", raw.get("trainable_modules"))
        if not isinstance(train, (list, tuple)):
            raise RunConfigError(f"freeze_schedule[{index}].train must be a list")
        stages.append(FreezeStageConfig(until_value, tuple(str(name) for name in train)))
    if stages and stages[-1].until_step is not None:
        raise RunConfigError("the final freeze_schedule stage must have until_step: null")
    return tuple(stages)


@dataclass(frozen=True)
class RunConfig:
    """Complete typed run contract consumed by the staged pipeline."""

    model_id: str = "tiny"
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    freeze_schedule: tuple[FreezeStageConfig, ...] = ()
    peft: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    tracking: dict[str, Any] = field(default_factory=lambda: {"kind": "local_json"})
    checkpoint: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    max_steps: int | None = 1
    epochs: int = 1
    save_every_steps: int = 0
    eval_every_steps: int = 0
    output_dir: str = "artifacts/runs"
    dataset_hash: str | None = None
    preprocessing_hash: str | None = None
    base_model_revision: str | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RunConfigError(
                f"unsupported RunConfig schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.model_id:
            raise RunConfigError("model_id must be non-empty")
        if self.max_steps is not None:
            object.__setattr__(self, "max_steps", _positive_int(self.max_steps, "max_steps"))
        object.__setattr__(self, "epochs", _positive_int(self.epochs, "epochs"))
        object.__setattr__(self, "save_every_steps", _nonnegative_int(self.save_every_steps, "save_every_steps"))
        object.__setattr__(self, "eval_every_steps", _nonnegative_int(self.eval_every_steps, "eval_every_steps"))
        self.batch.resolved_global_batch(self.accelerator.world_size)
        # This is a pre-allocation policy check.  It does not import a model or
        # query CUDA/XLA runtime state.
        validate_backend_combination(
            self.peft,
            self.quantization,
            self.accelerator.backend,
            model_family=self.model.get("family") if isinstance(self.model, Mapping) else None,
        )
        object.__setattr__(self, "model", dict(self.model))
        object.__setattr__(self, "dataset", dict(self.dataset))
        object.__setattr__(self, "task", dict(self.task))
        object.__setattr__(self, "tracking", dict(self.tracking))
        object.__setattr__(self, "checkpoint", dict(self.checkpoint))
        object.__setattr__(self, "recipe", dict(self.recipe))
        object.__setattr__(self, "extensions", dict(self.extensions))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        raw = _mapping(data, "run config")
        version = int(raw.get("schema_version", SCHEMA_VERSION))
        model_value = raw.get("model", {})
        if isinstance(model_value, str):
            model = {"id": model_value}
            model_id = model_value
        else:
            model = _mapping(model_value, "model")
            model_id = str(raw.get("model_id", model.get("id", model.get("model_id", "tiny"))))
        training = _mapping(raw.get("training"), "training")
        accelerator = AcceleratorConfig.from_dict(raw.get("accelerator"))
        batch = BatchConfig.from_dict(raw.get("batch"))
        memory = MemoryConfig.from_dict(raw.get("memory"))
        # Training-level aliases are accepted but canonical output is flat.
        max_steps_value = raw.get("max_steps", training.get("max_steps", 1))
        epochs_value = raw.get("epochs", training.get("epochs", 1))
        peft_value = raw.get("peft", raw.get("lora", {}))
        peft = LoRAConfig.from_dict(_mapping(peft_value, "peft")) if peft_value else LoRAConfig()
        quant_value = raw.get("quantization", {})
        quantization = (
            QuantizationConfig.from_dict(_mapping(quant_value, "quantization")) if quant_value else QuantizationConfig()
        )
        return cls(
            model_id=model_id,
            model=model,
            dataset=_mapping(raw.get("dataset"), "dataset"),
            task=(
                {"name": str(raw["task"])} if isinstance(raw.get("task"), str) else _mapping(raw.get("task"), "task")
            ),
            accelerator=accelerator,
            batch=batch,
            memory=memory,
            optimizer=OptimizerConfig.from_dict(raw.get("optimizer")),
            freeze_schedule=_parse_freeze_schedule(raw.get("freeze_schedule")),
            peft=peft,
            quantization=quantization,
            tracking=_mapping(raw.get("tracking"), "tracking") or {"kind": "local_json"},
            checkpoint=_mapping(raw.get("checkpoint"), "checkpoint"),
            seed=int(raw.get("seed", training.get("seed", 0))),
            max_steps=(int(max_steps_value) if max_steps_value is not None else None),
            epochs=int(epochs_value),
            save_every_steps=int(raw.get("save_every_steps", training.get("save_every_steps", 0))),
            eval_every_steps=int(raw.get("eval_every_steps", training.get("eval_every_steps", 0))),
            output_dir=str(raw.get("output_dir", training.get("output_dir", "artifacts/runs"))),
            dataset_hash=raw.get("dataset_hash"),
            preprocessing_hash=raw.get("preprocessing_hash"),
            base_model_revision=raw.get("base_model_revision"),
            recipe=_mapping(raw.get("recipe"), "recipe"),
            extensions=_mapping(raw.get("extensions"), "extensions"),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = Path(path)
        if not source.exists():
            raise RunConfigError(f"configuration file does not exist: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RunConfigError(f"could not parse configuration {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise RunConfigError(f"configuration {source} must contain a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model": dict(self.model),
            "dataset": dict(self.dataset),
            "task": dict(self.task),
            "accelerator": self.accelerator.to_dict(),
            "batch": self.batch.to_dict(),
            "memory": self.memory.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "freeze_schedule": [stage.to_dict() for stage in self.freeze_schedule],
            "peft": self.peft.to_dict(),
            "quantization": self.quantization.to_dict(),
            "tracking": dict(self.tracking),
            "checkpoint": dict(self.checkpoint),
            "seed": self.seed,
            "max_steps": self.max_steps,
            "epochs": self.epochs,
            "save_every_steps": self.save_every_steps,
            "eval_every_steps": self.eval_every_steps,
            "output_dir": self.output_dir,
            "dataset_hash": self.dataset_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "base_model_revision": self.base_model_revision,
            "recipe": dict(self.recipe),
            "extensions": dict(self.extensions),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def config_hash(self) -> str:
        return config_hash(self.to_dict())

    @property
    def global_batch_size(self) -> int:
        return self.batch.resolved_global_batch(self.accelerator.world_size)

    def with_runtime_resolution(
        self,
        *,
        world_size: int | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        host_index: int | None = None,
        topology: str | None = None,
        sharding_mesh: Mapping[str, Any] | None = None,
    ) -> RunConfig:
        """Return a resolved copy after launcher discovery, never mutate in place."""
        acc = replace(
            self.accelerator,
            world_size=self.accelerator.world_size if world_size is None else world_size,
            rank=self.accelerator.rank if rank is None else rank,
            local_rank=self.accelerator.local_rank if local_rank is None else local_rank,
            host_index=self.accelerator.host_index if host_index is None else host_index,
            topology=self.accelerator.topology if topology is None else topology,
            sharding_mesh=(dict(self.accelerator.sharding_mesh) if sharding_mesh is None else dict(sharding_mesh)),
        )
        return replace(self, accelerator=acc)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a typed configuration from YAML/JSON."""
    return RunConfig.load(path)


__all__ = [
    "SCHEMA_VERSION",
    "AcceleratorConfig",
    "BatchConfig",
    "MemoryConfig",
    "OptimizerConfig",
    "FreezeStageConfig",
    "RunConfig",
    "RunConfigError",
    "load_run_config",
]
