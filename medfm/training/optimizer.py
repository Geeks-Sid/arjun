"""Optimizer construction, staged freezing, and trainability audits."""

from __future__ import annotations

import importlib.util
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from medfm.peft.lora import is_quantized_parameter
from medfm.training.config import FreezeStageConfig, OptimizerConfig


class OptimizerConfigurationError(ValueError):
    """The requested optimizer policy is unavailable or unsafe."""


class TrainabilityAuditError(RuntimeError):
    """A parameter-group or gradient audit failed."""


ROLE_ALIASES = {
    "task": "task_head",
    "head": "task_head",
    "task_head": "task_head",
    "segmentation_decoder": "decoder",
    "decoder": "decoder",
    "vision_adapter": "vision_lora",
    "language_adapter": "language_lora",
    "language": "language_lora",
}
ROLE_ORDER = ("bridge", "task_head", "decoder", "vision_lora", "language_lora", "other")


def canonical_role(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    return ROLE_ALIASES.get(raw, raw)


def role_for_parameter(name: str, role_map: Mapping[str, str] | None = None) -> str:
    """Classify a parameter into one explicit recipe group."""
    if role_map:
        for prefix, role in sorted(role_map.items(), key=lambda pair: len(str(pair[0])), reverse=True):
            if name == prefix or name.startswith(str(prefix) + "."):
                return canonical_role(role)
    lowered = name.lower()
    has_lora = "lora_a" in lowered or "lora_b" in lowered or "dora_magnitude" in lowered
    if has_lora and any(token in lowered for token in ("vision", "visual", "encoder", "backbone")):
        return "vision_lora"
    if has_lora and any(token in lowered for token in ("language", "lang", "lm", "text", "causal")):
        return "language_lora"
    if any(token in lowered for token in ("language", "lang", "causal")):
        # A recipe may explicitly train a tiny/full language adapter before
        # introducing LoRA.  Keep that path in the same optimizer/stage role;
        # once LoRA is injected, base parameters remain frozen by PEFT.
        return "language_lora"
    if any(token in lowered for token in ("bridge", "projector", "boundary", "vl_bridge")):
        return "bridge"
    if any(token in lowered for token in ("segmentation_decoder", "decoder", "unet", "fpn")):
        return "decoder"
    if any(token in lowered for token in ("task_head", "classifier", "classification_head", "head", "lm_head")):
        return "task_head"
    return "other"


@dataclass(frozen=True)
class ParameterGroupInfo:
    name: str
    parameter_names: tuple[str, ...]
    parameter_count: int
    learning_rate: float
    weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_names": list(self.parameter_names),
            "parameter_count": self.parameter_count,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class GradientAudit:
    trainable_names: tuple[str, ...]
    gradient_names: tuple[str, ...]
    missing_gradient_names: tuple[str, ...]
    nonfinite_gradient_names: tuple[str, ...]
    component_norms: dict[str, float]

    @property
    def valid(self) -> bool:
        return not self.nonfinite_gradient_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable_names": list(self.trainable_names),
            "gradient_names": list(self.gradient_names),
            "missing_gradient_names": list(self.missing_gradient_names),
            "nonfinite_gradient_names": list(self.nonfinite_gradient_names),
            "component_norms": dict(self.component_norms),
            "valid": self.valid,
        }

    def assert_valid(self, *, require_all_gradients: bool = False) -> None:
        if self.nonfinite_gradient_names:
            raise TrainabilityAuditError(
                "non-finite gradients: " + ", ".join(self.nonfinite_gradient_names[:8])
            )
        if require_all_gradients and self.missing_gradient_names:
            raise TrainabilityAuditError(
                "trainable parameters did not receive gradients: " + ", ".join(self.missing_gradient_names[:8])
            )


@dataclass
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    groups: tuple[ParameterGroupInfo, ...]
    parameter_roles: dict[str, str]
    config: OptimizerConfig
    stage: tuple[str, ...] = ()

    def group_summary(self) -> list[dict[str, Any]]:
        return [group.to_dict() for group in self.groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "groups": self.group_summary(),
            "parameter_roles": dict(self.parameter_roles),
            "stage": list(self.stage),
        }


@dataclass(frozen=True)
class FreezeSchedule:
    stages: tuple[FreezeStageConfig, ...] = ()

    @classmethod
    def from_config(cls, stages: Iterable[FreezeStageConfig] | None) -> FreezeSchedule:
        return cls(tuple(stages or ()))

    def active_stage(self, step: int) -> FreezeStageConfig | None:
        if step < 0:
            raise ValueError("step must be >= 0")
        for stage in self.stages:
            if stage.until_step is None or step < stage.until_step:
                return stage
        return self.stages[-1] if self.stages else None

    def trainable_roles(self, step: int) -> tuple[str, ...]:
        stage = self.active_stage(step)
        if stage is None:
            return ()
        return tuple(canonical_role(name) for name in stage.train)

    def to_dict(self) -> list[dict[str, Any]]:
        return [stage.to_dict() for stage in self.stages]


def _group_hyperparameters(config: OptimizerConfig, role: str) -> tuple[float, float]:
    values = config.groups.get(role)
    if values is None and role == "decoder":
        values = config.groups.get("segmentation_decoder")
    if values is None:
        values = config.groups.get("default")
    if values is None:
        return config.lr, config.weight_decay
    return float(values.get("lr", config.lr)), float(values.get("weight_decay", config.weight_decay))


def _named_parameters(
    model: nn.Module,
    components: Mapping[str, nn.Module] | None = None,
) -> Iterable[tuple[str, nn.Parameter]]:
    yield from model.named_parameters()
    for prefix, component in (components or {}).items():
        for name, parameter in component.named_parameters():
            yield f"{prefix}.{name}", parameter


def build_parameter_groups(
    model: nn.Module,
    config: OptimizerConfig | None = None,
    *,
    role_map: Mapping[str, str] | None = None,
    components: Mapping[str, nn.Module] | None = None,
) -> tuple[list[dict[str, Any]], tuple[ParameterGroupInfo, ...], dict[str, str]]:
    """Build disjoint groups and reject frozen/quantized base parameters."""
    resolved = config or OptimizerConfig()
    grouped: dict[str, list[nn.Parameter]] = defaultdict(list)
    names: dict[str, list[str]] = defaultdict(list)
    roles: dict[str, str] = {}
    seen: set[int] = set()
    for name, parameter in _named_parameters(model, components):
        if not parameter.requires_grad:
            continue
        if is_quantized_parameter(parameter):
            raise OptimizerConfigurationError(f"quantized base parameter {name!r} cannot enter optimizer groups")
        identity = id(parameter)
        if identity in seen:
            raise TrainabilityAuditError(f"trainable parameter {name!r} appears more than once")
        seen.add(identity)
        role = role_for_parameter(name, role_map)
        if role not in ROLE_ORDER:
            role = "other"
        roles[name] = role
        grouped[role].append(parameter)
        names[role].append(name)
    if not seen:
        raise OptimizerConfigurationError("cannot build optimizer groups: no trainable parameters")
    parameter_groups: list[dict[str, Any]] = []
    infos: list[ParameterGroupInfo] = []
    for role in ROLE_ORDER:
        params = grouped.get(role, [])
        if not params:
            continue
        lr, decay = _group_hyperparameters(resolved, role)
        group_names = tuple(names[role])
        parameter_groups.append(
            {
                "name": role,
                "params": params,
                "lr": lr,
                "weight_decay": decay,
                "parameter_names": group_names,
            }
        )
        infos.append(
            ParameterGroupInfo(
                name=role,
                parameter_names=group_names,
                parameter_count=sum(int(parameter.numel()) for parameter in params),
                learning_rate=lr,
                weight_decay=decay,
            )
        )
    flattened = [id(parameter) for group in parameter_groups for parameter in group["params"]]
    if len(flattened) != len(set(flattened)):
        raise TrainabilityAuditError("optimizer groups are not disjoint")
    return parameter_groups, tuple(infos), roles


def _make_optimizer(parameter_groups: list[dict[str, Any]], config: OptimizerConfig, *, backend: str) -> torch.optim.Optimizer:
    if config.use_8bit:
        if backend != "cuda":
            raise OptimizerConfigurationError("8-bit optimizer state is supported only on CUDA")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise OptimizerConfigurationError("8-bit optimizer requested but bitsandbytes is unavailable")
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            return bnb.optim.AdamW8bit(parameter_groups, betas=config.betas, eps=config.eps)
        except (ImportError, AttributeError) as exc:
            raise OptimizerConfigurationError("installed bitsandbytes lacks AdamW8bit") from exc
    if config.sync_free and backend != "xla_tpu":
        raise OptimizerConfigurationError("XLA sync-free optimizer variants are not portable to this backend")
    if config.fused and backend != "cuda":
        raise OptimizerConfigurationError("fused AdamW is a CUDA-only optimization")
    kwargs: dict[str, Any] = {"betas": config.betas, "eps": config.eps}
    if config.fused:
        kwargs["fused"] = True
    cls: type[torch.optim.Optimizer] = torch.optim.AdamW if config.name == "adamw" else torch.optim.Adam
    try:
        return cls(parameter_groups, **kwargs)
    except (TypeError, RuntimeError) as exc:
        if config.fused:
            raise OptimizerConfigurationError("requested fused AdamW is unavailable on this CUDA runtime") from exc
        raise


def _make_scheduler(optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.warmup_steps <= 0 and config.total_steps is None:
        return None
    warmup = config.warmup_steps
    total = config.total_steps

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        if total is None or total <= warmup:
            return 1.0
        progress = min(1.0, max(0.0, float(step - warmup + 1) / float(total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig | None = None,
    *,
    backend: str = "cpu",
    role_map: Mapping[str, str] | None = None,
    components: Mapping[str, nn.Module] | None = None,
) -> OptimizerBundle:
    resolved = config or OptimizerConfig()
    parameter_groups, infos, roles = build_parameter_groups(
        model,
        resolved,
        role_map=role_map,
        components=components,
    )
    optimizer = _make_optimizer(parameter_groups, resolved, backend=backend)
    scheduler = _make_scheduler(optimizer, resolved)
    return OptimizerBundle(optimizer, scheduler, infos, roles, resolved)

def _enable_module(module: nn.Module) -> None:
    module.requires_grad_(True)
def apply_freeze_schedule(
    model: nn.Module,
    schedule: FreezeSchedule | Iterable[FreezeStageConfig],
    step: int,
    *,
    role_map: Mapping[str, str] | None = None,
    components: Mapping[str, nn.Module] | None = None,
) -> tuple[str, ...]:
    """Apply a stage at a boundary step and return trainable names."""
    resolved = schedule if isinstance(schedule, FreezeSchedule) else FreezeSchedule.from_config(schedule)
    stage = resolved.active_stage(step)
    modules = {"model": model, **dict(components or {})}
    if stage is None:
        return tuple(name for name, parameter in _named_parameters(model, components) if parameter.requires_grad)
    requested_roles = {canonical_role(name) for name in stage.train}
    for module in modules.values():
        module.requires_grad_(False)
    for name, parameter in _named_parameters(model, components):
        role = role_for_parameter(name, role_map)
        if role in requested_roles or name in stage.train or any(name.startswith(path + ".") for path in stage.train):
            parameter.requires_grad_(True)
    for module_name in stage.train:
        if "." in module_name and module_name.split(".", 1)[0] in modules:
            prefix, child = module_name.split(".", 1)
            try:
                _enable_module(modules[prefix].get_submodule(child))
            except AttributeError:
                continue
            continue
        try:
            _enable_module(model.get_submodule(module_name))
        except AttributeError:
            continue
    return tuple(name for name, parameter in _named_parameters(model, components) if parameter.requires_grad)


def audit_trainable_parameters(
    model: nn.Module,
    *,
    expected_roles: Iterable[str] | None = None,
    role_map: Mapping[str, str] | None = None,
    qlora: bool = False,
    components: Mapping[str, nn.Module] | None = None,
) -> dict[str, Any]:
    """Return a serializable construction audit and fail on unsafe state."""
    expected = {canonical_role(name) for name in (expected_roles or ())}
    trainable: list[str] = []
    quantized: list[str] = []
    roles: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    for name, parameter in _named_parameters(model, components):
        if parameter.requires_grad:
            trainable.append(name)
            role = role_for_parameter(name, role_map)
            roles[name] = role
            counts[role] += int(parameter.numel())
            if is_quantized_parameter(parameter):
                quantized.append(name)
    if quantized:
        raise TrainabilityAuditError("quantized base parameters are trainable: " + ", ".join(quantized[:8]))
    if qlora and any(role == "other" for role in roles.values()):
        raise TrainabilityAuditError("QLoRA exposes non-adapter trainable parameters")
    if expected and not expected.intersection(roles.values()) and trainable:
        raise TrainabilityAuditError(
            f"expected trainable roles {sorted(expected)} are absent; observed {sorted(set(roles.values()))}"
        )
    return {
        "trainable_names": trainable,
        "trainable_parameters": sum(int(parameter.numel()) for _, parameter in _named_parameters(model, components) if parameter.requires_grad),
        "roles": roles,
        "counts": dict(counts),
        "quantized_trainable_names": quantized,
    }


def audit_gradients(
    model: nn.Module,
    *,
    role_map: Mapping[str, str] | None = None,
    require_all_gradients: bool = False,
    components: Mapping[str, nn.Module] | None = None,
) -> GradientAudit:
    trainable: list[str] = []
    gradients: list[str] = []
    missing: list[str] = []
    nonfinite: list[str] = []
    squared: dict[str, torch.Tensor] = {}
    for name, parameter in _named_parameters(model, components):
        if not parameter.requires_grad:
            continue
        trainable.append(name)
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        gradients.append(name)
        if not torch.isfinite(gradient.detach()).all():
            nonfinite.append(name)
            continue
        role = role_for_parameter(name, role_map)
        value = gradient.detach().float().pow(2).sum()
        squared[role] = value if role not in squared else squared[role] + value
    norms = {role: float(torch.sqrt(value).cpu()) for role, value in squared.items()}
    audit = GradientAudit(tuple(trainable), tuple(gradients), tuple(missing), tuple(nonfinite), norms)
    audit.assert_valid(require_all_gradients=require_all_gradients)
    return audit


def rebuild_optimizer(
    model: nn.Module,
    previous: OptimizerBundle,
    *,
    backend: str = "cpu",
    role_map: Mapping[str, str] | None = None,
    components: Mapping[str, nn.Module] | None = None,
) -> OptimizerBundle:
    """Rebuild groups after a stage transition while preserving known state."""
    rebuilt = build_optimizer(
        model,
        previous.config,
        backend=backend,
        role_map=role_map,
        components=components,
    )
    old_state = previous.optimizer.state
    for parameter in (p for group in rebuilt.optimizer.param_groups for p in group["params"]):
        if parameter in old_state:
            rebuilt.optimizer.state[parameter] = old_state[parameter]
    if previous.scheduler is not None and rebuilt.scheduler is not None:
        try:
            rebuilt.scheduler.load_state_dict(previous.scheduler.state_dict())
        except (KeyError, ValueError):
            # Group topology changed; retaining optimizer state is safe, but a
            # mismatched scheduler state is not.  It restarts from its current
            # optimizer learning rates rather than mutating the recipe.
            pass
    rebuilt.stage = previous.stage
    return rebuilt


__all__ = [
    "FreezeSchedule",
    "GradientAudit",
    "OptimizerBundle",
    "OptimizerConfigurationError",
    "ParameterGroupInfo",
    "TrainabilityAuditError",
    "apply_freeze_schedule",
    "audit_gradients",
    "audit_trainable_parameters",
    "build_optimizer",
    "build_parameter_groups",
    "canonical_role",
    "rebuild_optimizer",
    "role_for_parameter",
]
