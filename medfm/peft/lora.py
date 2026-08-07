"""Backend-neutral LoRA injection, named adapters, and trainability audits.

Hugging Face PEFT remains an optional interoperability dependency.  The local
implementation is intentionally small and ordinary PyTorch so tiny contract
fixtures and visual/3D adapters remain testable without importing PEFT.  When a
model is quantized, quantization validation still happens in ``quantization``;
this module never treats a quantized base parameter as trainable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.peft.config import LoRAConfig
from medfm.peft.errors import (
    QuantizedParameterError,
    TargetMatchError,
    TrainabilityError,
)
from medfm.peft.resolver import TargetResolution, resolve_targets

_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _validate_adapter_name(name: str) -> str:
    if not _ADAPTER_NAME_RE.fullmatch(name):
        raise TrainabilityError(
            f"adapter_name {name!r} is unsafe; use letters, digits, '_' or '-' and start with a letter"
        )
    return name


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus one or more named low-rank updates."""

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base_layer).__name__}")
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self._adapter_configs: dict[str, dict[str, Any]] = {}
        self._active_adapters: list[str] = []
        self._merged_adapters: set[str] = set()
        self.add_adapter(config)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    @property
    def active_adapters(self) -> tuple[str, ...]:
        return tuple(self._active_adapters)

    @property
    def merged(self) -> bool:
        return bool(self._merged_adapters)

    def add_adapter(self, config: LoRAConfig) -> None:
        name = _validate_adapter_name(config.adapter_name)
        if name in self.lora_A:
            raise TrainabilityError(f"LoRA adapter {name!r} already exists on this module")
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        rank = int(config.rank)
        a = nn.Linear(self.in_features, rank, bias=False, device=device, dtype=dtype)
        b = nn.Linear(rank, self.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(a.weight, a=5**0.5)
        nn.init.zeros_(b.weight)
        self.lora_A[name] = a
        self.lora_B[name] = b
        self.lora_dropout[name] = nn.Dropout(float(config.dropout)) if config.dropout else nn.Identity()
        self._adapter_configs[name] = config.to_dict()
        self._active_adapters = [name]
        if config.use_dora:
            magnitude = torch.linalg.vector_norm(self.base_layer.weight.detach().float(), dim=1).to(dtype=dtype)
            self.register_parameter(f"dora_magnitude_{name}", nn.Parameter(magnitude))

    def adapter_config(self, name: str | None = None) -> dict[str, Any]:
        selected = name or (self._active_adapters[0] if self._active_adapters else None)
        if selected is None or selected not in self._adapter_configs:
            raise TrainabilityError(f"unknown LoRA adapter {selected!r}")
        return dict(self._adapter_configs[selected])

    def set_active_adapters(self, names: str | Sequence[str]) -> None:
        selected = [names] if isinstance(names, str) else list(names)
        unknown = sorted(set(selected) - set(self.lora_A))
        if unknown:
            raise TrainabilityError(f"unknown LoRA adapters {unknown}; available={sorted(self.lora_A)}")
        self._active_adapters = [_validate_adapter_name(name) for name in selected]

    def _delta_weight(self, name: str) -> torch.Tensor:
        config = self._adapter_configs[name]
        a = self.lora_A[name].weight
        b = self.lora_B[name].weight
        scaling = float(config["alpha"]) / (
            float(config["rank"]) ** 0.5 if bool(config.get("use_rslora", False)) else float(config["rank"])
        )
        return (b @ a) * scaling

    def _forward_adapter(self, x: torch.Tensor, name: str) -> torch.Tensor:
        update = self.lora_B[name](self.lora_A[name](self.lora_dropout[name](x)))
        config = self._adapter_configs[name]
        scaling = float(config["alpha"]) / (
            float(config["rank"]) ** 0.5 if bool(config.get("use_rslora", False)) else float(config["rank"])
        )
        update = update * scaling
        magnitude_name = f"dora_magnitude_{name}"
        if bool(config.get("use_dora", False)) and hasattr(self, magnitude_name):
            # DoRA's direction is the base-plus-update direction.  The
            # magnitude remains a small trainable vector and is initialized to
            # the base row norms, preserving the initial function.
            direction = self.base_layer(x) + update
            norm = direction.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(direction.dtype).eps)
            magnitude = getattr(self, magnitude_name)
            return direction / norm * magnitude.mean()
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        for name in self._active_adapters:
            if name not in self._merged_adapters:
                output = output + self._forward_adapter(x, name)
        return output

    def merge_adapter(self, name: str | None = None) -> None:
        selected = name or (self._active_adapters[0] if self._active_adapters else None)
        if selected is None:
            raise TrainabilityError("cannot merge without an active LoRA adapter")
        if selected in self._merged_adapters:
            return
        config = self._adapter_configs[selected]
        if bool(config.get("use_dora", False)):
            raise TrainabilityError("DoRA merge requires the inference backend's native merge implementation")
        with torch.no_grad():
            self.base_layer.weight.add_(self._delta_weight(selected).to(self.base_layer.weight.dtype))
        self._merged_adapters.add(selected)

    def unmerge_adapter(self, name: str | None = None) -> None:
        selected = name or (self._active_adapters[0] if self._active_adapters else None)
        if selected is None or selected not in self._merged_adapters:
            return
        config = self._adapter_configs[selected]
        if bool(config.get("use_dora", False)):
            raise TrainabilityError("DoRA unmerge requires the inference backend's native merge implementation")
        with torch.no_grad():
            self.base_layer.weight.sub_(self._delta_weight(selected).to(self.base_layer.weight.dtype))
        self._merged_adapters.remove(selected)

    def adapter_parameter_names(self, name: str | None = None) -> tuple[str, ...]:
        selected = name or (self._active_adapters[0] if self._active_adapters else None)
        if selected is None:
            return ()
        names = [f"lora_A.{selected}.weight", f"lora_B.{selected}.weight"]
        if bool(self._adapter_configs[selected].get("use_dora", False)):
            names.append(f"dora_magnitude_{selected}")
        return tuple(names)


@dataclass(frozen=True)
class InjectionResult:
    """Result of an injection, with the resolution retained for auditing."""

    model: nn.Module
    resolution: TargetResolution
    adapter_name: str
    matched_module_names: tuple[str, ...]
    modules_to_save: tuple[str, ...] = ()

    @property
    def selected_modules(self) -> tuple[str, ...]:
        return self.matched_module_names

    def __getattr__(self, name: str) -> Any:
        # Keeps the result ergonomic for callers that historically expected
        # injection to return the mutated model.
        return getattr(self.model, name)


@dataclass(frozen=True)
class ParameterAudit:
    total_parameters: int
    frozen_parameters: int
    trainable_parameters: int
    adapter_parameters: int
    bridge_parameters: int
    head_parameters: int
    decoder_parameters: int
    other_trainable_parameters: int
    quantized_parameters: int
    trainable_names: tuple[str, ...] = ()
    quantized_trainable_names: tuple[str, ...] = ()

    @property
    def trainable_percentage(self) -> float:
        return 0.0 if self.total_parameters == 0 else 100.0 * self.trainable_parameters / self.total_parameters

    @property
    def total(self) -> int:
        return self.total_parameters

    @property
    def frozen(self) -> int:
        return self.frozen_parameters

    @property
    def trainable(self) -> int:
        return self.trainable_parameters

    @property
    def adapter(self) -> int:
        return self.adapter_parameters

    @property
    def bridge(self) -> int:
        return self.bridge_parameters

    @property
    def head(self) -> int:
        return self.head_parameters

    @property
    def decoder(self) -> int:
        return self.decoder_parameters

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "frozen_parameters": self.frozen_parameters,
            "trainable_parameters": self.trainable_parameters,
            "adapter_parameters": self.adapter_parameters,
            "bridge_parameters": self.bridge_parameters,
            "head_parameters": self.head_parameters,
            "decoder_parameters": self.decoder_parameters,
            "other_trainable_parameters": self.other_trainable_parameters,
            "quantized_parameters": self.quantized_parameters,
            "trainable_percentage": self.trainable_percentage,
            "trainable_names": list(self.trainable_names),
            "quantized_trainable_names": list(self.quantized_trainable_names),
        }

    def assert_valid(self, *, qlora: bool = False) -> None:
        if self.trainable_parameters <= 0:
            raise TrainabilityError("zero trainable parameters: attach an adapter, bridge, head, or decoder first")
        if self.quantized_trainable_names:
            raise QuantizedParameterError(
                "quantized base parameters are trainable: " + ", ".join(self.quantized_trainable_names[:8])
            )
        if qlora and self.other_trainable_parameters:
            # Adapter/heads/bridges are expected; a base parameter is classified
            # as "other" only when it is trainable outside those roles.
            raise TrainabilityError(
                f"QLoRA unexpectedly exposes {self.other_trainable_parameters} non-adapter trainable parameters"
            )


def _is_adapter_name(name: str) -> bool:
    return any(token in name for token in ("lora_A.", "lora_B.", "dora_magnitude_"))


def is_quantized_parameter(parameter: nn.Parameter) -> bool:
    return bool(
        getattr(parameter, "_medfm_quantized", False)
        or getattr(parameter, "is_quantized", False)
        or parameter.__class__.__name__ in {"Params4bit", "Int8Params", "Int4Params"}
    )


def _role_for_name(name: str, role_map: Mapping[str, str] | None) -> str:
    if role_map:
        for prefix, role in role_map.items():
            if name == prefix or name.startswith(prefix + "."):
                return role
    lowered = name.lower()
    if _is_adapter_name(name):
        return "adapter"
    if any(token in lowered for token in ("bridge", "projector", "boundary")):
        return "bridge"
    if any(token in lowered for token in ("decoder", "segmentation_decoder")):
        return "decoder"
    if any(token in lowered for token in ("head", "classifier", "lm_head")):
        return "head"
    return "other"


def audit_trainable_parameters(
    model: nn.Module,
    *,
    role_map: Mapping[str, str] | None = None,
    qlora: bool = False,
) -> ParameterAudit:
    """Count trainable/frozen/base parameters and reject unsafe QLoRA state."""
    counts: MutableMapping[str, int] = defaultdict(int)
    total = 0
    trainable_names: list[str] = []
    quantized_names: list[str] = []
    for name, parameter in model.named_parameters():
        amount = int(parameter.numel())
        total += amount
        quantized = is_quantized_parameter(parameter)
        if quantized:
            counts["quantized"] += amount
        if parameter.requires_grad:
            counts["trainable"] += amount
            trainable_names.append(name)
            role = _role_for_name(name, role_map)
            counts[role] += amount
            if quantized:
                quantized_names.append(name)
        else:
            counts["frozen"] += amount
    audit = ParameterAudit(
        total_parameters=total,
        frozen_parameters=counts["frozen"],
        trainable_parameters=counts["trainable"],
        adapter_parameters=counts["adapter"],
        bridge_parameters=counts["bridge"],
        head_parameters=counts["head"],
        decoder_parameters=counts["decoder"],
        other_trainable_parameters=counts["other"],
        quantized_parameters=counts["quantized"],
        trainable_names=tuple(trainable_names),
        quantized_trainable_names=tuple(quantized_names),
    )
    audit.assert_valid(qlora=qlora)
    return audit


def _get_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise TargetMatchError(f"invalid module path {path!r}")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, nn.Sequential | nn.ModuleList):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TargetMatchError(f"module path {path!r} traverses a non-module at {part!r}")
    return parent, parts[-1]


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent, leaf = _get_parent(root, path)
    if leaf.isdigit() and isinstance(parent, nn.Sequential | nn.ModuleList):
        parent[int(leaf)] = replacement
    else:
        current = getattr(parent, leaf)
        if not isinstance(current, nn.Module):
            raise TargetMatchError(f"target {path!r} is not an nn.Module")
        setattr(parent, leaf, replacement)


def _matching_module_names(root: nn.Module, patterns: Iterable[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for name, _ in root.named_modules():
        if not name:
            continue
        if any(
            name == pattern or name.endswith("." + pattern) or _safe_pattern_match(pattern, name)
            for pattern in patterns
        ):
            selected.append(name)
    return tuple(selected)


def _safe_pattern_match(pattern: str, name: str) -> bool:
    normalized = pattern.replace(r"\\.", r"\.").replace(r"\\d", r"\d")
    if not any(token in normalized for token in r"\.^$*+?{}[]()|"):
        return name == normalized or name.endswith("." + normalized)
    try:
        return re.fullmatch(normalized, name) is not None or re.search(normalized, name) is not None
    except re.error as exc:
        raise TargetMatchError(f"invalid module pattern {pattern!r}: {exc}") from exc


def _existing_target_names(root: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in root.named_modules()
        if isinstance(module, LoRALinear)
        and config.adapter_name not in module.lora_A
        and (
            config.target_modules is None
            or any(_safe_pattern_match(pattern, name) for pattern in config.target_modules)
        )
    )


def _set_module_trainability(
    root: nn.Module,
    *,
    adapter_name: str,
    modules_to_save: tuple[str, ...],
    bias: str,
) -> tuple[str, ...]:
    # Base model parameters are always frozen first.  Explicit heads/bridges
    # can be enabled by passing them as a separate module or by modules_to_save.
    root.requires_grad_(False)
    saved = _matching_module_names(root, modules_to_save) if modules_to_save else ()
    if modules_to_save and not saved:
        raise TargetMatchError(f"modules_to_save patterns matched zero modules: {list(modules_to_save)}")
    for module_name in saved:
        module = root.get_submodule(module_name)
        module.requires_grad_(True)
    for module in root.modules():
        if not isinstance(module, LoRALinear):
            continue
        for name, parameter in module.named_parameters():
            if name.startswith(f"lora_A.{adapter_name}.") or name.startswith(f"lora_B.{adapter_name}."):
                parameter.requires_grad_(True)
            if name == f"dora_magnitude_{adapter_name}":
                parameter.requires_grad_(True)
        if bias == "all" and module.base_layer.bias is not None:
            module.base_layer.bias.requires_grad_(True)
        elif bias == "lora_only" and module.base_layer.bias is not None:
            module.base_layer.bias.requires_grad_(True)
    return tuple(saved)


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    *,
    architecture: str | None = None,
    confirm_unknown: bool | None = None,
) -> InjectionResult:
    """Inject native LoRA modules into a model before distributed wrapping."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    if not config.enabled:
        raise TrainabilityError("cannot inject a disabled LoRA configuration")
    _validate_adapter_name(config.adapter_name)

    existing = _existing_target_names(model, config)
    resolution: TargetResolution | None = None
    try:
        resolution = resolve_targets(model, config, architecture=architecture, confirm_unknown=confirm_unknown)
    except TargetMatchError:
        if not existing:
            raise
    if resolution is None:
        # A second named adapter may be added to already wrapped modules; make
        # a minimal resolution while retaining the original architecture/policy.
        # Existing wrappers have a stable target set.  Resolve using their
        # names through a small explicit policy only after verifying they are
        # the requested scope.
        selected = tuple(existing)
        if not selected:
            raise TargetMatchError(f"adapter {config.adapter_name!r} matched no existing LoRA modules")
        # Keep a real resolution from the first adapter when available.
        previous = getattr(model, "_medfm_peft_resolution", None)
        if isinstance(previous, TargetResolution):
            resolution = TargetResolution(
                architecture=previous.architecture,
                policy="named_adapter",
                records=previous.records,
                selected_names=selected,
            )
        else:
            raise TargetMatchError("cannot add a named adapter without a prior resolver record")

    matched: list[str] = []
    for name in resolution.selected_names:
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            if config.adapter_name in module.lora_A:
                raise TrainabilityError(f"LoRA adapter {config.adapter_name!r} already exists at {name}")
            module.add_adapter(config)
        elif isinstance(module, nn.Linear):
            _replace_module(model, name, LoRALinear(module, config))
        else:
            raise TargetMatchError(f"resolved target {name!r} is not an nn.Linear: {type(module).__name__}")
        matched.append(name)

    saved = _set_module_trainability(
        model,
        adapter_name=config.adapter_name,
        modules_to_save=config.modules_to_save,
        bias=str(config.bias),
    )
    metadata = getattr(model, "_medfm_peft_adapters", None)
    if not isinstance(metadata, dict):
        metadata = {}
        model._medfm_peft_adapters = metadata  # type: ignore[attr-defined]
    metadata[config.adapter_name] = config.to_dict()
    model._medfm_peft_resolution = resolution  # type: ignore[attr-defined]
    model._medfm_active_adapter = config.adapter_name  # type: ignore[attr-defined]
    return InjectionResult(model, resolution, config.adapter_name, tuple(matched), saved)


def _visual_architecture(adapter: nn.Module) -> str:
    backbone = getattr(adapter, "backbone", adapter)
    names = tuple(name for name, _ in backbone.named_modules())
    if any("patch_embed" in name or name.startswith("blocks") for name in names):
        return "3d_transformer"
    return "vision"


def inject_visual_lora(
    adapter: nn.Module,
    config: LoRAConfig,
    *,
    confirm_unknown: bool | None = None,
) -> InjectionResult:
    """Inject adapters into a visual adapter's backbone only."""
    gate = getattr(adapter, "check_lora_allowed", None)
    if callable(gate):
        gate()
    backbone = getattr(adapter, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise TargetMatchError("visual adapter does not expose an nn.Module backbone")
    result = inject_lora(
        backbone,
        config,
        architecture=config.architecture or _visual_architecture(adapter),
        confirm_unknown=confirm_unknown,
    )
    adapter._medfm_peft_result = result  # type: ignore[attr-defined]
    adapter._lora_state = config.to_dict()  # type: ignore[attr-defined]
    return result


def inject_language_lora(
    adapter: nn.Module,
    config: LoRAConfig,
    *,
    confirm_unknown: bool | None = None,
) -> InjectionResult:
    """Inject adapters into a language adapter's causal model, not its bridge."""
    language_model = getattr(adapter, "model", adapter)
    if not isinstance(language_model, nn.Module):
        raise TargetMatchError("language adapter does not expose an nn.Module model")
    result = inject_lora(
        language_model,
        config,
        architecture=config.architecture or "llm",
        confirm_unknown=confirm_unknown,
    )
    adapter._medfm_peft_result = result  # type: ignore[attr-defined]
    return result


def add_named_adapter(model: nn.Module, config: LoRAConfig, *, architecture: str | None = None) -> InjectionResult:
    """Add a second named adapter to an already-injected model."""
    return inject_lora(model, config, architecture=architecture, confirm_unknown=True)


def set_active_adapter(model: nn.Module, adapter_name: str) -> None:
    """Activate one named adapter across all wrapped modules."""
    _validate_adapter_name(adapter_name)
    found = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.set_active_adapters(adapter_name)
            found = True
    if not found:
        raise TrainabilityError("model contains no LoRA modules")
    model._medfm_active_adapter = adapter_name  # type: ignore[attr-defined]
    for parameter_name, parameter in model.named_parameters():
        if "lora_A." in parameter_name or "lora_B." in parameter_name or "dora_magnitude_" in parameter_name:
            parameter.requires_grad_(adapter_name in parameter_name)


def merge_lora_adapters(model: nn.Module, adapter_name: str | None = None) -> nn.Module:
    """Merge active LoRA updates for inference; the caller keeps canonical unmerged state."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_adapter(adapter_name)
    return model


def unmerge_lora_adapters(model: nn.Module, adapter_name: str | None = None) -> nn.Module:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_adapter(adapter_name)
    return model


def configure_trainability(
    model: nn.Module,
    *,
    trainable_modules: Iterable[str] = (),
    role_map: Mapping[str, str] | None = None,
) -> ParameterAudit:
    """Freeze deterministically, then enable explicit module paths and adapters."""
    model.requires_grad_(False)
    paths = tuple(str(path) for path in trainable_modules)
    for path in paths:
        try:
            module = model.get_submodule(path)
        except AttributeError as exc:
            raise TrainabilityError(f"trainable module path not found: {path!r}") from exc
        module.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            active = module.active_adapters
            for name in active:
                for parameter_name, parameter in module.named_parameters():
                    if f".{name}." in f".{parameter_name}." or parameter_name == f"dora_magnitude_{name}":
                        parameter.requires_grad_(True)
    return audit_trainable_parameters(model, role_map=role_map)


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    learning_rates: Mapping[str, float] | None = None,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Publish disjoint optimizer groups and reject quantized base weights."""
    rates = dict(learning_rates or {})
    groups: dict[str, list[nn.Parameter]] = defaultdict(list)
    names: dict[str, list[str]] = defaultdict(list)
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_quantized_parameter(parameter):
            raise QuantizedParameterError(f"quantized parameter {name!r} cannot enter optimizer groups")
        identity = id(parameter)
        if identity in seen:
            raise TrainabilityError(f"parameter {name!r} appears more than once in the optimizer model")
        seen.add(identity)
        role = _role_for_name(name, None)
        groups[role].append(parameter)
        names[role].append(name)
    if not seen:
        raise TrainabilityError("cannot build optimizer groups: zero trainable parameters")
    output: list[dict[str, Any]] = []
    for role in sorted(groups):
        output.append(
            {
                "name": role,
                "params": groups[role],
                "lr": float(rates.get(role, rates.get("default", 1e-4))),
                "weight_decay": float(weight_decay),
                "parameter_names": tuple(names[role]),
            }
        )
    return output


def attach_trainable_module(
    parent: nn.Module,
    name: str,
    module: nn.Module,
    *,
    role: str,
) -> nn.Module:
    """Attach a new bridge/head/decoder and make its trainability explicit."""
    if not name or "." in name:
        raise TrainabilityError("attached module names must be non-empty top-level names")
    if not isinstance(module, nn.Module):
        raise TypeError(f"attached component must be nn.Module, got {type(module).__name__}")
    setattr(parent, name, module)
    module.requires_grad_(True)
    roles = getattr(parent, "_medfm_component_roles", None)
    if not isinstance(roles, dict):
        roles = {}
        parent._medfm_component_roles = roles
    roles[name] = role
    return module


def set_requires_grad_deterministically(
    model: nn.Module,
    trainable_modules: Iterable[str],
    *,
    qlora: bool = False,
    role_map: Mapping[str, str] | None = None,
) -> ParameterAudit:
    """Apply one complete freeze/unfreeze stage and immediately audit it."""
    audit = configure_trainability(model, trainable_modules=trainable_modules, role_map=role_map)
    audit.assert_valid(qlora=qlora)
    return audit


def verify_adapter_state(model: nn.Module, *, adapter_name: str | None = None) -> tuple[str, ...]:
    """Return visible adapter parameter names and fail if the selected state vanished."""
    names = tuple(
        name
        for name, parameter in model.named_parameters()
        if _is_adapter_name(name)
        and (
            adapter_name is None
            or f"lora_A.{adapter_name}." in name
            or f"lora_B.{adapter_name}." in name
            or name.endswith(f"dora_magnitude_{adapter_name}")
        )
    )
    if not names:
        selected = f" {adapter_name!r}" if adapter_name else ""
        raise TrainabilityError(f"no visible LoRA parameters after wrapping; adapter{selected} state was lost")
    return names


# Public aliases used in handoff text and recipes.
apply_lora = inject_lora
trainable_parameter_audit = audit_trainable_parameters
build_optimizer_parameter_groups = optimizer_parameter_groups

__all__ = [
    "InjectionResult",
    "LoRALinear",
    "ParameterAudit",
    "add_named_adapter",
    "apply_lora",
    "attach_trainable_module",
    "audit_trainable_parameters",
    "build_optimizer_parameter_groups",
    "configure_trainability",
    "inject_language_lora",
    "inject_lora",
    "inject_visual_lora",
    "is_quantized_parameter",
    "merge_lora_adapters",
    "optimizer_parameter_groups",
    "set_active_adapter",
    "set_requires_grad_deterministically",
    "trainable_parameter_audit",
    "unmerge_lora_adapters",
    "verify_adapter_state",
]
