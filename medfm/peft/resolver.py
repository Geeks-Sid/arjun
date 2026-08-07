"""Architecture-aware LoRA target inspection and fail-closed resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn

from medfm.peft.config import LoRAConfig, TargetPolicy
from medfm.peft.errors import (
    BroadTargetError,
    TargetMatchError,
    TargetResolutionError,
    UnknownArchitectureError,
)


@dataclass(frozen=True)
class TargetRule:
    pattern: str
    reason: str


@dataclass(frozen=True)
class TargetPolicySpec:
    """Reviewed defaults for one architecture family."""

    name: str
    rules: tuple[TargetRule, ...]
    aliases: tuple[str, ...] = ()
    late_stage_only: bool = False
    late_stage_fraction: float = 1.0 / 3.0
    description: str = ""


@dataclass(frozen=True)
class ModuleInspection:
    """One inspectable parameterized module and its selection rationale."""

    name: str
    module_type: str
    parameter_shape: tuple[int, ...]
    parameter_count: int
    selected: bool
    reason: str
    suggested_lora_target: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.parameter_shape

    @property
    def count(self) -> int:
        return self.parameter_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.name,
            "module_type": self.module_type,
            "parameter_shape": list(self.parameter_shape),
            "parameter_count": self.parameter_count,
            "suggested_lora_target": self.suggested_lora_target,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TargetResolution:
    architecture: str
    policy: str
    records: tuple[ModuleInspection, ...]
    selected_names: tuple[str, ...]

    @property
    def selected(self) -> tuple[str, ...]:
        return self.selected_names

    @property
    def selected_count(self) -> int:
        return len(self.selected_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "policy": self.policy,
            "selected_names": list(self.selected_names),
            "modules": [record.to_dict() for record in self.records],
        }


# Patterns deliberately name projections, never broad "all Linear" defaults.
# The tiny local fixtures use nn.TransformerEncoder names (linear1/linear2 and
# self_attn.out_proj), while HF architectures expose q_proj/.../o_proj.
_LLM_RULES = (
    TargetRule(
        r"(?:^|\\.)(q_proj|k_proj|v_proj|o_proj|out_proj)$",
        "causal-language attention projection",
    ),
    TargetRule(
        r"(?:^|\\.)(gate_proj|up_proj|down_proj|linear1|linear2)$",
        "causal-language MLP projection",
    ),
)
_VISION_RULES = (
    TargetRule(
        r"(?:^|\\.)(qkv|query|key|value|q_proj|k_proj|v_proj|proj|out_proj)$",
        "2D visual-transformer attention projection",
    ),
    TargetRule(
        r"(?:^|\\.)(fc1|fc2|linear1|linear2)$",
        "2D visual-transformer MLP projection",
    ),
)
_3D_RULES = (
    TargetRule(
        r"(?:^|\\.)(qkv|query|key|value|q_proj|k_proj|v_proj|proj|out_proj)$",
        "3D transformer attention projection; patch stem excluded",
    ),
    TargetRule(
        r"(?:^|\\.)(fc1|fc2|linear1|linear2)$",
        "3D transformer MLP projection; patch stem excluded",
    ),
)

_TARGET_POLICIES: tuple[TargetPolicySpec, ...] = (
    TargetPolicySpec(
        "llm",
        _LLM_RULES,
        aliases=("language", "causal_lm", "causal-language", "gemma", "llama", "mistral", "qwen", "phi3"),
        description="Language attention and MLP projections; embeddings and norms remain frozen.",
    ),
    TargetPolicySpec(
        "vision",
        _VISION_RULES,
        aliases=("vit", "2d_vit", "siglip", "dinov2", "visual", "hoptimus"),
        description="2D vision-transformer attention and optional MLP projections.",
    ),
    TargetPolicySpec(
        "3d_transformer",
        _3D_RULES,
        aliases=("3d", "3d_swin", "swin3d", "native_3d", "volume_transformer"),
        late_stage_only=True,
        description="Reviewed late-stage 3D transformer projections; patch/convolution stems and decoders excluded.",
    ),
    TargetPolicySpec(
        "segmentation",
        _3D_RULES + _VISION_RULES,
        aliases=("segmentation_encoder",),
        late_stage_only=True,
        description="Transformer encoder projections only; newly initialized decoders are full-train modules.",
    ),
)

_POLICY_BY_NAME: dict[str, TargetPolicySpec] = {}
for _policy in _TARGET_POLICIES:
    _POLICY_BY_NAME[_policy.name] = _policy
    for _alias in _policy.aliases:
        _POLICY_BY_NAME[_alias] = _policy

_EXCLUDED_NAME_RE = re.compile(
    r"(?:^|\.)(?:patch_embed(?:ding)?|conv_stem|stem|norm(?:alization)?|decoder|decode|head|classifier)(?:\.|$)",
    re.IGNORECASE,
)


def available_target_policies() -> tuple[str, ...]:
    return tuple(policy.name for policy in _TARGET_POLICIES)


def _normalise_regex(pattern: str) -> str:
    """Accept legacy raw strings containing doubled regex separators."""
    return pattern.replace(r"\\.", r"\.").replace(r"\\d", r"\d")
def _matches(pattern: str, name: str) -> bool:
    normalized = _normalise_regex(pattern)
    # Literal target names are module paths, not substring regexes: matching
    # "0" must not accidentally select 0.base_layer after a second adapter is
    # added to an already wrapped module.
    if not any(token in normalized for token in r"\.^$*+?{}[]()|"):
        return name == normalized or name.endswith("." + normalized)
    try:
        if re.fullmatch(normalized, name) is not None:
            return True
        return re.search(normalized, name) is not None
    except re.error as exc:
        raise TargetResolutionError(f"invalid LoRA target regex {pattern!r}: {exc}") from exc


def _architecture_from_model(model: nn.Module, architecture: str | None) -> str | None:
    if architecture:
        return str(architecture).lower()
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "model_type", None),
        getattr(config, "architectures", None),
        getattr(model, "model_type", None),
    ):
        if isinstance(value, tuple | list):
            values: Iterable[Any] = value
        else:
            values = (value,)
        for item in values:
            if item is None:
                continue
            text = str(item).lower()
            if text in _POLICY_BY_NAME:
                return text
            for alias, policy in _POLICY_BY_NAME.items():
                if alias in text:
                    return policy.name
    return None


def _policy_for(model: nn.Module, architecture: str | None, *, confirm_unknown: bool) -> TargetPolicySpec:
    resolved = _architecture_from_model(model, architecture)
    if resolved is None or resolved not in _POLICY_BY_NAME:
        if not confirm_unknown:
            raise UnknownArchitectureError(
                f"no reviewed PEFT target policy for architecture {architecture or '<unknown>'!r}; "
                "provide explicit target_modules and confirm_target_modules=true"
            )
        # Confirmed unknown architectures use the caller's explicit patterns;
        # an empty policy is intentional and handled by the explicit-target gate.
        return TargetPolicySpec(
            "unknown",
            (),
            description="Unknown architecture explicitly confirmed by the caller; only explicit targets are allowed.",
        )
    return _POLICY_BY_NAME[resolved]


def _stage_index(name: str) -> int | None:
    # Covers HF Swin stages, native blocks, and TransformerEncoder layers.
    matches = re.findall(
        _normalise_regex(r"(?:^|\.)(?:blocks|stages|layers)(?:\.layers)?\.(\d+)(?:\.|$)"), name
    )
    if not matches:
        return None
    return int(matches[-1])


def _late_stage_allowed(name: str, module_names: tuple[str, ...], policy: TargetPolicySpec) -> bool:
    if not policy.late_stage_only:
        return True
    index = _stage_index(name)
    if index is None:
        # A path without a stage cannot be proven to be a reviewed late stage.
        return False
    indices = [value for value in (_stage_index(candidate) for candidate in module_names) if value is not None]
    if not indices:
        return False
    max_stage = max(indices)
    min_late = max(0, int((max_stage + 1) * (1.0 - policy.late_stage_fraction)))
    return index >= min_late


def _module_record(
    name: str,
    module: nn.Module,
    *,
    policy: TargetPolicySpec,
    module_names: tuple[str, ...],
    explicit_patterns: tuple[str, ...] | None,
) -> ModuleInspection:
    parameters = tuple(module.parameters(recurse=False))
    if not parameters:
        return ModuleInspection(name, type(module).__name__, (), 0, False, "no direct parameters")
    first_shape = tuple(int(value) for value in parameters[0].shape)
    count = sum(int(parameter.numel()) for parameter in parameters)
    if not isinstance(module, nn.Linear):
        return ModuleInspection(
            name,
            type(module).__name__,
            first_shape,
            count,
            False,
            "not an nn.Linear projection; convolution, embedding, normalization, and decoder modules stay frozen",
        )
    if _EXCLUDED_NAME_RE.search(name):
        return ModuleInspection(
            name,
            type(module).__name__,
            first_shape,
            count,
            False,
            "excluded by policy (patch embedding/stem/normalization/decoder/head)",
        )
    rules = tuple(TargetRule(pattern, "explicit caller target") for pattern in explicit_patterns or ()) or policy.rules
    matches = [rule for rule in rules if _matches(rule.pattern, name)]
    if not matches:
        return ModuleInspection(
            name,
            type(module).__name__,
            first_shape,
            count,
            False,
            "linear module is not an attention/MLP target for this architecture",
        )
    reason = "; ".join(rule.reason for rule in matches)
    if not _late_stage_allowed(name, module_names, policy):
        return ModuleInspection(
            name,
            type(module).__name__,
            first_shape,
            count,
            False,
            "excluded by 3D late-stage policy; only reviewed final stages are selected",
            suggested_lora_target=matches[0].pattern,
        )
    return ModuleInspection(
        name,
        type(module).__name__,
        first_shape,
        count,
        True,
        reason,
        suggested_lora_target=matches[0].pattern,
    )


def inspect_modules(
    model: nn.Module,
    *,
    architecture: str | None = None,
    config: LoRAConfig | None = None,
    confirm_unknown: bool | None = None,
    include_unselected: bool = True,
) -> TargetResolution:
    """Inspect parameterized modules and select reviewed LoRA targets."""
    if not isinstance(model, nn.Module):
        raise TargetResolutionError(f"model must be torch.nn.Module, got {type(model).__name__}")
    resolved_architecture = _architecture_from_model(model, architecture)
    explicit_confirmation = bool(
        config.confirm_target_modules if config is not None else False
    ) if confirm_unknown is None else bool(confirm_unknown)
    policy = _policy_for(model, architecture, confirm_unknown=explicit_confirmation)
    explicit_patterns: tuple[str, ...] | None = None
    selected_policy = TargetPolicy.ARCHITECTURE_DEFAULT.value
    if config is not None:
        selected_policy = str(config.target_policy)
        if config.target_policy == TargetPolicy.EXPLICIT.value:
            explicit_patterns = tuple(config.target_modules or ())
            if not explicit_patterns:
                raise TargetResolutionError("explicit target policy requires target_modules")
            if policy.name == "unknown" and not config.confirm_target_modules:
                raise UnknownArchitectureError(
                    "unknown architectures require confirm_target_modules=true alongside explicit targets"
                )
    if policy.name == "unknown" and not explicit_patterns:
        raise UnknownArchitectureError("confirmed unknown architectures still require explicit target_modules")

    module_names = tuple(name for name, _ in model.named_modules() if name)
    records = tuple(
        _module_record(
            name,
            module,
            policy=policy,
            module_names=module_names,
            explicit_patterns=explicit_patterns,
        )
        for name, module in model.named_modules()
        if name
    )
    if not include_unselected:
        records = tuple(record for record in records if record.selected)
    selected_names = tuple(record.name for record in records if record.selected)
    if not selected_names:
        target_hint = explicit_patterns or tuple(rule.pattern for rule in policy.rules)
        raise TargetMatchError(
            f"LoRA target patterns matched zero modules for architecture {policy.name!r}: {list(target_hint)}"
        )
    max_targets = config.max_target_modules if config is not None else 512
    if len(selected_names) > max_targets:
        raise BroadTargetError(
            f"LoRA target selection is unexpectedly broad: {len(selected_names)} modules > "
            f"max_target_modules={max_targets}; inspect and narrow target_modules"
        )
    return TargetResolution(
        architecture=resolved_architecture or policy.name,
        policy=selected_policy,
        records=records,
        selected_names=selected_names,
    )


def resolve_targets(
    model: nn.Module,
    config: LoRAConfig,
    *,
    architecture: str | None = None,
    confirm_unknown: bool | None = None,
) -> TargetResolution:
    """Resolve a LoRA config, preserving all inspection reasons."""
    return inspect_modules(
        model,
        architecture=architecture or config.architecture,
        config=config,
        confirm_unknown=confirm_unknown,
        include_unselected=True,
    )


# Names used by CLI and older callers.
inspect_model_modules = inspect_modules
resolve_lora_targets = resolve_targets


__all__ = [
    "ModuleInspection",
    "TargetPolicySpec",
    "TargetResolution",
    "TargetRule",
    "available_target_policies",
    "inspect_model_modules",
    "inspect_modules",
    "resolve_lora_targets",
    "resolve_targets",
]
