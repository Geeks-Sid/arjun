"""Phase 09 training-stage declarations (not optimizer/trainer internals)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

from torch import nn

from medfm.core.errors import ShapeContractError


class TrainingStage(IntEnum):
    """Progressive unfreezing stages shared with later training phases."""

    BRIDGE_ONLY = 1
    BRIDGE_LANGUAGE = 2
    BRIDGE_LANGUAGE_VISION = 3
    MULTITASK = 4


@dataclass(frozen=True)
class TrainableModuleDeclaration:
    """A stable name and rationale for a trainable component."""

    name: str
    rationale: str
    fully_trainable: bool = True


@dataclass(frozen=True)
class StageConfig:
    """Resolved stage policy consumed by PEFT/trainer phases."""

    stage: TrainingStage
    trainable_modules: tuple[str, ...]
    declarations: tuple[TrainableModuleDeclaration, ...]
    task_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.trainable_modules:
            raise ShapeContractError("a stage must declare at least one trainable module")
        if any(not name for name in self.trainable_modules):
            raise ShapeContractError("trainable module names must be non-empty")
        if any(weight <= 0 for _, weight in self.task_weights):
            raise ShapeContractError("task weights must be positive")

    @property
    def task_weight_map(self) -> dict[str, float]:
        return dict(self.task_weights)


def stage_config(stage: TrainingStage | int, *, task_weights: Mapping[str, float] | None = None) -> StageConfig:
    """Build the phase contract without modifying modules or optimizer state."""
    resolved = TrainingStage(int(stage))
    declarations = [
        TrainableModuleDeclaration("bridge", "learn the encoder-to-language representation", True),
        TrainableModuleDeclaration("boundary", "learn model-neutral visual span boundaries", True),
    ]
    names = ["bridge", "boundary"]
    if resolved >= TrainingStage.BRIDGE_LANGUAGE:
        declarations.append(
            TrainableModuleDeclaration("language_lora", "adapt the language model after bridge evidence", False)
        )
        names.append("language_lora")
    if resolved >= TrainingStage.BRIDGE_LANGUAGE_VISION:
        declarations.append(
            TrainableModuleDeclaration("vision_lora", "late vision adaptation after Stage 2 evidence", False)
        )
        names.append("vision_lora")
    if resolved == TrainingStage.MULTITASK:
        weights = tuple(sorted((str(k), float(v)) for k, v in (task_weights or {}).items()))
        if not weights:
            raise ShapeContractError("Stage 4 requires explicit positive task_weights")
    else:
        weights = ()
    return StageConfig(resolved, tuple(names), tuple(declarations), weights)


def apply_stage_freeze(
    modules: Mapping[str, nn.Module],
    config: StageConfig,
    *,
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Set ``requires_grad`` exactly according to a stage declaration.

    This is deliberately limited to module freezing; optimizer construction,
    PEFT injection, and scheduler behavior remain outside Phase 09.
    """
    aliases = dict(aliases or {})
    # Until Phase 10 injects actual PEFT modules, the declared LoRA names may
    # resolve to their base module for freeze-audit smoke tests.
    if "language_lora" not in modules and "language" in modules:
        aliases.setdefault("language_lora", "language")
    if "vision_lora" not in modules and "vision" in modules:
        aliases.setdefault("vision_lora", "vision")
    resolved: dict[str, nn.Module] = dict(modules)
    for alias, target in aliases.items():
        if target in modules:
            resolved[alias] = modules[target]
    missing = [name for name in config.trainable_modules if name not in resolved]
    if missing:
        raise ShapeContractError(f"stage references unavailable modules: {missing}")
    trainable = set(config.trainable_modules)
    for name, module in resolved.items():
        if name in {"vision", "language", "language_model"} and name not in trainable:
            module.requires_grad_(False)
        elif name not in trainable and name in ("bridge", "boundary", "language_lora", "vision_lora"):
            module.requires_grad_(False)
        elif name in trainable:
            module.requires_grad_(True)
    return tuple(
        name for name, module in resolved.items() if any(parameter.requires_grad for parameter in module.parameters())
    )


def trainable_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Expose deterministic names for the Phase 10/12 gradient audit."""
    return tuple(name for name, parameter in module.named_parameters() if parameter.requires_grad)


__all__ = [
    "StageConfig",
    "TrainableModuleDeclaration",
    "TrainingStage",
    "apply_stage_freeze",
    "stage_config",
    "trainable_parameter_names",
]
