"""CLI for auditable PEFT target inspection and capability reporting."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from medfm.peft import LoRAConfig, inspect_modules


def _tiny_vision_inspection_model() -> Any:
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8)
            self.self_attn.k_proj = nn.Linear(8, 8)
            self.self_attn.v_proj = nn.Linear(8, 8)
            self.self_attn.out_proj = nn.Linear(8, 8)
            self.mlp = nn.Module()
            self.mlp.fc1 = nn.Linear(8, 16)
            self.mlp.fc2 = nn.Linear(16, 8)

    class _Vision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([_Block(), _Block()])
            self.norm = nn.LayerNorm(8)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return pixel_values

    return _Vision()


def _build_inspection_model(model_id: str) -> tuple[Any, str]:
    """Build the registry's tiny plugin, never download or allocate real weights."""
    try:
        from medfm.registry import ModelRegistry, catalog
        from medfm.registry.plugins import get_plugin

        catalog.ensure_v1_catalog()
        spec = ModelRegistry.get(model_id)
        plugin = get_plugin(spec.model_id)
        if plugin is not None:
            return plugin.build(spec), model_id
    except Exception:
        # The inspection command remains useful for an unregistered local
        # family; unknown architecture errors below still fail closed.
        pass

    lowered = model_id.lower()
    if any(token in lowered for token in ("gemma", "llama", "mistral", "qwen", "phi", "lamed", "causal")):
        from medfm.models.language import GenericHFCausalLMAdapter

        return GenericHFCausalLMAdapter.build_tiny(), model_id
    if any(token in lowered for token in ("3d", "ct-fm", "flexict", "triad", "merlin", "segment")):
        from medfm.models.visual.native_3d import GenericMONAI3DAdapter

        return GenericMONAI3DAdapter.build_tiny(), model_id
    if any(token in lowered for token in ("vision", "vit", "siglip", "dino", "pathology", "medsiglip", "rad")):
        return _tiny_vision_inspection_model(), model_id
    raise KeyError(
        f"no tiny inspection plugin for model {model_id!r}; register a model plugin or pass a reviewed architecture"
    )


def inspect_command(args: argparse.Namespace) -> int:
    try:
        model, resolved_id = _build_inspection_model(args.model)
        root = getattr(model, "model", None) or getattr(model, "backbone", None) or model
        lowered = args.model.lower()
        if any(token in lowered for token in ("3d", "ct-fm", "flexict", "triad", "merlin", "segment")):
            architecture = "3d_transformer"
        elif any(token in lowered for token in ("gemma", "llama", "mistral", "qwen", "phi", "lamed", "causal")):
            architecture = "llm"
        else:
            architecture = "vision"
        config = LoRAConfig(
            architecture=architecture,
            target_policy="architecture_default",
            max_target_modules=args.max_targets,
        )
        resolution = inspect_modules(root, architecture=architecture, config=config)
    except Exception as exc:
        print(f"peft inspect failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = {
        "model": args.model,
        "resolved_model_id": resolved_id,
        "architecture": resolution.architecture,
        "policy": resolution.policy,
        "selected_count": resolution.selected_count,
        "modules": [record.to_dict() for record in resolution.records],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(
        f"PEFT inspection: {args.model} (resolved {resolved_id}, architecture={resolution.architecture}, "
        f"selected={resolution.selected_count})"
    )
    print("module_name\tmodule_type\tparameter_shape\tparameter_count\tselected\treason")
    for record in resolution.records:
        print(
            f"{record.name}\t{record.module_type}\t{record.parameter_shape}\t{record.parameter_count}\t"
            f"{str(record.selected).lower()}\t{record.reason}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm peft")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="inspect LoRA candidates before injection")
    inspect.add_argument("--model", required=True, help="registered model ID or alias")
    inspect.add_argument("--format", choices=("text", "json"), default="text")
    inspect.add_argument("--max-targets", type=int, default=512)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return inspect_command(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
