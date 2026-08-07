"""Accelerator validation CLI: medfm accelerator ...

``validate-model`` checks a model/backend/loading-mode combination *before*
weight allocation (registry policy) and then runs a tiny backend-specific
forward pass against the exact registered revision. A passing validation
records per-backend smoke evidence; it never mutates other backends.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import replace
from typing import Any

from medfm.core.enums import LoadingMode
from medfm.registry import ModelRegistry
from medfm.registry.schema import BACKEND_KEYS
from medfm.training.backend import BackendUnavailableError
from medfm.training.config import RunConfig, RunConfigError


def validate_model(args: argparse.Namespace) -> int:
    from medfm.registry import catalog
    from medfm.registry.smoke import BackendUnavailableError, NoAdapterError, run_smoke

    catalog.ensure_v1_catalog()

    model_id = args.model_id or getattr(args, "model_flag", None)
    if not model_id:
        print("validate-model requires a model id (positional or --model)", file=sys.stderr)
        return 2
    try:
        spec = ModelRegistry.get(model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    loading_mode = LoadingMode(args.loading_mode) if args.loading_mode else next(iter(spec.memory.loading_modes))

    # 1. Policy gate: reject unsupported combinations before any allocation.
    try:
        ModelRegistry.validate_backend(spec, args.backend, loading_mode)
    except ValueError as e:
        print(f"validate-model REJECTED (pre-allocation): {e}", file=sys.stderr)
        return 2

    # 2. Runtime smoke with the exact pinned revision and a tiny input.
    try:
        result = run_smoke(
            spec.model_id,
            backend=args.backend,
            artifact_dir=args.artifact_dir,
        )
    except NoAdapterError as e:
        print(f"validate-model: no adapter yet: {e}", file=sys.stderr)
        return 3
    except (BackendUnavailableError, RuntimeError) as e:
        print(f"validate-model smoke failed: {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(
            f"validate-model OK: {result.model_id}@{result.revision} "
            f"on {result.backend} ({loading_mode.value}): {result.detail}"
        )
    return 0


def _run_tiny(config: RunConfig, backend: str, *, max_steps: int = 1) -> Any:
    from medfm.cli.train import tiny_builders
    from medfm.training.pipeline import TrainingPipeline

    resolved = replace(
        config,
        accelerator=replace(config.accelerator, backend=backend),
        max_steps=max_steps,
    )
    built = TrainingPipeline(resolved, builders=tiny_builders()).build()
    trainer = built.trainer
    if trainer is None:
        raise RuntimeError("tiny accelerator recipe did not build a trainer")
    return trainer.train()


def parity(args: argparse.Namespace) -> int:
    try:
        config = RunConfig.load(args.config)
        left = _run_tiny(config, args.left)
        right = _run_tiny(config, args.right)
        left_loss = float(left.metrics.get("train/loss", float("nan")))
        right_loss = float(right.metrics.get("train/loss", float("nan")))
        delta = abs(left_loss - right_loss)
        payload = {
            "left": args.left,
            "right": args.right,
            "left_loss": left_loss,
            "right_loss": right_loss,
            "absolute_loss_delta": delta,
            "atol": args.atol,
            "passed": delta <= args.atol,
        }
    except (BackendUnavailableError, RunConfigError, RuntimeError, ValueError) as exc:
        print(f"accelerator parity failed: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"parity {'OK' if payload['passed'] else 'FAILED'}: "
            f"{args.left} vs {args.right} | loss delta={payload['absolute_loss_delta']:.6g}"
        )
    return 0 if payload["passed"] else 2


def profile(args: argparse.Namespace) -> int:
    try:
        config = RunConfig.load(args.config)
        result = _run_tiny(config, args.backend, max_steps=args.steps)
    except (BackendUnavailableError, RunConfigError, RuntimeError, ValueError) as exc:
        print(f"accelerator profile failed: {exc}", file=sys.stderr)
        return 1
    payload = result.to_dict()
    payload["profile_backend"] = args.backend
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"profile OK: backend={args.backend} optimizer_steps={payload['optimizer_steps']} "
            f"peak_memory={payload['peak_memory']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm accelerator")
    sub = parser.add_subparsers(dest="command", required=True)

    vm = sub.add_parser(
        "validate-model",
        help="pre-allocation policy check + tiny backend-specific smoke",
    )
    vm.add_argument("model_id", nargs="?")
    vm.add_argument("--model", dest="model_flag", help="model id (alternative to positional)")
    vm.add_argument("--backend", choices=BACKEND_KEYS, required=True)
    vm.add_argument("--loading-mode", help="LoadingMode name; default: first declared")
    vm.add_argument("--artifact-dir", help="write the smoke run artifact here")
    vm.add_argument("--format", choices=["text", "json"], default="text")

    parity_parser = sub.add_parser("parity", help="run a tiny one-step cross-backend loss parity check")
    parity_parser.add_argument("--config", required=True)
    parity_parser.add_argument("--left", choices=["cpu", "cuda", "xla_tpu"], required=True)
    parity_parser.add_argument("--right", choices=["cpu", "cuda", "xla_tpu"], required=True)
    parity_parser.add_argument("--atol", type=float, default=1e-3)
    parity_parser.add_argument("--format", choices=["text", "json"], default="text")

    profile_parser = sub.add_parser("profile", help="run a tiny backend profile and emit provenance")
    profile_parser.add_argument("--config", required=True)
    profile_parser.add_argument("--backend", choices=["cpu", "cuda", "xla_tpu"], required=True)
    profile_parser.add_argument("--steps", type=int, default=1)
    profile_parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args(argv)
    if args.command == "validate-model":
        return validate_model(args)
    if args.command == "parity":
        return parity(args)
    if args.command == "profile":
        return profile(args)
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
