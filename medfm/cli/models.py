"""Model registry CLI: medfm models ...

Metadata commands (list/show/validate without --local-weights,
estimate-memory, inspect-modules, accelerator-report) never touch the
network. Only ``download`` performs network I/O, and it is explicit.
``smoke`` runs a local tiny forward pass; it does not download weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from medfm.core.enums import LoadingMode, Modality, TaskType
from medfm.registry import ModelRegistry
from medfm.registry.schema import BACKEND_KEYS, LicenseClass


def _ensure_catalog() -> None:
    from medfm.registry import catalog

    catalog.ensure_v1_catalog()


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(getattr(k, "value", k)): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, StrEnum):
        return obj.value
    return obj


def list_models(args: argparse.Namespace) -> int:
    modality = Modality(args.modality) if args.modality else None
    task = TaskType(args.task) if args.task else None
    loading_mode = LoadingMode(args.loading_mode) if args.loading_mode else None
    license_class = LicenseClass(args.license_class) if args.license_class else None

    models = ModelRegistry.list_models(
        modality=modality,
        task=task,
        loading_mode=loading_mode,
        license_class=license_class,
        backend=args.backend,
        # CLI is an inspection surface: show blocked models with their reasons
        # by default; --ready-only restricts to production-loadable records.
        include_blocked=not args.ready_only,
        include_deprecated=args.include_deprecated,
    )

    if args.format == "json":
        print(json.dumps([_to_jsonable(m) for m in models], indent=2))
    else:
        for m in models:
            print(f"{m.model_id} ({m.status.value}): {m.license.class_type.value}")
    return 0


def show_model(args: argparse.Namespace) -> int:
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(_to_jsonable(model), indent=2))
    else:
        print(f"Model ID: {model.model_id}")
        print(f"Status: {model.status.value}")
        print(f"License: {model.license.name} ({model.license.status.value}, {model.license.class_type.value})")
        print(f"  Approved use cases: {', '.join(model.license.approved_use_cases) or 'none'}")
        print(f"  Prohibited use cases: {', '.join(model.license.prohibited_use_cases) or 'none'}")
        print(f"Repository: {model.repository}")
        print(f"Revision: {model.revision}")
        print(f"Schema version: {model.schema_version}")
        if model.deprecated:
            print(f"Deprecated: replaced by {model.replaced_by}")
        if model.blocked_reason:
            print(f"Blocked Reason: {model.blocked_reason}")
        print("Backend support:")
        for key in BACKEND_KEYS:
            support = model.backend_support.get(key)
            status = support.status.value if support else "UNTESTED"
            print(f"  {key}: {status}")
    return 0


def validate_model(args: argparse.Namespace) -> int:
    """Metadata validation; with --local-weights, local weight validation too.

    Neither mode uses the network.
    """
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    rc = 0
    if model.status.value == "BLOCKED":
        print(f"Model {args.model_id} metadata is BLOCKED: {model.blocked_reason}")
        rc = 1
    else:
        print(f"Model {args.model_id} metadata is valid.")

    if args.local_weights:
        from medfm.registry.weights import inspect_weights

        report = inspect_weights(Path(args.local_weights))
        if args.format == "json":
            print(json.dumps(report, indent=2))
        if not report["integrity_ok"]:
            print(f"Local weight validation FAILED for {args.local_weights}", file=sys.stderr)
            rc = 1
        else:
            print(f"Local weight validation passed for {args.local_weights}.")
    return rc


def download_model(args: argparse.Namespace) -> int:
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from medfm.registry.weights import download_weights

    try:
        path = download_weights(
            spec=model,
            cache_dir=args.cache_dir,
            token=args.token,
            allow_unsafe_formats=args.allow_unsafe,
        )
        print(f"Downloaded weights to {path}")
        return 0
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1


def accept_terms(args: argparse.Namespace) -> int:
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not model.license.acceptance_required:
        print(f"{model.model_id} does not require provider terms acceptance.")
        return 0

    from medfm.registry.acceptance import record_acceptance

    path = record_acceptance(model.model_id, model.repository, accepted_by=args.by)
    print(f"Recorded acceptance for {model.model_id} by {args.by} at {path}")
    return 0


def estimate_memory(args: argparse.Namespace) -> int:
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    mode = LoadingMode(args.loading_mode)
    if mode not in model.memory.loading_modes:
        print(f"Error: {mode.value} not supported by {model.model_id}", file=sys.stderr)
        return 1

    est = model.memory.loading_modes[mode]
    if args.format == "json":
        print(json.dumps(_to_jsonable(est), indent=2))
    else:
        print(f"Memory estimate for {model.model_id} in {mode.value}:")
        print(f"  Host Bytes: {est.host_bytes}")
        print(f"  Device Bytes: {est.device_bytes}")
        print(f"  Weight Format: {est.weight_format.value}")
        print(f"  Topology: {est.topology.value}")
        if est.measured_peak_bytes is not None:
            print(f"  Measured Peak Bytes: {est.measured_peak_bytes}")
        if est.uncertainty_note:
            print(f"  Note: {est.uncertainty_note}")
        if model.memory.compile_risk_note:
            print(f"  Compile Risk: {model.memory.compile_risk_note}")
    return 0


def smoke_model(args: argparse.Namespace) -> int:
    from medfm.registry.smoke import BackendUnavailableError, NoAdapterError, run_smoke

    try:
        result = run_smoke(
            args.model_id,
            backend=args.backend,
            artifact_dir=args.artifact_dir,
        )
    except NoAdapterError as e:
        print(f"Smoke unavailable: {e}", file=sys.stderr)
        return 2
    except (BackendUnavailableError, RuntimeError) as e:
        print(f"Smoke failed: {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(f"Smoke OK: {result.model_id}@{result.revision} on {result.backend}: {result.detail}")
    return 0


def inspect_modules(args: argparse.Namespace) -> int:
    try:
        model = ModelRegistry.get(args.model_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not model.capabilities.peft.supported:
        print(f"Model {model.model_id} does not support PEFT.")
        return 1

    if model.capabilities.peft.known_target_modules:
        print(f"Known target modules: {', '.join(model.capabilities.peft.known_target_modules)}")
    else:
        print(
            f"Model {model.model_id}: LoRA target modules unknown for this family; "
            f"confirmation required before PEFT training."
        )
    return 0


def accelerator_report(args: argparse.Namespace) -> int:
    report = ModelRegistry.accelerator_report()
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        header = ["model_id", *BACKEND_KEYS]
        rows = [[mid, *[str(s) for s in backends.values()]] for mid, backends in report.items()]
        widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        for row in rows:
            print(fmt.format(*row))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm models")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list models")
    list_parser.add_argument("--modality", type=str)
    list_parser.add_argument("--task", type=str)
    list_parser.add_argument("--loading-mode", type=str)
    list_parser.add_argument("--license-class", type=str)
    list_parser.add_argument("--backend", choices=BACKEND_KEYS, help="filter to SUPPORTED_* on this backend")
    list_parser.add_argument("--ready-only", action="store_true", help="hide BLOCKED models (deployment-ready view)")
    list_parser.add_argument("--include-deprecated", action="store_true")
    list_parser.add_argument("--format", choices=["text", "json"], default="text")

    show_parser = sub.add_parser("show", help="show model details incl. use cases")
    show_parser.add_argument("model_id")
    show_parser.add_argument("--format", choices=["text", "json"], default="text")

    validate_parser = sub.add_parser("validate", help="validate metadata and optionally local weights")
    validate_parser.add_argument("model_id")
    validate_parser.add_argument("--local-weights", help="directory of local weights to validate")
    validate_parser.add_argument("--format", choices=["text", "json"], default="text")

    download_parser = sub.add_parser("download", help="download weights (network, explicit)")
    download_parser.add_argument("model_id")
    download_parser.add_argument("--cache-dir", required=True)
    download_parser.add_argument("--token", help="HF token (prefer env; never logged)")
    download_parser.add_argument("--allow-unsafe", action="store_true", help="allow pickle formats (reviewed)")

    terms_parser = sub.add_parser("accept-terms", help="record provider terms acceptance")
    terms_parser.add_argument("model_id")
    terms_parser.add_argument("--by", required=True, help="name of the individual accepting")

    mem_parser = sub.add_parser("estimate-memory", help="estimate memory usage")
    mem_parser.add_argument("model_id")
    mem_parser.add_argument("--loading-mode", required=True)
    mem_parser.add_argument("--format", choices=["text", "json"], default="text")

    smoke_parser = sub.add_parser("smoke", help="run a local tiny-forward smoke test")
    smoke_parser.add_argument("model_id")
    smoke_parser.add_argument("--backend", choices=BACKEND_KEYS, default="cpu")
    smoke_parser.add_argument("--artifact-dir", help="write a run artifact here")
    smoke_parser.add_argument("--format", choices=["text", "json"], default="text")

    inspect_parser = sub.add_parser("inspect-modules", help="inspect PEFT target modules")
    inspect_parser.add_argument("model_id")

    report_parser = sub.add_parser("accelerator-report", help="per-model per-backend compatibility report")
    report_parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args(argv)
    _ensure_catalog()

    handlers = {
        "list": list_models,
        "show": show_model,
        "validate": validate_model,
        "download": download_model,
        "accept-terms": accept_terms,
        "estimate-memory": estimate_memory,
        "smoke": smoke_model,
        "inspect-modules": inspect_modules,
        "accelerator-report": accelerator_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
