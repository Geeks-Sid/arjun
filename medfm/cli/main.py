"""medfm command-line entry point.

Subcommands deliberately stay thin: each one delegates to a module under
``medfm.tools`` so the same functionality is importable and testable.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm", description="medfm framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="print runtime diagnostics")
    doctor.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda", "xla_tpu"],
        default="auto",
        help="accelerator backend to diagnose (default: auto)",
    )
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    sub.add_parser("data", help="dataset manifest tools (fingerprint/inspect/migrate/split)")
    sub.add_parser("models", help="model registry and capabilities tools")
    sub.add_parser("peft", help="LoRA/QLoRA inspection and adapter tools")
    sub.add_parser("accelerator", help="accelerator validation and parity tools")
    sub.add_parser("infer", help="run bounded model inference")
    sub.add_parser("export", help="export validated deployment bundles")
    sub.add_parser("train", help="run a training recipe")

    args, rest = parser.parse_known_args(argv)
    if args.command == "doctor":
        from medfm.tools import doctor as doctor_mod

        forwarded = ["--backend", args.backend]
        if args.json:
            forwarded.append("--json")
        return doctor_mod.main([*forwarded, *rest])
    if args.command == "data":
        from medfm.tools import data_tools

        return data_tools.main(rest)
    if args.command == "models":
        from medfm.cli import models as models_cli

        return models_cli.main(rest)
    if args.command == "peft":
        from medfm.cli import peft as peft_cli

        return peft_cli.main(rest)
    if args.command == "accelerator":
        from medfm.cli import accelerator as accel_cli

        return accel_cli.main(rest)
    if args.command == "evaluate":
        from medfm.cli import evaluate as evaluate_cli

        return evaluate_cli.main(rest)
    if args.command == "infer":
        from medfm.cli import infer as infer_cli

        return infer_cli.main(rest)
    if args.command == "export":
        from medfm.cli import export as export_cli

        return export_cli.main(rest)
    if args.command == "train":
        from medfm.cli import train as train_cli

        return train_cli.main(rest)

    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
