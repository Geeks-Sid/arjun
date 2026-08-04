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

    args, rest = parser.parse_known_args(argv)
    if args.command == "doctor":
        from medfm.tools import doctor as doctor_mod

        forwarded = ["--backend", args.backend]
        if args.json:
            forwarded.append("--json")
        return doctor_mod.main([*forwarded, *rest])

    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
