"""CLI for the Phase 18 release gate.

Subcommands:
  validate  - run every CPU-runnable release check (registry backend status,
              license/scope consistency, backend-neutral import scan, TPU/NF4
              policy, clinical-claims scan, and phase reports 00..18).
  matrix    - regenerate docs/release/support_matrix.md from the live catalog.
  checksums - regenerate docs/release/checksums.txt over the release docs.
"""

from __future__ import annotations

import argparse
import sys

from medfm.tools import release


def validate_command(args: argparse.Namespace) -> int:
    del args
    errors = release.validate()
    if errors:
        print(f"release validate FAILED ({len(errors)} problem(s)):")
        for message in errors:
            print(f"  - {message}")
        return 1
    print("release validate OK")
    return 0


def matrix_command(args: argparse.Namespace) -> int:
    del args
    path = release.generate_support_matrix()
    print(f"wrote support matrix: {path.relative_to(release.REPO_ROOT)}")
    return 0


def checksums_command(args: argparse.Namespace) -> int:
    del args
    path = release.write_checksums()
    print(f"wrote checksums: {path.relative_to(release.REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 18 release gate tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="run all CPU-runnable release checks")
    sub.add_parser("matrix", help="regenerate the model x task x backend support matrix")
    sub.add_parser("checksums", help="regenerate release checksums over docs/release")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return validate_command(args)
    if args.command == "matrix":
        return matrix_command(args)
    if args.command == "checksums":
        return checksums_command(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
