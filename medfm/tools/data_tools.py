"""``medfm data`` subcommands: fingerprint, inspect, migrate, split.

Thin, testable implementations behind :mod:`medfm.cli.data`. Every command
validates manifests fail-closed, writes atomically enough for CLI use, and
emits deterministic machine-readable output (no timestamps/paths in hashes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medfm.core.enums import SplitName
from medfm.data.errors import DataError
from medfm.data.fingerprint import fingerprint_manifest
from medfm.data.manifests.io import inspect_manifest, read_manifest, read_manifest_with_version, write_manifest
from medfm.data.splits import (
    DEFAULT_SITE_RATIOS,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEMPORAL_RATIOS,
    SplitPolicy,
    assert_no_split_leakage,
    build_split_report,
    generate_split_assignment,
)


def _parse_ratios(raw: str | None) -> tuple[tuple[SplitName, float], ...] | None:
    if raw is None:
        return None
    ratios: list[tuple[SplitName, float]] = []
    for part in raw.split(","):
        name, _, weight = part.partition(":")
        if not name or not weight:
            raise DataError(f"invalid ratio entry {part!r}; expected NAME:WEIGHT[,NAME:WEIGHT...]")
        ratios.append((SplitName.from_value(name.strip().upper()), float(weight)))
    return tuple(ratios)


def _emit(payload: dict[str, Any], output: str | None) -> int:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        Path(output).write_text(text, encoding="utf-8")
    return 0


def cmd_fingerprint(args: argparse.Namespace) -> int:
    df = read_manifest(args.manifest)
    report = fingerprint_manifest(df, leakage_temporal_policy=args.temporal)
    if not args.no_leakage_gate and not report["split_leakage"]["ok"]:
        sys.stderr.write(
            "split leakage detected; fingerprint written but the manifest must not train without a fix "
            "or a recorded ResearchOverride\n"
        )
    return _emit(report, args.output)


def cmd_inspect(args: argparse.Namespace) -> int:
    return _emit(inspect_manifest(args.manifest), args.output)


def cmd_migrate(args: argparse.Namespace) -> int:
    df, version = read_manifest_with_version(args.input)
    # Validate before writing so migration never produces an unusable manifest.
    write_manifest(df, args.output, base_dir=Path(args.base_dir) if args.base_dir else None)
    from medfm.data.manifests.schema import MANIFEST_SCHEMA_VERSION

    sys.stderr.write(f"migrated manifest schema v{version} -> v{MANIFEST_SCHEMA_VERSION}: {args.output}\n")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    df = read_manifest(args.manifest)
    policy = SplitPolicy(args.policy.upper())
    ratios = (
        _parse_ratios(args.ratios)
        or {
            SplitPolicy.PATIENT: DEFAULT_SPLIT_RATIOS,
            SplitPolicy.SITE: DEFAULT_SITE_RATIOS,
            SplitPolicy.TEMPORAL: DEFAULT_TEMPORAL_RATIOS,
        }[policy]
    )
    assigned = generate_split_assignment(df, policy=policy, seed=args.seed, ratios=ratios)
    report = build_split_report(assigned, policy=policy, seed=args.seed, ratios=ratios)
    assert_no_split_leakage(assigned, temporal_policy=policy is SplitPolicy.TEMPORAL)
    base_dir = Path(args.base_dir) if args.base_dir else None
    write_manifest(assigned, args.output, base_dir=base_dir)
    _emit(report.to_dict(), args.report)
    sys.stderr.write(
        f"split assignment written to {args.output} (policy {policy.value}, seed {args.seed}, "
        f"report {args.report or 'stdout'})\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medfm data", description="Dataset manifest tools (Phase 03).")
    sub = parser.add_subparsers(dest="command", required=True)

    fingerprint = sub.add_parser("fingerprint", help="compute the deterministic dataset fingerprint report")
    fingerprint.add_argument("--manifest", required=True, help="manifest path (.parquet or .jsonl)")
    fingerprint.add_argument("--output", default=None, help="write JSON report here instead of stdout")
    fingerprint.add_argument("--temporal", action="store_true", help="treat patient overlap as temporal policy")
    fingerprint.add_argument("--no-leakage-gate", action="store_true", help="suppress the leakage warning note")
    fingerprint.set_defaults(func=cmd_fingerprint)

    inspect_cmd = sub.add_parser("inspect", help="schema-inspect a manifest without validation")
    inspect_cmd.add_argument("--manifest", required=True)
    inspect_cmd.add_argument("--output", default=None)
    inspect_cmd.set_defaults(func=cmd_inspect)

    migrate = sub.add_parser("migrate", help="migrate a manifest to the current schema version")
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--base-dir", default=None, help="dataset root anchoring relative URIs")
    migrate.set_defaults(func=cmd_migrate)

    split = sub.add_parser("split", help="generate a patient/site/temporal split assignment")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True, help="where to write the split-assigned manifest")
    split.add_argument("--report", default=None, help="where to write the split report JSON (default stdout)")
    split.add_argument("--policy", default="patient", choices=["patient", "site", "temporal"])
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--ratios", default=None, help="e.g. TRAIN:0.7,VAL:0.15,TEST:0.15")
    split.add_argument("--base-dir", default=None, help="dataset root anchoring relative URIs")
    split.set_defaults(func=cmd_split)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DataError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
