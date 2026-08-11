#!/usr/bin/env python
"""Prepare cached TotalSegmentator thoracic priors for ReXGroundingCT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from train_rexgrounding import load_entries, resolve_local_volume

from medfm.data.totalsegmentator import (
    DEFAULT_THORACIC_LABELS,
    load_total_segmentator_prior,
    prior_case_dir,
    run_totalsegmentator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../RexGroundingData"))
    parser.add_argument("--volume-root", type=Path, default=None)
    parser.add_argument("--prior-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--labels", nargs="+", default=DEFAULT_THORACIC_LABELS)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    if not args.labels:
        raise ValueError("--labels must not be empty")
    data_dir = args.data_dir.resolve()
    volume_root = (args.volume_root or data_dir / "ct_rate_fixed").resolve()
    prior_dir = (args.prior_dir or data_dir / "totalsegmentator_priors").resolve()
    entries = [entry for split in args.splits for entry in load_entries(data_dir, split, require_masks=False)]
    entries = sorted({entry.name: entry for entry in entries}.values(), key=lambda entry: entry.name)
    if args.max_cases:
        entries = entries[: args.max_cases]
    manifest: dict[str, object] = {
        "data_dir": str(data_dir),
        "volume_root": str(volume_root),
        "prior_dir": str(prior_dir),
        "splits": list(args.splits),
        "labels": list(args.labels),
        "device": args.device,
        "fast": args.fast,
        "cases_requested": len(entries),
        "cases_completed": 0,
        "dry_run": args.dry_run,
    }
    for entry in entries:
        volume_path = resolve_local_volume(volume_root, entry.name)
        if volume_path is None:
            raise FileNotFoundError(f"missing staged CT volume for {entry.name} under {volume_root}")
        case_dir = prior_case_dir(prior_dir, entry.name)
        command = [
            "TotalSegmentator",
            "-i",
            str(volume_path),
            "-o",
            str(case_dir),
            "-ta",
            "total",
            "--roi_subset",
            *args.labels,
            "--device",
            str(args.device),
        ]
        if args.fast:
            command.append("--fast")
        print(json.dumps({"case": entry.name, "command": command}, sort_keys=True))
        if args.dry_run:
            continue
        volume_affine = nib.load(str(volume_path)).affine
        run_totalsegmentator(
            volume_path,
            case_dir,
            args.labels,
            device=args.device,
            fast=args.fast,
        )
        load_total_segmentator_prior(prior_dir, entry.name, args.labels, entry.shape, volume_affine)
        manifest["cases_completed"] = int(manifest["cases_completed"]) + 1
    prior_dir.mkdir(parents=True, exist_ok=True)
    (prior_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
