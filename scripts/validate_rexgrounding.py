#!/usr/bin/env python
"""Validate ReXGroundingCT metadata, released masks, and staged CT headers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from train_rexgrounding import resolve_local_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../RexGroundingData"))
    parser.add_argument("--volume-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--require-volumes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    volume_root = (args.volume_root or data_dir / "ct_rate_fixed").resolve()
    metadata_path = data_dir / "MICCAI_challenge_dataset.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    masks_dir = data_dir / "segmentations"
    expected_masks = {str(entry["name"]) for split in args.splits for entry in payload[split] if split != "test"}
    expected_volumes = {
        str(entry["name"]): tuple(int(v) for v in entry["shape"]) for split in args.splits for entry in payload[split]
    }
    missing_masks = sorted(name for name in expected_masks if not (masks_dir / name).is_file())
    mask_shape_errors: list[dict[str, object]] = []
    for name in sorted(expected_masks - set(missing_masks)):
        shape = tuple(int(v) for v in nib.load(str(masks_dir / name)).shape)
        metadata_entry = next(
            entry for split in args.splits if split != "test" for entry in payload[split] if entry["name"] == name
        )
        expected_shape = (len(metadata_entry["findings"]), *tuple(int(v) for v in metadata_entry["shape"]))
        if shape != expected_shape:
            mask_shape_errors.append({"name": name, "actual": shape, "expected": expected_shape})
    missing_volumes: list[str] = []
    volume_shape_errors: list[dict[str, object]] = []
    if args.require_volumes:
        for name, expected_shape in sorted(expected_volumes.items()):
            path = resolve_local_volume(volume_root, name)
            if path is None:
                missing_volumes.append(name)
                continue
            actual_shape = tuple(int(v) for v in nib.load(str(path)).shape)
            if actual_shape != expected_shape:
                volume_shape_errors.append({"name": name, "actual": actual_shape, "expected": expected_shape})
    result = {
        "metadata": str(metadata_path),
        "splits": list(args.splits),
        "expected_masks": len(expected_masks),
        "missing_masks": missing_masks,
        "mask_shape_errors": mask_shape_errors,
        "expected_volumes": len(expected_volumes),
        "missing_volumes": missing_volumes,
        "volume_shape_errors": volume_shape_errors,
        "volume_root": str(volume_root),
    }
    print(json.dumps(result, sort_keys=True))
    if missing_masks or mask_shape_errors or (args.require_volumes and (missing_volumes or volume_shape_errors)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
