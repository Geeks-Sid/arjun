"""CLI for producing validated adapter-only deployment bundles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from medfm.inference import BaseModelReference, BundleBuilder, RuntimeSupport


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("export config must be a mapping")
    return dict(value)


def _tensor_groups(config: Mapping[str, Any], root: Path) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {}
    raw = config.get("tensor_groups", {})
    if not isinstance(raw, Mapping):
        raise ValueError("tensor_groups must be a mapping")
    for group, source in raw.items():
        path = (root / str(source)).resolve()
        if not path.is_file() or path.suffix not in {".pt", ".pth"}:
            raise ValueError("tensor groups accept only reviewed .pt/.pth state dictionaries")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in value.items()
        ):
            raise ValueError("tensor group file must contain a string-to-tensor mapping")
        groups[str(group)] = dict(value)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm export", description="Build an adapter-only deployment bundle.")
    sub = parser.add_subparsers(dest="command", required=True)
    bundle = sub.add_parser("bundle", help="write and validate a deployment bundle")
    bundle.add_argument("--config", required=True, type=Path)
    bundle.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "bundle":  # pragma: no cover
        return 2
    try:
        config = _load(args.config)
        bases = [BaseModelReference.from_dict(value) for value in config["base_models"]]
        runtime = RuntimeSupport.from_dict(config.get("runtime"))
        builder = BundleBuilder(
            args.output,
            bundle_id=str(config["bundle_id"]),
            model_id=str(config["model_id"]),
            model_revision=str(config["model_revision"]),
            task=str(config["task"]),
            base_models=bases,
            model_card=str(config["model_card"]),
            license_summary=str(config["license_summary"]),
            preprocessing=dict(config.get("preprocessing", {})),
            postprocessing=dict(config.get("postprocessing", {})),
            task_schema=dict(config.get("task_schema", {"type": "object"})),
            inference_config=dict(config.get("inference_config", {})),
            modalities=tuple(str(value) for value in config.get("modalities", ())),
            runtime=runtime,
            metadata=dict(config.get("metadata", {})),
        )
        for group, tensors in _tensor_groups(config, args.config.parent).items():
            builder.add_tensor_group(group, tensors)
        bundle = builder.build()
        print(
            json.dumps({"bundle": str(bundle.root), "bundle_id": bundle.bundle_id, "checksums": len(bundle.checksums)})
        )
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "bundle export failed", "type": type(exc).__name__}))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
