#!/usr/bin/env python3
"""Deterministic golden-fixture generator (Phase 18, Level 4 regression).

Regenerate pinned fixtures with:

    uv run --frozen python scripts/generate_golden.py

Every value below is CPU-deterministic (seeded torch/numpy, no hardware noise).
The test suite (tests/phase_18/test_golden_regression.py) re-derives each value
and compares it to the committed fixture, so any upstream drift (torch,
torchvision, MONAI) that changes shapes, preprocess statistics, logits, masks,
structured fields, or memory envelopes fails the gate.

All fixtures are exact JSON (float ``repr`` round-trips). A committed
``manifest.json`` pins each file's SHA-256 so an accidental fixture edit is
detected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "phase_18" / "golden"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tiny_2d_logits() -> dict[str, object]:
    """Pinned [2, 4] logits for a tiny seeded MLP on a fixed input."""
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )
    generator = torch.Generator().manual_seed(4321)
    x = torch.randn(2, 64, generator=generator)
    with torch.no_grad():
        logits = model(x)
    return {
        "shape": list(logits.shape),
        "logits_flat": [float(value) for value in logits.flatten()],
    }


def preprocess_stats() -> dict[str, object]:
    """Pinned CT preprocess statistics (clip + window on a fixed HU volume)."""
    hu = torch.linspace(-200.0, 400.0, 8 * 8 * 8).reshape(1, 8, 8, 8)
    from medfm.data.transforms.base import TransformData
    from medfm.data.transforms.ct import ClipHU, WindowChannels

    data = TransformData(image=hu.contiguous())
    data = ClipHU(-1024.0, 3071.0).apply(data, None)
    data = WindowChannels(((40.0, 400.0),)).apply(data, None)
    out = data.image
    return {
        "shape": list(out.shape),
        "mean": float(out.mean()),
        "std": float(out.std()),
        "min": float(out.min()),
        "max": float(out.max()),
        "history_stages": [record.stage for record in data.history],
    }


def native3d_identity() -> dict[str, object]:
    """Pinned sliding-window identity reconstruction bound."""
    from medfm.inference.sliding_window import sliding_window_inference

    volume = torch.arange(1.0, 1 + 2 * 4 * 4).reshape(1, 1, 2, 4, 4)
    restored = sliding_window_inference(volume, lambda crop: crop, window_shape=(2, 2, 2), overlap=0.5)
    return {
        "output_shape": list(restored.shape),
        "max_abs_error": float((restored - volume).abs().max()),
    }


def metrics_dice_iou() -> dict[str, object]:
    """Pinned Dice/IoU for a fixed mask pair (Phase 16 metric kernel)."""
    from medfm.evaluation.advanced import _spatial_summary

    pred = np.zeros((16, 16), dtype=bool)
    pred[3:8, 4:10] = True
    truth = np.zeros((16, 16), dtype=bool)
    truth[4:9, 5:11] = True
    summary = _spatial_summary(pred, truth, (1.0, 1.0), 1.0)
    return {
        "dice": float(summary["dice"]),
        "iou": float(summary["iou"]),
        "empty_case": str(summary["empty_case"]),
    }


def structured_findings() -> dict[str, object]:
    """Pinned structured-findings parse (Phase 16/17 structured fields)."""
    from medfm.evaluation.advanced import _finding_set, _parse_structured

    payload = {
        "findings": [
            {
                "finding": "left lower lobe opacity",
                "negation": "present",
                "laterality": "left",
                "anatomy": "lung",
            },
            {"finding": "pleural effusion", "negation": "absent"},
        ]
    }
    parsed, valid = _parse_structured(json.dumps(payload))
    findings = sorted(_finding_set(payload))
    return {
        "parse_valid": valid,
        "parsed_keys": sorted(parsed) if isinstance(parsed, dict) else None,
        "findings": [tuple(str(part) for part in item) for item in findings],
    }


def memory_envelope() -> dict[str, object]:
    """Pinned parameter memory envelope for a tiny 2D model."""
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 6 * 6, 4),
    )
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    return {
        "parameter_bytes": parameter_bytes,
        "envelope_bytes": parameter_bytes * 2,
    }


FIXTURES: dict[str, callable] = {
    "tiny_2d_logits.json": tiny_2d_logits,
    "preprocess_stats.json": preprocess_stats,
    "native3d_identity.json": native3d_identity,
    "metrics_dice_iou.json": metrics_dice_iou,
    "structured_findings.json": structured_findings,
    "memory_envelope.json": memory_envelope,
}


def build_goldens() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, producer in FIXTURES.items():
        path = GOLDEN_DIR / name
        path.write_text(json.dumps(producer(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": {name: _sha256(GOLDEN_DIR / name) for name in FIXTURES},
        "note": "pinned by scripts/generate_golden.py; do not hand-edit without regenerating",
    }
    (GOLDEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    build_goldens()
    print(f"wrote {len(FIXTURES)} golden fixtures + manifest to {GOLDEN_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
