#!/usr/bin/env python
"""Evaluate ReXGroundingCT predictions with the released ReXrank scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../RexGroundingData"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--cc-connectivity", type=int, choices=(1, 2, 3), default=2)
    return parser.parse_args()


def load_official_scorer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("rexrank_eval_fast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    pred_dir = args.pred_dir.resolve()
    metadata_path = data_dir / "MICCAI_challenge_dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata.get(args.split)
    if not isinstance(entries, list):
        raise ValueError(f"{metadata_path} has no list for split {args.split!r}")
    scorer = load_official_scorer(data_dir / "rexrank_eval_fast.py")
    cases: list[dict[str, Any]] = []
    gt_dir = data_dir / "segmentations"
    for entry in sorted(entries, key=lambda value: str(value.get("name", ""))):
        name = str(entry.get("name", ""))
        if not name:
            raise ValueError("metadata entry has no name")
        gt_path = gt_dir / name
        pred_path = pred_dir / name
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing released ground truth mask: {gt_path}")
        if not pred_path.is_file():
            cases.append({"file": name, "error": "missing_prediction"})
            continue
        findings = scorer.evaluate_volume(
            str(gt_path), str(pred_path), args.global_only, args.min_size, args.cc_connectivity
        )
        cases.append(
            {
                "file": name,
                "findings": findings,
                "case_stats": scorer.compute_case_stats(findings, args.global_only),
            }
        )
    summary = scorer.compute_summary_overall(cases, args.global_only, args.min_size, args.cc_connectivity)
    output = {
        "summary": summary,
        "cases": cases,
        "scorer": str((data_dir / "rexrank_eval_fast.py").resolve()),
        "scorer_sha256": __import__("hashlib").sha256((data_dir / "rexrank_eval_fast.py").read_bytes()).hexdigest(),
        "split": args.split,
        "prediction_dir": str(pred_dir),
        "global_only": args.global_only,
        "min_size": args.min_size,
        "cc_connectivity": args.cc_connectivity,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
