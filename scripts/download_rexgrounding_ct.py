#!/usr/bin/env python
"""Download only CT-RATE fixed volumes referenced by ReXGroundingCT."""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_DATA_DIR = Path("../RexGroundingData")
DEFAULT_REPO = "ibrahimhamamci/CT-RATE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    secret_path = Path.home() / ".secrets.txt"
    if secret_path.is_file():
        for line in secret_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN=") and line.partition("=")[2].strip():
                return line.partition("=")[2].strip()
    raise RuntimeError("HF_TOKEN or ~/.secrets.txt with HF_TOKEN=... is required")


def ct_rate_path(name: str) -> str:
    stem = name.removesuffix(".nii.gz")
    pieces = stem.split("_")
    if len(pieces) < 3:
        raise ValueError(f"cannot infer CT-RATE path from {name!r}")
    split = "valid_fixed" if pieces[0] == "valid" else "train_fixed"
    study = "_".join(pieces[:2])
    series = "_".join(pieces[:-1])
    return f"dataset/{split}/{study}/{series}/{name}"


def load_names(data_dir: Path, splits: tuple[str, ...], max_cases: int) -> list[str]:
    payload = json.loads((data_dir / "MICCAI_challenge_dataset.json").read_text(encoding="utf-8"))
    names = sorted({str(entry["name"]) for split in splits for entry in payload[split]})
    if max_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    return names if max_cases == 0 else names[:max_cases]


def download_one(repo: str, name: str, output_dir: Path, token: str) -> tuple[str, int]:
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=ct_rate_path(name),
        token=token,
        local_dir=output_dir,
    )
    return name, Path(path).stat().st_size


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "ct_rate_fixed").resolve()
    names = load_names(data_dir, tuple(args.splits), args.max_cases)
    paths = [ct_rate_path(name) for name in names]
    print(json.dumps({"repo": args.repo, "cases": len(names), "output_dir": str(output_dir), "dry_run": args.dry_run}))
    if args.dry_run:
        print(json.dumps({"first": paths[:3], "last": paths[-3:]}))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    token = read_token()
    lock = threading.Lock()
    completed = 0
    failures: list[dict[str, str]] = []
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, args.repo, name, output_dir, token): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                downloaded_name, size = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve every failed source path
                failures.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps({"status": "error", "name": name, "error": str(exc)}))
                continue
            with lock:
                completed += 1
                sizes[downloaded_name] = size
                if completed == 1 or completed % 25 == 0 or completed == len(names):
                    print(
                        json.dumps(
                            {
                                "status": "downloaded",
                                "completed": completed,
                                "total": len(names),
                                "bytes": size,
                            }
                        )
                    )
    manifest = {
        "repo": args.repo,
        "splits": list(args.splits),
        "cases_requested": len(names),
        "cases_downloaded": completed,
        "cases_failed": len(failures),
        "bytes_downloaded": sum(sizes.values()),
        "files": sorted(sizes),
        "failures": sorted(failures, key=lambda item: item["name"]),
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} CT-RATE volumes failed; see {manifest_path}")
    print(json.dumps({key: manifest[key] for key in ("cases_requested", "cases_downloaded", "bytes_downloaded")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
