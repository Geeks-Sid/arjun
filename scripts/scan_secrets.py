#!/usr/bin/env python3
"""Repository secret / forbidden-data scanner (Phase 18 security gate).

Scans git-tracked text files for high-signal secret patterns (cloud keys,
private key material, JWT-shaped tokens, credential assignments) and refuses
committed credentials / forbidden patient-data shapes. Exits 1 on any match.

Usage:
    python scripts/scan_secrets.py [--root DIR]

Skips: vendored/external repos, virtualenvs, git metadata, bytecode caches,
model/dataset caches, and everything git does not track.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# High-signal patterns only: low signal-to-noise, easy to hit false positives.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("gcp_service_account", re.compile(r"\b[0-9]{12}-[a-z0-9]{30,40}@[a-z0-9-]+\.iam\.gserviceaccount\.com\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("private_key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    (
        "generic_secret",
        re.compile(
            r"\b(?:secret|api[_-]?key|password|passwd|token|auth[_-]?key)\s*[:=]\s*"
            r"['\"][^'\"]{10,}['\"]",
            re.IGNORECASE,
        ),
    ),
]

# Patterns that would specifically indicate patient data in a general-purpose
# source tree. Bare digit runs are NOT flagged: they appear legitimately in
# hashes/versions (bare runs are screened at the data layer by
# medfm/data/manifests/schema.py and medfm/data/textprep/phi.py).
FORBIDDEN_DATA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("labeled_mrn", re.compile(r"\bMRN[ :#-]*[0-9]{4,}\b", re.IGNORECASE)),
    ("dicom_uid", re.compile(r"\b[1-9][0-9]{0,30}(?:\.[0-9]{1,30}){4,}\b")),
]

SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
    ".dcm",
    ".svs",
    ".ndpi",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".parquet",
    ".pyc",
    ".so",
    ".o",
    ".a",
    ".whl",
    ".tar",
    ".gz",
    ".zip",
)

# Inline markers that indicate an intentional placeholder/fake value.
_HARMLESS_MARKERS = ("example", "placeholder", "changeme", "xxxxx", "fake", "dummy", "test")


def _harmless(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in _HARMLESS_MARKERS)


def tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - git required in CI
        raise RuntimeError(f"cannot enumerate tracked files under {root}: {exc}") from exc
    return [root / Path(name) for name in result.stdout.split("\0") if name]


def _test_surface(path: Path, root: Path) -> bool:
    """Test/synthetic surface: patient-data-shaped fixtures intentionally live
    here to test the runtime screening (secret patterns are never exempt)."""
    parts: list[str] = list(path.relative_to(root).parts)
    return (
        any(part.startswith("test_") for part in parts)
        or path.name in {"synthetic.py", "conftest.py"}
        or "fixtures" in parts
    )


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in tracked_files(root):
        if path.suffix in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        test_surface = _test_surface(path, root)
        for number, line in enumerate(text.splitlines(), start=1):
            if _harmless(line):
                continue
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path.relative_to(root)}:{number}: {label}")
            if test_surface:
                continue
            for label, pattern in FORBIDDEN_DATA_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path.relative_to(root)}:{number}: {label}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scan the repository for committed secrets/forbidden data")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan(root)
    if findings:
        print(f"secret scan FAILED ({len(findings)} finding(s)):")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("secret scan OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
