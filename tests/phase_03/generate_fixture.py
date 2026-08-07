"""Generate the committed fingerprint fixture manifest (deterministic).

Usage::

    python tests/phase_03/generate_fixture.py

Writes ``tests/fixtures/manifests/mixed_synthetic.parquet`` — the Phase 03
smoke target (``python -m medfm.cli.data fingerprint --manifest ...``). The
payload URIs are synthetic placeholders; payloads themselves are never
committed (.gitignore medical-imaging patterns).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from medfm.data.manifests.io import write_manifest  # noqa: E402
from phase_03.synthetic import build_mixed_manifest  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "manifests" / "mixed_synthetic.parquet"


def main() -> int:
    df = build_mixed_manifest()
    write_manifest(df, FIXTURE_PATH, base_dir=FIXTURE_PATH.parent)
    print(f"wrote {FIXTURE_PATH.relative_to(REPO_ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
