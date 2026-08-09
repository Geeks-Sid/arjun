#!/usr/bin/env python3
"""Fail on JUnit-skipped tests that lack an explanation (Phase 18).

Consumes ``pytest --junitxml`` output: every ``<skipped>`` element without a
``message`` (or inline text) means the suite skipped a test silently, which is
exactly the "unexplained skipped tests" regression the Phase 18 gate forbids.

Usage:
    uv run --frozen pytest tests/ -q --junitxml=artifacts/junit.xml
    uv run --frozen python scripts/audit_skips.py artifacts/junit.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def audit(junit: Path) -> list[str]:
    tree = ET.parse(junit)
    offenders: list[str] = []
    for case in tree.iter("testcase"):
        for skipped in case.iter("skipped"):
            message = skipped.get("message") or (skipped.text or "").strip()
            if not message:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                offenders.append(name)
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fail on JUnit skips without a reason")
    parser.add_argument("junit", help="path to pytest --junitxml output")
    args = parser.parse_args(argv)
    offenders = audit(Path(args.junit))
    if offenders:
        print(f"skip audit FAILED ({len(offenders)} unexplained skip(s)):")
        for name in offenders:
            print(f"  - {name}")
        return 1
    print("skip audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
