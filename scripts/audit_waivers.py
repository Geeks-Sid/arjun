#!/usr/bin/env python3
"""Fail on expired release waivers (Phase 18).

`docs/release/waivers.md` lists time-bound security/hardening waivers. Each
waiver row is ``| <id> | <expiry YYYY-MM-DD> | <reason> | <owner> |``. Any row
whose expiry is in the past fails the release gate until the waiver is renewed
or resolved.

Usage:
    uv run --frozen python scripts/audit_waivers.py docs/release/waivers.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

_LINE = re.compile(r"^\|\s*([A-Za-z0-9_-]+)\s*\|\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\|")


def audit(path: Path, today: dt.date | None = None) -> list[str]:
    today = today or dt.date.today()
    offenders: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _LINE.match(line.strip())
        if match is None:
            continue
        waiver_id, expiry = match.group(1), dt.date.fromisoformat(match.group(2))
        if expiry < today:
            offenders.append(f"{waiver_id} (line {number}) expired {expiry.isoformat()}; renew or resolve")
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fail on expired release waivers")
    parser.add_argument("waivers", help="path to docs/release/waivers.md")
    args = parser.parse_args(argv)
    offenders = audit(Path(args.waivers))
    if offenders:
        print(f"waiver audit FAILED ({len(offenders)} expired waiver(s)):")
        for message in offenders:
            print(f"  - {message}")
        return 1
    print("waiver audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
