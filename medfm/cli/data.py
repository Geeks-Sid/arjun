"""``medfm data`` CLI entry point (runnable as ``python -m medfm.cli.data``).

Thin wrapper over :mod:`medfm.tools.data_tools`; kept executable directly so
the Phase 03 smoke command works without installing the console script::

    python -m medfm.cli.data fingerprint --manifest <path>
"""

from __future__ import annotations

import sys

from medfm.tools.data_tools import main as _tools_main


def main(argv: list[str] | None = None) -> int:
    return _tools_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
