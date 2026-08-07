"""Gated-access acceptance records, stored outside source control.

License acceptance for gated repositories is a named-individual act performed
on the provider's site. This module only records *that* acceptance happened —
it never accepts terms automatically, and a successful download never implies
acceptance. Records live outside the repository (default
``~/.cache/medfm/gated_access.json``, override with
``MEDFM_GATED_ACCESS_FILE``) so source-controlled model records stay free of
per-user state.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

ENV_VAR = "MEDFM_GATED_ACCESS_FILE"


def default_store_path() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "medfm" / "gated_access.json"


def _load(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"corrupt gated-access store at {path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"corrupt gated-access store at {path}: not a JSON object")
    return data


def has_accepted(model_id: str, repository: str, store_path: Path | None = None) -> bool:
    """True iff an acceptance record exists for this exact model+repository."""
    path = store_path or default_store_path()
    record = _load(path).get(model_id)
    return bool(record and record.get("repository") == repository)


def record_acceptance(
    model_id: str,
    repository: str,
    accepted_by: str,
    store_path: Path | None = None,
) -> Path:
    """Record that ``accepted_by`` accepted the provider terms for ``model_id``.

    This is an explicit user action; nothing calls it as a side effect of
    downloading or loading weights.
    """
    if not accepted_by:
        raise ValueError("accepted_by must name the individual who accepted the terms")
    path = store_path or default_store_path()
    data = _load(path)
    data[model_id] = {
        "repository": repository,
        "accepted_by": accepted_by,
        "accepted_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)  # atomic on POSIX
    return path
