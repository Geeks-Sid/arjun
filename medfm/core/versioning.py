"""Schema versioning and payload migration hooks.

Every serializable contract object carries an explicit ``schema_version``
field. Payloads written at older versions are upgraded through a chain of
explicitly registered migrations — one step per version — so support for old
payloads is a deliberate, auditable decision rather than silent leniency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from medfm.core.errors import SchemaVersionError

#: Current contract schema version. Bump on any breaking schema change and
#: register a migration from the previous version.
SCHEMA_VERSION = 1

#: (type name, from version) -> function upgrading a payload dict to the next
#: version. Chains are applied one version at a time.
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]
_SCHEMA_MIGRATIONS: dict[tuple[str, int], MigrationFn] = {}


def register_schema_migration(type_name: str, from_version: int, migrate: MigrationFn) -> None:
    """Register a one-step migration for ``type_name`` payloads."""
    key = (type_name, from_version)
    if key in _SCHEMA_MIGRATIONS:
        raise SchemaVersionError(f"duplicate schema migration registered for {key}")
    _SCHEMA_MIGRATIONS[key] = migrate


def migrate_payload(type_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade ``data`` to :data:`SCHEMA_VERSION` or raise.

    Payloads newer than this code understands are rejected: downgrading is
    never attempted silently.
    """
    version = int(data.get("schema_version", SCHEMA_VERSION))
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{type_name} payload has schema_version {version}, but this code supports at most "
            f"{SCHEMA_VERSION}; upgrade medfm instead of downgrading the payload"
        )
    while version < SCHEMA_VERSION:
        migrate = _SCHEMA_MIGRATIONS.get((type_name, version))
        if migrate is None:
            raise SchemaVersionError(
                f"{type_name} payload has schema_version {version} and no migration to {version + 1} "
                "is registered; refusing to guess"
            )
        data = migrate(data)
        new_version = int(data.get("schema_version", version + 1))
        if new_version <= version:
            raise SchemaVersionError(
                f"{type_name} migration from v{version} did not advance schema_version (got {new_version})"
            )
        version = new_version
    return data
