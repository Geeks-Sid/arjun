"""Canonical dataset manifests: schema, validation, and IO (Phase 03)."""

from __future__ import annotations

from medfm.data.manifests.io import (
    JSONL_VERSION_KEY,
    SCHEMA_VERSION_METADATA_KEY,
    inspect_manifest,
    manifest_content_hash,
    read_manifest,
    read_manifest_with_version,
    write_manifest,
)
from medfm.data.manifests.schema import (
    COLUMN_SPECS,
    DEFAULT_URI_SCHEMES,
    LIST_VALUED_COLUMNS,
    MANIFEST_MIGRATIONS,
    MANIFEST_SCHEMA_VERSION,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    ColumnSpec,
    coerce_list_cell,
    manifest_schema_dict,
    migrate_manifest_frame,
    validate_manifest,
    validate_uri,
)

__all__ = [
    "COLUMN_SPECS",
    "DEFAULT_URI_SCHEMES",
    "JSONL_VERSION_KEY",
    "LIST_VALUED_COLUMNS",
    "MANIFEST_MIGRATIONS",
    "MANIFEST_SCHEMA_VERSION",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION_METADATA_KEY",
    "ColumnSpec",
    "coerce_list_cell",
    "inspect_manifest",
    "manifest_content_hash",
    "manifest_schema_dict",
    "migrate_manifest_frame",
    "read_manifest",
    "read_manifest_with_version",
    "validate_manifest",
    "validate_uri",
    "write_manifest",
]
