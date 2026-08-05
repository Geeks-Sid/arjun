"""On-disk manifest IO: Parquet (canonical) and JSONL (debugging interchange).

Formats:

- ``.parquet`` is the canonical on-disk format. The schema version is stored
  as Parquet file metadata under the key ``medfm.manifest_schema_version``
  (via pyarrow schema metadata), so it travels with the file rather than with
  a column. A v1 file WITHOUT this metadata key is assumed to be version 1
  (early writers did not stamp it); newer-than-supported versions are
  rejected and older ones go through :func:`migrate_manifest_frame`.
- ``.jsonl`` is a DEBUGGING INTERCHANGE ONLY (line-delimited, human
  diffable). :func:`write_manifest` emits a :class:`UserWarning` on every
  JSONL write. The first line is a header record
  ``{"manifest_schema_version": <int>}`` carrying the version (a record is
  treated as the header when it has ``manifest_schema_version`` and no
  ``sample_id``); data rows follow, one JSON object per line.

List-valued columns (``secondary_image_uris``, ``shape``, ``spacing_mm``,
``shape_bucket_shape``) are stored natively in Parquet (list types) and as
JSON arrays in JSONL. On read they are normalized to plain Python lists via
:func:`coerce_list_cell`, so downstream code sees the same cell types
regardless of format (Parquet otherwise yields numpy arrays per cell).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from medfm.core.serialization import canonical_json
from medfm.data.errors import ManifestError
from medfm.data.manifests.schema import (
    LIST_VALUED_COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_COLUMNS,
    coerce_list_cell,
    migrate_manifest_frame,
    validate_manifest,
)

#: Parquet file-metadata key carrying the manifest schema version.
SCHEMA_VERSION_METADATA_KEY = b"medfm.manifest_schema_version"

#: JSONL header-record key carrying the manifest schema version.
JSONL_VERSION_KEY = "manifest_schema_version"

_PARQUET_SUFFIX = ".parquet"
_JSONL_SUFFIX = ".jsonl"


def _clean_value(value: Any) -> Any:
    """Convert a cell to a canonical JSON-safe Python value.

    Nulls (None/NaN/NaT) become ``None``; numpy scalars become Python scalars;
    arrays/lists/tuples become lists recursively. Used for JSONL writes and
    content hashing so both formats hash identically.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [_clean_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None  # NaN / inf have no canonical JSON form; NaN is the null marker here
        return float(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _clean_value(v) for k, v in record.items()}


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize list-valued columns to Python lists (post-read)."""
    df = df.copy()
    for column in LIST_VALUED_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(coerce_list_cell)
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[SCHEMA_VERSION_METADATA_KEY] = str(MANIFEST_SCHEMA_VERSION).encode("ascii")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    warnings.warn(
        "JSONL manifests are a DEBUGGING INTERCHANGE ONLY; Parquet is the canonical on-disk format "
        "(implementation_plan/phase_03). Do not train from .jsonl manifests.",
        UserWarning,
        stacklevel=3,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({JSONL_VERSION_KEY: MANIFEST_SCHEMA_VERSION}, sort_keys=True) + "\n")
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(_clean_record(record), sort_keys=True) + "\n")


def write_manifest(df: pd.DataFrame, path: str | Path, *, base_dir: Path | None = None) -> None:
    """Validate ``df`` (fail closed) and write it as ``.parquet`` or ``.jsonl``.

    The schema version is stamped into the file (Parquet metadata key
    ``medfm.manifest_schema_version``; JSONL header record). ``base_dir``
    anchors URI path-traversal validation; see :func:`validate_manifest`.
    """
    path = Path(path)
    validate_manifest(df, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == _PARQUET_SUFFIX:
        _write_parquet(df, path)
    elif path.suffix == _JSONL_SUFFIX:
        _write_jsonl(df, path)
    else:
        raise ManifestError(
            f"unsupported manifest suffix {path.suffix!r}; use '.parquet' (canonical) or '.jsonl' "
            "(debugging interchange only)"
        )


def _read_parquet(path: Path) -> tuple[pd.DataFrame, int]:
    schema = pq.read_schema(path)
    raw_version = (schema.metadata or {}).get(SCHEMA_VERSION_METADATA_KEY)
    # Missing metadata on a v1-era file means version 1 (see module docstring).
    version = int(raw_version.decode("ascii")) if raw_version is not None else 1
    return pq.read_table(path).to_pandas(), version


def _read_jsonl(path: Path) -> tuple[pd.DataFrame, int]:
    version = 1
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ManifestError(f"{path.name} line {line_number + 1} is not a JSON object; corrupt JSONL manifest")
        if line_number == 0 and JSONL_VERSION_KEY in record and "sample_id" not in record:
            version = int(record[JSONL_VERSION_KEY])
            continue
        records.append(record)
    return pd.DataFrame(records), version


def read_manifest_with_version(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Read a manifest and return ``(frame, declared_schema_version)``.

    The returned version is the one stamped in the file (1 if absent); the
    frame itself is already migrated to :data:`MANIFEST_SCHEMA_VERSION` and
    its list-valued columns are normalized to Python lists. Raises
    :class:`ManifestVersionError` for newer-than-supported or unmigratable
    files. Does NOT validate contents — use :func:`read_manifest` for that.
    """
    path = Path(path)
    if path.suffix == _PARQUET_SUFFIX:
        df, version = _read_parquet(path)
    elif path.suffix == _JSONL_SUFFIX:
        df, version = _read_jsonl(path)
    else:
        raise ManifestError(f"unsupported manifest suffix {path.suffix!r}; expected '.parquet' or '.jsonl'")
    df = migrate_manifest_frame(df, version)
    return _normalize_frame(df), version


def read_manifest(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Read a manifest, migrate it if older, and validate it (default).

    Validation runs without a ``base_dir``: relative URIs containing ``..``
    are rejected fail-closed. Pass ``validate=False`` only for tooling that
    inspects broken manifests (and never feed the result to training).
    """
    df, _ = read_manifest_with_version(path)
    if validate:
        validate_manifest(df)
    return df


def inspect_manifest(path: str | Path) -> dict[str, Any]:
    """Machine-readable manifest summary for the ``medfm data`` inspect path.

    Reads WITHOUT validation (an inspection tool must work on broken
    manifests) but still enforces the version/migration rules. The output is
    deterministic (sorted keys and value sets) and safe to hash with
    ``medfm.core.serialization.config_hash``. It intentionally excludes the
    file path so identical manifests inspect identically.
    """
    path = Path(path)
    df, version = read_manifest_with_version(path)
    columns = {str(c) for c in df.columns}

    def _value_counts(column: str) -> dict[str, int]:
        if column not in columns:
            return {}
        counts = df[column].value_counts(dropna=True)
        return {str(key): int(counts[key]) for key in sorted(counts.index, key=str)}

    datasets: list[dict[str, str]] = []
    if {"dataset_name", "dataset_version"} <= columns:
        pairs = {
            (str(name), str(ver))
            for name, ver in zip(df["dataset_name"], df["dataset_version"], strict=True)
            if pd.notna(name) and pd.notna(ver)
        }
        datasets = [{"dataset_name": name, "dataset_version": ver} for name, ver in sorted(pairs)]

    licenses: list[str] = []
    if "license" in columns:
        licenses = sorted({str(value) for value in df["license"] if pd.notna(value)})

    return {
        "schema_version": version,
        "format": path.suffix.lstrip("."),
        "row_count": int(len(df)),
        "columns_present": sorted(columns),
        "columns_missing": [name for name in REQUIRED_COLUMNS if name not in columns],
        "modality_counts": _value_counts("modality"),
        "split_counts": _value_counts("split"),
        "datasets": datasets,
        "licenses": licenses,
    }


def manifest_content_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 over the canonical serialization of ``df``.

    Rows are sorted by ``sample_id``; each row is canonicalized with
    :func:`medfm.core.serialization.canonical_json` after null/numpy/list
    normalization, so a validated frame hashes identically whether it was
    read from Parquet or JSONL. Expects a validated (or at least
    schema-conformant) frame; used for provenance and run metadata
    (``docs/reproducibility_policy.md`` records the dataset-manifest hash per
    run).
    """
    records = [_clean_record(record) for record in _normalize_frame(df).to_dict(orient="records")]
    records.sort(key=lambda record: str(record.get("sample_id", "")))
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()
