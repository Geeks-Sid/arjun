"""Canonical dataset-manifest schema (Phase 03).

A manifest is a tabular dataset index: one row per sample, carrying
references (URIs) and metadata only — never payloads and never restricted
report text (``docs/data_governance.md`` section 3: manifests store
``report_uri`` references ONLY). Validation fails closed: every problem is
collected into a single actionable :class:`ManifestError` (row numbers and
``sample_id`` values included, capped at ~20 problems).

Identifier hygiene mirrors ``medfm.core.sample``: ``*_id_hash`` columns accept
only lowercase hex digests of 32-128 chars; raw MRNs (all-digit strings of
length >= 4) and DICOM-UID dotted-numeric patterns are rejected.

``shape_bucket_kind`` / ``shape_bucket_shape`` are fingerprint-derived hints
written by ``medfm data fingerprint`` to help preprocessing pick static-shape
buckets. They are NOT authoritative raw metadata: readers and preprocessing
must re-derive geometry from the payload, and a missing/stale bucket hint is
never an error.

Versioning note: manifests version independently of the core contract schema
(``medfm.core.versioning``) — a dataset artifact's lifecycle (regeneration,
migration of whole Parquet files) differs from per-sample payload migration,
so this module deliberately keeps its own ``MANIFEST_SCHEMA_VERSION`` and
migration registry rather than importing the core one. The philosophy is the
same: upgrades happen only through explicitly registered one-step migrations;
silent leniency is refused.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from medfm.core.enums import Modality, SplitName
from medfm.core.errors import UnknownEnumValueError
from medfm.data.errors import ManifestError, ManifestSecurityError, ManifestVersionError

#: Current manifest schema version. Bump on any breaking column change and
#: register a one-step migration in :data:`MANIFEST_MIGRATIONS`.
MANIFEST_SCHEMA_VERSION = 1

#: Columns every manifest must have (from idea.md, Phase 3). Nullability is
#: per-column; see :data:`COLUMN_SPECS`.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "patient_id_hash",
    "study_id_hash",
    "series_id_hash",
    "modality",
    "image_uri",
    "secondary_image_uris",
    "mask_uri",
    "annotation_uri",
    "report_uri",
    "label_json",
    "split",
    "site_id",
    "scanner_vendor",
    "acquisition_date_bucket",
    "dataset_name",
    "dataset_version",
    "license",
    "provenance_uri",
)

#: Columns that may be absent; validated when present.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "image_sha256",
    "group_id_hash",
    "shape",
    "spacing_mm",
    "num_slices",
    "num_tiles",
    "microns_per_pixel",
    "magnification",
    "report_chars",
    "intensity_stats_json",
    "seg_class_volumes_json",
    "shape_bucket_kind",
    "shape_bucket_shape",
)

#: Columns whose cells hold a list of values (natively in Parquet, as JSON
#: arrays — or JSON-array strings — in JSONL). Reads normalize them to Python
#: lists via :func:`coerce_list_cell`.
LIST_VALUED_COLUMNS: tuple[str, ...] = ("secondary_image_uris", "shape", "spacing_mm", "shape_bucket_shape")

#: URI schemes a manifest may reference without extra policy review.
DEFAULT_URI_SCHEMES: frozenset[str] = frozenset({"file", "s3", "gs", "https"})

#: URI columns validated cell-by-cell with :func:`validate_uri`.
_URI_COLUMNS: tuple[str, ...] = ("image_uri", "mask_uri", "annotation_uri", "report_uri", "provenance_uri")

#: Columns that must be non-empty strings on every row (governance audit rule,
#: ``docs/data_governance.md`` section 6).
_AUDIT_STRING_COLUMNS: tuple[str, ...] = ("dataset_name", "dataset_version", "license")

#: Nullable free-form string columns (non-empty when present).
_NULLABLE_STRING_COLUMNS: tuple[str, ...] = ("site_id", "scanner_vendor", "acquisition_date_bucket")

#: JSON-string columns (nullable; must parse as JSON when present).
_JSON_COLUMNS: tuple[str, ...] = ("label_json", "intensity_stats_json", "seg_class_volumes_json")

#: Maximum number of individual problems reported in one ManifestError.
_MAX_PROBLEMS = 20

_HASH_RE = re.compile(r"[0-9a-f]{32,128}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
#: Patterns that indicate a raw (unhashed) clinical identifier.
_DICOM_UID_RE = re.compile(r"[0-9]+(\.[0-9]+)+")
_MRN_RE = re.compile(r"[0-9]{4,}")
_SCHEME_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class ColumnSpec:
    """Machine-readable description of one manifest column."""

    name: str
    kind: str  # string | hash | sha256 | uri | uri_list | enum:* | json | int | float | int_list | float_list
    required_column: bool  # column must exist in the manifest
    nullable: bool  # cells may be null
    description: str


def _spec(name: str, kind: str, nullable: bool, description: str, *, required: bool = True) -> ColumnSpec:
    return ColumnSpec(name=name, kind=kind, required_column=required, nullable=nullable, description=description)


#: Canonical column table, in declared order (required columns first, then
#: optional columns). The single source of truth for validation and
#: :func:`manifest_schema_dict`.
COLUMN_SPECS: tuple[ColumnSpec, ...] = (
    _spec("sample_id", "string", False, "Unique non-empty sample identifier (de-identified)."),
    _spec("patient_id_hash", "hash", False, "Salted hash of the patient identifier; split-membership key (ADR 0004)."),
    _spec("study_id_hash", "hash", True, "Salted hash of the study identifier."),
    _spec("series_id_hash", "hash", True, "Salted hash of the series identifier."),
    _spec("modality", "enum:Modality", False, "Canonical modality value (medfm.core.enums.Modality)."),
    _spec("image_uri", "uri", True, "Primary image/series payload reference; required unless modality is TEXT_ONLY."),
    _spec("secondary_image_uris", "uri_list", True, "Additional image references (list of URIs), e.g. lateral views."),
    _spec("mask_uri", "uri", True, "Segmentation mask payload reference."),
    _spec("annotation_uri", "uri", True, "Structured annotation payload reference."),
    _spec("report_uri", "uri", True, "Reference to restricted report text in an approved store; never the text."),
    _spec("label_json", "json", True, "JSON-encoded label payload (task + values); null for unlabeled samples."),
    _spec("split", "enum:SplitName", True, "Canonical split assignment (medfm.core.enums.SplitName)."),
    _spec("site_id", "string", True, "De-identified acquisition-site identifier."),
    _spec("scanner_vendor", "string", True, "Scanner vendor label."),
    _spec("acquisition_date_bucket", "string", True, "Coarse acquisition bucket (e.g. '2019-Q3'); never exact dates."),
    _spec("dataset_name", "string", False, "Dataset identifier; required on every row (governance audit)."),
    _spec("dataset_version", "string", False, "Dataset version; required on every row (governance audit)."),
    _spec("license", "string", False, "Dataset license identifier; required on every row (governance audit)."),
    _spec("provenance_uri", "uri", True, "Reference to the row's provenance record."),
    _spec(
        "image_sha256",
        "sha256",
        True,
        "64-hex content hash of the image payload, when known; used for duplicate detection across splits.",
        required=False,
    ),
    _spec(
        "group_id_hash",
        "hash",
        True,
        "Explicit split-grouping override (e.g. slide/case hash); must keep group members in one split.",
        required=False,
    ),
    _spec("shape", "int_list", True, "Payload shape (positive ints) when known.", required=False),
    _spec("spacing_mm", "float_list", True, "Physical spacing per spatial axis (positive floats).", required=False),
    _spec("num_slices", "int", True, "Slice count for volumetric samples.", required=False),
    _spec("num_tiles", "int", True, "Tile count for WSI samples.", required=False),
    _spec("microns_per_pixel", "float", True, "Level-0 microns per pixel (pathology).", required=False),
    _spec("magnification", "float", True, "Objective magnification (pathology).", required=False),
    _spec(
        "report_chars",
        "int",
        True,
        "Character count of the restricted report; the ONLY report statistic allowed in a manifest.",
        required=False,
    ),
    _spec(
        "intensity_stats_json", "json", True, "JSON-encoded intensity statistics (fingerprint-derived).", required=False
    ),
    _spec("seg_class_volumes_json", "json", True, "JSON-encoded per-class segmentation volumes.", required=False),
    _spec(
        "shape_bucket_kind",
        "string",
        True,
        "Fingerprint-derived bucket kind hint (e.g. '2d_resolution', '3d_patch'); NOT authoritative metadata.",
        required=False,
    ),
    _spec(
        "shape_bucket_shape",
        "int_list",
        True,
        "Fingerprint-derived bucket shape hint; NOT authoritative metadata — re-derive geometry from the payload.",
        required=False,
    ),
)

_SPEC_BY_NAME: dict[str, ColumnSpec] = {spec.name: spec for spec in COLUMN_SPECS}

#: One-step manifest migrations: ``from_version -> function upgrading a frame
#: to from_version + 1``. Empty at v1; upgrades are deliberate and append-only.
MANIFEST_MIGRATIONS: dict[int, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def _is_null(value: Any) -> bool:
    """Scalar-aware null check (``pd.isna`` is elementwise on lists/arrays)."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def coerce_list_cell(value: Any) -> list[Any] | None:
    """Normalize a list-valued manifest cell to a Python list (``None`` if null).

    Accepts native lists/tuples, numpy arrays (Parquet reads), and JSON-array
    strings (hand-built frames / JSONL cells). Raises :class:`ManifestError`
    for anything else.
    """
    if _is_null(value):
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"list-valued manifest cell is neither a list nor a JSON array string: {value!r}; "
                "store a native list (Parquet) or a JSON array"
            ) from exc
        if not isinstance(parsed, list):
            raise ManifestError(f"list-valued manifest cell must be a JSON array; got {type(parsed).__name__}")
        return parsed
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ManifestError(
        f"list-valued manifest cell has unsupported type {type(value).__name__}; expected list or JSON array string"
    )


def _hash_problem(value: Any, field_name: str) -> str | None:
    """Return a problem message if ``value`` is not a clean identifier hash.

    Never echoes the offending value: it may be a raw patient identifier, and
    these messages propagate into logs (``medfm/data/errors.py`` privacy rule).
    """
    if not isinstance(value, str):
        return f"{field_name} must be a lowercase hex digest string; got {type(value).__name__}"
    if _DICOM_UID_RE.fullmatch(value):
        return f"{field_name} looks like a raw DICOM UID; store only a salted hash of the identifier"
    if _MRN_RE.fullmatch(value):
        return f"{field_name} looks like a raw numeric identifier (MRN); store only a hash"
    if not _HASH_RE.fullmatch(value):
        return (
            f"{field_name} must be a lowercase hex digest (32-128 chars); got a non-conforming "
            f"{len(value)}-char value (not echoed: possible identifier)"
        )
    return None


def _check_local_path(path_str: str, *, base_dir: Path | None, original: str) -> None:
    """Path-traversal guard for scheme-less and ``file://`` URIs.

    Absolute paths are allowed but are environment-specific: a manifest that
    uses them is not portable across machines (callers should prefer
    ``base_dir``-relative paths or object-store URIs).
    """
    if not path_str:
        raise ManifestSecurityError(f"URI has an empty path component: {original!r}")
    parts = PurePosixPath(path_str).parts
    if ".." not in parts:
        return
    if PurePosixPath(path_str).is_absolute():
        # An absolute path cannot escape a declared root it never claimed to
        # live under; normalization resolves the '..' components in place.
        return
    if base_dir is None:
        raise ManifestSecurityError(
            f"relative URI {original!r} contains '..' but no base_dir was provided to verify it stays "
            "inside the dataset root; pass base_dir or rewrite the path"
        )
    base_norm = posixpath.normpath(str(base_dir))
    resolved = posixpath.normpath(posixpath.join(base_norm, path_str))
    if resolved != base_norm and not resolved.startswith(base_norm + "/"):
        raise ManifestSecurityError(
            f"URI {original!r} resolves outside base_dir {base_norm!r}; manifest paths must stay inside "
            "the dataset root (path traversal is rejected fail-closed)"
        )


def validate_uri(
    uri: str,
    *,
    base_dir: Path | None,
    allowed_schemes: frozenset[str] = DEFAULT_URI_SCHEMES,
) -> None:
    """Validate one manifest URI, failing closed on anything unsafe.

    - Explicit schemes must be in ``allowed_schemes`` (default
      :data:`DEFAULT_URI_SCHEMES`); anything else raises
      :class:`ManifestSecurityError`.
    - Scheme-less and ``file://`` URIs are treated as local paths: control
      characters are rejected, and relative paths containing ``..`` must
      resolve inside ``base_dir`` (required in that case). Absolute paths are
      allowed but flagged as environment-specific (see :func:`_check_local_path`).
    """
    if not isinstance(uri, str) or not uri:
        raise ManifestSecurityError("URI must be a non-empty string; fix the manifest cell or drop the row")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in uri):
        raise ManifestSecurityError(f"URI contains control characters and is unsafe: {uri!r}")
    scheme: str | None = None
    match = _SCHEME_PREFIX_RE.match(uri)
    if match is not None and not _WINDOWS_DRIVE_RE.match(uri):
        scheme = match.group(1).lower()
    if scheme is not None:
        if scheme not in allowed_schemes:
            raise ManifestSecurityError(
                f"URI scheme {scheme!r} is not in the allowed set {sorted(allowed_schemes)}; store the payload in "
                "an approved store (file/s3/gs/https) or record an explicit policy exception"
            )
        if scheme == "file":
            parts = urlsplit(uri)
            if parts.netloc not in ("", "localhost"):
                raise ManifestSecurityError(
                    f"file:// URI with remote host {parts.netloc!r} is not allowed; use a local path or an "
                    "object-store URI"
                )
            _check_local_path(parts.path, base_dir=base_dir, original=uri)
        return
    _check_local_path(uri, base_dir=base_dir, original=uri)


def _row_label(position: int, row: pd.Series, columns: set[str]) -> str:
    if "sample_id" in columns:
        sample_id = row["sample_id"]
        if isinstance(sample_id, str) and sample_id:
            return f"row {position} (sample_id {sample_id!r})"
    return f"row {position}"


def _add_uri_problems(value: Any, column: str, base_dir: Path | None, label: str, problems: list[str]) -> None:
    if not isinstance(value, str) or not value:
        problems.append(f"{label}: {column} must be a non-empty URI string; got {value!r}")
        return
    try:
        validate_uri(value, base_dir=base_dir)
    except ManifestSecurityError as exc:
        problems.append(f"{label}: {column}: {exc}")


def _check_json_cell(value: Any, column: str, label: str, problems: list[str]) -> None:
    if not isinstance(value, str):
        problems.append(f"{label}: {column} must be a JSON string; got {type(value).__name__}")
        return
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        problems.append(f"{label}: {column} is not valid JSON ({exc.msg}); store a JSON-encoded payload")


def _check_int_list_cell(value: Any, column: str, label: str, problems: list[str]) -> None:
    try:
        items = coerce_list_cell(value)
    except ManifestError as exc:
        problems.append(f"{label}: {column}: {exc}")
        return
    if items is None:
        return
    if not items or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or int(v) <= 0 for v in items):
        problems.append(f"{label}: {column} must be a non-empty list of positive ints; got {items!r}")


def _check_float_list_cell(value: Any, column: str, label: str, problems: list[str]) -> None:
    try:
        items = coerce_list_cell(value)
    except ManifestError as exc:
        problems.append(f"{label}: {column}: {exc}")
        return
    if items is None:
        return
    if not items or any(
        not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(v, bool) or float(v) <= 0 for v in items
    ):
        problems.append(f"{label}: {column} must be a non-empty list of positive numbers; got {items!r}")


def _is_positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) > 0
    if isinstance(value, (float, np.floating)):
        # Parquet round-trips nullable int columns as float64; accept integral floats.
        return math.isfinite(float(value)) and float(value).is_integer() and float(value) > 0
    return False


def _is_non_negative_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) >= 0
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value)) and float(value).is_integer() and float(value) >= 0
    return False


def _is_positive_float(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def _check_optional_columns(row: pd.Series, columns: set[str], label: str, problems: list[str]) -> None:
    if "image_sha256" in columns:
        value = row["image_sha256"]
        if not _is_null(value) and (not isinstance(value, str) or not _SHA256_RE.fullmatch(value)):
            problems.append(f"{label}: image_sha256 must be a 64-char lowercase hex digest; got {value!r}")
    if "group_id_hash" in columns:
        value = row["group_id_hash"]
        if not _is_null(value):
            issue = _hash_problem(value, "group_id_hash")
            if issue is not None:
                problems.append(f"{label}: {issue}")
    for column in ("shape", "shape_bucket_shape"):
        if column in columns and not _is_null(row[column]):
            _check_int_list_cell(row[column], column, label, problems)
    if "spacing_mm" in columns and not _is_null(row["spacing_mm"]):
        _check_float_list_cell(row["spacing_mm"], "spacing_mm", label, problems)
    for column in ("num_slices", "num_tiles"):
        if column in columns:
            value = row[column]
            if not _is_null(value) and not _is_positive_int(value):
                problems.append(f"{label}: {column} must be a positive integer; got {value!r}")
    for column in ("microns_per_pixel", "magnification"):
        if column in columns:
            value = row[column]
            if not _is_null(value) and not _is_positive_float(value):
                problems.append(f"{label}: {column} must be a positive number; got {value!r}")
    if "report_chars" in columns:
        value = row["report_chars"]
        if not _is_null(value) and not _is_non_negative_int(value):
            # Never echo the cell: callers may have put free text (potentially
            # PHI) in this column; report only the expected type.
            problems.append(
                f"{label}: report_chars must be a non-negative integer; got {type(value).__name__} "
                "(value not echoed: possible report text)"
            )
    if "shape_bucket_kind" in columns:
        value = row["shape_bucket_kind"]
        if not _is_null(value) and (not isinstance(value, str) or not value):
            problems.append(f"{label}: shape_bucket_kind must be a non-empty string; got {value!r}")


def validate_manifest(df: pd.DataFrame, *, base_dir: Path | None = None) -> None:
    """Validate ``df`` against the canonical manifest schema; raise on any problem.

    All problems are collected into a single :class:`ManifestError` whose
    message carries row numbers / ``sample_id`` values and is capped at
    :data:`_MAX_PROBLEMS` entries. ``base_dir`` anchors path-traversal checks
    for relative URIs (see :func:`validate_uri`).

    Unknown extra columns are tolerated for forward compatibility, EXCEPT any
    column whose name starts with ``report`` other than ``report_uri`` /
    ``report_chars``: embedding report free text in a general-purpose manifest
    violates ``docs/data_governance.md`` section 3 and is always rejected.
    """
    problems: list[str] = []
    columns = {str(c) for c in df.columns}

    for column in df.columns:
        lowered = str(column).lower()
        if lowered.startswith("report") and lowered not in ("report_uri", "report_chars"):
            problems.append(
                f"column {column!r} looks like embedded report free text; manifests store report_uri references "
                "only (docs/data_governance.md section 3) — remove the column and reference the approved store"
            )

    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    for name in missing:
        problems.append(f"missing required column {name!r}; add it (nullable per the schema) and re-write the manifest")

    seen_sample_ids: set[str] = set()
    for position, (_, row) in enumerate(df.iterrows()):
        label = _row_label(position, row, columns)

        if "sample_id" in columns:
            value = row["sample_id"]
            if not isinstance(value, str) or not value:
                problems.append(f"{label}: sample_id must be a non-empty string; got {value!r}")
            elif value in seen_sample_ids:
                problems.append(f"{label}: duplicate sample_id {value!r}; sample_id must be unique")
            else:
                seen_sample_ids.add(value)

        if "patient_id_hash" in columns:
            value = row["patient_id_hash"]
            if _is_null(value):
                problems.append(f"{label}: patient_id_hash is required (splits key on it, ADR 0004); got null")
            else:
                issue = _hash_problem(value, "patient_id_hash")
                if issue is not None:
                    problems.append(f"{label}: {issue}")
        for column in ("study_id_hash", "series_id_hash"):
            if column in columns and not _is_null(row[column]):
                issue = _hash_problem(row[column], column)
                if issue is not None:
                    problems.append(f"{label}: {issue}")

        modality: Modality | None = None
        raw_modality = row["modality"] if "modality" in columns else None
        if _is_null(raw_modality) or not isinstance(raw_modality, str) or not raw_modality:
            problems.append(f"{label}: modality is required and must be a string; got {raw_modality!r}")
        else:
            try:
                modality = Modality.from_value(raw_modality)
            except UnknownEnumValueError as exc:
                problems.append(f"{label}: {exc}")

        if "split" in columns:
            value = row["split"]
            if not _is_null(value):
                if not isinstance(value, str):
                    problems.append(f"{label}: split must be a string; got {type(value).__name__}")
                else:
                    try:
                        SplitName.from_value(value)
                    except UnknownEnumValueError as exc:
                        problems.append(f"{label}: {exc}")

        if "image_uri" in columns:
            value = row["image_uri"]
            if _is_null(value):
                if modality is not Modality.TEXT_ONLY:
                    problems.append(
                        f"{label}: image_uri is required for modality {raw_modality!r}; only TEXT_ONLY rows may "
                        "leave it empty"
                    )
            else:
                _add_uri_problems(value, "image_uri", base_dir, label, problems)

        if "secondary_image_uris" in columns and not _is_null(row["secondary_image_uris"]):
            value = row["secondary_image_uris"]
            try:
                items = coerce_list_cell(value)
            except ManifestError as exc:
                problems.append(f"{label}: secondary_image_uris: {exc}")
                items = None
            if items is not None:
                for item in items:
                    _add_uri_problems(item, "secondary_image_uris", base_dir, label, problems)

        for column in ("mask_uri", "annotation_uri", "report_uri", "provenance_uri"):
            if column in columns and not _is_null(row[column]):
                _add_uri_problems(row[column], column, base_dir, label, problems)

        for column in _AUDIT_STRING_COLUMNS:
            if column in columns:
                value = row[column]
                if _is_null(value) or not isinstance(value, str) or not value:
                    problems.append(
                        f"{label}: {column} is required non-empty on every row (governance audit, "
                        f"docs/data_governance.md section 6); got {value!r}"
                    )

        for column in _NULLABLE_STRING_COLUMNS:
            if column in columns:
                value = row[column]
                if not _is_null(value) and (not isinstance(value, str) or not value):
                    problems.append(f"{label}: {column} must be a non-empty string when present; got {value!r}")

        for column in _JSON_COLUMNS:
            if column in columns and not _is_null(row[column]):
                _check_json_cell(row[column], column, label, problems)

        _check_optional_columns(row, columns, label, problems)

    if problems:
        shown = problems[:_MAX_PROBLEMS]
        message = f"manifest validation failed with {len(problems)} problem(s):\n- " + "\n- ".join(shown)
        if len(problems) > len(shown):
            message += f"\n- ... and {len(problems) - len(shown)} more problem(s); fix the above and re-validate"
        raise ManifestError(message)


def manifest_schema_dict() -> dict[str, Any]:
    """Machine-readable description of the canonical manifest schema.

    Deterministic (declared column order, sorted scheme list); safe to hash
    with ``medfm.core.serialization.config_hash``.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "columns": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "required_column": spec.required_column,
                "nullable": spec.nullable,
                "description": spec.description,
            }
            for spec in COLUMN_SPECS
        ],
        "list_valued_columns": list(LIST_VALUED_COLUMNS),
        "uri_columns": [*_URI_COLUMNS, "secondary_image_uris"],
        "allowed_uri_schemes": sorted(DEFAULT_URI_SCHEMES),
        "notes": [
            "Manifests store report_uri references only; embedded report text is rejected (data governance).",
            "shape_bucket_* columns are fingerprint-derived hints, NOT authoritative raw metadata.",
            "image_uri may be null only for TEXT_ONLY rows; dataset_name/dataset_version/license are always required.",
        ],
    }


def migrate_manifest_frame(df: pd.DataFrame, from_version: int) -> pd.DataFrame:
    """Upgrade ``df`` from ``from_version`` to :data:`MANIFEST_SCHEMA_VERSION`.

    Applies the registered one-step migrations in :data:`MANIFEST_MIGRATIONS`
    in order. Raises :class:`ManifestVersionError` when the frame is newer
    than this code supports or when the migration chain has a gap — upgrading
    is a deliberate, auditable decision, never a guess.
    """
    version = int(from_version)
    if version > MANIFEST_SCHEMA_VERSION:
        raise ManifestVersionError(
            f"manifest schema version {version} is newer than supported {MANIFEST_SCHEMA_VERSION}; upgrade medfm "
            "instead of downgrading the manifest"
        )
    while version < MANIFEST_SCHEMA_VERSION:
        migrate = MANIFEST_MIGRATIONS.get(version)
        if migrate is None:
            raise ManifestVersionError(
                f"manifest schema version {version} has no registered migration to {version + 1}; refusing to guess — "
                "regenerate the manifest or register an explicit migration"
            )
        df = migrate(df)
        version += 1
    return df
