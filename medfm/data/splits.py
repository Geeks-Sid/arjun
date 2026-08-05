"""Split generation and leakage protection (ADR 0004).

Splits are generated **patient-level first**, then site (external validation),
then time (temporal validation). Split membership is keyed on
``patient_id_hash``; ``group_id_hash`` is an explicit grouping override that
keeps slide tiles, slides of a case, or any caller-declared group inside one
split. Studies/series never cross splits because every grouping policy is a
coarsening of the patient key (a study/series belongs to exactly one patient)
or is declared through ``group_id_hash``.

Determinism: group-to-split assignment is a pure function of ``(seed, policy,
grouping key)`` — SHA-256 of ``"<seed>:<policy>:<key>"`` mapped to ``[0, 1)``
and bucketed by cumulative ratios. Row order never matters; re-running with
the same seed reproduces the same assignment. The :class:`SplitReport`
records seed, policy, grouping key, and per-split counts, and is hashable for
run metadata.

Leakage checks (:func:`check_split_leakage`) verify that no grouping key —
patient, study, series, explicit group, or image content hash — appears in
more than one split. Training on a manifest with known leakage is refused by
:func:`assert_no_split_leakage` unless a :class:`ResearchOverride` is
recorded (governance: ``docs/data_governance.md`` section 6).

Privacy: report/error content carries identifier *hashes* and capped
``sample_id`` lists only — never raw identifiers (``medfm/data/errors.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from medfm.core.enums import SplitName
from medfm.core.serialization import config_hash
from medfm.data.errors import SplitLeakageError

#: Default split ratios: 70% train, 15% val, 15% test.
DEFAULT_SPLIT_RATIOS: tuple[tuple[SplitName, float], ...] = (
    (SplitName.TRAIN, 0.70),
    (SplitName.VAL, 0.15),
    (SplitName.TEST, 0.15),
)

#: Default ratios for temporal holdouts: everything except the last bucket
#: trains; the last ratio fraction of the time range is held out.
DEFAULT_TEMPORAL_RATIOS: tuple[tuple[SplitName, float], ...] = (
    (SplitName.TRAIN, 0.85),
    (SplitName.TEMPORAL_VAL, 0.15),
)

#: Default ratios for site-disjoint splits (external validation).
DEFAULT_SITE_RATIOS: tuple[tuple[SplitName, float], ...] = (
    (SplitName.TRAIN, 0.70),
    (SplitName.VAL, 0.10),
    (SplitName.EXTERNAL_VAL, 0.20),
)

#: Grouping keys checked for leakage, in audit order.
_LEAKAGE_KEY_COLUMNS: tuple[str, ...] = (
    "patient_id_hash",
    "study_id_hash",
    "series_id_hash",
    "group_id_hash",
)

#: Maximum number of violations/sample ids rendered in one error message.
_MAX_VIOLATIONS = 20
_MAX_SAMPLE_IDS = 5


class SplitPolicy(StrEnum):
    """Splitting strategy; patient-disjoint is the default (ADR 0004)."""

    PATIENT = "PATIENT"
    SITE = "SITE"
    TEMPORAL = "TEMPORAL"


def _hash_bucket(key: str, seed: int, policy: SplitPolicy) -> float:
    """Deterministic ``[0, 1)`` bucket for a grouping key (row-order-free)."""
    digest = hashlib.sha256(f"{seed}:{policy.value}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _normalize_ratios(ratios: tuple[tuple[SplitName, float], ...]) -> tuple[tuple[SplitName, float], ...]:
    if not ratios:
        raise SplitLeakageError("split ratios must not be empty")
    names = [name for name, _ in ratios]
    if len(set(names)) != len(names):
        raise SplitLeakageError(f"split ratios contain duplicate split names: {names}")
    if any(weight <= 0 for _, weight in ratios):
        raise SplitLeakageError(f"split ratio weights must all be positive; got {ratios}")
    total = sum(weight for _, weight in ratios)
    return tuple((name, weight / total) for name, weight in ratios)


def _assign_by_bucket(bucket: float, ratios: tuple[tuple[SplitName, float], ...]) -> SplitName:
    cumulative = 0.0
    for name, weight in ratios:
        cumulative += weight
        if bucket < cumulative:
            return name
    return ratios[-1][0]  # float rounding guard: the last ratio owns the tail


def _group_key_column(df: pd.DataFrame, policy: SplitPolicy) -> str:
    """Effective grouping column for a policy.

    ``group_id_hash`` (when present on a row) always wins: it is the explicit
    override keeping slide tiles / slides / cases together. Otherwise the
    policy key applies.
    """
    if policy is SplitPolicy.SITE:
        if "site_id" not in df.columns:
            raise SplitLeakageError("SITE split policy requires a site_id column; the manifest has none")
        return "site_id"
    if policy is SplitPolicy.TEMPORAL:
        if "acquisition_date_bucket" not in df.columns:
            raise SplitLeakageError(
                "TEMPORAL split policy requires an acquisition_date_bucket column; the manifest has none"
            )
        return "acquisition_date_bucket"
    return "patient_id_hash"


def _validate_frame_columns(df: pd.DataFrame) -> None:
    columns = {str(c) for c in df.columns}
    if not columns:
        raise SplitLeakageError("manifest frame is empty; nothing to split")
    for required in ("sample_id", "patient_id_hash"):
        if required not in columns:
            raise SplitLeakageError(f"manifest is missing {required!r}; split generation keys on it")


def generate_split_assignment(
    df: pd.DataFrame,
    *,
    policy: SplitPolicy = SplitPolicy.PATIENT,
    seed: int,
    ratios: tuple[tuple[SplitName, float], ...] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a deterministic ``split`` assignment.

    - ``PATIENT`` (default): groups are ``group_id_hash`` (per row, when set)
      else ``patient_id_hash``; each group hashes into one split. Patient-
      disjoint, and group members (studies, tiles, slides, cases) stay
      together.
    - ``SITE``: whole sites hash into splits (external validation).
    - ``TEMPORAL``: date buckets are sorted chronologically and the tail
      ratio of the *time range* is held out; patients may legitimately
      appear on both sides (drift measurement), which the leakage checker
      treats as expected for this policy.
    """
    _validate_frame_columns(df)
    policy = SplitPolicy(policy)
    normalized = _normalize_ratios(
        ratios
        if ratios is not None
        else {
            SplitPolicy.PATIENT: DEFAULT_SPLIT_RATIOS,
            SplitPolicy.SITE: DEFAULT_SITE_RATIOS,
            SplitPolicy.TEMPORAL: DEFAULT_TEMPORAL_RATIOS,
        }[policy]
    )

    result = df.copy()
    if policy is SplitPolicy.TEMPORAL:
        # Date buckets sort chronologically ("2019-Q3" < "2020-Q1"); each
        # bucket's fractional position in the time range maps through the
        # cumulative ratios, so the tail ratio of the RANGE is held out.
        buckets = sorted({str(b) for b in df["acquisition_date_bucket"] if pd.notna(b)})
        if not buckets:
            raise SplitLeakageError("TEMPORAL split policy found no non-null acquisition_date_bucket values")
        if df["acquisition_date_bucket"].isna().any():
            raise SplitLeakageError("TEMPORAL split policy requires acquisition_date_bucket on every row")
        bucket_split = {
            bucket: _assign_by_bucket(index / len(buckets), normalized) for index, bucket in enumerate(buckets)
        }
        assignment = [bucket_split[str(b)] for b in df["acquisition_date_bucket"]]
    else:
        key_column = _group_key_column(df, policy)
        assignment = []
        for _, row in df.iterrows():
            override = row.get("group_id_hash") if "group_id_hash" in df.columns else None
            key = str(override) if override is not None and pd.notna(override) else str(row[key_column])
            assignment.append(_assign_by_bucket(_hash_bucket(key, seed, policy), normalized))
    result["split"] = [name.value for name in assignment]
    return result


@dataclass(frozen=True)
class SplitReport:
    """Auditable record of a split generation run (seed + grouping keys)."""

    policy: SplitPolicy
    seed: int
    ratios: tuple[tuple[SplitName, float], ...]
    grouping_key: str  # column the policy grouped on (group_id_hash override noted)
    group_count: int
    samples_per_split: dict[str, int]
    groups_per_split: dict[str, int]
    report_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_hash",
            config_hash(
                {
                    "policy": self.policy.value,
                    "seed": self.seed,
                    "ratios": [[name.value, weight] for name, weight in self.ratios],
                    "grouping_key": self.grouping_key,
                    "group_count": self.group_count,
                    "samples_per_split": self.samples_per_split,
                    "groups_per_split": self.groups_per_split,
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "seed": self.seed,
            "ratios": [[name.value, weight] for name, weight in self.ratios],
            "grouping_key": self.grouping_key,
            "group_count": self.group_count,
            "samples_per_split": dict(self.samples_per_split),
            "groups_per_split": dict(self.groups_per_split),
            "report_hash": self.report_hash,
        }


def build_split_report(
    df: pd.DataFrame,
    *,
    policy: SplitPolicy,
    seed: int,
    ratios: tuple[tuple[SplitName, float], ...],
) -> SplitReport:
    """Summarize the split assignment present in ``df`` (post-generation)."""
    _validate_frame_columns(df)
    if "split" not in df.columns:
        raise SplitLeakageError("frame has no split column; run generate_split_assignment first")
    policy = SplitPolicy(policy)
    key_column = _group_key_column(df, policy)

    def _effective_key(row: pd.Series) -> str:
        override = row.get("group_id_hash") if "group_id_hash" in df.columns else None
        if override is not None and pd.notna(override):
            return str(override)
        return str(row[key_column])

    groups: dict[str, set[str]] = {}
    samples_per_split: dict[str, int] = {}
    for _, row in df.iterrows():
        split = str(row["split"]) if pd.notna(row["split"]) else "UNASSIGNED"
        key = _effective_key(row)
        groups.setdefault(key, set()).add(split)
        samples_per_split[split] = samples_per_split.get(split, 0) + 1

    multi_split_groups = {key for key, splits in groups.items() if len(splits) > 1}
    if multi_split_groups and policy is not SplitPolicy.TEMPORAL:
        raise SplitLeakageError(
            f"{len(multi_split_groups)} grouping key(s) span multiple splits; this assignment leaks "
            "(e.g. group keys are not unique per split)"
        )

    groups_per_split: dict[str, int] = {}
    for splits in groups.values():
        for split in splits:
            groups_per_split[split] = groups_per_split.get(split, 0) + 1

    grouping_key = key_column if key_column != "patient_id_hash" else "patient_id_hash (group_id_hash override honored)"
    return SplitReport(
        policy=policy,
        seed=seed,
        ratios=_normalize_ratios(ratios),
        grouping_key=grouping_key,
        group_count=len(groups),
        samples_per_split={split: samples_per_split[split] for split in sorted(samples_per_split)},
        groups_per_split={split: groups_per_split[split] for split in sorted(groups_per_split)},
    )


@dataclass(frozen=True)
class LeakageViolation:
    """One grouping key observed in more than one split."""

    kind: str  # patient_id_hash | study_id_hash | series_id_hash | group_id_hash | image_sha256
    key: str  # the hash value (never a raw identifier)
    splits: tuple[str, ...]
    sample_ids: tuple[str, ...]  # capped, de-identified identifiers
    sample_count: int


@dataclass(frozen=True)
class LeakageReport:
    """Result of scanning a manifest for cross-split leakage."""

    violations: tuple[LeakageViolation, ...]
    rows_checked: int
    temporal_policy: bool  # patient overlap across temporal splits is expected

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rows_checked": self.rows_checked,
            "temporal_policy": self.temporal_policy,
            "violation_count": len(self.violations),
            "violations": [
                {
                    "kind": violation.kind,
                    "key": violation.key,
                    "splits": list(violation.splits),
                    "sample_ids": list(violation.sample_ids),
                    "sample_count": violation.sample_count,
                }
                for violation in self.violations
            ],
        }


def check_split_leakage(df: pd.DataFrame, *, temporal_policy: bool = False) -> LeakageReport:
    """Scan ``df`` for keys that appear in more than one split.

    Checked keys: ``patient_id_hash``, ``study_id_hash``, ``series_id_hash``,
    ``group_id_hash`` (covers studies, derived copies declared by the caller,
    adjacent slices, WSI tiles/slides/cases) and ``image_sha256`` (duplicate
    content crossing splits, e.g. derived/resampled copies). Rows with a null
    ``split`` are ignored (unassigned is not leakage). ``temporal_policy``
    exempts the identity keys (patient/study/series) — a temporal holdout
    legitimately re-images the same patient across the time boundary — while
    still checking group keys and content duplicates.
    """
    columns = {str(c) for c in df.columns}
    if "split" not in columns:
        return LeakageReport(violations=(), rows_checked=int(len(df)), temporal_policy=temporal_policy)

    assigned = df[df["split"].notna()]
    violations: list[LeakageViolation] = []

    key_columns = [c for c in _LEAKAGE_KEY_COLUMNS if c in columns]
    if temporal_policy:
        key_columns = [c for c in key_columns if c == "group_id_hash"]
    if "image_sha256" in columns:
        key_columns.append("image_sha256")

    for column in key_columns:
        grouped = assigned.groupby(column, dropna=True)
        for key, block in grouped:
            splits = sorted({str(s) for s in block["split"]})
            if len(splits) > 1:
                sample_ids = [str(s) for s in block["sample_id"].head(_MAX_SAMPLE_IDS)]
                violations.append(
                    LeakageViolation(
                        kind=column,
                        key=str(key),
                        splits=tuple(splits),
                        sample_ids=tuple(sample_ids),
                        sample_count=int(len(block)),
                    )
                )

    violations.sort(key=lambda v: (v.kind, v.key))
    return LeakageReport(violations=tuple(violations), rows_checked=int(len(assigned)), temporal_policy=temporal_policy)


@dataclass(frozen=True)
class ResearchOverride:
    """Explicit, recorded decision to proceed despite known leakage.

    Governance (``docs/data_governance.md`` section 6) requires leakage to
    fail ingestion; this override exists for documented research exceptions
    only and is recorded in the split/run metadata for audit.
    """

    reason: str
    recorded_by: str
    recorded_at: str  # ISO 8601

    def __post_init__(self) -> None:
        for name in ("reason", "recorded_by", "recorded_at"):
            if not getattr(self, name):
                raise SplitLeakageError(f"ResearchOverride.{name} must be non-empty for an auditable override")

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "recorded_by": self.recorded_by, "recorded_at": self.recorded_at}


def assert_no_split_leakage(
    df: pd.DataFrame,
    *,
    temporal_policy: bool = False,
    research_override: ResearchOverride | None = None,
) -> LeakageReport:
    """Gate training on leakage: raise :class:`SplitLeakageError` unless clean.

    A non-null ``research_override`` records an explicit exception (returned
    in-band via :attr:`LeakageReport.temporal_policy` is unaffected; the
    override itself is audit metadata the caller records in run metadata).
    """
    report = check_split_leakage(df, temporal_policy=temporal_policy)
    if report.ok or research_override is not None:
        return report
    shown = report.violations[:_MAX_VIOLATIONS]
    lines = [
        (
            f"- {v.kind} {v.key} spans splits {list(v.splits)} "
            f"({v.sample_count} row(s), e.g. sample_ids {list(v.sample_ids)})"
        )
        for v in shown
    ]
    message = (
        f"split leakage detected in {report.rows_checked} assigned row(s): {len(report.violations)} violation(s):\n"
        + "\n".join(lines)
    )
    if len(report.violations) > len(shown):
        message += f"\n- ... and {len(report.violations) - len(shown)} more violation(s)"
    message += (
        "\nRefusing to train on this manifest (ADR 0004). Fix the split assignment or record an explicit "
        "ResearchOverride for a documented research exception."
    )
    raise SplitLeakageError(message)
