"""Human review protocol and agreement metrics for generated clinical text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REVIEW_ERROR_CATEGORIES = (
    "correct",
    "minor_error",
    "major_error",
    "potentially_harmful_error",
    "unsupported_finding",
    "omitted_critical_finding",
    "incorrect_negation",
    "incorrect_laterality",
    "incorrect_severity",
    "incorrect_anatomy",
    "poor_uncertainty_expression",
    "lexical_mismatch",
)
MAJOR_ERROR_CATEGORIES = frozenset({"major_error", "potentially_harmful_error", "omitted_critical_finding"})
LEXICAL_ONLY_CATEGORIES = frozenset({"lexical_mismatch"})


class ReviewProtocolError(ValueError):
    """Raised when a review protocol is not reproducible or access-controlled."""


@dataclass(frozen=True)
class ReviewProtocol:
    """Instructions and sampling controls shared by all reviewers."""

    instructions: tuple[str, ...]
    sample_size: int
    sampling_seed: int
    sampling_strategy: str = "stratified_by_task_and_error_risk"
    blinding: str = "reviewers_blinded_to_model_id"
    disagreement_resolution: str = "adjudication_by_independent_senior_reviewer"
    access_control: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ReviewProtocolError("review sample_size must be positive")
        if not self.instructions:
            raise ReviewProtocolError("reviewer instructions are required")
        if not str(self.blinding).strip() or "blind" not in self.blinding.casefold():
            raise ReviewProtocolError("review sampling must declare blinding")
        if not str(self.disagreement_resolution).strip():
            raise ReviewProtocolError("disagreement resolution must be declared")
        required = {"classification", "authorized_roles", "storage_uri"}
        if not required.issubset(self.access_control):
            raise ReviewProtocolError(f"access_control must declare {sorted(required)}")
        if self.schema_version != 1:
            raise ReviewProtocolError("unsupported review protocol schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instructions": list(self.instructions),
            "sample_size": self.sample_size,
            "sampling_seed": self.sampling_seed,
            "sampling_strategy": self.sampling_strategy,
            "blinding": self.blinding,
            "disagreement_resolution": self.disagreement_resolution,
            "access_control": dict(self.access_control),
        }


@dataclass(frozen=True)
class ReviewRecord:
    """A reviewer label referencing protected content by URI, not embedding it."""

    item_id: str
    reviewer_id: str
    category: str
    artifact_uri: str
    blinded: bool = True
    lexical_mismatch: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not str(self.item_id).strip() or not str(self.reviewer_id).strip():
            raise ReviewProtocolError("review item and reviewer IDs must be non-empty")
        normalized = str(self.category).strip().lower().replace(" ", "_")
        if normalized not in REVIEW_ERROR_CATEGORIES:
            raise ReviewProtocolError(f"unsupported review category {self.category!r}")
        object.__setattr__(self, "category", normalized)
        if (
            not str(self.artifact_uri).strip()
            or "file://" in self.artifact_uri.casefold()
            and "protected" not in self.artifact_uri.casefold()
        ):
            raise ReviewProtocolError("reviewed content must use an access-controlled artifact URI")
        if not self.blinded:
            raise ReviewProtocolError("review records must be blinded")
        if self.schema_version != 1:
            raise ReviewProtocolError("unsupported review-record schema version")

    @property
    def is_major_or_harmful(self) -> bool:
        return self.category in MAJOR_ERROR_CATEGORIES

    @property
    def is_lexical_only(self) -> bool:
        return self.lexical_mismatch or self.category in LEXICAL_ONLY_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "reviewer_id": self.reviewer_id,
            "category": self.category,
            "artifact_uri": self.artifact_uri,
            "blinded": self.blinded,
            "lexical_mismatch": self.lexical_mismatch,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


def sample_review_items(
    item_ids: Iterable[str],
    *,
    sample_size: int,
    seed: int,
    strata: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Deterministically sample items, balancing declared strata when present."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    values = sorted({str(value) for value in item_ids})
    if sample_size >= len(values):
        return tuple(values)
    import random

    rng = random.Random(int(seed))
    if not strata:
        return tuple(sorted(rng.sample(values, sample_size)))
    grouped: dict[str, list[str]] = {}
    for item in values:
        grouped.setdefault(str(strata.get(item, "unstratified")), []).append(item)
    selected: list[str] = []
    keys = sorted(grouped)
    for index in range(min(sample_size, len(keys))):
        selected.append(rng.choice(grouped[keys[index]]))
    remaining = [item for item in values if item not in selected]
    if len(selected) < sample_size:
        selected.extend(rng.sample(remaining, sample_size - len(selected)))
    return tuple(sorted(selected))


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = sum((left.count(category) / len(left)) * (right.count(category) / len(right)) for category in categories)
    return None if expected == 1.0 else (observed - expected) / (1.0 - expected)


def inter_rater_agreement(records: Iterable[ReviewRecord]) -> dict[str, Any]:
    """Return pairwise agreement and Cohen's kappa for all reviewer pairs."""

    by_reviewer: dict[str, dict[str, str]] = {}
    for record in records:
        by_reviewer.setdefault(record.reviewer_id, {})[record.item_id] = record.category
    reviewers = sorted(by_reviewer)
    pairs: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(reviewers):
        for right_name in reviewers[left_index + 1 :]:
            items = sorted(set(by_reviewer[left_name]) & set(by_reviewer[right_name]))
            left = [by_reviewer[left_name][item] for item in items]
            right = [by_reviewer[right_name][item] for item in items]
            agreement = None if not items else sum(a == b for a, b in zip(left, right, strict=True)) / len(items)
            pairs.append(
                {
                    "reviewer_a": left_name,
                    "reviewer_b": right_name,
                    "items": len(items),
                    "agreement": agreement,
                    "cohen_kappa": _cohen_kappa(left, right),
                }
            )
    return {"reviewers": reviewers, "pairs": pairs, "pair_count": len(pairs)}


def summarize_reviews(records: Iterable[ReviewRecord]) -> dict[str, Any]:
    """Separate harmful/major errors from lexical mismatches."""

    values = list(records)
    counts = Counter(record.category for record in values)
    return {
        "reviewed_items": len(values),
        "category_counts": dict(sorted(counts.items())),
        "major_or_harmful_count": sum(record.is_major_or_harmful for record in values),
        "lexical_only_count": sum(record.is_lexical_only for record in values),
        "non_lexical_error_count": sum(
            not record.is_lexical_only and record.category != "correct" for record in values
        ),
        "inter_rater_agreement": inter_rater_agreement(values),
    }


def default_review_protocol(
    *, sample_size: int = 100, seed: int = 0, storage_uri: str = "protected://phase16/review"
) -> ReviewProtocol:
    return ReviewProtocol(
        instructions=(
            "Review the output against the reference and available evidence.",
            "Assign the most severe applicable category.",
            "Treat a lexical mismatch as distinct from a clinical error.",
            "Do not infer a clinical-validation claim from review labels.",
        ),
        sample_size=sample_size,
        sampling_seed=seed,
        access_control={
            "classification": "restricted",
            "authorized_roles": ("qualified_reviewer",),
            "storage_uri": storage_uri,
        },
    )


__all__ = [
    "LEXICAL_ONLY_CATEGORIES",
    "MAJOR_ERROR_CATEGORIES",
    "REVIEW_ERROR_CATEGORIES",
    "ReviewProtocol",
    "ReviewProtocolError",
    "ReviewRecord",
    "default_review_protocol",
    "inter_rater_agreement",
    "sample_review_items",
    "summarize_reviews",
]
