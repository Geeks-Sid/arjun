"""Distributed metric reductions and accelerator parity contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from medfm.evaluation.schemas import EvaluationSchemaError, PredictionRecord


@dataclass(frozen=True)
class ReducedCount:
    """A numerator/denominator pair reduced using true valid counts."""

    numerator: float
    denominator: int

    @property
    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": float(self.numerator),
            "denominator": int(self.denominator),
            "value": self.value,
        }


def reduce_metric_counts(partials: Iterable[Mapping[str, Any] | ReducedCount]) -> ReducedCount:
    """Sum true valid numerators/denominators across ranks."""

    numerator = 0.0
    denominator = 0
    for partial in partials:
        if isinstance(partial, ReducedCount):
            current_numerator, current_denominator = partial.numerator, partial.denominator
        else:
            current_numerator = float(partial.get("numerator", 0.0))
            current_denominator = int(partial.get("denominator", partial.get("valid_count", 0)))
        if current_denominator < 0 or not math.isfinite(current_numerator):
            raise ValueError("metric partials must have finite numerators and non-negative denominators")
        numerator += current_numerator
        denominator += current_denominator
    return ReducedCount(numerator, denominator)


def reduce_metric_mapping(
    partials_by_rank: Iterable[Mapping[str, Mapping[str, Any] | ReducedCount]],
) -> dict[str, ReducedCount]:
    """Reduce each metric independently without averaging rank averages."""

    grouped: dict[str, list[Mapping[str, Any] | ReducedCount]] = {}
    for rank in partials_by_rank:
        for name, partial in rank.items():
            grouped.setdefault(str(name), []).append(partial)
    return {name: reduce_metric_counts(values) for name, values in sorted(grouped.items())}


def remove_padded_duplicates(records: Iterable[PredictionRecord]) -> tuple[PredictionRecord, ...]:
    """Remove padded final-batch duplicates before clinical-unit aggregation."""

    result: list[PredictionRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.is_padding:
            continue
        if record.sample_id in seen:
            raise EvaluationSchemaError(f"duplicate valid sample {record.sample_id!r} across ranks")
        seen.add(record.sample_id)
        result.append(record)
    return tuple(result)


def gather_host_metadata(metadata_by_rank: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Gather variable-length metadata on the host, outside compiled steps."""

    rows: list[dict[str, Any]] = []
    for rank, metadata in enumerate(metadata_by_rank):
        for row in metadata.get("rows", ()):  # rows may have different lengths per rank
            rows.append({"rank": rank, **dict(row)})
    return tuple(sorted(rows, key=lambda row: (int(row["rank"]), str(row.get("sample_id", "")))))


def assert_shared_evaluation_seed(seeds: Iterable[int]) -> int:
    values = {int(seed) for seed in seeds}
    if len(values) != 1:
        raise ValueError("threshold, calibration, and bootstrap seeds must match across backends/ranks")
    return next(iter(values))


def coordinator_write_report(
    payload: Mapping[str, Any], path: str | Path, *, rank: int, coordinator_rank: int = 0
) -> Path | None:
    """Only the coordinator writes a final report; other ranks return ``None``."""

    if rank != coordinator_rank:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return destination


@dataclass(frozen=True)
class BackendTolerance:
    """Predeclared absolute/relative numerical tolerance for one backend."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        if self.absolute < 0 or self.relative < 0:
            raise ValueError("backend tolerances must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {"absolute": float(self.absolute), "relative": float(self.relative)}


_DEFAULT_BACKEND_TOLERANCE = BackendTolerance(1e-5, 1e-4)


@dataclass(frozen=True)
class ParityResult:
    """Comparison result that preserves, rather than normalizes, divergence."""

    backend: str
    reference_backend: str
    max_absolute_error: float
    max_relative_error: float
    within_tolerance: bool
    tolerance: BackendTolerance
    divergent_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reference_backend": self.reference_backend,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance.to_dict(),
            "divergent_indices": list(self.divergent_indices),
            "metadata": dict(self.metadata),
        }


def compare_backend_predictions(
    reference: Any,
    candidate: Any,
    *,
    backend: str,
    reference_backend: str = "cpu",
    tolerance: BackendTolerance | Mapping[str, float] = _DEFAULT_BACKEND_TOLERANCE,
) -> ParityResult:
    """Compare deterministic fixtures and retain larger divergences for review."""

    tol = (
        tolerance
        if isinstance(tolerance, BackendTolerance)
        else BackendTolerance(float(tolerance["absolute"]), float(tolerance["relative"]))
    )
    left = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(candidate, dtype=torch.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("backend prediction shapes must match")
    absolute = (left - right).abs()
    relative = absolute / left.abs().clamp_min(1e-12)
    allowed = absolute <= tol.absolute + tol.relative * left.abs()
    indices = tuple(int(index) for index in torch.nonzero(~allowed, as_tuple=False).reshape(-1).tolist())
    return ParityResult(
        backend=str(backend),
        reference_backend=str(reference_backend),
        max_absolute_error=float(absolute.max()) if absolute.numel() else 0.0,
        max_relative_error=float(relative.max()) if relative.numel() else 0.0,
        within_tolerance=not indices,
        tolerance=tol,
        divergent_indices=indices,
        metadata={"investigate_divergence": bool(indices)},
    )


def compare_backend_metrics(
    reference: Mapping[str, float | None],
    candidate: Mapping[str, float | None],
    *,
    backend: str,
    reference_backend: str = "cpu",
    tolerance: BackendTolerance | Mapping[str, float] = _DEFAULT_BACKEND_TOLERANCE,
) -> ParityResult:
    """Compare metric values by stable sorted key rather than rank-local means."""

    names = sorted(set(reference) | set(candidate))
    left: list[float] = []
    right: list[float] = []
    for name in names:
        reference_value = reference.get(name)
        candidate_value = candidate.get(name)
        if reference_value is None or candidate_value is None:
            continue
        left.append(float(reference_value))
        right.append(float(candidate_value))
    return compare_backend_predictions(
        left, right, backend=backend, reference_backend=reference_backend, tolerance=tolerance
    )


# Short names used by CLI/report code.
parity_evaluate = compare_backend_predictions
remove_padded_samples = remove_padded_duplicates


__all__ = [
    "BackendTolerance",
    "ParityResult",
    "ReducedCount",
    "assert_shared_evaluation_seed",
    "compare_backend_metrics",
    "compare_backend_predictions",
    "coordinator_write_report",
    "gather_host_metadata",
    "parity_evaluate",
    "reduce_metric_counts",
    "reduce_metric_mapping",
    "remove_padded_duplicates",
    "remove_padded_samples",
]
