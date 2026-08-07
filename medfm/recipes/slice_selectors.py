"""Deterministic host-side selectors for slice-sequence VLM experiments.

Slice selection is deliberately separate from native volumetric encoding.  A
selector returns the source index and all geometry/sequence fields needed to
interpret the selected 2D image later; it never silently turns a sequence into
native 3D tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch

SLICE_SELECTOR_VERSION = "phase14-slice-selectors-1"


@dataclass(frozen=True)
class SliceRecord:
    """One host-resident candidate slice and its selection signals."""

    index: int
    image: torch.Tensor
    physical_z_mm: float
    series_order: int = 0
    window: str = "default"
    mri_sequence: str | None = None
    normalized_z: float | None = None
    anatomy_score: float = 0.0
    report_score: float = 0.0
    lesion_score: float = 0.0
    entropy: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("slice index must be non-negative")
        if not isinstance(self.image, torch.Tensor) or self.image.ndim not in (2, 3):
            raise ValueError("slice image must be a [H,W] or [C,H,W] tensor")
        if not torch.isfinite(self.image.float()).all():
            raise ValueError("slice image must contain finite values")
        if not self.window:
            raise ValueError("slice window must be non-empty")
        if self.series_order < 0:
            raise ValueError("series_order must be non-negative")


@dataclass(frozen=True)
class SliceSelection:
    """Selection metadata preserved beside the selected 2D tensor."""

    index: int
    normalized_z: float
    physical_z_mm: float
    series_order: int
    window: str
    mri_sequence: str | None
    score: float
    selector: str
    selector_revision: str = SLICE_SELECTOR_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "normalized_z": self.normalized_z,
            "physical_z_mm": self.physical_z_mm,
            "series_order": self.series_order,
            "window": self.window,
            "mri_sequence": self.mri_sequence,
            "score": self.score,
            "selector": self.selector,
            "selector_revision": self.selector_revision,
        }


def _ordered_records(records: Iterable[SliceRecord]) -> tuple[SliceRecord, ...]:
    result = tuple(records)
    if not result:
        raise ValueError("slice selector requires at least one candidate")
    indices = [record.index for record in result]
    if len(set(indices)) != len(indices):
        raise ValueError("slice candidate indices must be unique")
    return tuple(sorted(result, key=lambda item: (item.series_order, item.index)))


def _selection(record: SliceRecord, records: tuple[SliceRecord, ...], score: float, selector: str) -> SliceSelection:
    if record.normalized_z is not None:
        normalized = float(record.normalized_z)
    else:
        positions = [float(item.physical_z_mm) for item in records]
        low, high = min(positions), max(positions)
        normalized = 0.0 if high == low else (float(record.physical_z_mm) - low) / (high - low)
    return SliceSelection(
        index=int(record.index),
        normalized_z=float(max(0.0, min(1.0, normalized))),
        physical_z_mm=float(record.physical_z_mm),
        series_order=int(record.series_order),
        window=str(record.window),
        mri_sequence=record.mri_sequence,
        score=float(score),
        selector=selector,
    )


def _uniform_positions(length: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("slice count must be positive")
    if count > length:
        raise ValueError(f"requested {count} slices but only {length} candidates are available")
    if count == 1:
        return (0,)
    # Round over the closed interval and repair the rare duplicate caused by
    # integer rounding.  This keeps exactly ``count`` positions and is stable
    # across CPU/CUDA/TPU because selection runs on the host.
    raw = torch.linspace(0, length - 1, count, dtype=torch.float64).round().to(torch.int64).tolist()
    chosen: list[int] = []
    for position in raw:
        candidate = int(position)
        if candidate in chosen:
            candidate = next(index for index in range(length) if index not in chosen)
        chosen.append(candidate)
    return tuple(sorted(chosen))


def _entropy(record: SliceRecord) -> float:
    if record.entropy is not None:
        return float(record.entropy)
    values = record.image.detach().float().reshape(-1)
    if values.numel() == 0:
        return 0.0
    # A normalized variance is a deterministic, dependency-free entropy proxy
    # suitable for ranking only.  The selected image remains unchanged.
    variance = float(values.var(unbiased=False))
    scale = float(values.abs().mean()) + 1e-6
    return variance / scale


class SliceSelector(ABC):
    """Host-only fixed-count selector interface."""

    name: str = "base"

    def __init__(self, count: int) -> None:
        if count <= 0:
            raise ValueError("slice selector count must be positive")
        self.count = int(count)

    @abstractmethod
    def _rank(self, records: tuple[SliceRecord, ...]) -> list[tuple[SliceRecord, float]]:
        """Return candidates ordered from most to least preferred."""

    def select(self, records: Iterable[SliceRecord]) -> tuple[SliceSelection, ...]:
        ordered = _ordered_records(records)
        ranked = self._rank(ordered)
        selected = ranked[: self.count]
        if len(selected) != self.count:
            raise ValueError(f"selector {self.name!r} could not produce {self.count} slices")
        # The visual tower consumes series order, not ranking order.  Selection
        # ranking is an experiment detail; source order is part of the data
        # contract and is therefore restored before collation.
        selected.sort(key=lambda item: (item[0].series_order, item[0].index))
        return tuple(_selection(record, ordered, score, self.name) for record, score in selected)


class UniformSliceSelector(SliceSelector):
    """Uniformly sample a fixed number of slices in source order."""

    name = "uniform"

    def _rank(self, records: tuple[SliceRecord, ...]) -> list[tuple[SliceRecord, float]]:
        positions = set(_uniform_positions(len(records), self.count))
        return [(record, 1.0) for position, record in enumerate(records) if position in positions]


class _ScoredSliceSelector(SliceSelector):
    score_field: str

    def _rank(self, records: tuple[SliceRecord, ...]) -> list[tuple[SliceRecord, float]]:
        scored = [(record, float(getattr(record, self.score_field))) for record in records]
        return sorted(scored, key=lambda item: (-item[1], item[0].series_order, item[0].index))


class AnatomyAwareSliceSelector(_ScoredSliceSelector):
    """Rank slices using a host-computed anatomy relevance score."""

    name = "anatomy_aware"
    score_field = "anatomy_score"


class ReportConditionedSliceSelector(_ScoredSliceSelector):
    """Rank slices using report/query relevance computed before collation."""

    name = "report_conditioned"
    score_field = "report_score"


class EntropySliceSelector(SliceSelector):
    """Select high-information slices using a deterministic entropy proxy."""

    name = "entropy"

    def _rank(self, records: tuple[SliceRecord, ...]) -> list[tuple[SliceRecord, float]]:
        scored = [(record, _entropy(record)) for record in records]
        return sorted(scored, key=lambda item: (-item[1], item[0].series_order, item[0].index))


class LesionAwareSliceSelector(_ScoredSliceSelector):
    """Rank slices by host-side lesion evidence without changing pixels."""

    name = "lesion_aware"
    score_field = "lesion_score"


class MultiWindowSliceSelector(SliceSelector):
    """Round-robin across acquisition windows, then fill by source order."""

    name = "multi_window"

    def _rank(self, records: tuple[SliceRecord, ...]) -> list[tuple[SliceRecord, float]]:
        windows: dict[str, list[SliceRecord]] = {}
        for record in records:
            windows.setdefault(record.window, []).append(record)
        ranked: list[tuple[SliceRecord, float]] = []
        keys = tuple(sorted(windows))
        cursor = 0
        while len(ranked) < len(records):
            added = False
            for key in keys:
                values = windows[key]
                if cursor < len(values):
                    ranked.append((values[cursor], 1.0))
                    added = True
            if not added:
                break
            cursor += 1
        return ranked


_SELECTOR_TYPES: dict[str, type[SliceSelector]] = {
    "uniform": UniformSliceSelector,
    "anatomy_aware": AnatomyAwareSliceSelector,
    "anatomy": AnatomyAwareSliceSelector,
    "report_conditioned": ReportConditionedSliceSelector,
    "report": ReportConditionedSliceSelector,
    "entropy": EntropySliceSelector,
    "lesion_aware": LesionAwareSliceSelector,
    "lesion": LesionAwareSliceSelector,
    "multi_window": MultiWindowSliceSelector,
    "window": MultiWindowSliceSelector,
}


def build_slice_selector(name: str, *, count: int) -> SliceSelector:
    """Build a named selector; all selection remains host-side."""

    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return _SELECTOR_TYPES[normalized](count)
    except KeyError as exc:
        raise ValueError(f"unknown slice selector {name!r}; expected {sorted(set(_SELECTOR_TYPES))}") from exc


def select_slices(
    records: Iterable[SliceRecord],
    *,
    selector: str = "uniform",
    count: int,
) -> tuple[SliceSelection, ...]:
    """Convenience wrapper used by host data preparation and tests."""

    return build_slice_selector(selector, count=count).select(records)


def selections_to_metadata(selections: Iterable[SliceSelection]) -> tuple[dict[str, object], ...]:
    """Serialize selection fields without retaining source pixels."""

    return tuple(selection.to_dict() for selection in selections)


__all__ = [
    "AnatomyAwareSliceSelector",
    "EntropySliceSelector",
    "LesionAwareSliceSelector",
    "MultiWindowSliceSelector",
    "ReportConditionedSliceSelector",
    "SLICE_SELECTOR_VERSION",
    "SliceRecord",
    "SliceSelection",
    "SliceSelector",
    "UniformSliceSelector",
    "build_slice_selector",
    "select_slices",
    "selections_to_metadata",
]
