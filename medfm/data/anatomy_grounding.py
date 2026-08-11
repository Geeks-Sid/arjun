"""Report-conditioned anatomical regions for CT grounding."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as connected_components

from medfm.data.totalsegmentator import DEFAULT_THORACIC_LABELS

LOBE_LABELS: tuple[str, ...] = DEFAULT_THORACIC_LABELS[:5]

RegionMode = Literal["exact_lobe", "lobe_group", "lung", "none"]


@dataclass(frozen=True)
class ReportRegion:
    """An anatomical hypothesis resolved from one finding description."""

    labels: tuple[str, ...]
    confidence: float
    matched_terms: tuple[str, ...]
    mode: RegionMode

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("report-region confidence must be in [0,1]")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("report-region labels must be unique")


_LOBE_BY_SIDE_POSITION = {
    ("left", "upper"): "lung_upper_lobe_left",
    ("left", "lower"): "lung_lower_lobe_left",
    ("right", "upper"): "lung_upper_lobe_right",
    ("right", "middle"): "lung_middle_lobe_right",
    ("right", "lower"): "lung_lower_lobe_right",
}
_SIDE_LABELS = {
    "left": ("lung_upper_lobe_left", "lung_lower_lobe_left"),
    "right": ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"),
}
_POSITION_LABELS = {
    "upper": ("lung_upper_lobe_left", "lung_upper_lobe_right"),
    "middle": ("lung_middle_lobe_right",),
    "lower": ("lung_lower_lobe_left", "lung_lower_lobe_right"),
}
_ABBREVIATIONS = {
    "lul": "left upper lobe",
    "lll": "left lower lobe",
    "rul": "right upper lobe",
    "rml": "right middle lobe",
    "rll": "right lower lobe",
}


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    for abbreviation, expansion in _ABBREVIATIONS.items():
        normalized = re.sub(rf"\b{abbreviation}\b", expansion, normalized)
    return f" {normalized} "


def resolve_report_region(text: str) -> ReportRegion:
    """Resolve explicit lobe/lung phrases without inventing unsupported anatomy.

    The resolver intentionally returns a soft confidence. Exact side+lobe phrases
    are high-confidence; broad lung mentions remain usable for model guidance but
    are not strong enough for anatomical postprocessing.
    """

    normalized = _normalize_text(text)
    scores: dict[str, float] = {}
    terms: set[str] = set()
    best_confidence = 0.0
    best_mode: RegionMode = "none"

    def add(labels: Sequence[str], confidence: float, term: str, mode: RegionMode) -> None:
        nonlocal best_confidence, best_mode
        for label in labels:
            scores[label] = max(scores.get(label, 0.0), confidence)
        if labels:
            terms.add(term)
            if confidence > best_confidence:
                best_confidence = confidence
                best_mode = mode

    exact_positions: set[str] = set()
    for (side, position), label in _LOBE_BY_SIDE_POSITION.items():
        patterns = (
            rf"\b{side}\s+{position}\s+lobes?\b",
            rf"\b{position}\s+lobes?\s+of\s+(?:the\s+)?{side}\s+lung\b",
            rf"\b{position}\s+lobes?\s+on\s+the\s+{side}\b",
            rf"\b{side}\s+{position}\s+lung\b",
        )
        if any(re.search(pattern, normalized) for pattern in patterns):
            add((label,), 1.0, f"{side} {position} lobe", "exact_lobe")
            exact_positions.add(position)

    if re.search(r"\blingula(?:r)?\b", normalized):
        add(("lung_upper_lobe_left",), 0.95, "lingula", "exact_lobe")
        exact_positions.add("upper")

    # Plural lobe groups are explicit enough to guide segmentation, but are
    # weaker than side-specific mentions.
    for position, labels in _POSITION_LABELS.items():
        if position in exact_positions:
            continue
        if re.search(rf"\b{position}\s+lobes\b", normalized):
            add(labels, 0.7, f"{position} lobes", "lobe_group")
        elif re.search(rf"\b{position}\s+lobe\b", normalized):
            add(labels, 0.55, f"{position} lobe", "lobe_group")

    # Segment terminology often omits the formal lobe name. Use it only when
    # laterality is explicit; this avoids turning an arbitrary "basal segment"
    # into both lungs.
    for side in ("left", "right"):
        if re.search(rf"\b(?:basal|basilar|dependent)\s+segments?\s+of\s+the\s+{side}\s+lung\b", normalized):
            add((_LOBE_BY_SIDE_POSITION[(side, "lower")],), 0.7, f"{side} basal segment", "lobe_group")
        if re.search(
            rf"\b(?:apical|apices|apex)\s+(?:segments?|regions?)?\s*(?:of\s+the\s+)?{side}\s+lung\b",
            normalized,
        ):
            add((_LOBE_BY_SIDE_POSITION[(side, "upper")],), 0.7, f"{side} apical region", "lobe_group")

    if not scores:
        if re.search(r"\b(?:both|bilateral)\s+lungs?\b", normalized):
            add(LOBE_LABELS, 0.65, "both lungs", "lung")
        elif re.search(r"\bright\s+lung\b", normalized):
            add(_SIDE_LABELS["right"], 0.6, "right lung", "lung")
        elif re.search(r"\bleft\s+lung\b", normalized):
            add(_SIDE_LABELS["left"], 0.6, "left lung", "lung")
        elif re.search(r"\blungs?\b", normalized):
            add(LOBE_LABELS, 0.35, "lung", "lung")

    labels = tuple(label for label in LOBE_LABELS if label in scores)
    if not labels:
        return ReportRegion((), 0.0, (), "none")
    return ReportRegion(labels, best_confidence, tuple(sorted(terms)), best_mode)


def select_report_prior(
    prior: np.ndarray,
    prior_labels: Sequence[str],
    region: ReportRegion,
) -> np.ndarray:
    """Keep only report-selected TotalSegmentator channels, scaled by confidence."""

    array = np.asarray(prior, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"prior must have shape [C,D,H,W], got {array.shape}")
    labels = tuple(str(label) for label in prior_labels)
    if len(labels) != array.shape[0]:
        raise ValueError(f"prior label count {len(labels)} does not match channels {array.shape[0]}")
    selected = np.zeros_like(array, dtype=np.float32)
    indices = {label: index for index, label in enumerate(labels)}
    for label in region.labels:
        index = indices.get(label)
        if index is not None:
            selected[index] = array[index] * float(region.confidence)
    return np.ascontiguousarray(selected)


def report_region_mask(
    prior: np.ndarray,
    prior_labels: Sequence[str],
    text: str,
) -> tuple[np.ndarray, ReportRegion]:
    """Return a report-selected binary region in ``[D,H,W]`` order."""

    region = resolve_report_region(text)
    selected = select_report_prior(prior, prior_labels, region)
    return np.any(selected > 0, axis=0), region


def filter_prediction_by_region(
    prediction_hwd: np.ndarray,
    finding_texts: Sequence[str],
    prior: np.ndarray,
    prior_labels: Sequence[str],
    affine: np.ndarray,
    *,
    min_confidence: float = 0.6,
    halo_mm: float = 15.0,
    min_component_overlap: float = 0.02,
    connectivity: int = 2,
) -> np.ndarray:
    """Conservatively remove predicted components far from report anatomy."""

    prediction = np.asarray(prediction_hwd)
    if prediction.ndim != 4:
        raise ValueError(f"prediction must have shape [F,H,W,D], got {prediction.shape}")
    if len(finding_texts) != prediction.shape[0]:
        raise ValueError("finding text count must match prediction channels")
    if halo_mm < 0.0:
        raise ValueError("halo_mm must be non-negative")
    if not 0.0 <= min_component_overlap <= 1.0:
        raise ValueError("min_component_overlap must be in [0,1]")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3")
    if np.asarray(affine).shape != (4, 4):
        raise ValueError("affine must have shape [4,4]")

    prior_array = np.asarray(prior)
    prior_shape = tuple(int(value) for value in prior_array.shape[1:])
    prediction_shape = (prediction.shape[3], prediction.shape[1], prediction.shape[2])
    if prior_array.ndim != 4 or prior_shape != prediction_shape:
        raise ValueError(
            "prior spatial shape [D,H,W] must match prediction [F,H,W,D], "
            f"got prior {prior_array.shape} and prediction {prediction.shape}"
        )
    output = (prediction > 0).astype(np.uint8, copy=True)
    spacing_hwd = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
    spacing_dhw = (float(spacing_hwd[2]), float(spacing_hwd[0]), float(spacing_hwd[1]))
    structure = generate_binary_structure(3, connectivity)

    for finding_index, text in enumerate(finding_texts):
        region_mask, region = report_region_mask(prior_array, prior_labels, text)
        if region.confidence < min_confidence or not region_mask.any():
            continue
        distance_mm = distance_transform_edt(~region_mask, sampling=spacing_dhw)
        allowed = distance_mm <= halo_mm
        components, count = connected_components(output[finding_index].transpose(2, 0, 1), structure=structure)
        filtered_dhw = np.zeros_like(components, dtype=np.uint8)
        for component_index in range(1, int(count) + 1):
            component = components == component_index
            overlap = float(np.count_nonzero(component & allowed)) / float(np.count_nonzero(component))
            if overlap >= min_component_overlap:
                filtered_dhw[component] = 1
        output[finding_index] = filtered_dhw.transpose(1, 2, 0)
    return output


__all__ = [
    "LOBE_LABELS",
    "ReportRegion",
    "filter_prediction_by_region",
    "report_region_mask",
    "resolve_report_region",
    "select_report_prior",
]
