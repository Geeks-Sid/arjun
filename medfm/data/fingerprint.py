"""Dataset fingerprinting: deterministic manifest statistics for Phase 04.

``fingerprint_manifest`` computes the statistics Phase 04 needs to configure
preprocessing: patient/study/sample counts, modality distributions, shape and
spacing distributions, intensity/label/missing-value/site/vendor/duplicate
statistics, report-length and WSI MPP/magnification statistics, segmentation
class volumes, split-leakage results, and recommended bounded shape buckets.

The report is fully deterministic (sorted keys/value sets, fixed float
rounding, no absolute paths or timestamps) and its ``fingerprint_hash`` is a
SHA-256 over the canonical serialization — safe to record in run metadata.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from medfm.core.serialization import config_hash
from medfm.data.splits import LeakageReport, check_split_leakage

#: Candidate 2D bucket edges (square-ish resolutions used by the v1 roster).
_RESOLUTION_CANDIDATES: tuple[int, ...] = (224, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
#: Candidate 3D patch edges (isotropic, bounded for 48 GB VRAM / TPU HBM).
_PATCH_CANDIDATES: tuple[int, ...] = (64, 96, 128, 160, 192, 224)
#: Candidate WSI tile-count caps.
_TILE_CANDIDATES: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
#: Candidate slice-count caps for slice-sequence models.
_SLICE_CANDIDATES: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128)
#: Text token-length cap policy: round p95 chars / 4 up to a multiple of 128.
_TEXT_MAX_CHARS_CAP = 8192

_2D_MODALITIES = ("XRAY_2D", "CT_2D_SLICE", "MRI_2D_SLICE", "PATHOLOGY_TILE", "MULTI_IMAGE_2D")
_3D_MODALITIES = ("CT_3D", "MRI_3D", "MULTI_SERIES_3D")

#: Columns the fingerprint reports missingness for.
_MISSINGNESS_COLUMNS: tuple[str, ...] = (
    "study_id_hash",
    "series_id_hash",
    "image_uri",
    "mask_uri",
    "annotation_uri",
    "report_uri",
    "label_json",
    "split",
    "site_id",
    "scanner_vendor",
    "acquisition_date_bucket",
    "image_sha256",
    "shape",
    "spacing_mm",
    "num_slices",
    "num_tiles",
    "microns_per_pixel",
    "magnification",
    "report_chars",
)


def _round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, digits) for v in value]
    return value


def _percentiles(values: list[float], marks: tuple[float, ...] = (0.0, 0.5, 0.95, 1.0)) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    result: dict[str, float] = {}
    for mark in marks:
        index = min(len(ordered) - 1, int(math.floor(mark * (len(ordered) - 1) + 0.5)))
        result[f"p{int(mark * 100)}"] = ordered[index]
    result["count"] = float(len(ordered))
    return result


def _cell_is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _list_cell(value: Any) -> list[Any] | None:
    if _cell_is_null(value):
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return list(value.tolist())
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    return None


def _shape_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Per-axis percentiles grouped by tensor rank."""
    if "shape" not in df.columns:
        return {}
    by_rank: dict[int, list[list[float]]] = {}
    for value in df["shape"]:
        items = _list_cell(value)
        if not items:
            continue
        dims = [float(d) for d in items if isinstance(d, (int, float))]
        if not dims or len(dims) != len(items):
            continue
        by_rank.setdefault(len(dims), []).append(dims)
    stats: dict[str, Any] = {}
    for rank in sorted(by_rank):
        rows = by_rank[rank]
        axis_stats = []
        for axis in range(rank):
            axis_stats.append(_percentiles([row[axis] for row in rows]))
        stats[f"rank_{rank}"] = {"sample_count": len(rows), "axes": axis_stats}
    return stats


def _spacing_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "spacing_mm" not in df.columns:
        return {}
    by_rank: dict[int, list[list[float]]] = {}
    for value in df["spacing_mm"]:
        items = _list_cell(value)
        if not items:
            continue
        dims = [float(d) for d in items if isinstance(d, (int, float))]
        if not dims or len(dims) != len(items):
            continue
        by_rank.setdefault(len(dims), []).append(dims)
    stats: dict[str, Any] = {}
    for rank in sorted(by_rank):
        rows = by_rank[rank]
        stats[f"rank_{rank}"] = {
            "sample_count": len(rows),
            "axes": [_percentiles([row[axis] for row in rows]) for axis in range(rank)],
        }
    return stats


def _label_prevalence(df: pd.DataFrame) -> dict[str, Any]:
    if "label_json" not in df.columns:
        return {"labeled_rows": 0, "unlabeled_rows": int(len(df))}
    task_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    unlabeled = 0
    for value in df["label_json"]:
        if _cell_is_null(value):
            unlabeled += 1
            continue
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        task = str(payload.get("task", "UNKNOWN"))
        task_counts[task] = task_counts.get(task, 0) + 1
        if task in ("BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION", "ORDINAL_CLASSIFICATION"):
            values = payload.get("values") or []
            if values:
                key = f"{task}:{int(values[0])}"
                class_counts[key] = class_counts.get(key, 0) + 1
        elif task == "MULTILABEL_CLASSIFICATION":
            for index, flag in enumerate(payload.get("values") or []):
                if flag:
                    key = f"{task}:{index}"
                    class_counts[key] = class_counts.get(key, 0) + 1
    return {
        "labeled_rows": int(sum(task_counts.values())),
        "unlabeled_rows": unlabeled,
        "task_counts": {k: task_counts[k] for k in sorted(task_counts)},
        "class_counts": {k: class_counts[k] for k in sorted(class_counts)},
    }


def _missingness(df: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in _MISSINGNESS_COLUMNS:
        if column not in df.columns:
            continue
        result[column] = int(sum(1 for value in df[column] if _cell_is_null(value)))
    return {k: result[k] for k in sorted(result)}


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts: dict[str, int] = {}
    for value in df[column]:
        if _cell_is_null(value):
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def _duplicate_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "image_sha256" not in df.columns:
        return {"duplicate_hash_count": 0, "duplicated_rows": 0, "cross_split_duplicates": 0}
    splits = df["split"] if "split" in df.columns else [None] * len(df)
    rows = [(str(h), str(s)) for h, s in zip(df["image_sha256"], splits, strict=True) if not _cell_is_null(h)]
    counts: dict[str, int] = {}
    splits_by_hash: dict[str, set[str]] = {}
    for digest, split in rows:
        counts[digest] = counts.get(digest, 0) + 1
        splits_by_hash.setdefault(digest, set()).add(split if split and split != "nan" else "UNASSIGNED")
    duplicates = {h for h, count in counts.items() if count > 1}
    cross_split = {h for h, splits_set in splits_by_hash.items() if len(splits_set - {"UNASSIGNED"}) > 1}
    return {
        "duplicate_hash_count": len(duplicates),
        "duplicated_rows": sum(counts[h] for h in duplicates),
        "cross_split_duplicates": len(cross_split),
        "cross_split_hashes": sorted(cross_split),
    }


def _numeric_column_stats(df: pd.DataFrame, column: str) -> dict[str, float]:
    if column not in df.columns:
        return {}
    values = [float(v) for v in df[column] if not _cell_is_null(v) and isinstance(v, (int, float))]
    return _percentiles(values)


def _intensity_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-row intensity_stats_json (fingerprint-derived upstream)."""
    if "intensity_stats_json" not in df.columns:
        return {"rows_with_stats": 0}
    collected: dict[str, list[float]] = {}
    rows_with = 0
    for value in df["intensity_stats_json"]:
        if _cell_is_null(value):
            continue
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        rows_with += 1
        for key, item in payload.items():
            if isinstance(item, (int, float)) and math.isfinite(float(item)):
                collected.setdefault(key, []).append(float(item))
    return {"rows_with_stats": rows_with, "aggregated": {k: _percentiles(v) for k, v in sorted(collected.items())}}


def _seg_volume_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "seg_class_volumes_json" not in df.columns:
        return {"rows_with_segmentation": 0}
    class_volumes: dict[str, list[float]] = {}
    rows_with = 0
    for value in df["seg_class_volumes_json"]:
        if _cell_is_null(value):
            continue
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        rows_with += 1
        for class_name, volume in payload.items():
            if isinstance(volume, (int, float)) and math.isfinite(float(volume)):
                class_volumes.setdefault(str(class_name), []).append(float(volume))
    return {
        "rows_with_segmentation": rows_with,
        "class_volume_mm3": {k: _percentiles(v) for k, v in sorted(class_volumes.items())},
    }


def _smallest_covering_candidate(values: list[float], candidates: tuple[int, ...], coverage: float) -> int | None:
    """Smallest candidate bucket covering ``coverage`` fraction of ``values``."""
    if not values:
        return None
    ordered = sorted(values)
    needed = ordered[min(len(ordered) - 1, int(math.ceil(coverage * len(ordered))) - 1)]
    for candidate in candidates:
        if candidate >= needed:
            return candidate
    return candidates[-1]


def recommend_shape_buckets(df: pd.DataFrame, *, coverage: float = 0.95) -> list[dict[str, Any]]:
    """Bounded, deterministic bucket recommendations for static-shape collation.

    Kinds follow the Phase 02 bucket vocabulary: ``2d_resolution`` (H, W),
    ``3d_patch`` (D, H, W), ``slice_count`` (I), ``tile_count`` (T),
    ``text_length`` (L). Each recommendation carries the observed coverage it
    was sized for; Phase 04 owns the final bucket table.
    """
    buckets: list[dict[str, Any]] = []

    heights: list[float] = []
    widths: list[float] = []
    volume_axes: list[list[float]] = []
    if "shape" in df.columns and "modality" in df.columns:
        for modality, value in zip(df["modality"], df["shape"], strict=True):
            items = _list_cell(value)
            if not items:
                continue
            dims = [float(d) for d in items if isinstance(d, (int, float))]
            if len(dims) != len(items):
                continue
            if str(modality) in _2D_MODALITIES and len(dims) >= 2:
                heights.append(dims[-2])
                widths.append(dims[-1])
            elif str(modality) in _3D_MODALITIES and len(dims) >= 3:
                volume_axes.append(dims[-3:])
    bucket_h = _smallest_covering_candidate(heights, _RESOLUTION_CANDIDATES, coverage)
    bucket_w = _smallest_covering_candidate(widths, _RESOLUTION_CANDIDATES, coverage)
    if bucket_h and bucket_w:
        buckets.append({"kind": "2d_resolution", "shape": [bucket_h, bucket_w], "samples_considered": len(heights)})

    if volume_axes:
        medians = [sorted(axis)[len(axis) // 2] for axis in volume_axes]
        patch = _smallest_covering_candidate(medians, _PATCH_CANDIDATES, coverage)
        if patch:
            buckets.append({"kind": "3d_patch", "shape": [patch, patch, patch], "samples_considered": len(volume_axes)})

    slice_values = [float(v) for v in df["num_slices"] if not _cell_is_null(v)] if "num_slices" in df.columns else []
    slice_bucket = _smallest_covering_candidate(slice_values, _SLICE_CANDIDATES, coverage)
    if slice_bucket:
        buckets.append({"kind": "slice_count", "shape": [slice_bucket], "samples_considered": len(slice_values)})

    tile_values = [float(v) for v in df["num_tiles"] if not _cell_is_null(v)] if "num_tiles" in df.columns else []
    tile_bucket = _smallest_covering_candidate(tile_values, _TILE_CANDIDATES, coverage)
    if tile_bucket:
        buckets.append({"kind": "tile_count", "shape": [tile_bucket], "samples_considered": len(tile_values)})

    if "report_chars" in df.columns:
        chars = [float(v) for v in df["report_chars"] if not _cell_is_null(v)]
        if chars:
            p95 = sorted(chars)[min(len(chars) - 1, int(math.ceil(0.95 * len(chars))) - 1)]
            capped = min(p95, _TEXT_MAX_CHARS_CAP)
            tokens_cap = int(math.ceil(capped / 4.0 / 128.0) * 128)
            buckets.append({"kind": "text_length", "shape": [max(tokens_cap, 128)], "samples_considered": len(chars)})

    return buckets


def fingerprint_manifest(df: pd.DataFrame, *, leakage_temporal_policy: bool = False) -> dict[str, Any]:
    """Compute the deterministic fingerprint report for a manifest frame."""
    patients = {str(v) for v in df["patient_id_hash"] if "patient_id_hash" in df.columns and not _cell_is_null(v)}
    studies = {str(v) for v in df["study_id_hash"] if "study_id_hash" in df.columns and not _cell_is_null(v)}
    series = {str(v) for v in df["series_id_hash"] if "series_id_hash" in df.columns and not _cell_is_null(v)}

    leakage: LeakageReport = check_split_leakage(df, temporal_policy=leakage_temporal_policy)

    report: dict[str, Any] = {
        "counts": {
            "samples": int(len(df)),
            "patients": len(patients),
            "studies": len(studies),
            "series": len(series),
        },
        "modality_counts": _value_counts(df, "modality"),
        "split_counts": _value_counts(df, "split"),
        "shape_stats": _shape_stats(df),
        "spacing_stats": _spacing_stats(df),
        "intensity_stats": _intensity_stats(df),
        "label_prevalence": _label_prevalence(df),
        "missing_values": _missingness(df),
        "site_distribution": _value_counts(df, "site_id"),
        "vendor_distribution": _value_counts(df, "scanner_vendor"),
        "duplicate_stats": _duplicate_stats(df),
        "split_leakage": leakage.to_dict(),
        "report_chars_stats": _numeric_column_stats(df, "report_chars"),
        "wsi_microns_per_pixel_stats": _numeric_column_stats(df, "microns_per_pixel"),
        "wsi_magnification_stats": _numeric_column_stats(df, "magnification"),
        "segmentation_volume_stats": _seg_volume_stats(df),
        "recommended_shape_buckets": recommend_shape_buckets(df),
    }
    report = _round_floats(report)
    report["fingerprint_hash"] = config_hash(report)
    return report
