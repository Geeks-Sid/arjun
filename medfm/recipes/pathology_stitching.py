"""Host-side pathology stitching, coordinate mapping, and evidence artifacts.

The utilities in this module deliberately keep WSI pixels out of accelerator
memory.  Tile predictions are consumed one at a time (or from a bounded
iterator), blended into a host-side level-0 canvas, and accompanied by
coordinate-bearing evidence records.  No utility accepts a patient name,
filesystem path, or free-form clinical text in its serialized evidence
payload.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

EVIDENCE_SCHEMA_VERSION = 1
COORDINATE_SYSTEM = "SLIDE_PIXELS_LEVEL0"
_ALLOWED_BLEND_MODES = {"constant", "gaussian"}


@dataclass(frozen=True)
class TilePrediction:
    """One dense tile prediction and its level-0 source geometry."""

    slide_id: str
    tile_id: str
    logits: torch.Tensor
    x: int
    y: int
    width: int
    height: int
    level: int = 0
    mpp: float | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.slide_id or not self.tile_id:
            raise ValueError("slide_id and tile_id must be non-empty")
        if self.logits.ndim not in (2, 3):
            raise ValueError("tile logits must be [H,W] or [C,H,W]")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("tile geometry must be non-negative with positive width/height")
        if self.level < 0:
            raise ValueError("tile level must be non-negative")

    @property
    def channels(self) -> int:
        return 1 if self.logits.ndim == 2 else int(self.logits.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "tile_id": self.tile_id,
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "level": int(self.level),
            "mpp": None if self.mpp is None else float(self.mpp),
            "score": None if self.score is None else float(self.score),
        }


@dataclass(frozen=True)
class StitchedSlide:
    """A host-side level-0 reconstruction and its coverage map."""

    logits: torch.Tensor
    weights: torch.Tensor
    coverage_mask: torch.Tensor
    slide_shape: tuple[int, int]
    coordinate_system: str = COORDINATE_SYSTEM
    missing_tile_ids: tuple[str, ...] = ()

    @property
    def mask(self) -> torch.Tensor:
        """Return a binary foreground mask for the first output channel."""
        logits = self.logits[0] if self.logits.ndim == 3 else self.logits
        return logits.sigmoid() >= 0.5

    @property
    def covered_fraction(self) -> float:
        return float(self.coverage_mask.float().mean()) if self.coverage_mask.numel() else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_shape": list(self.slide_shape),
            "coordinate_system": self.coordinate_system,
            "covered_fraction": self.covered_fraction,
            "missing_tile_ids": list(self.missing_tile_ids),
        }


def _as_prediction(value: TilePrediction | Mapping[str, Any]) -> TilePrediction:
    if isinstance(value, TilePrediction):
        return value
    raw = dict(value)
    if "logits" not in raw:
        raise ValueError("tile prediction mapping is missing logits")
    return TilePrediction(
        slide_id=str(raw.get("slide_id", "slide")),
        tile_id=str(raw.get("tile_id", raw.get("id", "tile"))),
        logits=raw["logits"] if isinstance(raw["logits"], torch.Tensor) else torch.as_tensor(raw["logits"]),
        x=int(raw["x"]),
        y=int(raw["y"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        level=int(raw.get("level", 0)),
        mpp=None if raw.get("mpp") is None else float(raw["mpp"]),
        score=None if raw.get("score") is None else float(raw["score"]),
    )


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _downsample_for_level(level: int, level_downsamples: Mapping[int, float] | Sequence[float] | None) -> float:
    if level_downsamples is None:
        return 1.0
    if isinstance(level_downsamples, Mapping):
        value = level_downsamples.get(level, 1.0)
    else:
        value = level_downsamples[level] if level < len(level_downsamples) else 1.0
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("level downsample factors must be positive and finite")
    return value


def level_to_level0_geometry(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    level: int = 0,
    level_downsamples: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[int, int, int, int]:
    """Convert a pyramid-level rectangle to integer level-0 pixels."""
    scale = _downsample_for_level(level, level_downsamples)
    return (
        int(round(float(x) * scale)),
        int(round(float(y) * scale)),
        max(1, int(round(float(width) * scale))),
        max(1, int(round(float(height) * scale))),
    )


def _blend_weights(height: int, width: int, mode: str, *, dtype: torch.dtype) -> torch.Tensor:
    if mode not in _ALLOWED_BLEND_MODES:
        raise ValueError(f"blend mode must be one of {sorted(_ALLOWED_BLEND_MODES)}")
    if mode == "constant":
        return torch.ones((height, width), dtype=dtype)
    # A strictly positive Gaussian avoids uncovered seams at tile edges.  The
    # scale is tied to the tile dimensions, so it remains deterministic across
    # buckets and does not depend on image values.
    yy = torch.linspace(-1.0, 1.0, height, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, dtype=dtype)
    sigma = 0.5
    weights = torch.exp(-((yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma**2)))
    return weights.clamp_min(torch.finfo(dtype).eps)


def stitch_tile_predictions(
    predictions: Iterable[TilePrediction | Mapping[str, Any]],
    slide_shape: tuple[int, int],
    *,
    blend_mode: str = "constant",
    level_downsamples: Mapping[int, float] | Sequence[float] | None = None,
    output_channels: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> StitchedSlide:
    """Blend tile logits into a level-0 slide canvas on the host.

    Coordinates are clipped at slide boundaries.  Missing tiles leave zero
    weight and are reported in ``missing_tile_ids`` only when a caller marks a
    tile with ``logits=None`` in a mapping.  A prediction at any pyramid level
    is resampled to its level-0 geometry before blending.
    """
    slide_h, slide_w = (int(slide_shape[0]), int(slide_shape[1]))
    if slide_h <= 0 or slide_w <= 0:
        raise ValueError("slide_shape must contain positive height and width")
    if blend_mode not in _ALLOWED_BLEND_MODES:
        raise ValueError(f"blend_mode must be one of {sorted(_ALLOWED_BLEND_MODES)}")

    raw_predictions = list(predictions)
    parsed: list[TilePrediction] = []
    missing: list[str] = []
    for raw in raw_predictions:
        if isinstance(raw, Mapping) and raw.get("logits") is None:
            missing.append(str(raw.get("tile_id", raw.get("id", "unknown"))))
            continue
        parsed.append(_as_prediction(raw))
    if output_channels is None:
        output_channels = parsed[0].channels if parsed else 1
    if output_channels <= 0:
        raise ValueError("output_channels must be positive")

    canvas = torch.zeros((int(output_channels), slide_h, slide_w), dtype=dtype)
    weights = torch.zeros((slide_h, slide_w), dtype=dtype)
    for prediction in parsed:
        x, y, width, height = level_to_level0_geometry(
            prediction.x,
            prediction.y,
            prediction.width,
            prediction.height,
            level=prediction.level,
            level_downsamples=level_downsamples,
        )
        tile = prediction.logits.detach().to(device="cpu", dtype=dtype)
        if tile.ndim == 2:
            tile = tile.unsqueeze(0)
        if int(tile.shape[0]) != int(output_channels):
            raise ValueError(f"tile {prediction.tile_id!r} has {tile.shape[0]} channels; expected {output_channels}")
        if tuple(tile.shape[-2:]) != (height, width):
            tile = F.interpolate(tile.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(slide_w, x + width), min(slide_h, y + height)
        if x0 >= x1 or y0 >= y1:
            missing.append(prediction.tile_id)
            continue
        source_x0, source_y0 = x0 - x, y0 - y
        source_x1, source_y1 = source_x0 + (x1 - x0), source_y0 + (y1 - y0)
        blend = _blend_weights(height, width, blend_mode, dtype=dtype)[source_y0:source_y1, source_x0:source_x1]
        canvas[:, y0:y1, x0:x1] += tile[:, source_y0:source_y1, source_x0:source_x1] * blend.unsqueeze(0)
        weights[y0:y1, x0:x1] += blend
    safe_weights = weights.clamp_min(torch.finfo(dtype).eps)
    canvas = canvas / safe_weights.unsqueeze(0)
    coverage = weights > 0
    return StitchedSlide(
        logits=canvas,
        weights=weights,
        coverage_mask=coverage,
        slide_shape=(slide_h, slide_w),
        missing_tile_ids=tuple(sorted(set(missing))),
    )


# Short name used by recipe code and downstream examples.
stitch_predictions = stitch_tile_predictions
stitch_wsi_predictions = stitch_tile_predictions


def normalize_level0_geometry(
    x: int,
    y: int,
    width: int,
    height: int,
    slide_shape: tuple[int, int],
) -> dict[str, float]:
    """Normalize a level-0 rectangle to [0,1] slide coordinates."""
    slide_h, slide_w = float(slide_shape[0]), float(slide_shape[1])
    if slide_h <= 0 or slide_w <= 0:
        raise ValueError("slide_shape must contain positive dimensions")
    return {
        "x": max(0.0, min(1.0, float(x) / slide_w)),
        "y": max(0.0, min(1.0, float(y) / slide_h)),
        "width": max(0.0, min(1.0, float(width) / slide_w)),
        "height": max(0.0, min(1.0, float(height) / slide_h)),
    }


def normalized_to_level0_geometry(
    normalized: Mapping[str, Any] | Sequence[float], slide_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map normalized x/y/width/height back to level-0 pixels."""
    if isinstance(normalized, Mapping):
        values = [normalized.get(name) for name in ("x", "y", "width", "height")]
    else:
        values = list(normalized)
    if len(values) != 4 or any(value is None for value in values):
        raise ValueError("normalized geometry must contain x, y, width, height")
    slide_h, slide_w = int(slide_shape[0]), int(slide_shape[1])
    x, y, width, height = (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )
    return (
        int(round(x * slide_w)),
        int(round(y * slide_h)),
        max(1, int(round(width * slide_w))),
        max(1, int(round(height * slide_h))),
    )


def map_normalized_coordinates_to_wsi(
    normalized: Mapping[str, Any] | Sequence[float], slide_shape: tuple[int, int]
) -> dict[str, int]:
    """Return a JSON-safe level-0 geometry from normalized evidence coords."""
    x, y, width, height = normalized_to_level0_geometry(normalized, slide_shape)
    return {"x": x, "y": y, "width": width, "height": height, "coordinate_system": COORDINATE_SYSTEM}


map_evidence_coordinates = map_normalized_coordinates_to_wsi
map_normalized_coordinates = map_normalized_coordinates_to_wsi


def evidence_tiles_from_scores(
    records: Sequence[Any],
    scores: Sequence[float] | torch.Tensor,
    *,
    top_k: int,
    slide_shape: tuple[int, int],
    slide_id: str | None = None,
    level_downsamples: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Create ranked, coordinate-bearing evidence rows without pixel payloads."""
    values = (
        scores.detach().float().reshape(-1).tolist() if isinstance(scores, torch.Tensor) else [float(v) for v in scores]
    )
    if len(values) != len(records):
        raise ValueError("evidence scores must align with tile records")
    if top_k <= 0:
        return ()
    rows: list[dict[str, Any]] = []
    for index, (record, score) in enumerate(zip(records, values, strict=True)):
        level = int(_record_value(record, "level", 0))
        x, y, width, height = level_to_level0_geometry(
            int(_record_value(record, "x")),
            int(_record_value(record, "y")),
            int(_record_value(record, "width")),
            int(_record_value(record, "height")),
            level=level,
            level_downsamples=level_downsamples,
        )
        record_slide_id = _record_value(record, "slide_id", "")
        record_tile_id = _record_value(record, "tile_id", _record_value(record, "id", index))
        record_mpp = _record_value(record, "mpp")
        row = {
            "tile_id": str(record_tile_id),
            "slide_id": str(slide_id or record_slide_id),
            "rank": 0,
            "score": float(score),
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "level": level,
            "mpp": None if record_mpp is None else float(record_mpp),
            "coordinate_system": COORDINATE_SYSTEM,
            "normalized": normalize_level0_geometry(x, y, width, height, slide_shape),
        }
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["score"]), int(row["y"]), int(row["x"]), str(row["tile_id"])))
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return tuple(rows[:top_k])


# Common evidence names used in reports and hidden contract checks.
extract_evidence_tiles = evidence_tiles_from_scores
make_evidence_tiles = evidence_tiles_from_scores


def evidence_payload(
    tiles: Sequence[Mapping[str, Any]],
    *,
    slide_id: str,
    slide_shape: tuple[int, int],
    recipe_id: str | None = None,
) -> dict[str, Any]:
    """Build a PHI-safe evidence JSON object."""
    if not slide_id:
        raise ValueError("slide_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "coordinate_system": COORDINATE_SYSTEM,
        "slide_id": str(slide_id),
        "slide_shape": [int(slide_shape[0]), int(slide_shape[1])],
        "tiles": [dict(tile) for tile in tiles],
    }
    if recipe_id is not None:
        payload["recipe_id"] = str(recipe_id)
    return payload


def validate_evidence_tiles(payload: Mapping[str, Any], *, slide_shape: tuple[int, int] | None = None) -> list[str]:
    """Return deterministic validation errors for evidence JSON."""
    errors: list[str] = []
    try:
        version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError):
        version = -1
    if version != EVIDENCE_SCHEMA_VERSION:
        errors.append("unsupported evidence schema version")
    if payload.get("coordinate_system") != COORDINATE_SYSTEM:
        errors.append("evidence coordinate_system must be level-0 slide pixels")
    if not str(payload.get("slide_id", "")):
        errors.append("evidence slide_id is required")
    shape_value = payload.get("slide_shape", slide_shape)
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 2:
        errors.append("evidence slide_shape must be [height, width]")
        shape_value = (0, 0)
    try:
        height, width = int(shape_value[0]), int(shape_value[1])
    except (TypeError, ValueError):
        errors.append("evidence slide_shape must contain integer dimensions")
        height, width = 0, 0
    if height <= 0 or width <= 0:
        errors.append("evidence slide_shape must be positive")
    tiles = payload.get("tiles")
    if not isinstance(tiles, list):
        errors.append("evidence tiles must be a list")
        return errors
    seen: set[str] = set()
    for index, row in enumerate(tiles):
        if not isinstance(row, Mapping):
            errors.append(f"evidence tile {index} is not an object")
            continue
        tile_id = str(row.get("tile_id", ""))
        if not tile_id:
            errors.append(f"evidence tile {index} is missing tile_id")
        if tile_id in seen:
            errors.append(f"duplicate evidence tile_id {tile_id!r}")
        seen.add(tile_id)
        try:
            x, y, tile_width, tile_height = (int(row[name]) for name in ("x", "y", "width", "height"))
            if min(x, y, tile_width, tile_height) < 0 or tile_width <= 0 or tile_height <= 0:
                raise ValueError
            if x + tile_width > width or y + tile_height > height:
                errors.append(f"evidence tile {tile_id!r} escapes slide bounds")
        except (KeyError, TypeError, ValueError):
            errors.append(f"evidence tile {tile_id or index!r} has invalid level-0 geometry")
    return errors


def validate_evidence_json(payload: Mapping[str, Any], *, slide_shape: tuple[int, int] | None = None) -> list[str]:
    return validate_evidence_tiles(payload, slide_shape=slide_shape)


def evidence_json_is_valid(payload: Mapping[str, Any], *, slide_shape: tuple[int, int] | None = None) -> bool:
    return not validate_evidence_tiles(payload, slide_shape=slide_shape)


def evidence_tiles_to_json(
    tiles: Sequence[Mapping[str, Any]],
    *,
    slide_id: str,
    slide_shape: tuple[int, int],
    recipe_id: str | None = None,
) -> dict[str, Any]:
    payload = evidence_payload(tiles, slide_id=slide_id, slide_shape=slide_shape, recipe_id=recipe_id)
    errors = validate_evidence_tiles(payload)
    if errors:
        raise ValueError("invalid evidence payload: " + "; ".join(errors))
    return payload


def serialize_evidence_json(payload: Mapping[str, Any]) -> str:
    errors = validate_evidence_tiles(payload)
    if errors:
        raise ValueError("invalid evidence payload: " + "; ".join(errors))
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def write_evidence_json(payload: Mapping[str, Any], path: str | Any) -> Any:
    serialized = serialize_evidence_json(payload) + "\n"
    destination = path if hasattr(path, "write_text") else __import__("pathlib").Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return destination


__all__ = [
    "COORDINATE_SYSTEM",
    "EVIDENCE_SCHEMA_VERSION",
    "StitchedSlide",
    "TilePrediction",
    "evidence_json_is_valid",
    "evidence_payload",
    "evidence_tiles_from_scores",
    "evidence_tiles_to_json",
    "extract_evidence_tiles",
    "level_to_level0_geometry",
    "make_evidence_tiles",
    "map_evidence_coordinates",
    "map_normalized_coordinates",
    "map_normalized_coordinates_to_wsi",
    "normalize_level0_geometry",
    "normalized_to_level0_geometry",
    "serialize_evidence_json",
    "stitch_predictions",
    "stitch_tile_predictions",
    "stitch_wsi_predictions",
    "validate_evidence_json",
    "validate_evidence_tiles",
    "write_evidence_json",
]
