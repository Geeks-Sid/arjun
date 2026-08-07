"""Pathology / whole-slide preprocessing: thumbnails, tissue masks, tile planning, stain ops.

This module works on in-memory RGB slide arrays (``(H, W, 3)`` uint8) plus a
level-0 microns-per-pixel (MPP). Slide I/O (pyramids, region reads, corrupt
tiles) belongs to the reader layer (``medfm.data.readers.pathology``); nothing
here imports OpenSlide/TiffSlide and every operation is deterministic and
host-only.

Coordinate contract: all tile origins and sizes are **level-0 slide pixels**,
so planned tiles always map back to the source slide regardless of scanner
MPP. :func:`plan_tiles` normalizes across scanners by sizing the level-0 grid
from ``tile_size * target_mpp / slide_mpp``, capping the plan at ``max_tiles``
in scan order (row-major, no RNG). A zero-tissue slide is a valid input at
every stage and yields an empty plan rather than an error.

Stain operations are :class:`~medfm.data.transforms.base.Transform` subclasses
over tile images packed in :class:`TransformData` as ``[C, H, W]`` float RGB in
``[0, 1]``. They are **opt-in**: no default pipeline includes them — they must
be configured explicitly, and their constructor configs hash independently via
``config_dict``/``config_hash``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from medfm.data.errors import TransformError
from medfm.data.transforms.base import Transform, TransformContext, TransformData


def _color_ops() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from skimage.color import hed2rgb, lab2rgb, rgb2hed, rgb2hsv, rgb2lab
    except ImportError as exc:
        raise TransformError("pathology stain/color operations require scikit-image; install medfm[medical]") from exc
    return hed2rgb, lab2rgb, rgb2hed, rgb2hsv, rgb2lab


#: Grayscale conversion weights (ITU-R BT.601 luma), used by blur scoring.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

#: Minimum saturation for a pixel to count as tissue even when Otsu finds no
#: foreground structure (guards against faint backgrounds on nearly-blank slides).
_MIN_TISSUE_SATURATION = 0.05

#: Brightness (HSV value) above which pixels are treated as slide background.
_BACKGROUND_VALUE = 0.95

#: Pen-mark-like artifact detection: highly saturated AND dark pixels.
_ARTIFACT_SAT_THRESHOLD = 0.8
_ARTIFACT_VAL_THRESHOLD = 0.6

#: Number of histogram bins used by the Otsu threshold over [0, 1] channels.
_OTSU_BINS = 256

#: Length (hex chars) of the short SHA-256 tile identifier.
_TILE_ID_LENGTH = 16


def _otsu_threshold(values: np.ndarray[Any, Any]) -> float:
    """Otsu threshold for a channel with values in [0, 1], via a fixed 256-bin histogram.

    Reimplemented in numpy (scipy/scikit-image filtering helpers are untyped)
    with fixed bins and first-maximum tie-breaking, so the result is a pure
    deterministic function of the input. A constant channel thresholds just
    above its value; combined with :data:`_MIN_TISSUE_SATURATION` this keeps
    uniform (blank) slides tissue-free.
    """
    hist, bin_edges = np.histogram(values, bins=_OTSU_BINS, range=(0.0, 1.0))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    probabilities = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centers)
    mu_total = float(mu[-1])
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between_class = np.where(
            denominator > 0.0, (mu_total * omega - mu) ** 2 / np.where(denominator > 0.0, denominator, 1.0), -1.0
        )
    return float(centers[int(np.argmax(between_class))])


def _require_rgb_uint8(name: str, array: np.ndarray[Any, Any]) -> None:
    """Validate an ``(H, W, 3)`` uint8 RGB array."""
    if not isinstance(array, np.ndarray) or array.ndim != 3 or array.shape[2] != 3:
        raise TransformError(f"{name} must have shape (H, W, 3); got {getattr(array, 'shape', None)}")
    if array.dtype != np.uint8:
        raise TransformError(f"{name} must be uint8; got {array.dtype}")


def _to_hsv(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """RGB uint8 ``(H, W, 3)`` -> HSV float64 with channels in [0, 1]."""
    _, _, _, rgb2hsv, _ = _color_ops()
    return np.asarray(rgb2hsv(image), dtype=np.float64)


def make_thumbnail(slide: np.ndarray[Any, Any], max_size: int) -> np.ndarray[Any, Any]:
    """Deterministically downscale a slide so its longest side is at most ``max_size``.

    Slides already within ``max_size`` are returned as a copy unchanged.
    Downscaling uses PIL bilinear resampling (fixed kernel, no RNG), so
    repeated calls on the same array are byte-identical.
    """
    _require_rgb_uint8("slide", slide)
    if max_size <= 0:
        raise TransformError(f"max_size must be positive; got {max_size}")
    height, width = int(slide.shape[0]), int(slide.shape[1])
    longest = max(height, width)
    if longest <= max_size:
        return slide.copy()
    scale = max_size / longest
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    resized = Image.fromarray(slide).resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def compute_tissue_mask(slide_or_thumbnail: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Boolean tissue mask via HSV saturation (Otsu) plus brightness thresholding.

    Tissue pixels are saturated (Otsu threshold on the saturation channel,
    floored at :data:`_MIN_TISSUE_SATURATION`) and not near-white. A blank or
    all-white slide yields an all-``False`` mask; downstream planning treats
    that as an empty grid, never an error. The output has the spatial shape of
    the input, so callers may run it on a thumbnail for speed and map the mask
    back to level-0 coordinates (:func:`plan_tiles` does this scaling itself).
    """
    _require_rgb_uint8("slide_or_thumbnail", slide_or_thumbnail)
    hsv = _to_hsv(slide_or_thumbnail)
    saturation, value = hsv[..., 1], hsv[..., 2]
    # Otsu on a constant channel thresholds just above that constant, so a
    # uniform white slide (saturation ~ 0 everywhere) hits the floor and the
    # mask stays empty instead of noise-selecting background pixels.
    threshold = max(_otsu_threshold(saturation), _MIN_TISSUE_SATURATION)
    return (saturation > threshold) & (value <= _BACKGROUND_VALUE)


def tissue_fraction(tile: np.ndarray[Any, Any], tissue_mask_of_tile: np.ndarray[Any, Any]) -> float:
    """Fraction of a tile covered by tissue; compare against a minimum to pass/fail.

    ``tissue_mask_of_tile`` is the boolean mask cropped to the tile's extent
    (from :func:`compute_tissue_mask`); it must match the tile's spatial shape.
    The tile array is validated for shape consistency but the score depends
    only on the mask, so the filter is a pure function of its inputs.
    """
    _require_rgb_uint8("tile", tile)
    if not isinstance(tissue_mask_of_tile, np.ndarray) or tissue_mask_of_tile.dtype != np.bool_:
        raise TransformError("tissue_mask_of_tile must be a boolean numpy array")
    if tuple(tissue_mask_of_tile.shape) != tuple(tile.shape[:2]):
        raise TransformError(
            f"tissue_mask_of_tile shape {tuple(tissue_mask_of_tile.shape)} does not match "
            f"tile spatial shape {tuple(tile.shape[:2])}"
        )
    if tissue_mask_of_tile.size == 0:
        raise TransformError("tissue_mask_of_tile must be non-empty")
    return float(tissue_mask_of_tile.mean(dtype=np.float64))


def _grayscale(tile: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """RGB uint8 -> float64 luma image."""
    return tile.astype(np.float64) @ _LUMA_WEIGHTS


def blur_score(tile: np.ndarray[Any, Any]) -> float:
    """Variance of the discrete Laplacian of the tile's luma channel; low = blurry.

    The Laplacian uses the 4-neighbour stencil evaluated on the interior only
    (border pixels excluded so edge handling cannot bias the variance). Pure
    numpy, no RNG: identical tiles give identical scores. A constant tile
    scores exactly 0.0.
    """
    _require_rgb_uint8("tile", tile)
    gray = _grayscale(tile)
    if min(gray.shape) < 3:
        raise TransformError(f"blur_score needs a tile of at least 3x3 pixels; got {gray.shape}")
    center = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(laplacian.var(dtype=np.float64))


def artifact_score(
    tile: np.ndarray[Any, Any],
    *,
    sat_threshold: float = _ARTIFACT_SAT_THRESHOLD,
    val_threshold: float = _ARTIFACT_VAL_THRESHOLD,
) -> float:
    """Fraction of highly saturated dark pixels (pen-mark-like artifacts).

    Pen ink and marker strokes are saturated and dark relative to H&E tissue,
    so a high score flags the tile for rejection (compare against a maximum to
    pass/fail). Deterministic in both the image and the thresholds.
    """
    _require_rgb_uint8("tile", tile)
    if not 0.0 <= sat_threshold <= 1.0 or not 0.0 <= val_threshold <= 1.0:
        raise TransformError(f"artifact thresholds must lie in [0, 1]; got sat={sat_threshold}, val={val_threshold}")
    hsv = _to_hsv(tile)
    hits = (hsv[..., 1] > sat_threshold) & (hsv[..., 2] < val_threshold)
    return float(hits.mean(dtype=np.float64))


def make_tile_id(slide_key: str, x: int, y: int, width: int, height: int, level: int) -> str:
    """Short SHA-256 tile identifier over (slide_key, x, y, width, height, level).

    Stable across hosts, runs, and processes (SHA-256, not Python's salted
    ``hash``); truncation keeps the first ``_TILE_ID_LENGTH`` hex chars.
    """
    payload = f"tile:{slide_key}:{x}:{y}:{width}:{height}:{level}".encode()
    return hashlib.sha256(payload).hexdigest()[:_TILE_ID_LENGTH]


@dataclass(frozen=True)
class TileRecord:
    """One planned tile: identity, level-0 geometry, MPP, and quality scores.

    ``x``/``y``/``width``/``height`` are always level-0 slide pixels and are
    validated against the slide bounds by :func:`plan_tiles`. ``mpp`` is the
    normalized target MPP the tile was planned at; ``level`` is the pyramid
    level the reader should extract from (advisory metadata — planning itself
    happens in level-0 geometry). ``quality`` carries per-tile scores
    (``"blur"``, ``"artifact"``) filled in after pixel extraction; planning
    works from the tissue mask alone, so :func:`plan_tiles` leaves it empty.
    """

    tile_id: str
    x: int
    y: int
    width: int
    height: int
    level: int
    mpp: float
    tissue_fraction: float
    quality: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tile_id:
            raise TransformError("TileRecord.tile_id must be non-empty")
        if self.x < 0 or self.y < 0:
            raise TransformError(f"TileRecord origin must be non-negative; got ({self.x}, {self.y})")
        if self.width <= 0 or self.height <= 0:
            raise TransformError(f"TileRecord size must be positive; got {self.width}x{self.height}")
        if self.level < 0:
            raise TransformError(f"TileRecord.level must be non-negative; got {self.level}")
        if not math.isfinite(self.mpp) or self.mpp <= 0:
            raise TransformError(f"TileRecord.mpp must be positive and finite; got {self.mpp}")
        if not 0.0 <= self.tissue_fraction <= 1.0:
            raise TransformError(f"TileRecord.tissue_fraction must lie in [0, 1]; got {self.tissue_fraction}")
        for key, value in self.quality.items():
            if not key or not math.isfinite(value):
                raise TransformError(f"TileRecord.quality entries must be named finite scores; got {key!r}={value!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-able representation for persistence alongside manifests."""
        return {
            "tile_id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "mpp": self.mpp,
            "tissue_fraction": self.tissue_fraction,
            "quality": dict(sorted(self.quality.items())),
        }


def plan_tiles(
    slide_shape: tuple[int, int],
    slide_mpp: float,
    tissue_mask: np.ndarray[Any, Any],
    *,
    tile_size: int,
    target_mpp: float,
    min_tissue_fraction: float,
    level: int = 0,
    max_tiles: int | None = None,
    slide_key: str = "",
) -> list[TileRecord]:
    """Plan a deterministic, MPP-normalized tile grid over a tissue mask.

    MPP normalization: the level-0 tile side is ``round(tile_size * target_mpp
    / slide_mpp)`` pixels, so tiles from scanners with different native MPPs
    cover the same physical area (``tile_size * target_mpp`` microns). The grid
    is non-overlapping in scan order (row-major); edge tiles are shifted
    in-bounds so every returned coordinate satisfies ``0 <= x, y`` and
    ``x + width <= W``, ``y + height <= H``.

    ``tissue_mask`` may be computed at thumbnail resolution; its pixels are
    mapped proportionally onto the level-0 slide shape when scoring each
    tile's tissue fraction. Tiles with ``tissue_fraction >=
    min_tissue_fraction`` are kept; ``max_tiles`` caps the plan at the first N
    tiles in scan order (no RNG). An all-``False`` mask (zero-tissue slide)
    returns an empty list.

    The function is pure: identical inputs give byte-identical
    :class:`TileRecord` lists across runs and processes.
    """
    if len(slide_shape) != 2:
        raise TransformError(f"slide_shape must be (height, width); got {slide_shape}")
    slide_h, slide_w = int(slide_shape[0]), int(slide_shape[1])
    if slide_h <= 0 or slide_w <= 0:
        raise TransformError(f"slide_shape entries must be positive; got {slide_shape}")
    if not math.isfinite(slide_mpp) or slide_mpp <= 0:
        raise TransformError(f"slide_mpp must be positive and finite; got {slide_mpp}")
    if not math.isfinite(target_mpp) or target_mpp <= 0:
        raise TransformError(f"target_mpp must be positive and finite; got {target_mpp}")
    if tile_size <= 0:
        raise TransformError(f"tile_size must be positive; got {tile_size}")
    if not 0.0 <= min_tissue_fraction <= 1.0:
        raise TransformError(f"min_tissue_fraction must lie in [0, 1]; got {min_tissue_fraction}")
    if level < 0:
        raise TransformError(f"level must be non-negative; got {level}")
    if max_tiles is not None and max_tiles < 0:
        raise TransformError(f"max_tiles must be non-negative or None; got {max_tiles}")
    if not isinstance(tissue_mask, np.ndarray) or tissue_mask.ndim != 2 or tissue_mask.dtype != np.bool_:
        raise TransformError("tissue_mask must be a 2D boolean numpy array")
    mask_h, mask_w = int(tissue_mask.shape[0]), int(tissue_mask.shape[1])

    resize_factor = target_mpp / slide_mpp
    side = max(1, int(round(tile_size * resize_factor)))
    tile_w, tile_h = min(side, slide_w), min(side, slide_h)

    records: list[TileRecord] = []
    for gy in range(0, slide_h, tile_h):
        y = min(gy, slide_h - tile_h)
        for gx in range(0, slide_w, tile_w):
            x = min(gx, slide_w - tile_w)
            # Map the tile's level-0 extent onto the (possibly coarser) mask.
            i0, i1 = (y * mask_h) // slide_h, -(-((y + tile_h) * mask_h) // slide_h)
            j0, j1 = (x * mask_w) // slide_w, -(-((x + tile_w) * mask_w) // slide_w)
            fraction = float(tissue_mask[i0:i1, j0:j1].mean(dtype=np.float64))
            if fraction < min_tissue_fraction:
                continue
            records.append(
                TileRecord(
                    tile_id=make_tile_id(slide_key, x, y, tile_w, tile_h, level),
                    x=x,
                    y=y,
                    width=tile_w,
                    height=tile_h,
                    level=level,
                    mpp=target_mpp,
                    tissue_fraction=fraction,
                )
            )
            if max_tiles is not None and len(records) >= max_tiles:
                return records

    # Defense in depth: the grid arithmetic above guarantees bounds, but the
    # contract ("every coordinate maps to the source slide") is checked here so
    # a future refactor cannot silently violate it.
    for record in records:
        if record.x + record.width > slide_w or record.y + record.height > slide_h:
            raise TransformError(
                f"internal error: planned tile {record.tile_id} at ({record.x}, {record.y}) with size "
                f"{record.width}x{record.height} escapes slide bounds {slide_w}x{slide_h}"
            )
    return records


def _tile_to_hwc_float(data: TransformData, name: str) -> tuple[np.ndarray[Any, Any], torch.dtype]:
    """Extract a ``[3, H, W]`` float RGB tile from TransformData as ``(H, W, 3)`` float64."""
    image = data.image
    if image.ndim != 3 or image.shape[0] != 3:
        raise TransformError(f"{name} expects a [3, H, W] RGB tile; got shape {tuple(image.shape)}")
    if not image.dtype.is_floating_point:
        raise TransformError(f"{name} expects a floating dtype; got {image.dtype}")
    array = image.detach().permute(1, 2, 0).cpu().numpy().astype(np.float64)
    if bool((array < 0.0).any()) or bool((array > 1.0).any()):
        raise TransformError(f"{name} expects RGB values in [0, 1]")
    return array, image.dtype


def _hwc_float_to_tensor(array: np.ndarray[Any, Any], dtype: torch.dtype) -> torch.Tensor:
    """Pack an ``(H, W, 3)`` float array back into a ``[3, H, W]`` CPU tensor."""
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(dtype)


class ReinhardStainNormalize(Transform):
    """Reinhard stain normalization: lab-space mean/std matching to a reference.

    Opt-in deterministic transform — never part of a default pipeline; it must
    be configured explicitly. ``reference_stats`` supplies the target lab
    statistics as ``{"mean": [L, a, b], "std": [L, a, b]}``; each tile is
    standardized to zero mean / unit std in lab space, then rescaled to the
    reference statistics. Constructor config flows into
    :meth:`config_dict`/:meth:`config_hash`, so a reference change invalidates
    preprocessing caches.
    """

    name = "reinhard_stain_normalize"
    stage = "deterministic"

    def __init__(self, reference_stats: dict[str, Any]) -> None:
        if not isinstance(reference_stats, dict):
            raise TransformError("reference_stats must be a mapping with 'mean' and 'std' entries")
        parsed: dict[str, tuple[float, float, float]] = {}
        for key in ("mean", "std"):
            values = reference_stats.get(key)
            if not isinstance(values, list | tuple) or len(values) != 3:
                raise TransformError(f"reference_stats[{key!r}] must be a list of 3 numbers; got {values!r}")
            floats = (float(values[0]), float(values[1]), float(values[2]))
            if any(not math.isfinite(v) for v in floats):
                raise TransformError(f"reference_stats[{key!r}] entries must be finite; got {values!r}")
            if key == "std" and any(v <= 0 for v in floats):
                raise TransformError(f"reference_stats['std'] entries must be positive; got {values!r}")
            parsed[key] = floats
        self._reference_mean = parsed["mean"]
        self._reference_std = parsed["std"]

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        """Match the tile's lab statistics to the configured reference."""
        _, lab2rgb, _, _, rgb2lab = _color_ops()
        array, dtype = _tile_to_hwc_float(data, self.name)
        lab = rgb2lab(array)
        source_mean = lab.reshape(-1, 3).mean(axis=0)
        source_std = lab.reshape(-1, 3).std(axis=0)
        safe_std = np.where(source_std > 1e-8, source_std, 1.0)
        matched = (lab - source_mean) / safe_std * self._reference_std + self._reference_mean
        normalized = np.clip(lab2rgb(matched), 0.0, 1.0)
        data.image = _hwc_float_to_tensor(normalized, dtype)
        data.record(self.name, self.stage, self.params())
        return data

    def config_dict(self) -> dict[str, Any]:
        return {
            "reference_stats": {
                "mean": list(self._reference_mean),
                "std": list(self._reference_std),
            }
        }


class StainAugment(Transform):
    """Stochastic stain augmentation in HED space, drawing only from ``ctx.rng``.

    Opt-in augmentation — never part of a default pipeline. Each hematoxylin /
    eosin / DAB concentration channel is scaled by ``1 + U(-alpha, alpha)`` and
    shifted by ``U(-beta, beta)`` with uniforms drawn from the seeded context
    generator, so augmentation is reproducible under fixed
    ``(base_seed, epoch, worker_id, sample_key)`` and never touches global RNG.
    Not spatially invertible; no inverter is registered (by design).
    """

    name = "stain_augment"
    stage = "stochastic"

    def __init__(self, alpha: float = 0.1, beta: float = 0.1) -> None:
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise TransformError(f"alpha must lie in [0, 1]; got {alpha}")
        if not math.isfinite(beta) or beta < 0.0:
            raise TransformError(f"beta must be non-negative and finite; got {beta}")
        self._alpha = float(alpha)
        self._beta = float(beta)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        """Perturb the tile's stain channels using the seeded context generator."""
        if ctx is None:
            raise TransformError(f"{self.name!r} requires a TransformContext with a seeded generator")
        hed2rgb, _, rgb2hed, _, _ = _color_ops()
        array, dtype = _tile_to_hwc_float(data, self.name)
        hed = rgb2hed(array)
        uniforms = torch.rand(2, 3, generator=ctx.rng).to(torch.float64).numpy() * 2.0 - 1.0
        scale = 1.0 + uniforms[0] * self._alpha
        shift = uniforms[1] * self._beta
        augmented = np.clip(hed2rgb(hed * scale + shift), 0.0, 1.0)
        data.image = _hwc_float_to_tensor(augmented, dtype)
        data.record(self.name, self.stage, self.params())
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"alpha": self._alpha, "beta": self._beta}
