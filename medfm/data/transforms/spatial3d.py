"""Shared volumetric spatial transforms: orientation, spacing, foreground crop.

These are the deterministic, host-only spatial canonicalization steps shared
by the CT and MRI pipelines (and any other volumetric modality). All operate
on ``[C, D, H, W]`` CPU tensors carrying :class:`SpatialMetadata` whose
``affine`` maps voxel indices ``(i, j, k)`` to patient-space millimetres (the
Phase 03 volumetric axis contract — axis identity is never inferred from
shape alone).

Orientation codes are three-letter strings (one letter per array axis) giving
the anatomical direction that increasing voxel index points to: ``R/L``
(patient right/left), ``A/P`` (anterior/posterior), ``S/I`` (superior/
inferior). Example: ``"RAS"`` means axis 0 increases toward patient right,
axis 1 toward anterior, axis 2 toward superior.

Every transform here is spatial and registers an exact inverter under its
``name`` (see :mod:`medfm.data.transforms.base`), so :func:`invert_history`
can map images (``order=3``) and labels (``order=0``) back to original
physical coordinates. Interpolation policy is mode-dependent by construction:
images resample trilinear-equivalent, labels/masks nearest — never mixed.

Determinism: no randomness anywhere in this module; outputs are pure
functions of (payload, config).
"""

from __future__ import annotations

from types import EllipsisType
from typing import Any, Literal

import numpy as np
import torch
from scipy import ndimage

from medfm.core.sample import SpatialMetadata
from medfm.data.errors import TransformError
from medfm.data.transforms.base import (
    InversionMode,
    Transform,
    TransformContext,
    TransformData,
    TransformRecord,
    register_inverter,
)

#: scipy.ndimage.zoom interpolation order for image intensities (trilinear).
IMAGE_ORDER = 3
#: scipy.ndimage.zoom interpolation order for labels/masks (nearest).
LABEL_ORDER = 0

#: Target names treated as label/mask tensors (nearest interpolation, zero
#: re-embed padding) regardless of dtype.
LABEL_TARGET_NAMES = ("mask", "label", "seg", "segmentation")

#: Default percentile used by ForegroundCrop3D when no explicit threshold is
#: configured (deterministic: a fixed quantile of the image intensities).
DEFAULT_FOREGROUND_PERCENTILE = 10.0

_AXIS_FAMILIES = (("R", "L"), ("A", "P"), ("S", "I"))
_AXIS_INDEX = {letter: family for family, letters in enumerate(_AXIS_FAMILIES) for letter in letters}


def _order_for_mode(mode: InversionMode) -> int:
    return LABEL_ORDER if mode == "label" else IMAGE_ORDER


def _require_volume(data: TransformData, name: str) -> SpatialMetadata:
    """Return the spatial metadata of a ``[C, D, H, W]`` payload or raise."""
    if data.image.ndim != 4:
        raise TransformError(f"{name} requires a [C, D, H, W] volume; got shape {tuple(data.image.shape)}")
    if data.spatial is None:
        raise TransformError(
            f"{name} requires SpatialMetadata (affine/spacing/orientation); geometry is never assumed from shape alone"
        )
    if data.spatial.spatial_rank != 3:
        raise TransformError(f"{name} requires rank-3 spatial metadata; got rank {data.spatial.spatial_rank}")
    if data.spatial.slice_positions_mm is not None:
        raise TransformError(
            f"{name} does not support non-uniform slice positions (slice_positions_mm is set); "
            "resample/canonicalize such series offline into a uniform grid first"
        )
    return data.spatial


def _spatial_target_names(data: TransformData) -> list[str]:
    """Names of targets sharing the image's trailing 3 spatial dims."""
    spatial_shape = data.spatial_shape
    return [
        name
        for name, tensor in data.targets.items()
        if tensor.ndim >= 3 and tuple(int(d) for d in tensor.shape[-3:]) == spatial_shape
    ]


def _is_label_like(name: str, tensor: torch.Tensor) -> bool:
    """Label policy: declared label/mask names or any non-floating dtype."""
    return name in LABEL_TARGET_NAMES or not tensor.dtype.is_floating_point


def _orientation_from_affine(affine: torch.Tensor) -> str:
    """Derive the axis orientation code from the dominant affine directions."""
    matrix = affine.detach().to(torch.float64).numpy()[:3, :3]
    letters: list[str] = []
    used: set[int] = set()
    for column in range(3):
        magnitudes = np.abs(matrix[:, column])
        order = np.argsort(-magnitudes, kind="stable")
        row = next(int(r) for r in order if int(r) not in used)
        used.add(row)
        positive = matrix[row, column] > 0
        letters.append(_AXIS_FAMILIES[row][0 if positive else 1])
    return "".join(letters)


def _validate_orientation_code(code: str, what: str) -> str:
    letters = tuple(code.upper())
    if len(letters) != 3 or any(letter not in _AXIS_INDEX for letter in letters):
        raise TransformError(f"{what} must be three letters from R/L/A/P/S/I (one per axis); got {code!r}")
    families = [_AXIS_INDEX[letter] for letter in letters]
    if sorted(families) != [0, 1, 2]:
        raise TransformError(
            f"{what} must cover exactly one direction per anatomical axis family (R/L, A/P, S/I); got {code!r}"
        )
    return "".join(letters)


def _current_orientation(spatial: SpatialMetadata) -> str:
    """Orientation code of the payload: explicit metadata first, affine fallback."""
    if spatial.orientation is not None:
        return _validate_orientation_code(spatial.orientation, "SpatialMetadata.orientation")
    if spatial.anatomical_axes is not None:
        return _validate_orientation_code("".join(spatial.anatomical_axes), "SpatialMetadata.anatomical_axes")
    if spatial.affine is not None:
        return _orientation_from_affine(spatial.affine)
    raise TransformError(
        "cannot determine volume orientation: SpatialMetadata carries neither orientation, "
        "anatomical_axes, nor an affine; orientation is never guessed from shape alone"
    )


def _current_spacing(spatial: SpatialMetadata) -> tuple[float, float, float]:
    """Per-axis spacing in mm: explicit spacing_mm first, affine column norms fallback."""
    if spatial.spacing_mm is not None:
        return (float(spatial.spacing_mm[0]), float(spatial.spacing_mm[1]), float(spatial.spacing_mm[2]))
    if spatial.affine is not None:
        matrix = spatial.affine.detach().to(torch.float64).numpy()[:3, :3]
        norms = np.linalg.norm(matrix, axis=0)
        if bool((norms <= 0).any()):
            raise TransformError("affine has a degenerate (zero-norm) column; cannot derive spacing")
        return (float(norms[0]), float(norms[1]), float(norms[2]))
    raise TransformError("cannot determine voxel spacing: SpatialMetadata carries neither spacing_mm nor an affine")


def _affine_to_nested(affine: torch.Tensor | None) -> list[list[float]] | None:
    if affine is None:
        return None
    return [[float(v) for v in row] for row in affine.detach().to(torch.float64).tolist()]


def _affine_from_nested(payload: list[list[float]]) -> torch.Tensor:
    return torch.as_tensor(payload, dtype=torch.float64)


def _updated_spatial(
    spatial: SpatialMetadata,
    *,
    current_shape: tuple[int, int, int],
    affine: torch.Tensor | None,
    spacing_mm: tuple[float, float, float] | None,
    orientation: str | None,
    anatomical_axes: tuple[str, ...] | None,
) -> SpatialMetadata:
    """Copy SpatialMetadata with updated current geometry (originals preserved)."""
    return SpatialMetadata(
        original_shape=spatial.original_shape,
        current_shape=current_shape,
        affine=affine,
        original_affine=spatial.original_affine,
        spacing_mm=spacing_mm,
        orientation=orientation,
        anatomical_axes=anatomical_axes,
        slice_positions_mm=None,
        frame_of_reference_hash=spatial.frame_of_reference_hash,
    )


def _spatial_flip_dims(ndim: int, axes: list[int]) -> list[int]:
    """Flip dims for spatial ``axes`` of a tensor whose last 3 dims are spatial."""
    return [ndim - 3 + axis for axis in sorted(axes)]


def _permute_spatial(tensor: torch.Tensor, permutation: list[int]) -> torch.Tensor:
    """Permute the trailing 3 spatial dims; ``permutation[t]`` is the source axis of new axis ``t``."""
    leading = list(range(tensor.ndim - 3))
    return tensor.permute(*leading, *[tensor.ndim - 3 + p for p in permutation])


def _invert_permutation(permutation: list[int]) -> list[int]:
    inverse = [0, 0, 0]
    for new_axis, source_axis in enumerate(permutation):
        inverse[source_axis] = new_axis
    return inverse


def _zoom_tensor(tensor: torch.Tensor, zoom_factors: tuple[float, float, float], order: int) -> torch.Tensor:
    """Zoom the trailing 3 spatial dims of ``tensor`` with explicit interpolation order."""
    leading_shape = tuple(int(d) for d in tensor.shape[:-3])
    spatial_shape = tuple(int(d) for d in tensor.shape[-3:])
    flat = tensor.reshape(-1, *spatial_shape)
    output_shape = tuple(max(1, int(round(n * z))) for n, z in zip(spatial_shape, zoom_factors, strict=True))
    zoomed = [
        ndimage.zoom(channel.numpy(), zoom_factors, order=order, mode="nearest", grid_mode=False) for channel in flat
    ]
    for array in zoomed:
        if tuple(int(d) for d in array.shape) != output_shape:
            raise TransformError(
                f"scipy zoom produced shape {tuple(array.shape)} but deterministic rounding expects "
                f"{output_shape}; refusing to proceed with ambiguous geometry"
            )
    stacked = np.stack(zoomed, axis=0)
    result = torch.as_tensor(np.ascontiguousarray(stacked))
    return result.reshape(*leading_shape, *output_shape).to(tensor.dtype)


def _crop_or_pad_to(tensor: torch.Tensor, target_shape: tuple[int, int, int], pad_value: float) -> torch.Tensor:
    """Force the trailing 3 spatial dims to ``target_shape`` by cropping or zero/constant padding."""
    current = tuple(int(d) for d in tensor.shape[-3:])
    slices: tuple[slice, ...] = tuple(slice(0, min(c, t)) for c, t in zip(current, target_shape, strict=True))
    index: tuple[EllipsisType, slice, slice, slice] = (Ellipsis, slices[0], slices[1], slices[2])
    result = tensor[index]
    if tuple(int(d) for d in result.shape[-3:]) == target_shape:
        return result
    output = torch.full((*result.shape[:-3], *target_shape), pad_value, dtype=result.dtype)
    output[index] = result
    return output


class CanonicalizeOrientation(Transform):
    """Reorient a volume to a target orientation code (default ``"RAS"``).

    Computes the axis permutation and per-axis flips that map the current
    orientation (from ``SpatialMetadata.orientation`` / ``anatomical_axes`` /
    the affine, in that precedence) onto ``target``, then applies them to the
    image and every spatial target. Flips and permutations are exact for both
    images and labels (no interpolation), so inversion restores the original
    tensor bit-for-bit.

    ``SpatialMetadata`` is updated: ``affine`` (permuted columns, flip signs,
    flip translations), ``orientation``, ``anatomical_axes``, permuted
    ``spacing_mm``, and ``current_shape``. The record carries the permutation,
    flips, and prior affine/shape/orientation needed to invert exactly.
    """

    name = "canonicalize_orientation"
    stage: Literal["deterministic"] = "deterministic"
    spatial = True

    def __init__(self, target: str = "RAS") -> None:
        self.target = _validate_orientation_code(target, "CanonicalizeOrientation.target")

    def config_dict(self) -> dict[str, Any]:
        return {"target": self.target}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        spatial = _require_volume(data, self.name)
        current = _current_orientation(spatial)
        # permutation[t] = current axis whose direction family matches target axis t.
        permutation = [
            next(axis for axis, letter in enumerate(current) if _AXIS_INDEX[letter] == _AXIS_INDEX[target_letter])
            for target_letter in self.target
        ]
        flips = [current[permutation[t]] != self.target[t] for t in range(3)]

        old_shape = data.spatial_shape
        new_shape = (old_shape[permutation[0]], old_shape[permutation[1]], old_shape[permutation[2]])
        flip_axes = [t for t in range(3) if flips[t]]

        def _reorient(tensor: torch.Tensor) -> torch.Tensor:
            result = _permute_spatial(tensor, permutation)
            if flip_axes:
                result = torch.flip(result, dims=_spatial_flip_dims(result.ndim, flip_axes))
            return result

        image = _reorient(data.image).contiguous()
        targets = dict(data.targets)
        for name in _spatial_target_names(data):
            targets[name] = _reorient(data.targets[name]).contiguous()

        affine = spatial.affine
        if affine is not None:
            basis = torch.zeros((4, 4), dtype=torch.float64)
            for t in range(3):
                basis[permutation[t], t] = -1.0 if flips[t] else 1.0
                if flips[t]:
                    basis[permutation[t], 3] = float(new_shape[t] - 1)
            basis[3, 3] = 1.0
            affine = affine.detach().to(torch.float64) @ basis

        spacing: tuple[float, float, float] | None = None
        if spatial.spacing_mm is not None:
            spacing = (
                float(spatial.spacing_mm[permutation[0]]),
                float(spatial.spacing_mm[permutation[1]]),
                float(spatial.spacing_mm[permutation[2]]),
            )

        data.image = image
        data.targets = targets
        data.spatial = _updated_spatial(
            spatial,
            current_shape=(int(new_shape[0]), int(new_shape[1]), int(new_shape[2])),
            affine=affine,
            spacing_mm=spacing,
            orientation=self.target,
            anatomical_axes=tuple(self.target),
        )
        data.record(
            self.name,
            self.stage,
            {
                "target": self.target,
                "prior_orientation": current,
                "permutation": [int(p) for p in permutation],
                "flips": [bool(f) for f in flips],
                "prior_shape": [int(d) for d in old_shape],
                "prior_affine": _affine_to_nested(spatial.affine),
            },
            spatial=True,
        )
        return data


def _invert_canonicalize_orientation(
    record: TransformRecord, tensor: torch.Tensor, mode: InversionMode
) -> torch.Tensor:
    """Undo flips then the axis permutation; exact for images and labels alike."""
    flips = [bool(f) for f in record.params["flips"]]
    permutation = [int(p) for p in record.params["permutation"]]
    result = tensor
    flip_axes = [t for t in range(3) if flips[t]]
    if flip_axes:
        result = torch.flip(result, dims=_spatial_flip_dims(result.ndim, flip_axes))
    return _permute_spatial(result, _invert_permutation(permutation)).contiguous()


class ResampleToSpacing(Transform):
    """Resample a volume to a target isotropic/anisotropic spacing (mm).

    Zoom factors are computed from the *current* spacing (``spacing_mm``, or
    affine column norms as fallback): ``zoom = current / target`` per axis.
    Images zoom with ``order=3`` (trilinear-equivalent); label/mask targets
    (names in :data:`LABEL_TARGET_NAMES` or non-floating dtypes) zoom with
    ``order=0`` (nearest). Non-spatial targets pass through untouched.

    ``SpatialMetadata`` is updated: ``affine`` columns scaled by
    ``1 / zoom``, ``spacing_mm`` set to the target, ``current_shape`` set to
    the deterministically rounded output shape. The record carries the zoom
    factors, the original shape, and the prior affine/spacing; the inverter
    zooms back with mode-dependent order and then crops/pads to the recorded
    original shape exactly (pad value 0 for labels, ``pad_value`` for images).
    """

    name = "resample_to_spacing"
    stage: Literal["deterministic"] = "deterministic"
    spatial = True

    def __init__(self, spacing_mm: tuple[float, float, float], *, pad_value: float = 0.0) -> None:
        spacing = tuple(float(s) for s in spacing_mm)
        if len(spacing) != 3 or any(s <= 0 for s in spacing):
            raise TransformError(f"ResampleToSpacing.spacing_mm must be three positive floats; got {spacing_mm!r}")
        self.spacing_mm = spacing
        self.pad_value = float(pad_value)

    def config_dict(self) -> dict[str, Any]:
        return {"spacing_mm": list(self.spacing_mm), "pad_value": self.pad_value}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        spatial = _require_volume(data, self.name)
        original_shape = data.spatial_shape
        current_spacing = _current_spacing(spatial)
        zoom_factors: tuple[float, float, float] = (
            current_spacing[0] / self.spacing_mm[0],
            current_spacing[1] / self.spacing_mm[1],
            current_spacing[2] / self.spacing_mm[2],
        )

        image = _zoom_tensor(data.image, zoom_factors, IMAGE_ORDER)
        targets = dict(data.targets)
        for name in _spatial_target_names(data):
            order = LABEL_ORDER if _is_label_like(name, data.targets[name]) else IMAGE_ORDER
            targets[name] = _zoom_tensor(data.targets[name], zoom_factors, order)

        affine = spatial.affine
        if affine is not None:
            scale = torch.diag(torch.as_tensor([1.0 / z for z in zoom_factors] + [1.0], dtype=torch.float64))
            affine = affine.detach().to(torch.float64) @ scale

        new_shape = tuple(int(d) for d in image.shape[-3:])
        data.image = image
        data.targets = targets
        data.spatial = _updated_spatial(
            spatial,
            current_shape=(new_shape[0], new_shape[1], new_shape[2]),
            affine=affine,
            spacing_mm=self.spacing_mm,
            orientation=spatial.orientation,
            anatomical_axes=spatial.anatomical_axes,
        )
        data.record(
            self.name,
            self.stage,
            {
                "target_spacing_mm": list(self.spacing_mm),
                "prior_spacing_mm": list(current_spacing),
                "zoom_factors": [float(z) for z in zoom_factors],
                "original_shape": [int(d) for d in original_shape],
                "prior_affine": _affine_to_nested(spatial.affine),
                "pad_value": self.pad_value,
            },
            spatial=True,
        )
        return data


def _invert_resample_to_spacing(record: TransformRecord, tensor: torch.Tensor, mode: InversionMode) -> torch.Tensor:
    """Zoom back with mode-dependent order, then crop/pad to the recorded shape."""
    zoom_factors = tuple(float(z) for z in record.params["zoom_factors"])
    original_shape = tuple(int(d) for d in record.params["original_shape"])
    pad_value = 0.0 if mode == "label" else float(record.params.get("pad_value", 0.0))
    inverse = (1.0 / zoom_factors[0], 1.0 / zoom_factors[1], 1.0 / zoom_factors[2])
    result = _zoom_tensor(tensor, inverse, _order_for_mode(mode))
    return _crop_or_pad_to(result, (original_shape[0], original_shape[1], original_shape[2]), pad_value)


class ForegroundCrop3D(Transform):
    """Crop to the foreground bounding box with a fixed margin (invertible).

    Foreground voxels are those with intensity strictly above ``threshold``
    in any image channel. When ``threshold`` is ``None`` the threshold is the
    :data:`DEFAULT_FOREGROUND_PERCENTILE`-th percentile of all image
    intensities — deterministic, data-derived, never random. The bounding box
    is padded by ``margin`` voxels on every side, clipped to the volume
    bounds; an empty foreground degenerates to the full extent (a recorded
    no-op, never a silent failure).

    The crop applies to the image and every spatial target. ``SpatialMetadata``
    is updated (``current_shape``, affine translation by the crop origin).
    The record carries the crop origin, the original shape, and the pad
    value; the inverter re-embeds the cropped tensor into a full-size volume
    filled with ``pad_value`` for images and ``0`` for labels — the
    foreground/body crop with invertible coordinates.
    """

    name = "foreground_crop_3d"
    stage: Literal["deterministic"] = "deterministic"
    spatial = True

    def __init__(self, margin: int = 4, threshold: float | None = None, *, pad_value: float = 0.0) -> None:
        if margin < 0:
            raise TransformError(f"ForegroundCrop3D.margin must be non-negative; got {margin}")
        self.margin = int(margin)
        self.threshold = None if threshold is None else float(threshold)
        self.pad_value = float(pad_value)

    def config_dict(self) -> dict[str, Any]:
        return {"margin": self.margin, "threshold": self.threshold, "pad_value": self.pad_value}

    def _foreground_threshold(self, image: torch.Tensor) -> float:
        if self.threshold is not None:
            return self.threshold
        flat = image.detach().to(torch.float32).flatten()
        return float(torch.quantile(flat, DEFAULT_FOREGROUND_PERCENTILE / 100.0))

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        spatial = _require_volume(data, self.name)
        threshold = self._foreground_threshold(data.image)
        foreground = (data.image > threshold).any(dim=0)
        shape = data.spatial_shape
        if bool(foreground.any()):
            coords = foreground.nonzero()
            lo = [max(0, int(coords[:, axis].min()) - self.margin) for axis in range(3)]
            hi = [min(shape[axis], int(coords[:, axis].max()) + self.margin + 1) for axis in range(3)]
        else:
            lo, hi = [0, 0, 0], [int(d) for d in shape]
        box: tuple[slice, slice, slice] = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        crop_index: tuple[EllipsisType, slice, slice, slice] = (Ellipsis, box[0], box[1], box[2])

        target_names = _spatial_target_names(data)  # capture before data.image is reassigned
        data.image = data.image[crop_index].contiguous()
        targets = dict(data.targets)
        for name in target_names:
            targets[name] = data.targets[name][crop_index].contiguous()
        data.targets = targets

        affine = spatial.affine
        if affine is not None:
            shift = torch.eye(4, dtype=torch.float64)
            shift[:3, 3] = torch.as_tensor([float(v) for v in lo], dtype=torch.float64)
            affine = affine.detach().to(torch.float64) @ shift

        new_shape = tuple(int(d) for d in data.image.shape[-3:])
        data.spatial = _updated_spatial(
            spatial,
            current_shape=(new_shape[0], new_shape[1], new_shape[2]),
            affine=affine,
            spacing_mm=(
                (float(spatial.spacing_mm[0]), float(spatial.spacing_mm[1]), float(spatial.spacing_mm[2]))
                if spatial.spacing_mm is not None
                else None
            ),
            orientation=spatial.orientation,
            anatomical_axes=spatial.anatomical_axes,
        )
        data.record(
            self.name,
            self.stage,
            {
                "margin": self.margin,
                "threshold": threshold,
                "origin": [int(v) for v in lo],
                "original_shape": [int(d) for d in shape],
                "prior_affine": _affine_to_nested(spatial.affine),
                "pad_value": self.pad_value,
            },
            spatial=True,
        )
        return data


def _invert_foreground_crop_3d(record: TransformRecord, tensor: torch.Tensor, mode: InversionMode) -> torch.Tensor:
    """Re-embed the cropped tensor at the recorded origin in a full-size volume."""
    origin = [int(v) for v in record.params["origin"]]
    original_shape = tuple(int(d) for d in record.params["original_shape"])
    pad_value = 0.0 if mode == "label" else float(record.params.get("pad_value", 0.0))
    cropped_shape = tuple(int(d) for d in tensor.shape[-3:])
    output = torch.full((*tensor.shape[:-3], *original_shape), pad_value, dtype=tensor.dtype)
    box: tuple[slice, slice, slice] = (
        slice(origin[0], origin[0] + cropped_shape[0]),
        slice(origin[1], origin[1] + cropped_shape[1]),
        slice(origin[2], origin[2] + cropped_shape[2]),
    )
    output[(Ellipsis, box[0], box[1], box[2])] = tensor
    return output


register_inverter(CanonicalizeOrientation.name, _invert_canonicalize_orientation)
register_inverter(ResampleToSpacing.name, _invert_resample_to_spacing)
register_inverter(ForegroundCrop3D.name, _invert_foreground_crop_3d)
