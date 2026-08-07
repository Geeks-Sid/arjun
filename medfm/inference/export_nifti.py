"""NIfTI mask export and original-coordinate validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medfm.core.sample import SpatialMetadata
from medfm.data.transforms.base import TransformRecord, invert_history
from medfm.inference.errors import OptionalDependencyError, RequestValidationError


def restore_mask_to_original(
    mask: torch.Tensor,
    metadata: SpatialMetadata,
    *,
    history: Sequence[TransformRecord] | None = None,
) -> torch.Tensor:
    """Invert recorded transforms and/or restore the declared original shape."""

    if not isinstance(mask, torch.Tensor):
        raise RequestValidationError(details={"field": "mask"})
    restored = mask
    if history:
        restored = invert_history(list(history), restored, mode="label", strict=True)
    target_shape = tuple(int(value) for value in metadata.original_shape)
    current_shape = tuple(int(value) for value in metadata.current_shape)
    if restored.ndim < len(target_shape):
        raise RequestValidationError(details={"field": "mask", "reason": "rank does not match spatial metadata"})
    spatial = tuple(int(value) for value in restored.shape[-len(target_shape) :])
    if spatial != target_shape:
        if spatial != current_shape and history is None:
            raise RequestValidationError(details={"field": "mask", "reason": "mask shape does not match metadata"})
        # A history inverter should already produce original dimensions.  When
        # no history is available, exact zero-padding/cropping is safer than an
        # unrecorded interpolation and preserves label semantics.
        result = torch.zeros(
            (*restored.shape[: -len(target_shape)], *target_shape), dtype=restored.dtype, device=restored.device
        )
        slices_src: list[slice] = []
        slices_dst: list[slice] = []
        for source, target in zip(spatial, target_shape, strict=True):
            size = min(source, target)
            slices_src.append(slice(0, size))
            slices_dst.append(slice(0, size))
        result[(..., *slices_dst)] = restored[(..., *slices_src)]
        restored = result
    return restored


def _affine_for(metadata: SpatialMetadata) -> np.ndarray:
    affine = metadata.original_affine if metadata.original_affine is not None else metadata.affine
    if affine is None:
        return np.eye(4, dtype=np.float32)
    matrix = affine.detach().to("cpu").numpy().astype(np.float32, copy=False)
    if matrix.shape != (4, 4):
        raise RequestValidationError(details={"field": "metadata.affine", "reason": "expected 4x4 affine"})
    return matrix


def export_nifti(
    mask: torch.Tensor,
    path: str | Path,
    *,
    metadata: SpatialMetadata,
    history: Sequence[TransformRecord] | None = None,
    dtype: np.dtype[Any] | type[np.generic] = np.uint8,
) -> Path:
    """Write a mask with original affine and reopen it to verify geometry."""

    try:
        import nibabel as nib
    except ImportError as exc:
        raise OptionalDependencyError(details={"extra": "medical", "format": "NIfTI"}) from exc
    restored = restore_mask_to_original(mask, metadata, history=history)
    spatial_rank = len(metadata.original_shape)
    array = restored.detach().to("cpu")
    while array.ndim > spatial_rank:
        if int(array.shape[0]) != 1:
            raise RequestValidationError(
                details={"field": "mask", "reason": "NIfTI export requires singleton batch/channel axes"}
            )
        array = array[0]
    if tuple(int(value) for value in array.shape) != tuple(metadata.original_shape):
        raise RequestValidationError(
            details={"field": "mask", "reason": "NIfTI mask shape differs from source geometry"}
        )
    array_np = array.numpy().astype(dtype, copy=False)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array_np, _affine_for(metadata))
    nib.save(image, str(output))
    reopened = nib.load(str(output))
    if tuple(int(value) for value in reopened.shape) != tuple(metadata.original_shape):
        raise RequestValidationError(details={"field": "output", "reason": "reopened NIfTI shape differs from source"})
    if not np.allclose(np.asarray(reopened.affine), _affine_for(metadata), atol=1e-5):
        raise RequestValidationError(details={"field": "output", "reason": "reopened NIfTI affine differs from source"})
    return output


def reopen_and_validate_nifti(path: str | Path, metadata: SpatialMetadata) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise OptionalDependencyError(details={"extra": "medical", "format": "NIfTI"}) from exc
    image = nib.load(str(path))
    expected_shape = tuple(metadata.original_shape)
    if tuple(int(value) for value in image.shape[-len(expected_shape) :]) != expected_shape:
        raise RequestValidationError(details={"field": "path", "reason": "NIfTI shape does not match source geometry"})
    if not np.allclose(np.asarray(image.affine), _affine_for(metadata), atol=1e-5):
        raise RequestValidationError(details={"field": "path", "reason": "NIfTI affine does not match source geometry"})


__all__ = ["export_nifti", "reopen_and_validate_nifti", "restore_mask_to_original"]
