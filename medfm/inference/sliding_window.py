"""Bounded 3D sliding-window inference with Gaussian blending."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from medfm.core.errors import ShapeContractError
from medfm.core.sample import SpatialMetadata


def _window_starts(size: int, window: int, stride: int) -> tuple[int, ...]:
    if size <= 0 or window <= 0 or stride <= 0:
        raise ShapeContractError("volume and window dimensions must be positive")
    if window >= size:
        return (0,)
    starts = list(range(0, size - window + 1, stride))
    last = size - window
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def gaussian_importance_map(
    window_shape: tuple[int, int, int],
    *,
    sigma_scale: float = 0.125,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a positive separable Gaussian map shaped ``[1,1,D,H,W]``."""

    if len(window_shape) != 3 or any(int(value) <= 0 for value in window_shape):
        raise ShapeContractError("window_shape must contain three positive dimensions")
    if not math.isfinite(float(sigma_scale)) or sigma_scale <= 0:
        raise ShapeContractError("sigma_scale must be finite and positive")
    axes: list[torch.Tensor] = []
    for size in window_shape:
        coordinates = torch.arange(int(size), device=device, dtype=torch.float32)
        center = (float(size) - 1.0) / 2.0
        sigma = max(float(size) * sigma_scale, 1e-6)
        axes.append(torch.exp(-0.5 * ((coordinates - center) / sigma) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return weight.clamp_min(1e-6).to(dtype=dtype).unsqueeze(0).unsqueeze(0)


def _predict_window(
    predictor: Callable[..., Any],
    crop: torch.Tensor,
    metadata: Sequence[SpatialMetadata | None] | None,
) -> torch.Tensor:
    try:
        result = predictor(crop, metadata)
    except TypeError:
        result = predictor(crop)
    if isinstance(result, Mapping):
        for key in ("logits", "prediction", "predictions", "mask", "output"):
            if key in result:
                result = result[key]
                break
    if not isinstance(result, torch.Tensor):
        raise ShapeContractError("sliding-window predictor must return a tensor or mapping containing one")
    return result


def sliding_window_inference(
    volume: torch.Tensor,
    predictor: Callable[..., Any],
    *,
    window_shape: tuple[int, int, int],
    overlap: float = 0.25,
    sigma_scale: float = 0.125,
    sw_batch_size: int = 1,
    metadata: Sequence[SpatialMetadata | None] | None = None,
) -> torch.Tensor:
    """Infer a ``[B,C,D,H,W]`` volume in bounded windows.

    Windows larger than a volume dimension are zero-padded before prediction
    and cropped back.  Gaussian importance weights reduce seams while keeping
    every voxel covered.  Only one window batch and one output accumulator are
    retained, so memory is bounded by the configured ``sw_batch_size`` and
    output volume.
    """

    if not isinstance(volume, torch.Tensor) or volume.ndim != 5:
        raise ShapeContractError("sliding_window_inference expects a [B,C,D,H,W] tensor")
    if len(window_shape) != 3 or any(int(value) <= 0 for value in window_shape):
        raise ShapeContractError("window_shape must contain three positive dimensions")
    if not 0 <= float(overlap) < 1:
        raise ShapeContractError("overlap must be in [0, 1)")
    if int(sw_batch_size) <= 0:
        raise ShapeContractError("sw_batch_size must be positive")
    batch, _, depth, height, width = (int(value) for value in volume.shape)
    if metadata is not None and len(metadata) != batch:
        raise ShapeContractError("metadata length must equal volume batch size")
    metas = list(metadata) if metadata is not None else [None] * batch
    strides = tuple(max(1, int(window * (1.0 - float(overlap)))) for window in window_shape)
    starts = tuple(
        _window_starts(size, int(window), stride)
        for size, window, stride in zip((depth, height, width), window_shape, strides, strict=True)
    )
    locations = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    output: torch.Tensor | None = None
    weight_sum: torch.Tensor | None = None
    importance = gaussian_importance_map(window_shape, sigma_scale=sigma_scale, device=volume.device)

    for offset in range(0, len(locations), int(sw_batch_size)):
        chunk = locations[offset : offset + int(sw_batch_size)]
        crops: list[torch.Tensor] = []
        crop_specs: list[tuple[int, int, int, int, int, int]] = []
        for z, y, x in chunk:
            z1, y1, x1 = (
                min(z + window_shape[0], depth),
                min(y + window_shape[1], height),
                min(x + window_shape[2], width),
            )
            crop = volume[:, :, z:z1, y:y1, x:x1]
            pad = (0, window_shape[2] - (x1 - x), 0, window_shape[1] - (y1 - y), 0, window_shape[0] - (z1 - z))
            if any(pad):
                crop = torch.nn.functional.pad(crop, pad)
            crops.append(crop)
            crop_specs.append((z, z1, y, y1, x, x1))
        # Flatten [windows, B, C, ...] to one bounded predictor batch.  This
        # avoids unbounded Python/device state while preserving sample order.
        stacked = torch.cat(crops, dim=0)
        prediction = _predict_window(predictor, stacked, metas)
        if prediction.ndim != 5 or int(prediction.shape[0]) != len(chunk) * batch:
            raise ShapeContractError("sliding-window predictor must return [windows*B,K,D,H,W]")
        if tuple(int(value) for value in prediction.shape[-3:]) != tuple(int(value) for value in window_shape):
            raise ShapeContractError("sliding-window predictor must return the declared window spatial shape")
        channels = int(prediction.shape[1])
        if output is None:
            output = torch.zeros(
                (batch, channels, depth, height, width), device=prediction.device, dtype=prediction.dtype
            )
            weight_sum = torch.zeros((batch, 1, depth, height, width), device=prediction.device, dtype=torch.float32)
        assert weight_sum is not None
        prediction = prediction.reshape(len(chunk), batch, channels, *window_shape)
        for index, (z, z1, y, y1, x, x1) in enumerate(crop_specs):
            d, h, w = z1 - z, y1 - y, x1 - x
            local_weight = importance[:, :, :d, :h, :w].to(device=prediction.device)
            output[:, :, z:z1, y:y1, x:x1] += prediction[index, :, :, :d, :h, :w] * local_weight
            weight_sum[:, :, z:z1, y:y1, x:x1] += local_weight
    if output is None or weight_sum is None:
        raise ShapeContractError("sliding-window inference generated no windows")
    return output / weight_sum.to(dtype=output.dtype).clamp_min(torch.finfo(output.dtype).eps)


__all__ = ["gaussian_importance_map", "sliding_window_inference"]
