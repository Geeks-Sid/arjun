"""2D radiology transforms: grayscale canonicalization, geometry, augmentation.

Scope: single-frame 2D radiographs (e.g. chest X-ray) already decoded to
host-resident float tensors of shape ``[C, H, W]``. All transforms in this
module reject 3D volumes and non-CPU payloads — volume canonicalization lives
in the CT/MRI modules, and tensors transfer to accelerators only after
collation.

Deterministic stage (cacheable):

- :class:`DecodeGrayscale` — MONOCHROME1 → MONOCHROME2 inversion correction.
- :class:`LetterboxResize` — aspect-preserving resize + symmetric padding,
  exactly invertible back to the original ``H x W``.
- :class:`BodyRegionCrop` — foreground crop with re-embedding inversion.
- :class:`ToChannels`, :class:`NormalizeImage`, :class:`RescaleIntensity`.

Stochastic stage (augmentation, draws only from ``ctx.rng``):

- :class:`RandomRotate2D`, :class:`RandomTranslate2D`, :class:`RandomScale2D`,
  :class:`RandomIntensityShift`, :class:`RandomGaussianNoise`,
  :class:`RandomFlip2D` — horizontal flips are opt-in; vertical flips are
  disabled by default because they are not anatomy-preserving for most
  radiographs.

Policy: natural-image color jitter is deliberately *not* implemented here —
grayscale radiology pipelines exclude it from defaults (see
``implementation_plan/phase_04_preprocessing_and_collators.md``).

Metadata policy: no transform in this module reads or writes
``data.metadata`` — view position, multi-view ordering, and longitudinal
indices pass through untouched. Spatial transforms update
``data.spatial.current_shape`` when present and record enough geometry in
their :class:`TransformRecord` for :func:`invert_history` to reconstruct
original coordinates (nearest interpolation for ``mode="label"``, bilinear
for ``mode="image"``). Stochastic augmentations register no inverters —
their history records are audit metadata, not inversion sources.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from medfm.data.errors import TransformError
from medfm.data.transforms.base import (
    InversionMode,
    Transform,
    TransformContext,
    TransformData,
    TransformRecord,
    register_inverter,
)
from medfm.data.transforms.specs import NormalizationSpec

#: Photometric interpretations this module can canonicalize.
_SUPPORTED_PHOTOMETRIC = ("MONOCHROME1", "MONOCHROME2")


def _check_2d(data: TransformData, name: str) -> torch.Tensor:
    """Return the image, requiring a host ``[C, H, W]`` float32 tensor."""
    image = data.image
    if image.ndim != 3:
        raise TransformError(
            f"{name} operates on 2D images of shape [C, H, W]; got shape {tuple(image.shape)}. "
            "Volumes are canonicalized by the CT/MRI transform modules."
        )
    if image.dtype != torch.float32:
        raise TransformError(f"{name} expects float32 images; got {image.dtype}")
    return image


def _update_spatial_shape(data: TransformData, new_shape: tuple[int, ...]) -> None:
    """Keep ``data.spatial.current_shape`` consistent after a spatial transform."""
    if data.spatial is not None:
        data.spatial = replace(data.spatial, current_shape=tuple(int(d) for d in new_shape))


class DecodeGrayscale(Transform):
    """Canonicalize a single-channel grayscale image to MONOCHROME2 convention.

    MONOCHROME1 pixels are white-is-low (bone bright in display space); the
    canonical pipeline convention is MONOCHROME2 (higher pixel value = higher
    intensity), so MONOCHROME1 input is corrected as ``max - image``.
    MONOCHROME2 input passes through unchanged. Multi-channel (e.g. RGB)
    input is rejected — channel conversion is an explicit later step.
    """

    name = "decode_grayscale"
    stage = "deterministic"

    def __init__(self, photometric_interpretation: str) -> None:
        interpretation = photometric_interpretation.strip().upper()
        if interpretation not in _SUPPORTED_PHOTOMETRIC:
            raise TransformError(
                f"DecodeGrayscale supports {_SUPPORTED_PHOTOMETRIC}; got {photometric_interpretation!r}. "
                "Convert other photometric interpretations (e.g. RGB, PALETTE COLOR) before this transform."
            )
        self.photometric_interpretation = interpretation

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        if image.shape[0] != 1:
            raise TransformError(
                f"DecodeGrayscale expects a single-channel image; got {image.shape[0]} channels. "
                "DecodeGrayscale never infers a grayscale projection from multi-channel data."
            )
        params: dict[str, Any] = self.config_dict()
        if self.photometric_interpretation == "MONOCHROME1":
            maximum = float(image.max())
            data.image = maximum - image
            params["inversion_applied"] = True
            params["source_max"] = maximum
        else:
            params["inversion_applied"] = False
        data.record(self.name, self.stage, params)
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"photometric_interpretation": self.photometric_interpretation}


def _invert_letterbox(record: TransformRecord, tensor: torch.Tensor, mode: InversionMode) -> torch.Tensor:
    """Undo letterboxing: crop the symmetric padding, then resize back."""
    params = record.params
    pad_top, pad_left = int(params["pad_top"]), int(params["pad_left"])
    new_h, new_w = (int(v) for v in params["content_size"])
    orig_h, orig_w = (int(v) for v in params["original_size"])
    cropped = tensor[..., pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    if (new_h, new_w) == (orig_h, orig_w):
        return cropped
    batched = cropped.unsqueeze(0)
    if mode == "label":
        resized = F.interpolate(batched, size=(orig_h, orig_w), mode="nearest")
    else:
        resized = F.interpolate(batched, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


class LetterboxResize(Transform):
    """Aspect-preserving resize to fit ``size``, then symmetric zero padding.

    The image is scaled by ``min(H/h, W/w)``, so content never distorts; the
    remaining rows/columns are padded with ``pad_value``, split as evenly as
    possible between both sides (the extra odd pixel goes to the
    bottom/right). The record carries the scale, content size, padding
    offsets, and original size; :func:`invert_history` maps tensors back to
    the original ``H x W`` (nearest for labels, bilinear for images — the
    shape roundtrip is exact, intensities approximate for downscaled
    images).
    """

    name = "letterbox_resize"
    stage = "deterministic"
    spatial = True

    def __init__(self, size: tuple[int, int], pad_value: float = 0.0) -> None:
        target_h, target_w = (int(d) for d in size)
        if target_h <= 0 or target_w <= 0:
            raise TransformError(f"LetterboxResize size must be two positive ints; got {size}")
        self.size = (target_h, target_w)
        self.pad_value = float(pad_value)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        orig_h, orig_w = int(image.shape[1]), int(image.shape[2])
        target_h, target_w = self.size
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = max(1, min(target_h, int(round(orig_h * scale))))
        new_w = max(1, min(target_w, int(round(orig_w * scale))))
        if (new_h, new_w) != (orig_h, orig_w):
            image = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_right = target_w - new_w - pad_left
        if pad_top or pad_bottom or pad_left or pad_right:
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=self.pad_value)
        data.image = image
        _update_spatial_shape(data, (target_h, target_w))
        data.record(
            self.name,
            self.stage,
            {
                **self.config_dict(),
                "scale": scale,
                "original_size": [orig_h, orig_w],
                "content_size": [new_h, new_w],
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            },
            spatial=True,
        )
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"size": list(self.size), "pad_value": self.pad_value}


def _invert_body_region_crop(record: TransformRecord, tensor: torch.Tensor, mode: InversionMode) -> torch.Tensor:  # noqa: ARG001
    """Re-embed the cropped tensor into a zero canvas of the original size."""
    top, left, bottom, right = (int(v) for v in record.params["crop_box"])
    orig_h, orig_w = (int(v) for v in record.params["original_size"])
    canvas = tensor.new_zeros((*tensor.shape[:-2], orig_h, orig_w))
    canvas[..., top:bottom, left:right] = tensor[..., : bottom - top, : right - left]
    return canvas


class BodyRegionCrop(Transform):
    """Crop to the non-background body region plus a margin.

    The foreground mask is ``image > threshold``; when ``threshold`` is
    ``None`` a deterministic midpoint threshold ``(min + max) / 2`` of the
    image's own intensity range is used (sufficient for radiographs with a
    dark surround, and a pure function of the payload). The bounding box is
    expanded by ``margin`` pixels on every side and clamped to the image. An
    empty foreground mask (or a box equal to the full image) is a recorded
    no-op. The inverter re-embeds the crop into a zero canvas at the
    recorded box, so crop → invert restores the original image exactly.
    """

    name = "body_region_crop"
    stage = "deterministic"
    spatial = True

    def __init__(self, margin: int = 8, threshold: float | None = None) -> None:
        if margin < 0:
            raise TransformError(f"BodyRegionCrop margin must be non-negative; got {margin}")
        self.margin = int(margin)
        self.threshold = None if threshold is None else float(threshold)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        orig_h, orig_w = int(image.shape[1]), int(image.shape[2])
        intensity = image.amax(dim=0)  # [H, W]: brightest channel wins
        if self.threshold is None:
            threshold = (float(intensity.min()) + float(intensity.max())) / 2.0
        else:
            threshold = self.threshold
        foreground = intensity > threshold
        rows = foreground.any(dim=1).nonzero()
        cols = foreground.any(dim=0).nonzero()
        if rows.numel() == 0 or cols.numel() == 0:
            top, left, bottom, right = 0, 0, orig_h, orig_w
        else:
            top = max(0, int(rows.min()) - self.margin)
            left = max(0, int(cols.min()) - self.margin)
            bottom = min(orig_h, int(rows.max()) + 1 + self.margin)
            right = min(orig_w, int(cols.max()) + 1 + self.margin)
        data.image = image[:, top:bottom, left:right]
        _update_spatial_shape(data, (bottom - top, right - left))
        data.record(
            self.name,
            self.stage,
            {
                **self.config_dict(),
                "effective_threshold": threshold,
                "original_size": [orig_h, orig_w],
                "crop_box": [top, left, bottom, right],
            },
            spatial=True,
        )
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"margin": self.margin, "threshold": self.threshold}


class ToChannels(Transform):
    """Set the channel count: single-channel passthrough or repeat to three.

    Three-channel output repeats the grayscale channel (the standard recipe
    for adapters pretrained on natural images). Any other conversion —
    multi-channel input, or channel reduction — is rejected rather than
    silently averaged.
    """

    name = "to_channels"
    stage = "deterministic"

    def __init__(self, channels: int) -> None:
        if channels not in (1, 3):
            raise TransformError(f"ToChannels supports 1 or 3 channels; got {channels}")
        self.channels = int(channels)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        source_channels = int(image.shape[0])
        if source_channels == self.channels:
            pass
        elif self.channels == 3 and source_channels == 1:
            data.image = image.repeat(3, 1, 1)
        else:
            raise TransformError(
                f"ToChannels({self.channels}) cannot convert a {source_channels}-channel image; "
                "only 1 -> 3 (repeat) and identity passthrough are supported"
            )
        data.record(self.name, self.stage, {**self.config_dict(), "source_channels": source_channels})
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"channels": self.channels}


class NormalizeImage(Transform):
    """Per-channel z-score normalization ``(x - mean) / std``.

    The normalization statistics come from the model adapter's
    :class:`~medfm.data.transforms.specs.NormalizationSpec` — they are
    model-specific configuration, never computed from the sample.
    """

    name = "normalize_image"
    stage = "deterministic"

    def __init__(self, normalization: NormalizationSpec) -> None:
        self.normalization = normalization

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        if image.shape[0] != self.normalization.channels:
            raise TransformError(
                f"NormalizeImage has statistics for {self.normalization.channels} channels but the image "
                f"has {image.shape[0]}; run ToChannels first"
            )
        mean = torch.tensor(self.normalization.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.normalization.std, dtype=torch.float32).view(-1, 1, 1)
        data.image = (image - mean) / std
        data.record(self.name, self.stage, self.config_dict())
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"normalization": self.normalization.to_dict()}


class RescaleIntensity(Transform):
    """Deterministic intensity rescale into ``out_range`` (default ``[0, 1]``).

    By default the scale anchors are the image's own min/max; with
    ``percentiles=(p_low, p_high)`` the anchors are those quantiles of the
    flattened image and the result is clamped to ``out_range`` (robust to
    outlier pixels). Anchors are computed over the whole image, never per
    channel, so multi-channel stacks keep a common scale. A constant image
    maps to ``out_range[0]`` rather than dividing by zero.
    """

    name = "rescale_intensity"
    stage = "deterministic"

    def __init__(
        self,
        out_range: tuple[float, float] = (0.0, 1.0),
        percentiles: tuple[float, float] | None = None,
    ) -> None:
        low, high = (float(v) for v in out_range)
        if not low < high:
            raise TransformError(f"RescaleIntensity out_range must be (low < high); got {out_range}")
        if percentiles is not None:
            p_low, p_high = (float(p) for p in percentiles)
            if not (0.0 <= p_low < p_high <= 100.0):
                raise TransformError(
                    f"RescaleIntensity percentiles must satisfy 0 <= p_low < p_high <= 100; got {percentiles}"
                )
            percentiles = (p_low, p_high)
        self.out_range = (low, high)
        self.percentiles = percentiles

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:  # noqa: ARG002
        image = _check_2d(data, self.name)
        if self.percentiles is None:
            anchor_low, anchor_high = float(image.min()), float(image.max())
        else:
            flat = image.flatten()
            anchor_low = float(torch.quantile(flat, self.percentiles[0] / 100.0))
            anchor_high = float(torch.quantile(flat, self.percentiles[1] / 100.0))
        out_low, out_high = self.out_range
        if anchor_high <= anchor_low:
            rescaled = torch.full_like(image, out_low)
        else:
            rescaled = (image - anchor_low) / (anchor_high - anchor_low)
            rescaled = rescaled * (out_high - out_low) + out_low
            if self.percentiles is not None:
                rescaled = rescaled.clamp(out_low, out_high)
        data.image = rescaled
        data.record(
            self.name,
            self.stage,
            {**self.config_dict(), "anchor_low": anchor_low, "anchor_high": anchor_high},
        )
        return data

    def config_dict(self) -> dict[str, Any]:
        return {
            "out_range": list(self.out_range),
            "percentiles": list(self.percentiles) if self.percentiles is not None else None,
        }


def _draw_uniform(ctx: TransformContext, low: float, high: float) -> float:
    """One uniform draw in ``[low, high]`` from the context generator."""
    return low + (high - low) * float(torch.rand((), generator=ctx.rng))


def _require_ctx(ctx: TransformContext | None, name: str) -> TransformContext:
    if ctx is None:
        raise TransformError(f"stochastic transform {name!r} requires a seeded TransformContext")
    return ctx


def _affine_resample(image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Warp ``[C, H, W]`` by a 2x3 affine (output-to-input sampling grid)."""
    batched = image.unsqueeze(0)
    size = [1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
    grid = F.affine_grid(theta.unsqueeze(0), size=size, align_corners=False)
    return F.grid_sample(batched, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)


class RandomRotate2D(Transform):
    """In-plane rotation by a uniform angle in ``[-max_degrees, max_degrees]``.

    Conservative by configuration (small ``max_degrees``); the center of
    rotation is the image center and exposed corners are zero-filled.
    Records the drawn angle for audit; no inverter is registered — exact
    inversion of a resampled rotation is lossy.
    """

    name = "random_rotate_2d"
    stage = "stochastic"
    spatial = True

    def __init__(self, max_degrees: float) -> None:
        if max_degrees < 0:
            raise TransformError(f"RandomRotate2D max_degrees must be non-negative; got {max_degrees}")
        self.max_degrees = float(max_degrees)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        angle_degrees = _draw_uniform(context, -self.max_degrees, self.max_degrees)
        radians = math.radians(angle_degrees)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        theta = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32)
        data.image = _affine_resample(image, theta)
        data.record(self.name, self.stage, {**self.config_dict(), "angle_degrees": angle_degrees}, spatial=True)
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"max_degrees": self.max_degrees}


class RandomTranslate2D(Transform):
    """Translation by fractions of (H, W), each uniform in ``[-max_fraction, max_fraction]``.

    Exposed borders are zero-filled. Records the drawn pixel shifts; no
    inverter is registered (border content is lost by construction).
    """

    name = "random_translate_2d"
    stage = "stochastic"
    spatial = True

    def __init__(self, max_fraction: float) -> None:
        if not 0.0 <= max_fraction <= 1.0:
            raise TransformError(f"RandomTranslate2D max_fraction must be in [0, 1]; got {max_fraction}")
        self.max_fraction = float(max_fraction)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        height, width = int(image.shape[1]), int(image.shape[2])
        shift_h = _draw_uniform(context, -self.max_fraction, self.max_fraction) * height
        shift_w = _draw_uniform(context, -self.max_fraction, self.max_fraction) * width
        # affine_grid works in normalized [-1, 1] coordinates: 2 px-of-range per axis.
        theta = torch.tensor(
            [[1.0, 0.0, -2.0 * shift_w / width], [0.0, 1.0, -2.0 * shift_h / height]],
            dtype=torch.float32,
        )
        data.image = _affine_resample(image, theta)
        data.record(
            self.name,
            self.stage,
            {**self.config_dict(), "shift_h_px": shift_h, "shift_w_px": shift_w},
            spatial=True,
        )
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"max_fraction": self.max_fraction}


class RandomScale2D(Transform):
    """Isotropic zoom around the image center by a factor uniform in ``scale_range``.

    Factors above 1 zoom in (cropping the field of view); below 1 zoom out
    with zero-filled borders. Records the drawn factor; no inverter is
    registered.
    """

    name = "random_scale_2d"
    stage = "stochastic"
    spatial = True

    def __init__(self, scale_range: tuple[float, float]) -> None:
        low, high = (float(v) for v in scale_range)
        if not 0.0 < low <= high:
            raise TransformError(f"RandomScale2D scale_range must satisfy 0 < low <= high; got {scale_range}")
        self.scale_range = (low, high)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        factor = _draw_uniform(context, self.scale_range[0], self.scale_range[1])
        # The grid maps output -> input coordinates, so divide by the visual zoom factor.
        theta = torch.tensor([[1.0 / factor, 0.0, 0.0], [0.0, 1.0 / factor, 0.0]], dtype=torch.float32)
        data.image = _affine_resample(image, theta)
        data.record(self.name, self.stage, {**self.config_dict(), "scale_factor": factor}, spatial=True)
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"scale_range": list(self.scale_range)}


class RandomIntensityShift(Transform):
    """Random affine intensity transform ``x * scale + shift``.

    Draws ``shift`` uniformly from ``shift_range`` and ``scale`` uniformly
    from ``scale_range``. Applied to all channels with the same draw so
    repeated-channel stacks stay identical across channels.
    """

    name = "random_intensity_shift"
    stage = "stochastic"

    def __init__(self, shift_range: tuple[float, float], scale_range: tuple[float, float] = (1.0, 1.0)) -> None:
        shift_low, shift_high = (float(v) for v in shift_range)
        scale_low, scale_high = (float(v) for v in scale_range)
        if shift_low > shift_high:
            raise TransformError(f"RandomIntensityShift shift_range must satisfy low <= high; got {shift_range}")
        if not 0.0 <= scale_low <= scale_high:
            raise TransformError(f"RandomIntensityShift scale_range must satisfy 0 <= low <= high; got {scale_range}")
        self.shift_range = (shift_low, shift_high)
        self.scale_range = (scale_low, scale_high)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        shift = _draw_uniform(context, self.shift_range[0], self.shift_range[1])
        scale = _draw_uniform(context, self.scale_range[0], self.scale_range[1])
        data.image = image * scale + shift
        data.record(self.name, self.stage, {**self.config_dict(), "shift": shift, "scale": scale})
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"shift_range": list(self.shift_range), "scale_range": list(self.scale_range)}


class RandomGaussianNoise(Transform):
    """Additive i.i.d. Gaussian noise with std drawn uniformly from ``std_range``."""

    name = "random_gaussian_noise"
    stage = "stochastic"

    def __init__(self, std_range: tuple[float, float]) -> None:
        low, high = (float(v) for v in std_range)
        if not 0.0 <= low <= high:
            raise TransformError(f"RandomGaussianNoise std_range must satisfy 0 <= low <= high; got {std_range}")
        self.std_range = (low, high)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        std = _draw_uniform(context, self.std_range[0], self.std_range[1])
        noise = torch.randn(image.shape, generator=context.rng) * std
        data.image = image + noise
        data.record(self.name, self.stage, {**self.config_dict(), "std": std})
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"std_range": list(self.std_range)}


class RandomFlip2D(Transform):
    """Random axis flips, gated by explicit allow-lists.

    Horizontal flips happen only when ``allow_horizontal`` is set (a task
    decision — e.g. chest X-ray laterality matters); vertical flips default
    to off because they are not anatomy-preserving for most radiographs.
    Each allowed axis flips independently with probability ``p``. A fully
    gated transform is a no-op but still records, keeping history shape
    stable across task configurations. No inverter is registered.
    """

    name = "random_flip_2d"
    stage = "stochastic"
    spatial = True

    def __init__(self, allow_horizontal: bool, allow_vertical: bool = False, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise TransformError(f"RandomFlip2D p must be in [0, 1]; got {p}")
        self.allow_horizontal = bool(allow_horizontal)
        self.allow_vertical = bool(allow_vertical)
        self.p = float(p)

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        context = _require_ctx(ctx, self.name)
        image = _check_2d(data, self.name)
        flipped_horizontal = False
        flipped_vertical = False
        if self.allow_horizontal:
            flipped_horizontal = bool(torch.rand((), generator=context.rng) < self.p)
            if flipped_horizontal:
                image = torch.flip(image, dims=(-1,))
        if self.allow_vertical:
            flipped_vertical = bool(torch.rand((), generator=context.rng) < self.p)
            if flipped_vertical:
                image = torch.flip(image, dims=(-2,))
        data.image = image
        data.record(
            self.name,
            self.stage,
            {
                **self.config_dict(),
                "flipped_horizontal": flipped_horizontal,
                "flipped_vertical": flipped_vertical,
            },
            spatial=True,
        )
        return data

    def config_dict(self) -> dict[str, Any]:
        return {"allow_horizontal": self.allow_horizontal, "allow_vertical": self.allow_vertical, "p": self.p}


register_inverter(LetterboxResize.name, _invert_letterbox)
register_inverter(BodyRegionCrop.name, _invert_body_region_crop)

__all__ = [
    "BodyRegionCrop",
    "DecodeGrayscale",
    "LetterboxResize",
    "NormalizeImage",
    "RandomFlip2D",
    "RandomGaussianNoise",
    "RandomIntensityShift",
    "RandomRotate2D",
    "RandomScale2D",
    "RandomTranslate2D",
    "RescaleIntensity",
    "ToChannels",
]
