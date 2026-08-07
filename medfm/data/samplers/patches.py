"""Deterministic 3D patch sampling with explicit metadata (Phase 04).

Contract:

- **Explicit randomness**: every stochastic sampler draws exclusively from a
  ``torch.Generator`` built from its ``seed`` (or a caller-supplied
  generator). Python's global RNG and unseeded ``torch.rand*`` calls are
  never used, so two samplers built with the same seed yield identical
  origin sequences.
- **Full-shape patches**: origins are clamped so every extracted patch has
  exactly ``patch_shape``. Patches extending past the volume (or volumes
  smaller than the patch) are padded with ``pad_value``, and the padding is
  recorded explicitly on :class:`PatchInfo` (``padded``/``padding``).
- **Measurable positivity**: every :class:`PatchInfo` records
  ``target_positive`` (the extracted patch overlaps mask foreground) and
  ``sampling_probability`` (the probability mass the drawn patch type had
  under the sampler configuration), so the empirical positive-patch
  proportion can be measured against the configured one.
- **Physical geometry**: when ``spacing_mm`` is supplied,
  ``physical_min_mm``/``physical_max_mm`` give the patch bounding box in
  patient millimetres; otherwise both are ``None``.

All samplers operate on host CPU tensors: images are ``[C, D, H, W]`` and
masks/labels are ``[D, H, W]`` matching the image spatial shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import scipy.ndimage
import torch

from medfm.data.errors import PatchSamplingError
from medfm.data.transforms.base import make_generator

#: A seed accepted by every stochastic sampler: an integer seed or a
#: caller-owned ``torch.Generator``.
SeedLike = int | torch.Generator


def _validate_patch_shape(patch_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    dims = tuple(int(d) for d in patch_shape)
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise PatchSamplingError(f"patch_shape must be three positive ints (D, H, W); got {patch_shape!r}")
    return (dims[0], dims[1], dims[2])


def _validate_volume_shape(volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    dims = tuple(int(d) for d in volume_shape)
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise PatchSamplingError(f"volume shape must be three positive ints (D, H, W); got {volume_shape!r}")
    return (dims[0], dims[1], dims[2])


def _validate_spacing(spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    spacing = tuple(float(s) for s in spacing_mm)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise PatchSamplingError(f"spacing_mm must be three positive floats; got {spacing_mm!r}")
    return (spacing[0], spacing[1], spacing[2])


def _validate_inputs(image: torch.Tensor, mask: torch.Tensor | None) -> tuple[int, int, int]:
    if image.ndim != 4:
        raise PatchSamplingError(f"image must be [C, D, H, W]; got shape {tuple(image.shape)}")
    volume_shape = _validate_volume_shape((int(image.shape[1]), int(image.shape[2]), int(image.shape[3])))
    if mask is not None and (mask.ndim != 3 or tuple(int(d) for d in mask.shape) != volume_shape):
        raise PatchSamplingError(
            f"mask must be [D, H, W] matching the image spatial shape {volume_shape}; got {tuple(mask.shape)}"
        )
    return volume_shape


def _clamp_origin(
    origin: tuple[int, int, int], patch_shape: tuple[int, int, int], volume_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Clamp ``origin`` so the patch stays inside the volume where possible.

    Axes where the volume is smaller than the patch clamp to 0; the shortfall
    is reported as padding by :func:`_padding_amounts`.
    """
    clamped = []
    for axis in range(3):
        hi = volume_shape[axis] - patch_shape[axis]
        clamped.append(0 if hi < 0 else min(max(origin[axis], 0), hi))
    return (clamped[0], clamped[1], clamped[2])


def _padding_amounts(
    origin: tuple[int, int, int], patch_shape: tuple[int, int, int], volume_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Per-axis trailing pad needed for the patch to reach ``patch_shape``."""
    amounts = [max(0, origin[axis] + patch_shape[axis] - volume_shape[axis]) for axis in range(3)]
    return (amounts[0], amounts[1], amounts[2])


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    """Uniform integer in ``[low, high]`` (inclusive) from ``generator``."""
    if high <= low:
        return low
    return low + int(torch.randint(0, high - low + 1, (1,), generator=generator).item())


def _rand_float(generator: torch.Generator) -> float:
    """Uniform float in ``[0, 1)`` from ``generator``."""
    return float(torch.rand((), generator=generator).item())


@dataclass(frozen=True)
class PatchInfo:
    """Where a patch sits in its source volume, plus sampling metadata.

    ``origin`` is the voxel coordinate of the patch corner after clamping.
    ``physical_min_mm``/``physical_max_mm`` bound the patch (including any
    padded region) in patient millimetres and are only set when ``spacing_mm``
    was supplied to the sampler. ``padding`` holds per-axis trailing pad
    amounts and is ``()`` when the patch needed no padding.
    """

    origin: tuple[int, int, int]
    patch_shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    target_positive: bool
    sampling_probability: float
    physical_min_mm: tuple[float, float, float] | None = None
    physical_max_mm: tuple[float, float, float] | None = None
    padded: bool = False
    padding: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("origin", "patch_shape", "original_shape"):
            value = tuple(int(v) for v in getattr(self, name))
            if len(value) != 3:
                raise PatchSamplingError(f"PatchInfo.{name} must be a 3-tuple; got {getattr(self, name)!r}")
            object.__setattr__(self, name, (value[0], value[1], value[2]))
        if self.padding:
            padding = tuple(int(p) for p in self.padding)
            if len(padding) != 3 or any(p < 0 for p in padding):
                raise PatchSamplingError(
                    f"PatchInfo.padding must be () or three non-negative per-axis amounts; got {self.padding!r}"
                )
            object.__setattr__(self, "padding", (padding[0], padding[1], padding[2]))


@dataclass(frozen=True, eq=False)  # tensor fields break default structural equality
class Patch:
    """An extracted patch: metadata plus image ``[C, D, H, W]`` and optional mask ``[D, H, W]`` tensors."""

    info: PatchInfo
    image: torch.Tensor
    mask: torch.Tensor | None = None


def extract_patch(image: torch.Tensor, info: PatchInfo, pad_value: float = 0.0) -> torch.Tensor:
    """Extract ``info.patch_shape`` at ``info.origin`` from the last three dims of ``image``.

    The extracted region always has exactly ``info.patch_shape`` spatially:
    where the patch extends past the volume (per ``info.padding``), the
    trailing voxels are filled with ``pad_value``. Works for ``[C, D, H, W]``
    images and ``[D, H, W]`` masks alike.
    """
    if image.ndim < 3:
        raise PatchSamplingError(f"extract_patch needs at least 3 spatial dims; got shape {tuple(image.shape)}")
    spatial = tuple(int(d) for d in image.shape[-3:])
    slices = tuple(slice(info.origin[a], min(info.origin[a] + info.patch_shape[a], spatial[a])) for a in range(3))
    patch = image[..., slices[0], slices[1], slices[2]]
    if info.padded:
        pad: list[int] = []
        for axis in (2, 1, 0):
            pad.extend((0, info.padding[axis]))
        patch = torch.nn.functional.pad(patch, pad, value=pad_value)
    return patch


def _build_info(
    origin: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
    *,
    mask: torch.Tensor | None,
    spacing_mm: tuple[float, float, float] | None,
    sampling_probability: float,
) -> PatchInfo:
    """Assemble a :class:`PatchInfo` from a clamped origin (shared by all samplers)."""
    padding = _padding_amounts(origin, patch_shape, volume_shape)
    target_positive = False
    if mask is not None:
        slices = tuple(slice(origin[a], min(origin[a] + patch_shape[a], volume_shape[a])) for a in range(3))
        target_positive = bool((mask[slices[0], slices[1], slices[2]] > 0).any().item())
    physical_min: tuple[float, float, float] | None = None
    physical_max: tuple[float, float, float] | None = None
    if spacing_mm is not None:
        spacing = _validate_spacing(spacing_mm)
        mins = [float(origin[a]) * spacing[a] for a in range(3)]
        maxs = [float(origin[a] + patch_shape[a]) * spacing[a] for a in range(3)]
        physical_min = (mins[0], mins[1], mins[2])
        physical_max = (maxs[0], maxs[1], maxs[2])
    return PatchInfo(
        origin=origin,
        patch_shape=patch_shape,
        original_shape=volume_shape,
        target_positive=target_positive,
        sampling_probability=sampling_probability,
        physical_min_mm=physical_min,
        physical_max_mm=physical_max,
        padded=any(p > 0 for p in padding),
        padding=padding,
    )


class PatchSampler(ABC):
    """Base class for 3D patch samplers: shared validation and extraction flow.

    Subclasses implement :meth:`_propose_origin`, returning an (unclamped)
    origin and the probability mass of that draw; :meth:`sample` clamps the
    origin, builds the :class:`PatchInfo`, and extracts the full-shape patch.
    """

    def __init__(self, patch_shape: tuple[int, int, int]) -> None:
        self._patch_shape = _validate_patch_shape(patch_shape)

    @property
    def patch_shape(self) -> tuple[int, int, int]:
        return self._patch_shape

    @abstractmethod
    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        """Return ``(unclamped origin, sampling probability)`` for one patch."""

    def sample(
        self,
        image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        spacing_mm: tuple[float, float, float] | None = None,
        pad_value: float = 0.0,
    ) -> Patch:
        """Draw one patch from ``image`` ([C, D, H, W]); ``mask`` ([D, H, W]) drives positivity metadata."""
        volume_shape = _validate_inputs(image, mask)
        origin, probability = self._propose_origin(volume_shape, mask)
        origin = _clamp_origin(origin, self._patch_shape, volume_shape)
        info = _build_info(
            origin,
            self._patch_shape,
            volume_shape,
            mask=mask,
            spacing_mm=spacing_mm,
            sampling_probability=probability,
        )
        return Patch(
            info=info,
            image=extract_patch(image, info, pad_value=pad_value),
            mask=extract_patch(mask, info) if mask is not None else None,
        )


class _RandomPatchSampler(PatchSampler):
    """Base for samplers drawing from a seeded ``torch.Generator``."""

    def __init__(self, patch_shape: tuple[int, int, int], *, seed: SeedLike = 0) -> None:
        super().__init__(patch_shape)
        self._generator = seed if isinstance(seed, torch.Generator) else make_generator(seed)

    def _uniform_origin(self, volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        origin = [_randint(self._generator, 0, max(0, volume_shape[a] - self._patch_shape[a])) for a in range(3)]
        return (origin[0], origin[1], origin[2])


class RandomPatchSampler(_RandomPatchSampler):
    """Uniform random origins over the whole volume."""

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        return self._uniform_origin(volume_shape), 1.0


class ForegroundPatchSampler(_RandomPatchSampler):
    """Centers patches on foreground voxels with configurable probability.

    With probability ``positive_ratio`` the patch is centered on a uniformly
    drawn foreground voxel (``mask > 0``); otherwise the origin is uniform
    over the volume. The intended draw is recorded as
    ``PatchInfo.sampling_probability`` and the realized overlap as
    ``PatchInfo.target_positive``, so the empirical positive proportion is
    measurable against the configured ratio. Without a mask every draw is
    uniform (probability mass 1.0) and ``target_positive`` stays ``False``.

    Parameters
    ----------
    patch_shape:
        Patch size ``(D, H, W)``.
    positive_ratio:
        Probability of drawing a foreground-centred patch; must be in [0, 1].
    seed:
        Integer seed or caller-owned ``torch.Generator``.
    """

    def __init__(self, patch_shape: tuple[int, int, int], *, positive_ratio: float = 0.5, seed: SeedLike = 0) -> None:
        super().__init__(patch_shape, seed=seed)
        if not 0.0 <= positive_ratio <= 1.0:
            raise PatchSamplingError(f"positive_ratio must be in [0, 1]; got {positive_ratio}")
        self._positive_ratio = float(positive_ratio)

    @property
    def positive_ratio(self) -> float:
        return self._positive_ratio

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        if mask is None:
            return self._uniform_origin(volume_shape), 1.0
        if _rand_float(self._generator) < self._positive_ratio:
            foreground = (mask > 0).nonzero()
            if int(foreground.shape[0]) == 0:
                raise PatchSamplingError("positive patch requested but the mask has no foreground voxels")
            pick = _randint(self._generator, 0, int(foreground.shape[0]) - 1)
            voxel = foreground[pick]
            origin = [int(voxel[a].item()) - self._patch_shape[a] // 2 for a in range(3)]
            return (origin[0], origin[1], origin[2]), self._positive_ratio
        return self._uniform_origin(volume_shape), 1.0 - self._positive_ratio


class ClassBalancedPatchSampler(_RandomPatchSampler):
    """Cycles through ``classes``, centering each patch on a voxel of the current class.

    Each call advances an internal cursor round-robin over ``classes`` and
    centers the patch on a uniformly drawn voxel whose label equals that
    class, so every requested class is visited equally often regardless of
    its voxel count. Requires a mask whose values are the label indices;
    requesting a class absent from the mask raises :class:`PatchSamplingError`.

    Parameters
    ----------
    patch_shape:
        Patch size ``(D, H, W)``.
    classes:
        Non-empty tuple of non-negative label indices to balance across.
    seed:
        Integer seed or caller-owned ``torch.Generator``.
    """

    def __init__(self, patch_shape: tuple[int, int, int], *, classes: tuple[int, ...], seed: SeedLike = 0) -> None:
        super().__init__(patch_shape, seed=seed)
        if not classes or any(int(c) < 0 for c in classes):
            raise PatchSamplingError(
                f"classes must be a non-empty tuple of non-negative label indices; got {classes!r}"
            )
        self._classes = tuple(int(c) for c in classes)
        self._cursor = 0

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        if mask is None:
            raise PatchSamplingError("ClassBalancedPatchSampler requires a mask with class labels")
        cls = self._classes[self._cursor % len(self._classes)]
        self._cursor += 1
        voxels = (mask == cls).nonzero()
        if int(voxels.shape[0]) == 0:
            raise PatchSamplingError(f"class {cls} is not present in the mask; cannot balance across it")
        pick = _randint(self._generator, 0, int(voxels.shape[0]) - 1)
        voxel = voxels[pick]
        origin = [int(voxel[a].item()) - self._patch_shape[a] // 2 for a in range(3)]
        return (origin[0], origin[1], origin[2]), 1.0 / len(self._classes)


class BoxPatchSampler(_RandomPatchSampler):
    """Uniform random origins constrained inside a voxel bounding box.

    ``box`` is ``(z0, y0, x0, z1, y1, x1)`` with max-exclusive semantics per
    axis: patch boxes never extend past the box unless the box is smaller
    than the patch on that axis (the origin then clamps to the box minimum).
    A box outside the sampled volume raises :class:`PatchSamplingError`.
    """

    def __init__(
        self, patch_shape: tuple[int, int, int], *, box: tuple[int, int, int, int, int, int], seed: SeedLike = 0
    ) -> None:
        super().__init__(patch_shape, seed=seed)
        if len(box) != 6:
            raise PatchSamplingError(f"box must be (z0, y0, x0, z1, y1, x1); got {box!r}")
        mins = tuple(int(v) for v in box[:3])
        maxs = tuple(int(v) for v in box[3:])
        if any(m < 0 for m in mins) or any(maxs[a] < mins[a] for a in range(3)):
            raise PatchSamplingError(f"box must satisfy 0 <= min <= max per axis; got {box!r}")
        self._box_min = (mins[0], mins[1], mins[2])
        self._box_max = (maxs[0], maxs[1], maxs[2])

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        for axis in range(3):
            if self._box_max[axis] > volume_shape[axis]:
                raise PatchSamplingError(
                    f"box {(*self._box_min, *self._box_max)!r} exceeds volume shape {volume_shape} on axis {axis}"
                )
        origin = []
        for axis in range(3):
            lo = self._box_min[axis]
            box_hi = self._box_max[axis] - self._patch_shape[axis]
            vol_hi = volume_shape[axis] - self._patch_shape[axis]
            origin.append(_randint(self._generator, lo, max(lo, min(box_hi, vol_hi))))
        return (origin[0], origin[1], origin[2]), 1.0


class LesionCenteredPatchSampler(_RandomPatchSampler):
    """Centers patches on connected-component centroids of a label mask.

    Components are found with :func:`scipy.ndimage.label` on ``mask > 0``;
    one component is drawn uniformly per patch and the patch is centered on
    its (rounded) centroid, optionally offset by a deterministic per-axis
    jitter of up to ``jitter_voxels`` drawn from the sampler's generator.
    A mask with no components raises :class:`PatchSamplingError`.

    Parameters
    ----------
    patch_shape:
        Patch size ``(D, H, W)``.
    jitter_voxels:
        Maximum absolute per-axis jitter in voxels; 0 disables jitter.
    seed:
        Integer seed or caller-owned ``torch.Generator``.
    """

    def __init__(self, patch_shape: tuple[int, int, int], *, jitter_voxels: int = 0, seed: SeedLike = 0) -> None:
        super().__init__(patch_shape, seed=seed)
        if jitter_voxels < 0:
            raise PatchSamplingError(f"jitter_voxels must be >= 0; got {jitter_voxels}")
        self._jitter = int(jitter_voxels)

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        if mask is None:
            raise PatchSamplingError("LesionCenteredPatchSampler requires a label mask")
        mask_np = (mask > 0).to(torch.float32).numpy()
        labels, count = scipy.ndimage.label(mask_np)
        if count == 0:
            raise PatchSamplingError("no connected components (lesions) found in the mask")
        centroids = scipy.ndimage.center_of_mass(mask_np, labels, list(range(1, count + 1)))
        pick = _randint(self._generator, 0, count - 1)
        centroid = centroids[pick]
        origin = [int(round(float(centroid[a]))) - self._patch_shape[a] // 2 for a in range(3)]
        if self._jitter > 0:
            for axis in range(3):
                origin[axis] += _randint(self._generator, -self._jitter, self._jitter)
        return (origin[0], origin[1], origin[2]), 1.0 / count


class GridPatchSampler(PatchSampler):
    """Deterministic grid of patches covering the whole volume (no RNG).

    The stride per axis is ``max(1, round(patch * (1 - overlap)))``; a final
    flush position is appended per axis whenever the strided stops would
    leave the far edge uncovered, so the union of patch boxes covers every
    voxel. :meth:`iter_patches` yields every patch in scan order (D, then H,
    then W); :meth:`sample` walks the same sequence one patch per call,
    wrapping around after the last patch.

    Parameters
    ----------
    patch_shape:
        Patch size ``(D, H, W)``.
    overlap:
        Fractional overlap between neighbouring patches in ``[0, 1)``.
    """

    def __init__(self, patch_shape: tuple[int, int, int], *, overlap: float = 0.0) -> None:
        super().__init__(patch_shape)
        if not 0.0 <= overlap < 1.0:
            raise PatchSamplingError(f"overlap must be in [0, 1); got {overlap}")
        self._overlap = float(overlap)
        strides = [max(1, round(p * (1.0 - self._overlap))) for p in self._patch_shape]
        self._strides = (strides[0], strides[1], strides[2])
        self._cached_shape: tuple[int, int, int] | None = None
        self._pending: list[tuple[int, int, int]] = []

    @property
    def overlap(self) -> float:
        return self._overlap

    @staticmethod
    def _axis_positions(length: int, patch: int, stride: int) -> tuple[int, ...]:
        if length <= patch:
            return (0,)
        positions = []
        pos = 0
        while pos + patch < length:
            positions.append(pos)
            pos += stride
        last = length - patch
        if positions[-1] != last:
            positions.append(last)
        return tuple(positions)

    def origins_for(self, volume_shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
        """All grid origins for a volume of ``volume_shape``, in scan order."""
        shape = _validate_volume_shape(volume_shape)
        axes = [self._axis_positions(shape[a], self._patch_shape[a], self._strides[a]) for a in range(3)]
        return tuple((d, h, w) for d in axes[0] for h in axes[1] for w in axes[2])

    def _propose_origin(
        self, volume_shape: tuple[int, int, int], mask: torch.Tensor | None
    ) -> tuple[tuple[int, int, int], float]:
        if self._cached_shape != volume_shape or not self._pending:
            self._pending = list(self.origins_for(volume_shape))
            self._cached_shape = volume_shape
        origin = self._pending.pop(0)
        return origin, 1.0 / len(self.origins_for(volume_shape))

    def iter_patches(
        self,
        image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        spacing_mm: tuple[float, float, float] | None = None,
        pad_value: float = 0.0,
    ) -> Iterator[Patch]:
        """Yield every grid patch of ``image`` ([C, D, H, W]) in scan order."""
        volume_shape = _validate_inputs(image, mask)
        origins = self.origins_for(volume_shape)
        probability = 1.0 / len(origins)
        for origin in origins:
            info = _build_info(
                origin,
                self._patch_shape,
                volume_shape,
                mask=mask,
                spacing_mm=spacing_mm,
                sampling_probability=probability,
            )
            yield Patch(
                info=info,
                image=extract_patch(image, info, pad_value=pad_value),
                mask=extract_patch(mask, info) if mask is not None else None,
            )
