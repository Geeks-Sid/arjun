"""MRI preprocessing: sequence resolution, normalization, multi-sequence stacking.

Deterministic MRI canonicalization steps:

- :class:`SequenceResolver` — canonical sequence identification with explicit
  aliases (``"t1ce"`` / ``"T1C"`` -> ``"T1CE"``). Unknown names raise
  :class:`TransformError` listing the legal values; a different sequence is
  NEVER silently substituted.
- :class:`ForegroundZScoreNormalize` / :class:`RobustPercentileNormalize` —
  foreground-aware intensity normalization (background voxels stay 0).
- :func:`select_sequences` / :func:`stack_sequences` — per-sequence selection
  and channel stacking with an explicit missing-sequence mask; allowed-missing
  sequences must be configured by the caller, required sequences raise.
- :func:`apply_n4_bias_field_correction` — bias-field correction as an
  explicit, configured-only operation. It is *not* a pipeline default and
  must never be added to one implicitly; see its docstring.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from scipy import ndimage

from medfm.data.errors import TransformError
from medfm.data.transforms.base import Transform, TransformContext, TransformData

#: TransformData metadata key carrying the (raw) sequence name of a payload.
SEQUENCE_METADATA_KEY = "sequence"

#: Minimum foreground standard deviation / percentile span treated as
#: informative; below this the channel is left unchanged (documented no-op).
_EPS = 1e-8


class SequenceResolver:
    """Canonical MRI sequence identification with explicit aliases.

    ``aliases`` maps each canonical name (e.g. ``"T1CE"``) to the accepted
    aliases for it (e.g. ``("T1C", "T1GD", "T1-CE")``). Resolution is
    case-insensitive and ignores surrounding whitespace, but nothing else:
    an unknown name raises :class:`TransformError` listing every legal
    canonical name and alias. This is the anti-silent-subscription boundary —
    a requested sequence is either identified exactly or rejected.
    """

    def __init__(self, aliases: dict[str, tuple[str, ...]]) -> None:
        if not aliases:
            raise TransformError("SequenceResolver requires at least one canonical sequence")
        self._canonical: list[str] = []
        self._lookup: dict[str, str] = {}
        for canonical, names in aliases.items():
            if not canonical:
                raise TransformError("SequenceResolver canonical names must be non-empty")
            key = self._normalize(canonical)
            if key in self._lookup:
                raise TransformError(f"duplicate canonical sequence {canonical!r} in resolver configuration")
            self._canonical.append(canonical)
            self._lookup[key] = canonical
            for alias in names:
                alias_key = self._normalize(alias)
                if not alias_key:
                    raise TransformError(f"empty alias configured for canonical sequence {canonical!r}")
                owner = self._lookup.get(alias_key)
                if owner is not None and owner != canonical:
                    raise TransformError(
                        f"alias {alias!r} is configured for both {owner!r} and {canonical!r}; "
                        "aliases must be unambiguous"
                    )
                self._lookup[alias_key] = canonical

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().upper()

    @property
    def canonical_names(self) -> tuple[str, ...]:
        """Configured canonical sequence names, in configuration order."""
        return tuple(self._canonical)

    def is_known(self, name: str) -> bool:
        return self._normalize(name) in self._lookup

    def resolve(self, name: str) -> str:
        """Resolve ``name`` to its canonical sequence name or raise.

        Raises :class:`TransformError` for unknown names, listing the legal
        canonical names and aliases — unknown sequences are never mapped to a
        "close" sequence silently.
        """
        canonical = self._lookup.get(self._normalize(name))
        if canonical is None:
            legal = sorted(self._lookup)
            raise TransformError(
                f"unknown MRI sequence {name!r}; legal canonical names are {self.canonical_names} "
                f"with aliases {legal}. Requested sequences must match exactly — no silent substitution."
            )
        return canonical

    def config_dict(self) -> dict[str, Any]:
        return {
            "aliases": {
                canonical: sorted(alias for alias, owner in self._lookup.items() if owner == canonical)
                for canonical in self._canonical
            }
        }


def select_sequences(
    available: dict[str, TransformData],
    requested: tuple[str, ...],
    resolver: SequenceResolver,
) -> dict[str, TransformData]:
    """Return ``{canonical_name: data}`` for each requested sequence.

    Both the available keys and the requested names are resolved through
    ``resolver``. A requested sequence with no matching available entry raises
    :class:`TransformError` — missing sequences are rejected, never replaced
    by a different sequence. Two available keys resolving to the same
    canonical sequence are ambiguous and also raise.
    """
    resolved: dict[str, TransformData] = {}
    for key, data in available.items():
        canonical = resolver.resolve(key)
        if canonical in resolved:
            raise TransformError(
                f"two available sequences ({key!r} and another) both resolve to {canonical!r}; "
                "sequence selection must be unambiguous"
            )
        resolved[canonical] = data
    selected: dict[str, TransformData] = {}
    for name in requested:
        canonical = resolver.resolve(name)
        if canonical not in resolved:
            raise TransformError(
                f"requested MRI sequence {name!r} (canonical {canonical!r}) is not available; "
                f"available sequences: {sorted(resolved)}. Missing sequences are rejected, not substituted."
            )
        selected[canonical] = resolved[canonical]
    return selected


class ForegroundZScoreNormalize(Transform):
    """Z-score each channel over its foreground (nonzero) voxels only.

    Background (exactly zero) voxels stay exactly zero. Per-channel means and
    standard deviations are recorded in the history. A channel with no
    foreground voxels, or with foreground standard deviation below
    :data:`_EPS`, is left unchanged (a recorded, deterministic no-op — e.g.
    an empty or constant acquisition).
    """

    name = "foreground_zscore_normalize"
    stage: Literal["deterministic"] = "deterministic"

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        image = data.image.to(torch.float32)
        means: list[float] = []
        stds: list[float] = []
        try:
            from monai.transforms import NormalizeIntensity  # type: ignore[attr-defined]
        except ImportError:
            normalize_kernel: Any = None
        else:
            normalize_kernel = NormalizeIntensity(
                nonzero=True,
                channel_wise=True,
                dtype=torch.float32,  # type: ignore[arg-type]
            )
        for channel in range(image.shape[0]):
            foreground = image[channel] != 0
            if not bool(foreground.any()):
                means.append(0.0)
                stds.append(0.0)
                continue
            values = image[channel][foreground]
            mean = float(values.mean())
            std = float(values.std(unbiased=False))
            means.append(mean)
            stds.append(std)
            if std > _EPS:
                if normalize_kernel is None:
                    image[channel] = torch.where(foreground, (image[channel] - mean) / std, image[channel])
                else:
                    image[channel] = normalize_kernel(image[channel : channel + 1])[0]
        data.image = image.contiguous()
        data.record(self.name, self.stage, {"means": means, "stds": stds})
        return data


class RobustPercentileNormalize(Transform):
    """Clip foreground intensities to percentiles, then scale to ``[0, 1]``.

    Percentiles are computed per channel over foreground (nonzero) voxels
    only; background voxels stay exactly zero. A channel with no foreground,
    or whose percentile span is below :data:`_EPS`, is left unchanged
    (recorded no-op).
    """

    name = "robust_percentile_normalize"
    stage: Literal["deterministic"] = "deterministic"

    def __init__(self, lower: float = 1.0, upper: float = 99.0) -> None:
        if not 0.0 <= float(lower) < float(upper) <= 100.0:
            raise TransformError(
                f"RobustPercentileNormalize requires 0 <= lower < upper <= 100; got ({lower}, {upper})"
            )
        self.lower = float(lower)
        self.upper = float(upper)

    def config_dict(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        image = data.image.to(torch.float32)
        lows: list[float] = []
        highs: list[float] = []
        try:
            from monai.transforms import ScaleIntensityRangePercentiles  # type: ignore[attr-defined]
        except ImportError:
            percentile_kernel: Any = None
        else:
            percentile_kernel = ScaleIntensityRangePercentiles(
                self.lower,
                self.upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                relative=False,
                channel_wise=False,
                dtype=torch.float32,  # type: ignore[arg-type]
            )
        for channel in range(image.shape[0]):
            foreground = image[channel] != 0
            if not bool(foreground.any()):
                lows.append(0.0)
                highs.append(0.0)
                continue
            values = image[channel][foreground]
            low = float(torch.quantile(values, self.lower / 100.0))
            high = float(torch.quantile(values, self.upper / 100.0))
            lows.append(low)
            highs.append(high)
            if high - low > _EPS:
                if percentile_kernel is None:
                    normalized = ((image[channel].clamp(low, high) - low) / (high - low)).clamp(0.0, 1.0)
                    image[channel] = torch.where(foreground, normalized, image[channel])
                else:
                    normalized_values = percentile_kernel(values)
                    image[channel][foreground] = normalized_values
        data.image = image.contiguous()
        data.record(self.name, self.stage, {"lower": self.lower, "upper": self.upper, "lows": lows, "highs": highs})
        return data


def _sequence_of(data: TransformData, resolver: SequenceResolver) -> str:
    raw = data.metadata.get(SEQUENCE_METADATA_KEY)
    if not isinstance(raw, str) or not raw.strip():
        raise TransformError(
            f"TransformData.metadata[{SEQUENCE_METADATA_KEY!r}] must carry the sequence name for stacking; "
            f"got {raw!r}. Sequence identity is required — it is never inferred from order."
        )
    return resolver.resolve(raw)


def stack_sequences(
    datas: list[TransformData],
    required: tuple[str, ...],
    resolver: SequenceResolver,
    *,
    allowed_missing: tuple[str, ...] = (),
    fill_value: float = 0.0,
) -> TransformData:
    """Stack per-sequence volumes into one multi-channel TransformData.

    ``required`` is the channel layout (canonical order). Every input must
    carry its sequence name in ``metadata["sequence"]``; each is resolved
    through ``resolver`` and duplicate canonical sequences raise.

    Missing-sequence policy (explicit, never silent):

    - a required sequence that is absent raises :class:`TransformError`,
      UNLESS the caller configured it in ``allowed_missing``;
    - an allowed-missing sequence contributes a channel filled with
      ``fill_value`` and a ``False`` entry in the mask.

    The result carries ``targets["sequence_mask"]``, a bool tensor of shape
    ``[len(required)]`` marking present sequences, and
    ``metadata["sequences"]`` with the canonical channel order. All present
    sequences must be single-channel with identical spatial shapes.
    """
    if not required:
        raise TransformError("stack_sequences requires a non-empty channel layout")
    canonical_required = [resolver.resolve(name) for name in required]
    if len(set(canonical_required)) != len(canonical_required):
        raise TransformError(f"stack_sequences channel layout has duplicates: {canonical_required}")
    canonical_allowed = {resolver.resolve(name) for name in allowed_missing}
    unknown_allowed = canonical_allowed - set(canonical_required)
    if unknown_allowed:
        raise TransformError(
            f"allowed_missing sequences {sorted(unknown_allowed)} are not part of the required layout; "
            "configure allowed-missing only for sequences that can appear as channels"
        )

    by_sequence = (
        select_sequences(
            {_sequence_of(data, resolver): data for data in datas},
            tuple(_sequence_of(data, resolver) for data in datas),
            resolver,
        )
        if datas
        else {}
    )

    present_shapes: set[tuple[int, ...]] = set()
    for data in by_sequence.values():
        if data.image.ndim != 4 or data.image.shape[0] != 1:
            raise TransformError(
                f"each sequence must be a single-channel [1, D, H, W] volume; got shape {tuple(data.image.shape)}"
            )
        present_shapes.add(tuple(int(d) for d in data.image.shape[1:]))
    if len(present_shapes) > 1:
        raise TransformError(
            f"sequence volumes have mismatched spatial shapes {sorted(present_shapes)}; "
            "resample sequences to a common grid before stacking"
        )

    reference = next(iter(by_sequence.values()), None)
    if reference is None:
        raise TransformError("stack_sequences got no sequence data at all; at least one sequence must be present")
    spatial_shape = tuple(int(d) for d in reference.image.shape[1:])

    channels: list[torch.Tensor] = []
    mask: list[bool] = []
    for canonical in canonical_required:
        seq_data = by_sequence.get(canonical)
        if seq_data is None:
            if canonical not in canonical_allowed:
                raise TransformError(
                    f"required MRI sequence {canonical!r} is missing and is not configured as allowed-missing; "
                    "missing sequences are rejected, never substituted"
                )
            channels.append(torch.full((1, *spatial_shape), float(fill_value), dtype=reference.image.dtype))
            mask.append(False)
        else:
            channels.append(seq_data.image.to(torch.float32))
            mask.append(True)

    result = TransformData(
        image=torch.cat(channels, dim=0).contiguous(),
        targets=dict(reference.targets),
        spatial=reference.spatial,
        pathology=reference.pathology,
        metadata={**reference.metadata, "sequences": list(canonical_required)},
        history=list(reference.history),
    )
    result.targets["sequence_mask"] = torch.as_tensor(mask, dtype=torch.bool)
    result.record(
        "stack_sequences",
        "deterministic",
        {
            "required": list(canonical_required),
            "allowed_missing": sorted(canonical_allowed),
            "fill_value": float(fill_value),
            "present": mask,
        },
    )
    return result


def apply_n4_bias_field_correction(
    data: TransformData,
    *,
    smoothing_sigma_voxels: float = 16.0,
    convergence_note: str | None = None,
) -> TransformData:
    """Bias-field correction as an EXPLICIT, configured-only operation.

    This is a deterministic low-pass approximation of N4-style multiplicative
    bias-field estimation: per channel, the log-intensity of foreground
    (nonzero) voxels is smoothed with a normalized Gaussian convolution
    (``scipy.ndimage.gaussian_filter`` with a fixed sigma — no RNG, no
    iterative fitting) and the estimated bias is divided out. It is *not* the
    full SimpleITK N4 algorithm; wire SimpleITK-based N4 offline if exact N4
    is required.

    WARNING: bias-field correction is an offline / explicitly-configured
    operation. It must NEVER appear in a default pipeline — pipelines that
    need it must opt in by constructor configuration, so the step is visible
    in the pipeline's ``config_dict``/hash. Background voxels stay exactly 0.
    """
    del convergence_note  # reserved for future SimpleITK wiring; keeps the signature explicit-only
    if smoothing_sigma_voxels <= 0:
        raise TransformError(
            f"apply_n4_bias_field_correction smoothing_sigma_voxels must be positive; got {smoothing_sigma_voxels}"
        )
    image = data.image.to(torch.float32)
    corrected = image.clone()
    for channel in range(image.shape[0]):
        foreground = (image[channel] > 0).to(torch.float32)
        if not bool(foreground.any()):
            continue
        log_intensity = torch.log(image[channel].clamp(min=_EPS)) * foreground
        numerator = ndimage.gaussian_filter(log_intensity.numpy(), sigma=smoothing_sigma_voxels, mode="nearest")
        denominator = ndimage.gaussian_filter(foreground.numpy(), sigma=smoothing_sigma_voxels, mode="nearest")
        valid = denominator > _EPS
        bias_log = torch.zeros_like(image[channel])
        bias_log[torch.as_tensor(valid)] = torch.as_tensor(numerator[valid]) / torch.as_tensor(denominator[valid])
        field = torch.exp(bias_log)
        corrected[channel] = torch.where(foreground.bool(), image[channel] / field, image[channel])
    data.image = corrected.contiguous()
    data.record(
        "n4_bias_field_correction",
        "deterministic",
        {"smoothing_sigma_voxels": float(smoothing_sigma_voxels), "method": "low_pass_log_approximation"},
    )
    return data
