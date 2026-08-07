"""CT preprocessing: Hounsfield calibration, clipping, and window channels.

Deterministic CT canonicalization steps. The intensity chain for a CT volume
is typically::

    ToHounsfieldUnits -> ClipHU -> WindowChannels

after the shared spatial steps (:mod:`medfm.data.transforms.spatial3d`). All
transforms here are non-spatial intensity operations: they record their
parameters in the transform history but register no inverter (intensity
transforms are not inverted; spatial reconstruction skips them by design).

Unit verification policy: Hounsfield conversion *verifies* the declared units
of the incoming payload. Unknown or unsupported unit metadata raises
:class:`TransformError` — calibration status is never silently assumed.

Window convention: a window is a ``(center, width)`` pair in HU. Window
presets are constructor configuration (model/config-specific); this module
deliberately defines no module-level global preset dict.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from medfm.data.errors import TransformError
from medfm.data.transforms.base import Transform, TransformContext, TransformData

#: Units metadata values accepted as "already calibrated to HU".
_HU_UNITS = ("HU",)


class ToHounsfieldUnits(Transform):
    """Apply the DICOM-style rescale ``image = image * slope + intercept``.

    ``units`` declares the calibration status of the incoming payload (as
    reported by the reader / acquisition metadata):

    - ``units="HU"``: the payload is already calibrated; the transform is a
      recorded no-op (slope/intercept are ignored).
    - ``units=None``: the payload carries raw stored values; the
      slope/intercept conversion applies.
    - any other value: rejected with :class:`TransformError` at construction.
      Unknown units metadata is never silently assumed to be calibrated or
      raw.
    """

    name = "to_hounsfield_units"
    stage: Literal["deterministic"] = "deterministic"

    def __init__(self, slope: float, intercept: float, units: str | None = None) -> None:
        if units is not None and units not in _HU_UNITS:
            raise TransformError(
                f"ToHounsfieldUnits got unknown units metadata {units!r}; legal values are {list(_HU_UNITS)} "
                "(already calibrated) or None (raw stored values, slope/intercept applies). "
                "Unit metadata must be verified, never assumed."
            )
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.units = units

    def config_dict(self) -> dict[str, Any]:
        return {"slope": self.slope, "intercept": self.intercept, "units": self.units}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        applied = self.units is None
        if applied:
            data.image = (data.image.to(torch.float32) * self.slope + self.intercept).contiguous()
        data.record(
            self.name,
            self.stage,
            {"slope": self.slope, "intercept": self.intercept, "units": self.units, "applied": applied},
        )
        return data


class ClipHU(Transform):
    """Clamp HU intensities to a configurable ``[min_hu, max_hu]`` range."""

    name = "clip_hu"
    stage: Literal["deterministic"] = "deterministic"

    def __init__(self, min_hu: float, max_hu: float) -> None:
        if not float(min_hu) < float(max_hu):
            raise TransformError(f"ClipHU requires min_hu < max_hu; got ({min_hu}, {max_hu})")
        self.min_hu = float(min_hu)
        self.max_hu = float(max_hu)

    def config_dict(self) -> dict[str, Any]:
        return {"min_hu": self.min_hu, "max_hu": self.max_hu}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        data.image = data.image.clamp(self.min_hu, self.max_hu).contiguous()
        data.record(self.name, self.stage, {"min_hu": self.min_hu, "max_hu": self.max_hu})
        return data


class WindowChannels(Transform):
    """Map a single-channel HU volume to one ``[0, 1]`` channel per window.

    Each window is a ``(center, width)`` pair in HU (the convention used by
    this framework — *not* ``(min, max)``). A value ``v`` maps to
    ``clip((v - (center - width / 2)) / width, 0, 1)``. A single window yields
    a ``[1, D, H, W]`` output; ``n`` windows yield ``[n, D, H, W]`` channels
    stacked in the configured order.

    Window presets are model/config-specific constructor input; there is no
    global preset registry. The input must be single-channel ``[1, D, H, W]``
    (apply after :class:`ToHounsfieldUnits` / :class:`ClipHU`).
    """

    name = "window_channels"
    stage: Literal["deterministic"] = "deterministic"

    def __init__(self, windows: tuple[tuple[float, float], ...]) -> None:
        parsed = tuple((float(center), float(width)) for center, width in windows)
        if not parsed:
            raise TransformError("WindowChannels requires at least one (center, width) window")
        if any(width <= 0 for _, width in parsed):
            raise TransformError(f"WindowChannels window widths must be positive; got {parsed}")
        self.windows = parsed

    def config_dict(self) -> dict[str, Any]:
        return {"windows": [[center, width] for center, width in self.windows]}

    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        if data.image.shape[0] != 1:
            raise TransformError(
                f"WindowChannels expects a single-channel [1, D, H, W] HU volume; got shape {tuple(data.image.shape)}"
            )
        hu = data.image[0]
        channels = [((hu - (center - width / 2.0)) / width).clamp(0.0, 1.0) for center, width in self.windows]
        data.image = torch.stack(channels, dim=0).contiguous()
        data.record(self.name, self.stage, {"windows": [[c, w] for c, w in self.windows]})
        return data
