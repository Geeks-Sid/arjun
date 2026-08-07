"""Model-aware preprocessing contracts: NormalizationSpec and PreprocessSpec.

A :class:`PreprocessSpec` is the *declared input contract* of a model adapter:
spatial shape, channel count, dtype, expected value range, and the
normalization statistics the adapter was trained with. Pipelines validate
their final tensors against the selected spec, so an adapter always receives
exactly its declared tensor format.

Specs are configuration, not code: ``config_dict``/``spec_hash`` feed
``medfm.core.serialization.config_hash`` so a spec change invalidates
preprocessing caches. Window presets, crop policies, and augmentation
settings are deliberately *not* part of the spec — they live in transform
configs (model/config-specific, never global).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from medfm.core.serialization import canonical_dtype_name, config_hash, dtype_from_canonical
from medfm.data.errors import PreprocessSpecError


@dataclass(frozen=True)
class NormalizationSpec:
    """Per-channel z-score normalization a model expects (mean/std)."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.std):
            raise PreprocessSpecError("NormalizationSpec mean/std must be non-empty and equal length")
        if any(s <= 0 for s in self.std):
            raise PreprocessSpecError(f"NormalizationSpec std entries must be positive; got {self.std}")

    @property
    def channels(self) -> int:
        return len(self.mean)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationSpec:
        return cls(mean=tuple(float(v) for v in data["mean"]), std=tuple(float(v) for v in data["std"]))


@dataclass(frozen=True)
class PreprocessSpec:
    """Declared tensor contract of one model adapter.

    - ``spatial_shape``: exact spatial dims the adapter consumes — ``(H, W)``
      for 2D, ``(D, H, W)`` for volumes.
    - ``channels``: exact channel count (1 for grayscale, 3 for repeated
      three-channel or multi-window/multi-sequence stacks).
    - ``value_range``: inclusive intensity range expected *after*
      canonicalization but *before* normalization (e.g. ``(0, 1)`` for 2D
      radiology, ``(-1024, 3071)`` for clipped HU). ``None`` disables range
      validation.
    - ``normalization``: expected channel statistics; when present,
      :meth:`validate` can also check that a normalized tensor is
      approximately standardized (empirical conformance, used with dummy
      specs in tests).
    - ``dtype``: canonical (accelerator-neutral) dtype name.
    """

    model_id: str
    spatial_shape: tuple[int, ...]
    channels: int
    dtype: str = "float32"
    value_range: tuple[float, float] | None = None
    normalization: NormalizationSpec | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise PreprocessSpecError("PreprocessSpec.model_id must be non-empty")
        shape = tuple(int(d) for d in self.spatial_shape)
        if len(shape) not in (2, 3) or any(d <= 0 for d in shape):
            raise PreprocessSpecError(f"PreprocessSpec.spatial_shape must be 2 or 3 positive ints; got {shape}")
        object.__setattr__(self, "spatial_shape", shape)
        if self.channels <= 0:
            raise PreprocessSpecError(f"PreprocessSpec.channels must be positive; got {self.channels}")
        if self.normalization is not None and self.normalization.channels != self.channels:
            raise PreprocessSpecError(
                f"normalization has {self.normalization.channels} channels but spec expects {self.channels}"
            )
        if self.value_range is not None:
            low, high = (float(v) for v in self.value_range)
            if not low < high:
                raise PreprocessSpecError(f"PreprocessSpec.value_range must be (low < high); got {self.value_range}")
            object.__setattr__(self, "value_range", (low, high))
        try:
            dtype_from_canonical(self.dtype)
        except Exception as exc:
            raise PreprocessSpecError(f"PreprocessSpec.dtype {self.dtype!r} is not a canonical dtype name") from exc

    @property
    def spatial_rank(self) -> int:
        return len(self.spatial_shape)

    @property
    def torch_dtype(self) -> torch.dtype:
        return dtype_from_canonical(self.dtype)

    def expected_tensor_shape(self) -> tuple[int, ...]:
        """Expected per-sample tensor shape ``[C, *spatial]``."""
        return (self.channels, *self.spatial_shape)

    def validate(self, tensor: torch.Tensor, *, check_normalization: bool = False, atol: float = 1e-4) -> None:
        """Raise :class:`PreprocessSpecError` unless ``tensor`` matches the spec.

        Checks shape, dtype, and (when configured) value range. With
        ``check_normalization=True`` and a ``normalization`` spec, also
        verifies the tensor is approximately standardized per channel —
        the empirical model shape/range/normalization conformance check.
        """
        if tensor.ndim != len(self.spatial_shape) + 1:
            raise PreprocessSpecError(
                f"spec {self.model_id!r} expects a [C, *spatial] tensor of rank {len(self.spatial_shape) + 1}; "
                f"got rank {tensor.ndim} with shape {tuple(tensor.shape)}"
            )
        expected = self.expected_tensor_shape()
        if tuple(tensor.shape) != expected:
            raise PreprocessSpecError(
                f"spec {self.model_id!r} expects shape {expected}; got {tuple(tensor.shape)}. "
                "Adapters must receive exactly their declared tensor format."
            )
        if tensor.dtype != self.torch_dtype:
            raise PreprocessSpecError(f"spec {self.model_id!r} expects dtype {self.dtype}; got {tensor.dtype}")
        if self.value_range is not None:
            low, high = self.value_range
            t_min, t_max = float(tensor.min()), float(tensor.max())
            if t_min < low - atol or t_max > high + atol:
                raise PreprocessSpecError(
                    f"spec {self.model_id!r} expects values in [{low}, {high}]; tensor spans [{t_min}, {t_max}]"
                )
        if check_normalization:
            if self.normalization is None:
                raise PreprocessSpecError("check_normalization=True requires a normalization spec")
            flat = tensor.detach().float().flatten(start_dim=1)
            mean = flat.mean(dim=1)
            std = flat.std(dim=1, unbiased=False)
            if not bool(torch.allclose(mean, torch.zeros_like(mean), atol=1e-1)):
                raise PreprocessSpecError(
                    f"normalized tensor channel means {mean.tolist()} deviate from 0 beyond tolerance"
                )
            if not bool(torch.allclose(std, torch.ones_like(std), atol=1e-1)):
                raise PreprocessSpecError(
                    f"normalized tensor channel stds {std.tolist()} deviate from 1 beyond tolerance"
                )

    def config_dict(self) -> dict[str, Any]:
        """Canonical JSON-able configuration for hashing and cache keys."""
        return {
            "model_id": self.model_id,
            "spatial_shape": list(self.spatial_shape),
            "channels": self.channels,
            "dtype": self.dtype,
            "value_range": list(self.value_range) if self.value_range is not None else None,
            "normalization": self.normalization.to_dict() if self.normalization is not None else None,
        }

    def spec_hash(self) -> str:
        """SHA-256 over :meth:`config_dict`; the cache-invalidation identity."""
        return config_hash(self.config_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessSpec:
        norm = data.get("normalization")
        raw_range = data.get("value_range")
        value_range: tuple[float, float] | None = None
        if raw_range is not None:
            pair = tuple(float(v) for v in raw_range)
            if len(pair) != 2:
                raise PreprocessSpecError(f"value_range must have exactly 2 entries; got {raw_range!r}")
            value_range = (pair[0], pair[1])
        return cls(
            model_id=str(data["model_id"]),
            spatial_shape=tuple(int(d) for d in data["spatial_shape"]),
            channels=int(data["channels"]),
            dtype=str(data.get("dtype", "float32")),
            value_range=value_range,
            normalization=NormalizationSpec.from_dict(norm) if norm is not None else None,
        )

    def canonical_dtype_of(self, tensor: torch.Tensor) -> str:
        """Convenience: canonical name of a tensor's dtype (for metadata)."""
        return canonical_dtype_name(tensor.dtype)
