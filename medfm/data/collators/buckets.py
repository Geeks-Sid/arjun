"""Static-shape bucket policy for collators (ADR 0008).

XLA compiles per shape, so TPU training pads every variable dimension to a
bounded, documented set of buckets. A :class:`BucketPlan` declares those
buckets per :class:`BucketKind`, assigns the smallest covering bucket to a
shape, and decides what happens when no bucket covers a sample:

- ``out_of_bucket_policy="error"`` raises :class:`CollatorError` naming the
  unplanned shape — collating it would pad to an undeclared shape and trigger
  an unplanned TPU compilation.
- ``out_of_bucket_policy="pad_to_max"`` falls back to the largest declared
  bucket (collators crop the overflow, then pad).

Bucket sets are configuration: :meth:`BucketPlan.from_config` /
:meth:`BucketPlan.to_config` round-trip a JSON-able mapping and
:meth:`BucketPlan.config_hash` hashes it through
:func:`medfm.core.serialization.config_hash`, so every run pins the exact
bucket table it was compiled for. Validation bucket plans are separately
constructed from training plans: build them from their own config objects
(separate, independently hashed configs — never share one plan across
train/val) and keep validation shapes stable for the whole run.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from medfm.core.batch import BucketId, BucketKind
from medfm.core.serialization import config_hash
from medfm.data.errors import CollatorError

#: ``static`` pads to declared buckets; ``dynamic`` pads to the per-batch max.
BucketMode = Literal["static", "dynamic"]

#: Policy when no declared bucket covers a sample shape.
OutOfBucketPolicy = Literal["error", "pad_to_max"]


def _volume(shape: tuple[int, ...]) -> int:
    volume = 1
    for dim in shape:
        volume *= dim
    return volume


@dataclass(frozen=True)
class BucketPlan:
    """Bounded set of static-shape buckets plus the out-of-bucket policy.

    ``buckets`` maps each :class:`BucketKind` to its declared
    :class:`BucketId` shapes (normalized to sorted, de-duplicated tuples at
    construction). ``max_buckets`` bounds the total number of buckets across
    kinds so the compilation surface stays bounded. ``warn_utilization``
    (a fraction in ``(0, 1]``) emits a :class:`UserWarning` when an assigned
    bucket is filled below that fraction, and the first exercise of every
    bucket warns once so missing precompile/warm coverage is loud.
    """

    buckets: dict[BucketKind, tuple[BucketId, ...]]
    mode: BucketMode = "static"
    max_buckets: int = 64
    out_of_bucket_policy: OutOfBucketPolicy = "error"
    warn_utilization: float | None = None
    _exercised: set[BucketId] = field(default_factory=set, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("static", "dynamic"):
            raise CollatorError(f"bucket plan mode must be 'static' or 'dynamic'; got {self.mode!r}")
        if self.out_of_bucket_policy not in ("error", "pad_to_max"):
            raise CollatorError(
                f"out_of_bucket_policy must be 'error' or 'pad_to_max'; got {self.out_of_bucket_policy!r}"
            )
        if self.max_buckets <= 0:
            raise CollatorError(f"max_buckets must be positive; got {self.max_buckets}")
        if self.warn_utilization is not None and not 0.0 < self.warn_utilization <= 1.0:
            raise CollatorError(f"warn_utilization must lie in (0, 1]; got {self.warn_utilization}")
        normalized: dict[BucketKind, tuple[BucketId, ...]] = {}
        total = 0
        for key, ids in self.buckets.items():
            kind = key if isinstance(key, BucketKind) else BucketKind.from_value(str(key))
            if not ids:
                raise CollatorError(f"bucket plan declares an empty bucket set for {kind.value}")
            unique = tuple(sorted(set(ids), key=lambda b: (_volume(b.shape), b.shape)))
            for bucket in unique:
                if bucket.kind is not kind:
                    raise CollatorError(f"bucket {bucket} is filed under kind {kind.value}; kinds must match")
            normalized[kind] = unique
            total += len(unique)
        if total == 0:
            raise CollatorError("bucket plan declares no buckets")
        if total > self.max_buckets:
            raise CollatorError(
                f"bucket plan declares {total} buckets, exceeding max_buckets={self.max_buckets}; "
                "the TPU compilation surface must stay bounded (ADR 0008)"
            )
        object.__setattr__(self, "buckets", normalized)

    def has_kind(self, kind: BucketKind) -> bool:
        return kind in self.buckets

    def bucket_ids(self, kind: BucketKind) -> tuple[BucketId, ...]:
        return self.buckets.get(kind, ())

    def assign(self, kind: BucketKind, shape: Sequence[int]) -> BucketId:
        """Return the smallest declared bucket covering ``shape`` on every dim.

        Raises :class:`CollatorError` naming the unplanned shape when no
        bucket covers it and ``out_of_bucket_policy="error"`` — such a shape
        would trigger an unplanned TPU compilation. With ``"pad_to_max"`` the
        largest declared bucket is returned instead (the caller crops overflow
        before padding).
        """
        dims = tuple(int(d) for d in shape)
        if len(dims) != kind.rank or any(d <= 0 for d in dims):
            raise CollatorError(f"{kind.value} buckets require a positive rank-{kind.rank} shape; got {dims}")
        declared = self.buckets.get(kind)
        if not declared:
            raise CollatorError(
                f"no {kind.value} buckets declared in this plan; declared kinds: "
                f"{sorted(k.value for k in self.buckets)}"
            )
        covering = [b for b in declared if all(bd >= d for bd, d in zip(b.shape, dims, strict=True))]
        if covering:
            bucket = min(covering, key=lambda b: (_volume(b.shape), b.shape))
        elif self.out_of_bucket_policy == "pad_to_max":
            bucket = max(declared, key=lambda b: (_volume(b.shape), b.shape))
        else:
            raise CollatorError(
                f"shape {dims} exceeds every {kind.value} bucket {[b.shape for b in declared]}; collating it "
                "would pad to an unplanned shape and trigger an unplanned TPU compilation. Declare a covering "
                "bucket or set out_of_bucket_policy='pad_to_max'."
            )
        self._warn_on_assign(kind, dims, bucket)
        return bucket

    def _warn_on_assign(self, kind: BucketKind, shape: tuple[int, ...], bucket: BucketId) -> None:
        if bucket not in self._exercised:
            self._exercised.add(bucket)
            warnings.warn(
                f"bucket {bucket} exercised for the first time; ensure this shape was precompiled/warmed "
                "before training (ADR 0008)",
                UserWarning,
                stacklevel=3,
            )
        if self.warn_utilization is not None:
            utilization = _volume(shape) / _volume(bucket.shape)
            if utilization < self.warn_utilization:
                warnings.warn(
                    f"shape {shape} utilizes only {utilization:.1%} of assigned bucket {bucket} "
                    f"(warn_utilization={self.warn_utilization}); consider a smaller {kind.value} bucket",
                    UserWarning,
                    stacklevel=3,
                )

    # ------------------------------------------------------------------ #
    # Configuration (bucket sets are hashed config)
    # ------------------------------------------------------------------ #

    def to_config(self) -> dict[str, Any]:
        """JSON-able configuration; canonical ordering for stable hashes."""
        return {
            "mode": self.mode,
            "out_of_bucket_policy": self.out_of_bucket_policy,
            "max_buckets": self.max_buckets,
            "warn_utilization": self.warn_utilization,
            "buckets": {
                kind.value: [list(bucket.shape) for bucket in ids]
                for kind, ids in sorted(self.buckets.items(), key=lambda kv: kv[0].value)
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BucketPlan:
        """Build a plan from a :meth:`to_config` mapping.

        Rank-1 kinds accept either ``64`` or ``[64]`` per bucket. Training
        and validation plans are built from separate config objects so their
        hashes stay independent.
        """
        raw = config.get("buckets")
        if not isinstance(raw, dict):
            raise CollatorError("bucket plan config requires a 'buckets' mapping of kind -> shapes")
        buckets: dict[BucketKind, tuple[BucketId, ...]] = {}
        for kind_name, shapes in raw.items():
            kind = BucketKind.from_value(str(kind_name))
            ids = tuple(
                BucketId(kind=kind, shape=(int(entry),) if isinstance(entry, int) else tuple(int(d) for d in entry))
                for entry in shapes
            )
            buckets[kind] = ids
        return cls(
            buckets=buckets,
            mode=config.get("mode", "static"),
            max_buckets=int(config.get("max_buckets", 64)),
            out_of_bucket_policy=config.get("out_of_bucket_policy", "error"),
            warn_utilization=config.get("warn_utilization"),
        )

    def config_hash(self) -> str:
        """SHA-256 over the canonical bucket configuration."""
        return config_hash(self.to_config())


def pad_to_shape(
    tensor: torch.Tensor,
    target: Sequence[int],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``tensor``'s trailing ``len(target)`` dims to ``target``.

    Returns ``(padded, mask)`` where ``mask`` is a bool tensor of shape
    ``target`` with ``True`` over the real content — padded regions are always
    distinguishable (ADR 0008). Never crops: a target smaller than the tensor
    on any dim raises :class:`CollatorError`.
    """
    target_shape = tuple(int(d) for d in target)
    rank = len(target_shape)
    if rank == 0 or rank > tensor.ndim or any(d <= 0 for d in target_shape):
        raise CollatorError(
            f"pad target must be a positive shape of rank 1..{tensor.ndim}; got {target_shape} "
            f"for tensor shape {tuple(tensor.shape)}"
        )
    current = tuple(int(d) for d in tensor.shape[-rank:])
    if any(c > t for c, t in zip(current, target_shape, strict=True)):
        raise CollatorError(
            f"cannot pad tensor shape {tuple(tensor.shape)} down to {target_shape}; pad_to_shape never crops"
        )
    pads: list[int] = []
    for c, t in zip(current[::-1], target_shape[::-1], strict=True):
        pads.extend((0, t - c))
    padded = F.pad(tensor, pads, value=float(pad_value)) if any(pads) else tensor
    mask = torch.zeros(target_shape, dtype=torch.bool)
    mask[tuple(slice(0, c) for c in current)] = True
    return padded, mask
