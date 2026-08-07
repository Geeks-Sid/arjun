"""Transform base types: TransformData, TransformRecord, seeding, inversion.

Every transform is a callable ``TransformData -> TransformData`` with a
declared ``stage`` (``"deterministic"`` or ``"stochastic"``) and a JSON-able
``config_dict``. Applying a transform appends a :class:`TransformRecord` to
``data.history`` — the record carries enough parameters to invert spatial
transforms back to original physical coordinates.

Randomness contract: stochastic transforms draw exclusively from
``ctx.rng`` (a ``torch.Generator``). Seeds are derived per
``(base_seed, epoch, worker_id, sample_key)`` via :func:`derive_seed`, so
streams are reproducible under fixed seeds but never identical across
workers, epochs, or samples. Python's global RNG and unseeded
``torch.rand*`` calls are prohibited in this layer.

Inversion contract: spatial transforms register an inverter under their
``name`` with :func:`register_inverter`. :func:`invert_history` replays the
history in reverse, using ``mode="label"`` for masks so interpolation
policy differs correctly (nearest for labels, linear for images).
Transforms that cannot be inverted exactly (intensity/noise augmentation,
stain augmentation) register no inverter and are skipped by
:func:`invert_history` — the unsupported cases are recorded, never silent.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from medfm.core.sample import PathologyMetadata, SpatialMetadata
from medfm.core.serialization import canonical_json, config_hash
from medfm.data.errors import InversionError, TransformError

Stage = Literal["deterministic", "stochastic"]

#: Interpolation modes understood by the inversion registry.
InversionMode = Literal["image", "label"]


def derive_seed(base_seed: int, epoch: int, worker_id: int, sample_key: str) -> int:
    """Deterministic per-sample seed from (base_seed, epoch, worker, sample).

    Pure function of its inputs, stable across hosts and restarts (SHA-256
    based, not Python's salted ``hash``). ``sample_key`` should be a stable
    sample identifier or content hash so the same sample gets the same
    augmentation draw in the same epoch/worker context.
    """
    digest = hashlib.sha256(f"transform:{base_seed}:{epoch}:{worker_id}:{sample_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def make_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded with ``seed``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@dataclass
class TransformContext:
    """Per-sample execution context: the ONLY randomness source allowed."""

    rng: torch.Generator
    seed: int
    epoch: int = 0
    worker_id: int = 0
    sample_key: str = ""

    @classmethod
    def for_sample(
        cls,
        base_seed: int,
        epoch: int,
        worker_id: int,
        sample_key: str,
    ) -> TransformContext:
        """Build a context with a seed derived per worker/epoch/sample."""
        seed = derive_seed(base_seed, epoch, worker_id, sample_key)
        return cls(
            rng=make_generator(seed),
            seed=seed,
            epoch=epoch,
            worker_id=worker_id,
            sample_key=sample_key,
        )


@dataclass(frozen=True)
class TransformRecord:
    """One applied transform: name, stage, and the parameters needed to invert.

    ``params`` must be JSON-able (canonical_json-serializable). Spatial
    transforms set ``spatial=True`` and record enough geometry (origins,
    shapes, affines, crop boxes) for exact inversion.
    """

    name: str
    stage: Stage
    params: dict[str, Any] = field(default_factory=dict)
    spatial: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise TransformError("TransformRecord.name must be non-empty")
        if self.stage not in ("deterministic", "stochastic"):
            raise TransformError(f"TransformRecord.stage must be 'deterministic' or 'stochastic'; got {self.stage!r}")
        try:
            canonical_json(self.params)
        except (TypeError, ValueError) as exc:
            raise TransformError(f"TransformRecord {self.name!r} params are not JSON-able: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "params": self.params, "spatial": self.spatial}


#: Inverter signature: (record, tensor, mode) -> tensor in prior coordinates.
Inverter = Callable[[TransformRecord, torch.Tensor, InversionMode], torch.Tensor]

_INVERTERS: dict[str, Inverter] = {}


def register_inverter(name: str, inverter: Inverter) -> None:
    """Register the inversion function for a spatial transform name."""
    if not name:
        raise TransformError("inverter name must be non-empty")
    if name in _INVERTERS and _INVERTERS[name] is not inverter:
        raise TransformError(f"conflicting inverter registered for {name!r}")
    _INVERTERS[name] = inverter


def registered_inverter(name: str) -> Inverter | None:
    return _INVERTERS.get(name)


@dataclass
class TransformData:
    """Payload flowing through a transform pipeline.

    - ``image``: CPU tensor ``[C, *spatial]`` (2D: ``[C, H, W]``;
      volume: ``[C, D, H, W]``). Never batched — collation is separate.
    - ``targets``: auxiliary tensors transformed alongside the image
      (e.g. ``"mask"`` with label interpolation, ``"label"`` scalars).
    - ``spatial`` / ``pathology``: geometry metadata, updated by spatial
      transforms, never silently dropped.
    - ``history``: ordered records of applied transforms (inversion source).
    - ``metadata``: sample-level facts (view position, sequence name,
      longitudinal ordering) preserved through the pipeline.
    """

    image: torch.Tensor
    targets: dict[str, torch.Tensor] = field(default_factory=dict)
    spatial: SpatialMetadata | None = None
    pathology: PathologyMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[TransformRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.image.device.type != "cpu":
            raise TransformError(
                f"transform payloads are host-resident; got image on {self.image.device}. "
                "Decode/canonicalization runs on the host; tensors transfer only after collation."
            )
        if self.image.ndim not in (3, 4):
            raise TransformError(f"image must be [C, H, W] or [C, D, H, W]; got shape {tuple(self.image.shape)}")

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.image.shape[1:])

    def record(self, name: str, stage: Stage, params: dict[str, Any] | None = None, *, spatial: bool = False) -> None:
        """Append a transform record to the history."""
        self.history.append(TransformRecord(name=name, stage=stage, params=params or {}, spatial=spatial))

    def history_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.history]

    def history_hash(self) -> str:
        """SHA-256 over the transform history (deterministic ordering)."""
        return config_hash({"history": self.history_dicts()})


class Transform(ABC):
    """Base class for all Phase 04 transforms.

    Subclasses declare ``name``/``stage``, implement :meth:`apply`, and keep
    every constructor parameter JSON-able via :meth:`config_dict` so pipeline
    configs are hashable (cache keys) and replayable.
    """

    name: str = "transform"
    stage: Stage = "deterministic"
    #: Whether applying this transform changes spatial geometry.
    spatial: bool = False

    def __call__(self, data: TransformData, ctx: TransformContext | None = None) -> TransformData:
        if self.stage == "stochastic" and ctx is None:
            raise TransformError(
                f"stochastic transform {self.name!r} requires a TransformContext with a seeded generator"
            )
        result = self.apply(data, ctx)
        return result

    @abstractmethod
    def apply(self, data: TransformData, ctx: TransformContext | None) -> TransformData:
        """Apply the transform, appending a :class:`TransformRecord`."""

    def params(self) -> dict[str, Any]:
        """Instance parameters recorded in the history (defaults to config)."""
        return self.config_dict()

    def config_dict(self) -> dict[str, Any]:
        """JSON-able constructor configuration for hashing/replay."""
        return {}

    def config_hash(self) -> str:
        return config_hash({"name": self.name, "stage": self.stage, "config": self.config_dict()})


def invert_history(
    history: list[TransformRecord],
    image: torch.Tensor,
    *,
    mode: InversionMode = "image",
    strict: bool = False,
) -> torch.Tensor:
    """Replay ``history`` in reverse, mapping ``image`` back to original coordinates.

    Records without a registered inverter (non-spatial or non-invertible
    transforms, e.g. intensity/noise augmentation) are skipped by default;
    with ``strict=True`` a spatial record lacking an inverter raises
    :class:`InversionError`. ``mode="label"`` selects label interpolation
    (nearest) inside spatial inverters.
    """
    result = image
    for record in reversed(history):
        inverter = _INVERTERS.get(record.name)
        if inverter is None:
            if strict and record.spatial:
                raise InversionError(
                    f"spatial transform {record.name!r} has no registered inverter; "
                    "original-space reconstruction is impossible for this history"
                )
            continue
        result = inverter(record, result, mode)
    return result
