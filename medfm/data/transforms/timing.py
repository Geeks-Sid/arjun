"""Host/device preprocessing timing, measured independently.

Training throughput analysis needs host preprocessing time and device input
wait measured *separately* (``accelerator_training_strategy.md``): a slow
host pipeline and a slow transfer look identical in step time alone. This
module times:

- **host**: transform/collation work on CPU (decode, canonicalization,
  augmentation, collation) via a monotonic wall clock.
- **device wait**: the time to transfer a collated :class:`MedicalBatch` to
  the target device and synchronize, i.e. what the training step would wait
  on for inputs.

Both are pure measurements — they never change behavior, and device sync is
backend-neutral (``torch.cuda.synchronize`` only when CUDA is the target;
XLA synchronizes via ``torch_xla`` when present, imported lazily so the CPU
baseline never touches it).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from medfm.core.batch import MedicalBatch


@dataclass(frozen=True)
class PhaseTiming:
    """One measured phase: wall seconds and the number of items processed."""

    seconds: float
    items: int

    @property
    def per_item_seconds(self) -> float:
        return self.seconds / self.items if self.items else 0.0


@dataclass(frozen=True)
class PreprocessTimingReport:
    """Independent host/device measurements for one run of preprocessing."""

    host: PhaseTiming  # transform + collate time (CPU)
    device_wait: PhaseTiming  # transfer + synchronize time
    device: str
    backend: str  # "cpu" | "cuda" | "xla"

    def to_dict(self) -> dict[str, object]:
        """JSON-able summary for run metadata (no payloads, no text)."""
        return {
            "host_seconds": self.host.seconds,
            "host_items": self.host.items,
            "host_per_item_seconds": self.host.per_item_seconds,
            "device_wait_seconds": self.device_wait.seconds,
            "device_wait_items": self.device_wait.items,
            "device_wait_per_item_seconds": self.device_wait.per_item_seconds,
            "device": self.device,
            "backend": self.backend,
        }


@dataclass
class PreprocessTimer:
    """Accumulates independent host and device-wait timings.

    Usage::

        timer = PreprocessTimer(device="cpu")
        with timer.time_host(items=len(examples)):
            batch = collator(examples)
        timer.record_device_wait(batch)
        report = timer.report()
    """

    device: str = "cpu"
    _host_seconds: float = field(default=0.0, init=False)
    _host_items: int = field(default=0, init=False)
    _device_seconds: float = field(default=0.0, init=False)
    _device_items: int = field(default=0, init=False)

    def time_host(self, *, items: int) -> _HostTimer:
        """Context manager timing a block of host preprocessing work."""
        if items <= 0:
            raise ValueError(f"time_host items must be positive; got {items}")
        return _HostTimer(self, items)

    def record_device_wait(self, batch: MedicalBatch) -> float:
        """Transfer ``batch`` to ``self.device`` + synchronize; returns seconds."""
        start = time.perf_counter()
        moved = batch.to(self.device)
        _synchronize(self.device)
        elapsed = time.perf_counter() - start
        del moved
        self._device_seconds += elapsed
        self._device_items += 1
        return elapsed

    def report(self) -> PreprocessTimingReport:
        return PreprocessTimingReport(
            host=PhaseTiming(seconds=self._host_seconds, items=self._host_items),
            device_wait=PhaseTiming(seconds=self._device_seconds, items=self._device_items),
            device=self.device,
            backend=_backend_of(self.device),
        )


class _HostTimer:
    def __init__(self, timer: PreprocessTimer, items: int) -> None:
        self._timer = timer
        self._items = items
        self._start = 0.0

    def __enter__(self) -> _HostTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = time.perf_counter() - self._start
        self._timer._host_seconds += elapsed
        self._timer._host_items += self._items


def _backend_of(device: str) -> str:
    kind = str(device).split(":")[0]
    if kind == "cuda":
        return "cuda"
    if kind == "xla":
        return "xla"
    return "cpu"


def _synchronize(device: str) -> None:
    backend = _backend_of(device)
    if backend == "cuda":
        import torch

        torch.cuda.synchronize()
    elif backend == "xla":
        import torch_xla.core.xla_model as xm  # lazy: CPU baseline never imports torch_xla

        xm.mark_step()


def time_host_preprocessing(work: Callable[[], int], *, repeats: int = 1) -> PhaseTiming:
    """Time ``work`` (returning the item count) over ``repeats`` runs on the host."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive; got {repeats}")
    start = time.perf_counter()
    items = 0
    for _ in range(repeats):
        items += work()
    return PhaseTiming(seconds=time.perf_counter() - start, items=items)
