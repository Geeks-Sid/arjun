"""Backend-neutral input helpers for distributed and static-shape training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence, Sized
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from torch.utils.data import Dataset, Sampler

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.training.backend import AcceleratorBackend


class DataExecutionError(RuntimeError):
    """Input pipeline cannot satisfy the configured execution contract."""


class DeterministicDistributedSampler(Sampler[int]):
    """Distributed sampler with explicit, resumable epoch/seed state."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler rank/replica count")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        size = len(cast(Sized, dataset))
        if drop_last:
            self.num_samples = size // num_replicas
        else:
            self.num_samples = math.ceil(size / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(cast(Sized, self.dataset))))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=generator).tolist()
        if not self.drop_last:
            padding = self.total_size - len(indices)
            if padding:
                indices.extend((indices * math.ceil(padding / max(1, len(indices))))[:padding])
        else:
            indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("num_replicas", self.num_replicas)) != self.num_replicas:
            raise DataExecutionError("sampler checkpoint world size does not match current run")
        if int(state.get("rank", self.rank)) != self.rank:
            raise DataExecutionError("sampler checkpoint rank does not match current run")
        if int(state.get("seed", self.seed)) != self.seed:
            raise DataExecutionError("sampler checkpoint seed does not match current run")
        if bool(state.get("drop_last", self.drop_last)) != self.drop_last:
            raise DataExecutionError("sampler checkpoint drop_last does not match current run")
        epoch = int(state.get("epoch", 0))
        if epoch < 0:
            raise DataExecutionError("sampler checkpoint epoch must be >= 0")
        self.epoch = epoch


@dataclass(frozen=True)
class ShapeBucket:
    name: str
    dimensions: tuple[int, ...]
    kind: str = "generic"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("shape bucket name must be non-empty")
        if not self.dimensions or any(int(value) <= 0 for value in self.dimensions):
            raise ValueError("shape bucket dimensions must be positive")
        object.__setattr__(self, "dimensions", tuple(int(value) for value in self.dimensions))

    @property
    def bucket_id(self) -> BucketId | None:
        try:
            return BucketId(BucketKind(self.kind), self.dimensions)
        except (ValueError, KeyError):
            return None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimensions": list(self.dimensions), "kind": self.kind}


@dataclass(frozen=True)
class ShapeBucketPlan:
    """Bounded train/validation bucket sets for XLA static-shape execution."""

    train: tuple[ShapeBucket, ...] = ()
    validation: tuple[ShapeBucket, ...] = ()
    max_buckets: int = 32

    def __post_init__(self) -> None:
        if self.max_buckets < 1:
            raise ValueError("max_buckets must be positive")
        if len(self.train) > self.max_buckets or len(self.validation) > self.max_buckets:
            raise DataExecutionError("shape bucket set exceeds configured bound")
        if len({bucket.name for bucket in (*self.train, *self.validation)}) != len((*self.train, *self.validation)):
            raise DataExecutionError("shape bucket names must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, max_buckets: int = 32) -> ShapeBucketPlan:
        raw = data or {}

        def parse(values: Any, prefix: str) -> tuple[ShapeBucket, ...]:
            if values is None:
                return ()
            if not isinstance(values, (list, tuple)):
                raise DataExecutionError(f"shape_buckets.{prefix} must be a list")
            result: list[ShapeBucket] = []
            for index, value in enumerate(values):
                if isinstance(value, dict):
                    result.append(
                        ShapeBucket(
                            name=str(value.get("name", f"{prefix}_{index}")),
                            dimensions=tuple(int(d) for d in value["dimensions"]),
                            kind=str(value.get("kind", prefix)),
                        )
                    )
                else:
                    result.append(ShapeBucket(f"{prefix}_{index}", tuple(int(d) for d in value), prefix))
            return tuple(result)

        return cls(
            train=parse(raw.get("train"), "train"),
            validation=parse(raw.get("validation"), "validation"),
            max_buckets=max_buckets,
        )

    def all(self) -> tuple[ShapeBucket, ...]:
        return (*self.train, *self.validation)

    def select(self, dimensions: Sequence[int], *, split: str = "train") -> ShapeBucket:
        if split not in {"train", "validation"}:
            raise DataExecutionError(f"unknown shape-bucket split {split!r}")
        candidates = self.train if split == "train" else self.validation
        requested = tuple(int(value) for value in dimensions)
        for bucket in candidates:
            if len(requested) == len(bucket.dimensions) and all(
                actual <= target for actual, target in zip(requested, bucket.dimensions, strict=True)
            ):
                return bucket
        raise DataExecutionError(f"shape {requested} is not covered by {split} buckets")

    def warmup(self, loader: Iterable[Any], *, steps_per_bucket: int = 1) -> Iterator[Any]:
        """Yield a bounded prefix so callers can execute each bucket before measurement."""
        if steps_per_bucket < 1:
            raise ValueError("steps_per_bucket must be positive")
        planned = {bucket.name for bucket in self.all()}
        counts: dict[str, int] = {}
        for batch in loader:
            name = bucket_name(batch)
            if name is None:
                yield batch
                continue
            if name not in planned:
                yield batch
                continue
            if counts.get(name, 0) < steps_per_bucket:
                counts[name] = counts.get(name, 0) + 1
                yield batch
            if planned and all(counts.get(bucket, 0) >= steps_per_bucket for bucket in planned):
                break

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [bucket.to_dict() for bucket in self.train],
            "validation": [bucket.to_dict() for bucket in self.validation],
            "max_buckets": self.max_buckets,
        }


class BackendDataLoader:
    """Use the backend's device loader without exposing runtime APIs to tasks."""

    def __init__(self, loader: Iterable[Any], backend: AcceleratorBackend) -> None:
        self.loader = backend.prepare_dataloader(loader)
        self.backend = backend

    def __iter__(self) -> Iterator[Any]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def state_dict(self) -> dict[str, Any]:
        state = getattr(self.loader, "state_dict", None)
        return dict(state()) if callable(state) else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loader = getattr(self.loader, "load_state_dict", None)
        if callable(loader):
            loader(state)


def set_epoch(loader: Any, epoch: int) -> None:
    """Set every sampler-like object deterministically for a new epoch."""
    candidates = [getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None), loader]
    for candidate in candidates:
        setter = getattr(candidate, "set_epoch", None)
        if callable(setter):
            setter(epoch)


def mark_padded_batch(batch: MedicalBatch, real_count: int) -> MedicalBatch:
    """Attach a sample mask so padded distributed entries are loss-invisible."""
    batch_size = len(batch.sample_ids)
    if real_count < 0 or real_count > batch_size:
        raise DataExecutionError(f"real_count {real_count} is outside batch size {batch_size}")
    mask = torch.zeros(batch_size, dtype=torch.bool, device=_batch_reference_device(batch))
    mask[:real_count] = True
    targets = dict(batch.task_targets)
    targets["sample_mask"] = mask
    return replace(batch, task_targets=targets)


def bucket_name(batch: Any) -> str | None:
    bucket = getattr(batch, "bucket", None)
    if bucket is None:
        return None
    name = getattr(bucket, "name", None)
    return str(name) if name is not None else str(bucket)


def _batch_reference_device(batch: MedicalBatch) -> torch.device:
    for name in ("pixel_values", "input_ids", "labels"):
        value = getattr(batch, name, None)
        if isinstance(value, torch.Tensor):
            return value.device
    for value in batch.task_targets.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


__all__ = [
    "BackendDataLoader",
    "DataExecutionError",
    "DeterministicDistributedSampler",
    "ShapeBucket",
    "ShapeBucketPlan",
    "bucket_name",
    "mark_padded_batch",
    "set_epoch",
]
