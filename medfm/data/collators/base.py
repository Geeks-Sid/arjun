"""Collator base class, final-batch policy, and shared collation helpers.

Every collator consumes a list of per-example dicts (``Example``) and returns
a :class:`medfm.core.batch.MedicalBatch` — never a loose dict. Conventional
example keys: ``sample_id`` (required), ``modality`` (required), ``image``,
``images``, ``volumes``, ``tiles``, ``tile_coordinates``, ``label``, ``mask``,
``input_ids``/``attention_mask``, ``lm_labels``, ``visual_tokens``,
``visual_token_mask``, and ``spatial``.

Modality contract: single-modality collators reject any example whose
declared ``modality`` differs from the collator's — mixed incompatible
modalities fail; only :class:`~medfm.data.collators.multitask.MultitaskCollator`
may mix, and only across its declared modalities.

Final-batch contract (ADR 0008): distributed training needs stable per-replica
shapes, so a short final batch is padded (copies of the last example, fully
masked out) or dropped (``__call__`` returns ``None``) according to
:class:`FinalBatchPolicy`. The policy applies ONLY when ``training=True``;
when ``training=False`` collators never drop or pad — evaluation samples are
sacred and output counts always equal input counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.core.enums import Modality, StrictStrEnum
from medfm.core.sample import SpatialMetadata
from medfm.data.collators.buckets import BucketPlan, pad_to_shape
from medfm.data.errors import CollatorError

#: Per-example collation input: a dict of tensors/metadata keyed by convention.
Example = dict[str, Any]

#: Marker injected on synthetic examples created by final-batch padding.
PADDED_EXAMPLE_KEY = "_padded_example"


class FinalBatchPolicy(StrictStrEnum):
    """What a training collator does with a short final batch."""

    PAD = "pad"  # pad with fully-masked copies of the last example
    DROP = "drop"  # drop the batch (``__call__`` returns None)


def example_modality(example: Example) -> Modality:
    """The example's declared modality (required, strictly parsed)."""
    value = example.get("modality")
    if value is None:
        raise CollatorError(f"sample {example.get('sample_id')!r} does not declare a 'modality'")
    if isinstance(value, Modality):
        return value
    return Modality.from_value(str(value))


def is_padded_example(example: Example) -> bool:
    """True for synthetic examples injected by final-batch padding."""
    return bool(example.get(PADDED_EXAMPLE_KEY, False))


def require_tensor(example: Example, key: str) -> torch.Tensor:
    value = example.get(key)
    if not isinstance(value, torch.Tensor):
        raise CollatorError(f"sample {example.get('sample_id')!r} requires a tensor '{key}'")
    return value


def spatial_metadata_of(examples: Sequence[Example]) -> list[SpatialMetadata | None]:
    """Per-example ``spatial`` metadata, order preserved."""
    metadata: list[SpatialMetadata | None] = []
    for example in examples:
        value = example.get("spatial")
        if value is not None and not isinstance(value, SpatialMetadata):
            raise CollatorError(
                f"sample {example.get('sample_id')!r} 'spatial' must be SpatialMetadata or None; "
                f"got {type(value).__name__}"
            )
        metadata.append(value)
    return metadata


def max_shape(shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    """Elementwise maximum over same-rank shapes."""
    if not shapes:
        raise CollatorError("cannot reduce an empty shape list")
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise CollatorError(f"shape ranks differ across the batch: {list(shapes)}")
    return tuple(max(s[i] for s in shapes) for i in range(rank))


def fit_tensor(tensor: torch.Tensor, target: tuple[int, ...], pad_value: float = 0.0) -> torch.Tensor:
    """Crop trailing-dim overflow (``pad_to_max`` policy only), then pad to ``target``."""
    rank = len(target)
    current = tuple(int(d) for d in tensor.shape[-rank:])
    if current == target:
        return tensor
    index = [slice(None)] * (tensor.ndim - rank) + [slice(0, min(c, t)) for c, t in zip(current, target, strict=True)]
    cropped = tensor[tuple(index)]
    padded, _ = pad_to_shape(cropped, target, pad_value)
    return padded


@dataclass(frozen=True)
class TextCollate:
    """Collated text block: padded token ids, mask, optional LM labels, bucket."""

    input_ids: torch.Tensor  # [B, L] int64
    attention_mask: torch.Tensor  # [B, L] bool
    lm_labels: torch.Tensor | None  # [B, L] int64, -100 outside supervised spans
    bucket: BucketId | None


class Collator(ABC):
    """Base class for all Phase 04 collators.

    Subclasses declare ``supported_modalities``, implement :meth:`_collate`,
    and inherit modality checks, sample-order guards, final-batch policy, and
    bucket/text/image padding helpers. ``static=True`` requires a
    :class:`BucketPlan` with ``mode="static"``; every padded dimension is then
    covered by a mask and the batch carries its :class:`BucketId`.
    """

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset()

    modality: Modality | None
    bucket_plan: BucketPlan | None
    static: bool
    final_batch_policy: FinalBatchPolicy

    def __init__(
        self,
        modality: Modality,
        bucket_plan: BucketPlan | None = None,
        *,
        static: bool = False,
        final_batch_policy: FinalBatchPolicy = FinalBatchPolicy.PAD,
    ) -> None:
        if modality not in self.supported_modalities:
            raise CollatorError(
                f"{type(self).__name__} does not support modality {modality.value}; "
                f"supported: {sorted(m.value for m in self.supported_modalities)}"
            )
        if static and bucket_plan is None:
            raise CollatorError(f"{type(self).__name__}(static=True) requires a BucketPlan")
        if static and bucket_plan is not None and bucket_plan.mode != "static":
            raise CollatorError(f"{type(self).__name__}(static=True) requires a BucketPlan with mode='static'")
        self.modality = modality
        self.bucket_plan = bucket_plan
        self.static = static
        self.final_batch_policy = FinalBatchPolicy(final_batch_policy)

    def __call__(
        self,
        examples: Sequence[Example],
        *,
        training: bool = False,
        target_batch_size: int | None = None,
    ) -> MedicalBatch | None:
        """Collate ``examples`` into a :class:`MedicalBatch`.

        Returns ``None`` only when ``training=True``, ``target_batch_size`` is
        set, the batch is short, and the final-batch policy is ``DROP``. When
        ``training=False`` the output always contains exactly the input
        samples — evaluation samples are never dropped or padded.
        """
        prepared = self._prepare_examples(examples, training=training, target_batch_size=target_batch_size)
        if prepared is None:
            return None
        self._check_modalities(prepared)
        batch = self._collate(prepared)
        self._order_guard(prepared, batch)
        return batch

    @abstractmethod
    def _collate(self, examples: list[Example]) -> MedicalBatch:
        """Build the batch from validated, policy-adjusted examples."""

    # ------------------------------------------------------------------ #
    # Shared validation
    # ------------------------------------------------------------------ #

    def _check_modalities(self, examples: list[Example]) -> None:
        if self.modality is None:
            raise CollatorError(f"{type(self).__name__} must declare a modality to check examples against")
        for example in examples:
            actual = example_modality(example)
            if actual is not self.modality:
                raise CollatorError(
                    f"{type(self).__name__} for {self.modality.value} received a {actual.value} example "
                    f"(sample {example.get('sample_id')!r}); mixed incompatible modalities fail — only "
                    "MultitaskCollator may mix declared modalities"
                )

    def _order_guard(self, examples: list[Example], batch: MedicalBatch) -> None:
        expected = [str(e["sample_id"]) for e in examples]
        if list(batch.sample_ids) != expected:
            raise CollatorError(
                f"collator output sample order {list(batch.sample_ids)} does not preserve input order {expected}"
            )

    def _prepare_examples(
        self,
        examples: Sequence[Example],
        *,
        training: bool,
        target_batch_size: int | None,
    ) -> list[Example] | None:
        prepared = [dict(e) for e in examples]
        if not prepared:
            raise CollatorError("cannot collate an empty example list")
        for example in prepared:
            sample_id = example.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise CollatorError("every example requires a non-empty string 'sample_id'")
        if target_batch_size is not None and target_batch_size <= 0:
            raise CollatorError(f"target_batch_size must be positive; got {target_batch_size}")
        if not training or target_batch_size is None or len(prepared) == target_batch_size:
            # Evaluation samples are sacred: never dropped, never padded.
            return prepared
        if len(prepared) > target_batch_size:
            raise CollatorError(
                f"batch of {len(prepared)} exceeds target_batch_size={target_batch_size}; fix the sampler"
            )
        if self.final_batch_policy is FinalBatchPolicy.DROP:
            return None
        anchor = prepared[-1]
        for pad_index in range(target_batch_size - len(prepared)):
            replica = dict(anchor)
            replica["sample_id"] = f"{anchor['sample_id']}::pad{pad_index}"
            replica[PADDED_EXAMPLE_KEY] = True
            prepared.append(replica)
        return prepared

    # ------------------------------------------------------------------ #
    # Shared padding helpers
    # ------------------------------------------------------------------ #

    def _unified_target(
        self,
        kind: BucketKind,
        shapes: Sequence[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], BucketId | None]:
        """One padded shape covering every example, plus its bucket when static.

        Static mode assigns each shape its bucket and unifies to the smallest
        bucket covering all assignments, so every batch lands on a declared
        shape. Dynamic mode (or a kind absent from the plan) pads to the
        per-batch maximum and returns no bucket.
        """
        if not shapes:
            raise CollatorError(f"no shapes to collate for {kind.value}")
        for shape in shapes:
            if len(shape) != kind.rank or any(d <= 0 for d in shape):
                raise CollatorError(f"{kind.value} requires positive rank-{kind.rank} shapes; got {shape}")
        if self.static and self.bucket_plan is not None and self.bucket_plan.has_kind(kind):
            assigned = [self.bucket_plan.assign(kind, shape) for shape in shapes]
            upper = max_shape([bucket.shape for bucket in assigned])
            unified = self.bucket_plan.assign(kind, upper)
            return unified.shape, unified
        return max_shape(shapes), None

    def _collate_text(self, examples: list[Example]) -> TextCollate:
        """Pad ``input_ids``/``attention_mask``/``lm_labels`` to a shared length.

        Text is tokenized upstream; examples carry 1-D ``input_ids`` (required)
        and optional ``attention_mask`` (defaults to all-real) and ``lm_labels``
        (same length; padded with -100 so padding is never supervised).
        """
        ids_list: list[torch.Tensor] = []
        for example in examples:
            ids = require_tensor(example, "input_ids")
            if ids.ndim != 1:
                raise CollatorError(
                    f"sample {example['sample_id']!r} input_ids must be 1-D [L]; got {tuple(ids.shape)}"
                )
            ids_list.append(ids.to(torch.int64))
        target, bucket = self._unified_target(BucketKind.TEXT_TOKENS, [(int(ids.shape[0]),) for ids in ids_list])
        length = target[0]
        batch_size = len(examples)
        input_ids = torch.zeros(batch_size, length, dtype=torch.int64)
        attention_mask = torch.zeros(batch_size, length, dtype=torch.bool)
        wants_lm = any("lm_labels" in example for example in examples)
        lm_labels = torch.full((batch_size, length), -100, dtype=torch.int64) if wants_lm else None
        for row, (example, ids) in enumerate(zip(examples, ids_list, strict=True)):
            real = int(ids.shape[0])
            input_ids[row, :real] = ids
            if not is_padded_example(example):
                raw_mask = example.get("attention_mask")
                if raw_mask is None:
                    attention_mask[row, :real] = True
                else:
                    mask = torch.as_tensor(raw_mask)
                    if mask.ndim != 1 or int(mask.shape[0]) != real:
                        raise CollatorError(
                            f"sample {example['sample_id']!r} attention_mask must be 1-D with length {real}"
                        )
                    attention_mask[row, :real] = mask.to(torch.bool)
            if lm_labels is not None:
                raw_labels = example.get("lm_labels")
                if raw_labels is None:
                    raise CollatorError(
                        f"sample {example['sample_id']!r} is missing 'lm_labels' carried by the rest of the batch"
                    )
                labels = torch.as_tensor(raw_labels)
                if labels.ndim != 1 or int(labels.shape[0]) != real:
                    raise CollatorError(
                        f"sample {example['sample_id']!r} lm_labels must be 1-D with length {real} (matching input_ids)"
                    )
                lm_labels[row, :real] = labels.to(torch.int64)
        return TextCollate(input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels, bucket=bucket)
