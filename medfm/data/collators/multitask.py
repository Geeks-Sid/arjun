"""Multitask collator: the only place mixed modalities may meet.

Single-modality collators reject foreign modalities; this collator explicitly
declares the modality mix it supports (one delegate :class:`Collator` per
modality) and dispatches each modality group to its delegate. Any example
whose modality was not declared raises :class:`CollatorError` — mixing is
opt-in and bounded by configuration.

A single :class:`MedicalBatch` cannot hold mixed modalities (its ``modality``
field is authoritative), so the result is a :class:`MultitaskBatch`: one
validated ``MedicalBatch`` per modality plus a ``modality_index`` recording
each input sample's modality in original input order. Within each modality
group, sample order is preserved (delegate order guards apply).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality
from medfm.data.collators.base import Collator, Example, FinalBatchPolicy, example_modality
from medfm.data.errors import CollatorError


@dataclass(frozen=True)
class MultitaskBatch:
    """Per-modality batches plus the input-order modality assignment."""

    batches: dict[Modality, MedicalBatch]
    modality_index: tuple[str, ...]  # modality value per input sample, input order
    sample_ids: tuple[str, ...]  # input-order sample ids (before final-batch padding)

    def __post_init__(self) -> None:
        if not self.batches:
            raise CollatorError("MultitaskBatch requires at least one per-modality batch")
        if len(self.modality_index) != len(self.sample_ids):
            raise CollatorError("modality_index and sample_ids must cover the same samples")


class MultitaskCollator:
    """Dispatches mixed-modality examples to declared per-modality collators.

    Parameters
    ----------
    delegates:
        Mapping of modality to the collator responsible for it. The keys are
        the *complete* allowed modality mix — anything else fails.
    final_batch_policy:
        Forwarded to each delegate for training final batches (evaluation
        samples are never dropped, matching the base collator contract).
    """

    def __init__(self, delegates: dict[Modality, Collator]) -> None:
        if not delegates:
            raise CollatorError("MultitaskCollator requires at least one delegate collator")
        for modality, collator in delegates.items():
            if not isinstance(modality, Modality):
                raise CollatorError(f"MultitaskCollator delegate keys must be Modality; got {modality!r}")
            if collator.modality is not modality:
                raise CollatorError(
                    f"delegate for {modality.value} is a {type(collator).__name__} bound to "
                    f"{collator.modality}; delegates must be bound to their declared modality"
                )
        self.delegates = dict(delegates)
        self.allowed_modalities = frozenset(delegates)

    def __call__(
        self,
        examples: Sequence[Example],
        *,
        training: bool = False,
        target_batch_size: int | None = None,
    ) -> MultitaskBatch:
        """Collate a mixed batch; undeclared modalities raise.

        ``target_batch_size`` applies per modality group (each group is padded
        or dropped per its delegate's policy when ``training=True``). A group
        dropped under the DROP policy simply has no batch in the result.
        Evaluation (``training=False``) never drops: every input sample appears
        in exactly one per-modality batch.
        """
        if not examples:
            raise CollatorError("cannot collate an empty example list")
        groups: dict[Modality, list[Example]] = {}
        modality_index: list[str] = []
        sample_ids: list[str] = []
        for example in examples:
            modality = example_modality(example)
            if modality not in self.allowed_modalities:
                raise CollatorError(
                    f"sample {example.get('sample_id')!r} has undeclared modality {modality.value}; "
                    f"MultitaskCollator only mixes {sorted(m.value for m in self.allowed_modalities)}"
                )
            groups.setdefault(modality, []).append(example)
            modality_index.append(modality.value)
            sample_id = example.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise CollatorError("every example requires a non-empty string 'sample_id'")
            sample_ids.append(sample_id)
        batches: dict[Modality, MedicalBatch] = {}
        for modality, group in groups.items():
            per_group_target = target_batch_size if training else None
            batch = self.delegates[modality](group, training=training, target_batch_size=per_group_target)
            if batch is not None:  # DROP policy: group omitted entirely
                batches[modality] = batch
        if not batches:
            raise CollatorError(
                "every modality group was dropped by the DROP final-batch policy; "
                "a zero-modality multitask batch cannot be trained on"
            )
        return MultitaskBatch(
            batches=batches,
            modality_index=tuple(modality_index),
            sample_ids=tuple(sample_ids),
        )


#: Re-exported for convenience; the policy enum lives in collators.base.
__all__ = ["FinalBatchPolicy", "MultitaskBatch", "MultitaskCollator"]
