"""Group-aware distributed sampling: exact coverage, group integrity, padding."""

from __future__ import annotations

import pandas as pd
import pytest

from medfm.data.errors import DataError
from medfm.data.samplers import (
    PADDING_INDEX,
    GroupAwareDistributedSampler,
    combine_shards_for_metrics,
    resolve_samples_before_collective,
    worker_seed,
)
from phase_03.synthetic import hid, manifest_row


def _frame(patients: int = 9, samples_per_patient: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(patients):
        for s in range(samples_per_patient):
            rows.append(
                manifest_row(
                    sample_id=f"p{p}-s{s}",
                    patient_id_hash=hid(f"patient-{p}"),
                    split="TRAIN",
                )
            )
    return pd.DataFrame(rows)


def test_shards_are_disjoint_and_cover_exactly() -> None:
    df = _frame(patients=9, samples_per_patient=3)
    num_ranks = 3
    shards = [
        GroupAwareDistributedSampler(df, num_ranks=num_ranks, rank=r, seed=1).shard_for(r) for r in range(num_ranks)
    ]
    indices, padding_removed = combine_shards_for_metrics(shards)
    assert indices == sorted(df.index.tolist())  # exact coverage, no dupes
    assert padding_removed >= 0


def test_group_intact_per_rank() -> None:
    df = _frame(patients=12, samples_per_patient=4)
    sampler = GroupAwareDistributedSampler(df, num_ranks=4, rank=0, seed=2)
    for rank in range(4):
        shard = sampler.shard_for(rank)
        patients_on_rank = {df.loc[i, "patient_id_hash"] for i in shard.real_indices}
        for patient in patients_on_rank:
            members = df.index[df["patient_id_hash"] == patient].tolist()
            assert all(m in set(shard.real_indices) for m in members), "a patient group split across ranks"


def test_padding_equalizes_shard_lengths_with_sentinels() -> None:
    # 7 patients over 3 ranks -> unequal group counts -> padding needed.
    df = _frame(patients=7, samples_per_patient=2)
    sampler = GroupAwareDistributedSampler(df, num_ranks=3, rank=0, seed=3)
    shards = [sampler.shard_for(r) for r in range(3)]
    lengths = {len(s.indices) for s in shards}
    assert len(lengths) == 1, "all ranks must run the same number of steps"
    for shard in shards:
        for index, pad in zip(shard.indices, shard.is_padding, strict=True):
            if pad:
                assert index == PADDING_INDEX
            else:
                assert index != PADDING_INDEX
    total_real = sum(len(s.real_indices) for s in shards)
    assert total_real == len(df)


def test_epoch_is_deterministic_and_covers_dataset() -> None:
    df = _frame(patients=8, samples_per_patient=3)
    sampler = GroupAwareDistributedSampler(df, num_ranks=2, rank=0, seed=4)
    # Deterministic for a given epoch.
    sampler.set_epoch(1)
    first = sampler.shard_for(0).real_indices
    sampler.set_epoch(1)
    assert sampler.shard_for(0).real_indices == first
    # Every epoch still covers the dataset exactly with groups intact.
    for epoch in (0, 1, 2):
        sampler.set_epoch(epoch)
        shards = [sampler.shard_for(r) for r in range(2)]
        indices, _ = combine_shards_for_metrics(shards)
        assert indices == sorted(df.index.tolist()), f"epoch {epoch} lost or duplicated samples"


def test_iteration_yields_padding_sentinels_too() -> None:
    df = _frame(patients=5, samples_per_patient=1)
    sampler = GroupAwareDistributedSampler(df, num_ranks=3, rank=0, seed=5)
    produced = list(iter(sampler))
    assert len(produced) == len(sampler)
    assert len(sampler) >= len(sampler.shard_for(0).real_indices)


def test_split_filter_restricts_rows() -> None:
    df = _frame(patients=6, samples_per_patient=2)
    df.loc[df.index[:4], "split"] = "VAL"
    sampler = GroupAwareDistributedSampler(df, num_ranks=2, rank=0, seed=6, split="TRAIN")
    assert sampler.total_samples == len(df) - 4


def test_group_id_hash_override_wins() -> None:
    rows = []
    for tile in range(6):
        rows.append(
            manifest_row(
                sample_id=f"t{tile}",
                patient_id_hash=hid("shared-patient"),  # one patient for everything
                group_id_hash=hid(f"slide-{tile % 3}"),
            )
        )
    df = pd.DataFrame(rows)
    sampler = GroupAwareDistributedSampler(df, num_ranks=3, rank=0, seed=7)
    for rank in range(3):
        shard = sampler.shard_for(rank)
        slides = {df.loc[i, "group_id_hash"] for i in shard.real_indices}
        for slide in slides:
            members = df.index[df["group_id_hash"] == slide].tolist()
            assert all(m in set(shard.real_indices) for m in members)


def test_invalid_rank_arguments_rejected() -> None:
    df = _frame(patients=3)
    with pytest.raises(DataError, match="num_ranks"):
        GroupAwareDistributedSampler(df, num_ranks=0, rank=0, seed=0)
    with pytest.raises(DataError, match="out of range"):
        GroupAwareDistributedSampler(df, num_ranks=2, rank=2, seed=0)
    with pytest.raises(DataError, match="grouping column"):
        GroupAwareDistributedSampler(df, num_ranks=1, rank=0, seed=0, group_column="nope")


def test_worker_seed_is_deterministic_and_distinct() -> None:
    a = worker_seed(42, epoch=0, rank=1, worker_id=2)
    b = worker_seed(42, epoch=0, rank=1, worker_id=2)
    c = worker_seed(42, epoch=0, rank=1, worker_id=3)
    d = worker_seed(42, epoch=1, rank=1, worker_id=2)
    assert a == b
    assert a != c and a != d
    assert 0 <= a <= 0x7FFFFFFF


def test_corrupt_samples_resolve_before_collectives() -> None:
    corrupt = {3, 7}
    check = lambda index: "unreadable payload" if index in corrupt else None  # noqa: E731
    resolved = resolve_samples_before_collective([0, 1, PADDING_INDEX, 3, 7, 9], check)
    assert resolved.valid_indices == (0, 1, PADDING_INDEX, 9)
    assert [i for i, _ in resolved.quarantined] == [3, 7]
    assert all("unreadable payload" in reason for _, reason in resolved.quarantined)
