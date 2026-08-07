"""Split generation + leakage protection (ADR 0004)."""

from __future__ import annotations

import pandas as pd
import pytest

from medfm.core.enums import SplitName
from medfm.data.errors import SplitLeakageError
from medfm.data.splits import (
    ResearchOverride,
    SplitPolicy,
    assert_no_split_leakage,
    build_split_report,
    check_split_leakage,
    generate_split_assignment,
)
from phase_03.synthetic import hid, manifest_row


def _frame(n_patients: int = 12, studies_per_patient: int = 1) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        for s in range(studies_per_patient):
            rows.append(
                manifest_row(
                    sample_id=f"sample-{p}-{s}",
                    patient_id_hash=hid(f"patient-{p}"),
                    study_id_hash=hid(f"study-{p}-{s}"),
                    series_id_hash=hid(f"series-{p}-{s}"),
                    image_uri=f"images/{p}-{s}.nii.gz",
                    image_sha256=hid(f"payload-{p}-{s}"),
                    split=None,
                )
            )
    return pd.DataFrame(rows)


# --- generation ------------------------------------------------------------


def test_patient_policy_is_patient_disjoint() -> None:
    df = generate_split_assignment(_frame(20, studies_per_patient=2), policy=SplitPolicy.PATIENT, seed=42)
    per_patient = df.groupby("patient_id_hash")["split"].nunique()
    assert (per_patient == 1).all(), "one patient spans multiple splits"
    assert set(df["split"].unique()) <= {"TRAIN", "VAL", "TEST"}
    assert len(df) == 40


def test_assignment_is_deterministic_in_seed_and_row_order() -> None:
    df = _frame(15)
    first = generate_split_assignment(df, policy=SplitPolicy.PATIENT, seed=7)
    same = generate_split_assignment(df, policy=SplitPolicy.PATIENT, seed=7)
    shuffled = generate_split_assignment(df.iloc[::-1].reset_index(drop=True), policy=SplitPolicy.PATIENT, seed=7)
    assert (first["split"] == same["split"]).all()
    left = first.set_index("sample_id")["split"].sort_index()
    right = shuffled.set_index("sample_id")["split"].sort_index()
    assert (left == right).all(), "row order must not change assignment"
    other_seed = generate_split_assignment(df, policy=SplitPolicy.PATIENT, seed=8)
    assert not (first["split"] == other_seed["split"]).all()


def test_group_id_hash_keeps_tiles_together() -> None:
    rows = []
    for slide in range(6):
        for tile in range(4):
            rows.append(
                manifest_row(
                    sample_id=f"tile-{slide}-{tile}",
                    patient_id_hash=hid(f"patient-{slide % 2}"),
                    group_id_hash=hid(f"slide-{slide}"),
                    image_uri=f"tiles/{slide}-{tile}.png",
                )
            )
    df = generate_split_assignment(pd.DataFrame(rows), policy=SplitPolicy.PATIENT, seed=11)
    per_slide = df.groupby("group_id_hash")["split"].nunique()
    assert (per_slide == 1).all(), "tiles of one slide crossed splits"


def test_site_policy_is_site_disjoint() -> None:
    rows = []
    for i in range(120):
        rows.append(manifest_row(sample_id=f"s-{i}", patient_id_hash=hid(f"p-{i}"), site_id=f"site-{i % 20}"))
    df = generate_split_assignment(pd.DataFrame(rows), policy=SplitPolicy.SITE, seed=3)
    per_site = df.groupby("site_id")["split"].nunique()
    assert (per_site == 1).all(), "one site spans multiple splits"
    # 20 sites across 70/10/20 ratios: expect all three splits to be populated.
    assert {"TRAIN", "VAL", "EXTERNAL_VAL"} <= set(df["split"].unique())


def test_temporal_policy_holds_out_the_tail_of_time() -> None:
    rows = []
    for i in range(40):
        rows.append(
            manifest_row(
                sample_id=f"s-{i}",
                patient_id_hash=hid(f"p-{i % 10}"),
                acquisition_date_bucket=f"201{9 + i // 20}-Q{(i % 4) + 1}",
            )
        )
    df = pd.DataFrame(rows)
    assigned = generate_split_assignment(df, policy=SplitPolicy.TEMPORAL, seed=5)
    buckets = sorted(df["acquisition_date_bucket"].unique())
    last_bucket = buckets[-1]
    tail = assigned[assigned["acquisition_date_bucket"] == last_bucket]
    assert (tail["split"] == "TEMPORAL_VAL").all()
    assert "TRAIN" in set(assigned["split"].unique())


def test_invalid_ratios_rejected() -> None:
    df = _frame(4)
    with pytest.raises(SplitLeakageError, match="positive"):
        generate_split_assignment(df, seed=1, ratios=((SplitName.TRAIN, 0.0), (SplitName.VAL, 1.0)))
    with pytest.raises(SplitLeakageError, match="duplicate"):
        generate_split_assignment(df, seed=1, ratios=((SplitName.TRAIN, 0.5), (SplitName.TRAIN, 0.5)))


# --- leakage checks ---------------------------------------------------------


def test_clean_manifest_passes_leakage_check() -> None:
    df = generate_split_assignment(_frame(10, studies_per_patient=2), policy=SplitPolicy.PATIENT, seed=9)
    report = check_split_leakage(df)
    assert report.ok
    assert assert_no_split_leakage(df).ok


def test_patient_leakage_detected() -> None:
    df = generate_split_assignment(_frame(10), policy=SplitPolicy.PATIENT, seed=9)
    leaked = df.copy()
    # Force one patient into a second split.
    patient = leaked["patient_id_hash"].iloc[0]
    victim = leaked.index[leaked.index[-1]]
    leaked.loc[victim, "patient_id_hash"] = patient
    leaked.loc[victim, "split"] = "TEST" if leaked.loc[victim, "split"] != "TEST" else "VAL"
    report = check_split_leakage(leaked)
    assert not report.ok
    kinds = {v.kind for v in report.violations}
    assert "patient_id_hash" in kinds


def test_duplicate_content_hash_across_splits_detected() -> None:
    df = generate_split_assignment(_frame(10), policy=SplitPolicy.PATIENT, seed=9)
    leaked = df.copy()
    shared = hid("shared-payload")
    leaked.loc[leaked.index[0], "image_sha256"] = shared
    leaked.loc[leaked.index[-1], "image_sha256"] = shared
    report = check_split_leakage(leaked)
    assert any(v.kind == "image_sha256" for v in report.violations)


def test_group_leakage_detected() -> None:
    rows = []
    for tile in range(4):
        rows.append(
            manifest_row(
                sample_id=f"tile-{tile}",
                patient_id_hash=hid("patient-0"),
                group_id_hash=hid("slide-0"),
                split="TRAIN" if tile < 2 else "TEST",
            )
        )
    report = check_split_leakage(pd.DataFrame(rows))
    assert any(v.kind == "group_id_hash" for v in report.violations)


def test_assert_no_split_leakage_raises_with_hashes_only() -> None:
    df = generate_split_assignment(_frame(10), policy=SplitPolicy.PATIENT, seed=9)
    leaked = df.copy()
    patient = leaked["patient_id_hash"].iloc[0]
    leaked.loc[leaked.index[-1], "patient_id_hash"] = patient
    leaked.loc[leaked.index[-1], "split"] = "TEST" if leaked.loc[leaked.index[-1], "split"] != "TEST" else "VAL"
    with pytest.raises(SplitLeakageError, match="split leakage detected"):
        assert_no_split_leakage(leaked)


def test_research_override_allows_documented_exception() -> None:
    df = generate_split_assignment(_frame(10), policy=SplitPolicy.PATIENT, seed=9)
    leaked = df.copy()
    patient = leaked["patient_id_hash"].iloc[0]
    leaked.loc[leaked.index[-1], "patient_id_hash"] = patient
    leaked.loc[leaked.index[-1], "split"] = "TEST" if leaked.loc[leaked.index[-1], "split"] != "TEST" else "VAL"
    override = ResearchOverride(
        reason="benchmark comparability", recorded_by="maintainer", recorded_at="2026-08-04T00:00:00Z"
    )
    report = assert_no_split_leakage(leaked, research_override=override)
    assert not report.ok  # still reported; the override only gates training
    with pytest.raises(SplitLeakageError, match="must be non-empty"):
        ResearchOverride(reason="", recorded_by="", recorded_at="")


def test_temporal_policy_exempts_patient_overlap_but_not_content() -> None:
    rows = []
    for i in range(20):
        rows.append(
            manifest_row(
                sample_id=f"s-{i}",
                patient_id_hash=hid("p-0"),  # same patient on both sides
                acquisition_date_bucket="2019-Q1" if i < 10 else "2021-Q1",
                split="TRAIN" if i < 10 else "TEMPORAL_VAL",
                image_sha256=hid(f"payload-{i}"),
            )
        )
    df = pd.DataFrame(rows)
    assert check_split_leakage(df, temporal_policy=True).ok
    assert not check_split_leakage(df, temporal_policy=False).ok


# --- audit report -----------------------------------------------------------


def test_split_report_is_deterministic_and_auditable() -> None:
    df = generate_split_assignment(_frame(12), policy=SplitPolicy.PATIENT, seed=21)
    ratios = ((SplitName.TRAIN, 0.7), (SplitName.VAL, 0.15), (SplitName.TEST, 0.15))
    first = build_split_report(df, policy=SplitPolicy.PATIENT, seed=21, ratios=ratios)
    second = build_split_report(df, policy=SplitPolicy.PATIENT, seed=21, ratios=ratios)
    assert first.report_hash == second.report_hash
    payload = first.to_dict()
    assert payload["seed"] == 21
    assert payload["policy"] == "PATIENT"
    assert "patient_id_hash" in payload["grouping_key"]
    assert sum(payload["samples_per_split"].values()) == 12
    assert payload["report_hash"] == first.report_hash
