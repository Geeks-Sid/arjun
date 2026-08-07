"""Dataset fingerprinting: deterministic statistics + bounded bucket hints."""

from __future__ import annotations

import pandas as pd

from medfm.data.fingerprint import fingerprint_manifest, recommend_shape_buckets
from phase_03.synthetic import build_mixed_manifest, hid, manifest_row


def test_fingerprint_counts_and_distributions() -> None:
    df = build_mixed_manifest(patients=12, seed=7)
    report = fingerprint_manifest(df)
    assert report["counts"]["samples"] == 12
    assert report["counts"]["patients"] == 12
    assert report["modality_counts"]["CT_3D"] == 2
    assert report["split_leakage"]["ok"] is True
    assert report["fingerprint_hash"]


def test_fingerprint_is_deterministic() -> None:
    df = build_mixed_manifest(patients=10, seed=3)
    first = fingerprint_manifest(df)
    second = fingerprint_manifest(df.copy())
    assert first == second
    assert first["fingerprint_hash"] == second["fingerprint_hash"]
    # Reordering rows must not change the hash.
    shuffled = fingerprint_manifest(df.iloc[::-1].reset_index(drop=True))
    assert shuffled["fingerprint_hash"] == first["fingerprint_hash"]


def test_shape_spacing_and_intensity_stats_present() -> None:
    df = build_mixed_manifest(patients=8, seed=5)
    report = fingerprint_manifest(df)
    assert report["shape_stats"], "shape stats expected for volumetric rows"
    assert report["spacing_stats"]
    assert report["intensity_stats"]["rows_with_stats"] > 0
    assert report["missing_values"], "missing-value report expected"
    assert report["site_distribution"]
    assert report["vendor_distribution"]


def test_wsi_and_report_stats() -> None:
    df = build_mixed_manifest(patients=12, seed=7)
    report = fingerprint_manifest(df)
    assert report["wsi_microns_per_pixel_stats"]["count"] >= 1
    assert report["wsi_magnification_stats"]["count"] >= 1
    assert report["report_chars_stats"]["count"] >= 1
    assert report["segmentation_volume_stats"]["rows_with_segmentation"] >= 1


def test_label_prevalence_counts_tasks() -> None:
    df = build_mixed_manifest(patients=12, seed=7)
    report = fingerprint_manifest(df)
    prevalence = report["label_prevalence"]
    assert prevalence["labeled_rows"] > 0
    assert "BINARY_CLASSIFICATION" in prevalence["task_counts"]


def test_duplicate_content_hashes_flagged() -> None:
    rows = [
        manifest_row(sample_id="a", patient_id_hash=hid("p1"), image_sha256=hid("same")),
        manifest_row(sample_id="b", patient_id_hash=hid("p2"), image_sha256=hid("same")),
    ]
    report = fingerprint_manifest(pd.DataFrame(rows))
    assert report["duplicate_stats"]["duplicate_hash_count"] == 1
    assert report["duplicate_stats"]["duplicated_rows"] == 2


def test_leakage_results_embedded_in_report() -> None:
    df = build_mixed_manifest(patients=6, seed=1)
    leaked = df.copy()
    patient = leaked["patient_id_hash"].iloc[0]
    leaked.loc[leaked.index[-1], "patient_id_hash"] = patient
    leaked.loc[leaked.index[-1], "split"] = "TEST" if leaked.loc[leaked.index[-1], "split"] != "TEST" else "VAL"
    report = fingerprint_manifest(leaked)
    assert report["split_leakage"]["ok"] is False
    assert report["split_leakage"]["violation_count"] >= 1


def test_shape_bucket_recommendations_are_bounded() -> None:
    df = build_mixed_manifest(patients=24, seed=11)
    buckets = recommend_shape_buckets(df)
    kinds = {b["kind"] for b in buckets}
    assert {"2d_resolution", "tile_count", "text_length"} <= kinds
    for bucket in buckets:
        assert bucket["shape"], "every bucket needs a concrete shape"
        assert all(v > 0 for v in bucket["shape"])
    # Deterministic.
    assert recommend_shape_buckets(df) == buckets


def test_bucket_recommendation_covers_observed_shapes() -> None:
    rows = [
        manifest_row(sample_id=f"x{i}", patient_id_hash=hid(f"p{i}"), modality="XRAY_2D", shape=[1, 700, 700])
        for i in range(10)
    ]
    buckets = {b["kind"]: b for b in recommend_shape_buckets(pd.DataFrame(rows))}
    resolution = buckets["2d_resolution"]["shape"]
    assert resolution[0] >= 700 and resolution[1] >= 700


def test_fingerprint_handles_minimal_frame() -> None:
    df = pd.DataFrame([manifest_row(sample_id="only", patient_id_hash=hid("p"), modality="TEXT_ONLY", image_uri=None)])
    report = fingerprint_manifest(df)
    assert report["counts"]["samples"] == 1
    assert report["recommended_shape_buckets"] == []
