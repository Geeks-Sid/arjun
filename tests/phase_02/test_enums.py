"""Enum strictness, migration, and identifier hygiene tests."""

import pytest

from medfm.core import (
    IdentifierError,
    LoadingMode,
    Modality,
    PrecisionMode,
    SplitName,
    TaskType,
    UnknownEnumValueError,
    patient_id_hash,
    register_enum_migration,
    series_id_hash,
    study_id_hash,
)
from medfm.core.enums import CoordinateSystem


def test_modality_enum_matches_governance_doc():
    expected = {
        "XRAY_2D",
        "CT_2D_SLICE",
        "CT_3D",
        "MRI_2D_SLICE",
        "MRI_3D",
        "PATHOLOGY_TILE",
        "PATHOLOGY_WSI",
        "MULTI_IMAGE_2D",
        "MULTI_SERIES_3D",
        "TEXT_ONLY",
    }
    assert {m.value for m in Modality} == expected


def test_task_enum_matches_governance_doc():
    expected = {
        "BINARY_CLASSIFICATION",
        "MULTICLASS_CLASSIFICATION",
        "MULTILABEL_CLASSIFICATION",
        "ORDINAL_CLASSIFICATION",
        "IMAGE_TEXT_RETRIEVAL",
        "TEXT_IMAGE_RETRIEVAL",
        "SEMANTIC_SEGMENTATION",
        "INSTANCE_SEGMENTATION",
        "PROMPTABLE_SEGMENTATION",
        "LANGUAGE_CONDITIONED_SEGMENTATION",
        "BOUNDING_BOX_LOCALIZATION",
        "VISUAL_QUESTION_ANSWERING",
        "REPORT_GENERATION",
        "STRUCTURED_FINDING_GENERATION",
        "CONTRASTIVE_ALIGNMENT",
        "MULTITASK",
    }
    assert {t.value for t in TaskType} == expected


def test_auxiliary_enums_cover_loading_coordinate_precision_split():
    assert {m.value for m in LoadingMode} == {"FULL", "FROZEN", "LORA", "QLORA_NF4"}
    assert {c.value for c in CoordinateSystem} == {
        "NORMALIZED_IMAGE",
        "MILLIMETERS",
        "MICRONS",
        "SLIDE_PIXELS",
    }
    assert {p.value for p in PrecisionMode} == {"FP32", "FP16", "BF16"}
    assert {s.value for s in SplitName} == {"TRAIN", "VAL", "TEST", "EXTERNAL_VAL", "TEMPORAL_VAL"}


def test_unknown_values_rejected_with_legal_value_listing():
    with pytest.raises(UnknownEnumValueError, match="XRAY_2D"):
        Modality.from_value("CHEST_XRAY")
    with pytest.raises(UnknownEnumValueError):
        TaskType.from_value("detection")


def test_versioned_enum_migration_applies_only_with_version():
    register_enum_migration("Modality", 0, "RADIOGRAPH_2D", "XRAY_2D")
    assert Modality.from_value("RADIOGRAPH_2D", schema_version=0) is Modality.XRAY_2D
    # Without a version context the retired value is still rejected.
    with pytest.raises(UnknownEnumValueError):
        Modality.from_value("RADIOGRAPH_2D")


def test_pixel_rank_contract_is_per_modality():
    assert Modality.CT_3D.expected_pixel_rank == 5
    assert Modality.MULTI_IMAGE_2D.expected_pixel_rank == 5  # same rank, different meaning
    assert Modality.PATHOLOGY_WSI.expected_pixel_rank == 5
    assert Modality.MULTI_SERIES_3D.expected_pixel_rank == 6
    assert Modality.TEXT_ONLY.expected_pixel_rank is None


def test_hash_id_factories_accept_digests():
    digest = "ab12" * 16
    assert patient_id_hash(digest) == digest
    assert study_id_hash(digest) == digest
    assert series_id_hash(digest) == digest


@pytest.mark.parametrize(
    "raw",
    [
        "1.2.840.113619.2.55.3",  # DICOM UID
        "0001234567",  # numeric MRN
        "PATIENT-0042",  # non-hex identifier
        "AB12" * 16,  # uppercase hex (enforce canonical lowercase)
    ],
)
def test_hash_id_factories_reject_raw_identifiers(raw):
    with pytest.raises(IdentifierError):
        patient_id_hash(raw)
