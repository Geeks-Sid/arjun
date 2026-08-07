"""MedicalSample construction, modality/task validation, and round-trips."""

import pytest

from medfm.core import (
    BoxTarget,
    ConversationTurn,
    CoordinateSystem,
    IdentifierError,
    ImageReference,
    LabelTarget,
    MedicalSample,
    Modality,
    SchemaValidationError,
    SegmentationTarget,
    TaskType,
)
from phase_02.contract_fixtures import HASH64, make_provenance, make_sample, make_spatial

ALL_MODALITIES = tuple(Modality)


@pytest.mark.parametrize("modality", ALL_MODALITIES, ids=lambda m: m.value)
def test_synthetic_sample_validates_for_every_modality(modality):
    sample = make_sample(modality)
    assert sample.modality is modality
    assert sample.provenance.dataset_name == "synthetic"


def test_modality_specific_metadata_requirements():
    # 3D requires spatial metadata with geometry.
    with pytest.raises(SchemaValidationError, match="SpatialMetadata"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.CT_3D,
            image_references=(ImageReference(uri="s3://b/v.nii.gz"),),
            provenance=make_provenance(),
        )
    # Pathology requires pathology metadata.
    with pytest.raises(SchemaValidationError, match="PathologyMetadata"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.PATHOLOGY_WSI,
            image_references=(ImageReference(uri="s3://b/slide.svs"),),
            provenance=make_provenance(),
        )
    # 3D spatial metadata must carry affine or spacing.
    with pytest.raises(SchemaValidationError, match="affine or spacing"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.MRI_3D,
            image_references=(ImageReference(uri="s3://b/v.nii.gz"),),
            spatial=make_spatial().__class__(original_shape=(8, 8, 8), current_shape=(8, 8, 8)),
            provenance=make_provenance(),
        )


def test_text_only_forbids_images_and_requires_text():
    with pytest.raises(SchemaValidationError, match="image_references"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.TEXT_ONLY,
            image_references=(ImageReference(uri="s3://b/x.png"),),
            report="text",
            provenance=make_provenance(),
        )
    with pytest.raises(SchemaValidationError, match="text content"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.TEXT_ONLY,
            provenance=make_provenance(),
        )


def test_image_modalities_require_image_references():
    with pytest.raises(SchemaValidationError, match="image reference"):
        MedicalSample(
            sample_id="x",
            patient_id_hash=HASH64,
            modality=Modality.XRAY_2D,
            provenance=make_provenance(),
        )


def test_raw_patient_identifier_rejected():
    with pytest.raises(IdentifierError):
        MedicalSample(
            sample_id="x",
            patient_id_hash="1.2.840.113619.2.176.1",
            modality=Modality.TEXT_ONLY,
            report="r",
            provenance=make_provenance(),
        )


def test_validate_for_task_requirements():
    ct = make_sample(Modality.CT_3D)
    ct.validate_for_task(TaskType.BINARY_CLASSIFICATION)  # has labels
    with pytest.raises(SchemaValidationError, match="segmentation"):
        ct.validate_for_task(TaskType.SEMANTIC_SEGMENTATION)
    with pytest.raises(SchemaValidationError, match="boxes"):
        ct.validate_for_task(TaskType.BOUNDING_BOX_LOCALIZATION)

    vqa = make_sample(Modality.TEXT_ONLY)
    vqa.validate_for_task(TaskType.VISUAL_QUESTION_ANSWERING)  # question + answer
    with pytest.raises(SchemaValidationError, match="report"):
        make_sample(Modality.XRAY_2D).validate_for_task(TaskType.REPORT_GENERATION)


def test_label_target_semantics():
    LabelTarget(task=TaskType.BINARY_CLASSIFICATION, values=(1.0,))
    LabelTarget(task=TaskType.MULTILABEL_CLASSIFICATION, values=(1.0, 0.0, 1.0))
    LabelTarget(task=TaskType.MULTICLASS_CLASSIFICATION, values=(2.0,), class_names=("a", "b", "c"))
    with pytest.raises(SchemaValidationError):
        LabelTarget(task=TaskType.BINARY_CLASSIFICATION, values=(0.5,))
    with pytest.raises(SchemaValidationError):
        LabelTarget(task=TaskType.MULTILABEL_CLASSIFICATION, values=(2.0,))
    with pytest.raises(SchemaValidationError):
        LabelTarget(task=TaskType.SEMANTIC_SEGMENTATION, values=(1.0,))


def test_box_target_coordinate_system_and_geometry():
    boxes = BoxTarget(
        boxes=((0.1, 0.1, 0.5, 0.5),),
        labels=("nodule",),
        coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
    )
    assert boxes.coordinate_system is CoordinateSystem.NORMALIZED_IMAGE
    with pytest.raises(SchemaValidationError):
        BoxTarget(boxes=((0.5, 0.5, 0.1, 0.1),), labels=("x",), coordinate_system=CoordinateSystem.MILLIMETERS)
    with pytest.raises(SchemaValidationError):
        BoxTarget(boxes=((1.0, 2.0),), labels=("x",), coordinate_system=CoordinateSystem.MILLIMETERS)


def test_conversation_turn_roles():
    ConversationTurn(role="user", content="hi")
    with pytest.raises(SchemaValidationError):
        ConversationTurn(role="radiologist", content="hi")


def test_full_sample_roundtrip_through_canonical_dict():
    sample = make_sample(Modality.PATHOLOGY_WSI)
    sample.validate_for_task(TaskType.BINARY_CLASSIFICATION)
    restored = MedicalSample.from_dict(sample.to_dict())
    assert restored.to_dict() == sample.to_dict()


def test_segmentation_target_roundtrip():
    seg = SegmentationTarget(mask_uri="s3://b/m.nii.gz", class_index_to_name={0: "bg", 1: "tumor"})
    assert SegmentationTarget.from_dict(seg.to_dict()) == seg


def test_pathology_metadata_available_on_sample():
    sample = make_sample(Modality.PATHOLOGY_TILE)
    assert sample.pathology is not None
    assert sample.pathology.microns_per_pixel == 0.25
