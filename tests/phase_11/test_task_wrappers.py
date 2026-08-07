from __future__ import annotations

import torch

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderOutput
from medfm.core.enums import Modality
from medfm.models.decoders import LanguageConditionedMaskDecoder, UNetDecoder2D
from medfm.models.heads import BoxHead2D, ImageTextProjectionHead, LinearClassificationHead
from medfm.tasks import (
    BinaryClassificationTask,
    BinarySegmentationTask,
    LanguageConditionedSegmentationTask,
    LocalizationTask,
    RetrievalTask,
)


def _encoded() -> EncoderOutput:
    return EncoderOutput(
        pooled_embedding=torch.randn(2, 8, requires_grad=True),
        feature_maps=(torch.randn(2, 4, 4, 4), torch.randn(2, 8, 8, 8)),
    )


def test_classification_and_segmentation_tasks_honor_sample_mask() -> None:
    encoded = _encoded()
    classification_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["c0", "c1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        labels=torch.tensor([[0.0], [1.0]]),
        task_targets={"sample_mask": torch.tensor([True, False])},
    )
    classification = BinaryClassificationTask(LinearClassificationHead(8, 1))
    classification_loss = classification.compute_loss({"logits": torch.randn(2, 1)}, classification_batch)
    assert classification_loss.sample_count == 1

    segmentation_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["s0", "s1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        task_targets={
            "segmentation": torch.zeros(2, 1, 8, 8),
            "sample_mask": torch.tensor([True, False]),
        },
    )
    segmentation = BinarySegmentationTask(UNetDecoder2D((4, 8), hidden_channels=4))
    segmentation_loss = segmentation.compute_loss({"feature_maps": encoded.feature_maps}, segmentation_batch)
    assert segmentation_loss.sample_count == 1
    assert torch.isfinite(classification_loss.total + segmentation_loss.total)


def test_language_query_is_encoded_separately_and_missing_query_is_zeroed() -> None:
    encoded = _encoded()
    text = torch.randn(2, 3, 6, requires_grad=True)
    batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["l0", "l1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        task_targets={
            "segmentation": torch.zeros(2, 1, 8, 8),
            "text_embeddings": text,
            "query_mask": torch.tensor([True, False]),
        },
    )
    task = LanguageConditionedSegmentationTask(LanguageConditionedMaskDecoder(8, 6, hidden_dim=8))
    loss = task.compute_loss({"visual_features": encoded.feature_maps or ()}, batch)
    assert loss.diagnostics["text_conditioned"] is True
    assert torch.isfinite(loss.total)


def test_retrieval_and_localization_tasks_consume_shared_outputs() -> None:
    encoded = _encoded()
    retrieval_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["r0", "r1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        task_targets={"text_embeddings": torch.randn(2, 6)},
    )
    retrieval = RetrievalTask(ImageTextProjectionHead(8, 6, projection_dim=4))
    retrieval_loss = retrieval.compute_loss({"encoder_output": encoded}, retrieval_batch)
    assert torch.isfinite(retrieval_loss.total)

    localization_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["b0", "b1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        task_targets={"boxes": torch.full((2, 4), 0.25)},
    )
    localization = LocalizationTask(BoxHead2D(8))
    localization_loss = localization.compute_loss(encoded, localization_batch)
    assert set(localization_loss.components) == {"box_l1", "box_iou"}
    assert torch.isfinite(localization_loss.total)
