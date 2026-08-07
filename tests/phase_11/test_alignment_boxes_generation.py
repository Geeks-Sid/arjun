from __future__ import annotations

import pytest
import torch

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderOutput
from medfm.core.enums import Modality
from medfm.core.errors import ShapeContractError
from medfm.models.heads import (
    BoxHead2D,
    BoxHead3D,
    ImageTextProjectionHead,
    SymmetricContrastiveLoss,
    generalized_box_iou,
    normalized_to_physical_boxes,
)
from medfm.tasks.generation import StructuredGenerationTask
from medfm.tasks.multitask import FixedWeight, LinearWeightSchedule, MultiTaskLossComposer
from medfm.tasks.structured import (
    StructuredFindingsError,
    StructuredFindingsValidator,
    validate_generation_before_scoring,
)


def test_true_count_reduction_uses_summed_counts() -> None:
    from medfm.tasks import reduce_mean_by_count

    mean, count = reduce_mean_by_count(
        torch.tensor(2.0),
        1,
        reduce_fn=lambda stats: stats + torch.tensor([6.0, 3.0]),
    )
    assert torch.allclose(mean, torch.tensor(2.0))
    assert torch.allclose(count, torch.tensor(4.0))


def test_projection_normalization_and_same_patient_filter() -> None:
    head = ImageTextProjectionHead(4, 5, projection_dim=3)
    output = head(EncoderOutput(pooled_embedding=torch.randn(3, 4)), torch.randn(3, 5), patient_ids=["p0", "p0", "p1"])
    assert torch.allclose(output.image_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)
    value = SymmetricContrastiveLoss()(output, patient_ids=["p0", "p0", "p1"])
    assert torch.isfinite(value)
    with pytest.raises(ShapeContractError):
        SymmetricContrastiveLoss()(output, patient_ids=["same", "same", "same"])


def test_box_heads_and_physical_conversion() -> None:
    output = EncoderOutput(pooled_embedding=torch.randn(2, 8))
    assert BoxHead2D(8)(output).boxes.shape == (2, 4)
    assert BoxHead3D(8)(output).boxes.shape == (2, 6)
    boxes = torch.tensor([[0.25, 0.5, 0.75, 1.0]])
    physical = normalized_to_physical_boxes(boxes, spatial_shape=(10, 20), spacing=(2.0, 3.0))
    # x uses width 20 and 3 mm; y uses height 10 and 2 mm.
    assert torch.allclose(physical, torch.tensor([[15.0, 10.0, 45.0, 20.0]]))
    affine = torch.tensor([[2.0, 0.0, 5.0], [0.0, 3.0, 7.0], [0.0, 0.0, 1.0]])
    affine_physical = normalized_to_physical_boxes(boxes, spatial_shape=(10, 20), affine=affine)
    assert torch.allclose(affine_physical, torch.tensor([[15.0, 22.0, 35.0, 37.0]]))
    assert torch.allclose(generalized_box_iou(boxes, boxes), torch.ones(1))


def test_structured_invalid_outputs_are_counted_and_raw_not_retained() -> None:
    valid = {
        "findings": [],
        "impression": "clear",
    }
    result = validate_generation_before_scoring([valid, "not json", {"findings": []}])
    assert result.report.total == 3
    assert result.report.valid == 1
    assert result.report.invalid == 2
    assert result.report.parse_errors == 1
    assert result.report.schema_errors == 1
    assert all("not json" not in str(item) for item in result.report.results)
    with pytest.raises(StructuredFindingsError):
        StructuredFindingsValidator(debug_sink=lambda index, raw: None)


def test_structured_task_reports_errors() -> None:
    batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["a", "b"],
        pixel_values=torch.randn(2, 1, 4, 4),
    )
    task = StructuredGenerationTask()
    loss = task.compute_loss({"generated_texts": ['{"findings": [], "impression": "ok"}', "bad"]}, batch)
    assert loss.diagnostics["invalid_output_count"] == 1


def test_multitask_fixed_and_scheduled_weights_backward() -> None:
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)
    from medfm.core.task import LossOutput

    composer = MultiTaskLossComposer(
        {"classification": FixedWeight(1.0), "segmentation": LinearWeightSchedule(0.5, 1.0, 0, 10)}
    )
    output = composer(
        {
            "classification": LossOutput(first, sample_count=2),
            "segmentation": LossOutput(second, sample_count=3),
        },
        step=5,
    )
    assert torch.allclose(output.total, torch.tensor(2.5))
    assert output.diagnostics["sample_counts"] == {"classification": 2, "segmentation": 3}
    output.total.backward()
    assert first.grad is not None and second.grad is not None
    with pytest.raises(ShapeContractError):
        MultiTaskLossComposer({"classification": 0.0})
