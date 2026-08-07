"""Task/loss contract: LossOutput, lifecycle, typed errors."""

from dataclasses import replace

import pytest
import torch

from medfm.core import (
    LossOutput,
    Modality,
    ShapeContractError,
    TaskModule,
    TaskType,
    UnsupportedModalityError,
)
from phase_02.contract_fixtures import DummyTaskModule, DummyVisualEncoder, make_batch


def test_dummy_task_conforms_to_protocol():
    assert isinstance(DummyTaskModule(), TaskModule)


def test_end_to_end_loss_and_metric_lifecycle():
    encoder = DummyVisualEncoder()
    task = DummyTaskModule()
    batch = replace(make_batch(Modality.XRAY_2D, batch_size=4), labels=torch.tensor([0, 1, 0, 1]))
    output = encoder.encode(batch)

    loss = task.compute_loss(output, batch)
    assert loss.total.ndim == 0
    assert loss.sample_count == 4
    assert set(loss.components) == {"bce"}
    assert isinstance(loss.component_dict()["bce"], float)

    task.reset_metrics()
    task.update_metrics(output, batch)
    metrics = task.compute_metrics()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    task.reset_metrics()
    assert task.compute_metrics()["accuracy"] == 0.0


def test_unsupported_modality_raises_typed_error():
    task = DummyTaskModule()
    with pytest.raises(UnsupportedModalityError):
        task.check_supported(Modality.CT_3D)


def test_task_type_declared():
    assert DummyTaskModule().task_type is TaskType.BINARY_CLASSIFICATION


def test_loss_output_scalar_enforcement():
    with pytest.raises(ShapeContractError, match="scalar"):
        LossOutput(total=torch.tensor([1.0, 2.0]))
    with pytest.raises(ShapeContractError, match="scalar"):
        LossOutput(total=torch.tensor(1.0), components={"bad": torch.tensor([1.0])})
    with pytest.raises(ShapeContractError, match="sample_count"):
        LossOutput(total=torch.tensor(1.0), sample_count=-1)


def test_loss_output_counts_support_distributed_reduction():
    loss = LossOutput(
        total=torch.tensor(2.0),
        components={"ce": torch.tensor(2.0)},
        sample_count=3,
        token_count=17,
    )
    assert loss.sample_count == 3 and loss.token_count == 17
