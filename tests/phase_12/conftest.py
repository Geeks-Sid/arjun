from __future__ import annotations

import pytest
import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality
from medfm.tasks.classification import ClassificationTask
from medfm.training.config import RunConfig


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = nn.Linear(1, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, batch: MedicalBatch) -> dict[str, torch.Tensor]:
        assert batch.pixel_values is not None
        values = batch.pixel_values.flatten(1).mean(dim=1, keepdim=True)
        return {"logits": self.classifier(torch.tanh(self.bridge(values)))}


@pytest.fixture
def tiny_batch() -> MedicalBatch:
    return MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["a", "b"],
        pixel_values=torch.tensor([[[[1.0, 1.0]]], [[[0.0, 0.0]]]]),
        task_targets={"classification": torch.tensor([1, 0])},
    )


@pytest.fixture
def tiny_config(tmp_path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "model_id": "tiny",
            "task": {"name": "multiclass_classification"},
            "accelerator": {"backend": "cpu", "precision": "fp32"},
            "batch": {"microbatch_per_device": 2, "gradient_accumulation_steps": 1},
            "max_steps": 2,
            "output_dir": str(tmp_path / "runs"),
        }
    )


@pytest.fixture
def tiny_task() -> ClassificationTask:
    return ClassificationTask(nn.Identity())
