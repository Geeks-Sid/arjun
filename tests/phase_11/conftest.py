from __future__ import annotations

import pytest
import torch

from medfm.core.batch import MedicalBatch
from medfm.core.encoder import EncoderOutput
from medfm.core.enums import Modality


@pytest.fixture
def encoder_output_2d() -> EncoderOutput:
    return EncoderOutput(
        pooled_embedding=torch.randn(2, 8),
        spatial_tokens=torch.randn(2, 6, 8),
        token_mask=torch.tensor([[True, True, True, False, False, False], [True] * 6]),
        feature_maps=(torch.randn(2, 4, 4, 4), torch.randn(2, 8, 8, 8)),
    )


@pytest.fixture
def encoder_output_3d() -> EncoderOutput:
    return EncoderOutput(
        pooled_embedding=torch.randn(2, 8),
        spatial_tokens=torch.randn(2, 6, 8),
        token_mask=torch.ones(2, 6, dtype=torch.bool),
        feature_maps=(torch.randn(2, 4, 2, 4, 4), torch.randn(2, 8, 4, 8, 8)),
    )


def make_batch_2d(*, labels: torch.Tensor | None = None, segmentation: torch.Tensor | None = None) -> MedicalBatch:
    targets: dict[str, object] = {}
    if segmentation is not None:
        targets["segmentation"] = segmentation
    return MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["s0", "s1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        labels=labels,
        task_targets=targets,
    )


def make_batch_3d(*, segmentation: torch.Tensor | None = None) -> MedicalBatch:
    targets: dict[str, object] = {}
    if segmentation is not None:
        targets["segmentation"] = segmentation
    return MedicalBatch(
        modality=Modality.CT_3D,
        sample_ids=["v0", "v1"],
        pixel_values=torch.randn(2, 1, 4, 8, 8),
        task_targets=targets,
    )
