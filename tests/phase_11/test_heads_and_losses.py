from __future__ import annotations

import math

import pytest
import torch

from medfm.core.encoder import EncoderOutput
from medfm.core.errors import UnsupportedCapabilityError
from medfm.models.heads import (
    AttentionPoolingClassificationHead,
    CLSPooling,
    GeneralizedMeanPooling,
    LinearClassificationHead,
    MaskedMeanPooling,
    MILClassificationHead,
    MLPClassificationHead,
    MultiLabelClassificationHead,
    OrdinalClassificationHead,
    TopKPooling,
)
from medfm.tasks.losses import (
    AsymmetricMultilabelLoss,
    BinaryCrossEntropyWithLogitsLoss,
    CrossEntropyClassificationLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    OrdinalCumulativeLinkLoss,
)


def test_all_classification_heads_use_encoder_output(encoder_output_2d: EncoderOutput) -> None:
    heads = [
        LinearClassificationHead(8, 3),
        MLPClassificationHead(8, 3, hidden_dim=12),
        AttentionPoolingClassificationHead(8, 3),
        MultiLabelClassificationHead(8, 3),
        OrdinalClassificationHead(8, 4),
        MILClassificationHead(8, 3),
    ]
    for head in heads:
        result = head(encoder_output_2d)
        assert result.shape == (2, 3 if not isinstance(head, OrdinalClassificationHead) else 3)
        assert torch.isfinite(result).all()


def test_pooling_variants_and_capability_boundary(encoder_output_2d: EncoderOutput) -> None:
    assert CLSPooling()(encoder_output_2d).shape == (2, 8)
    assert MaskedMeanPooling()(encoder_output_2d).shape == (2, 8)
    assert GeneralizedMeanPooling()(encoder_output_2d).shape == (2, 8)
    assert TopKPooling(k=2)(encoder_output_2d).shape == (2, 8)
    pooled_only = EncoderOutput(pooled_embedding=torch.zeros(2, 8))
    with pytest.raises(UnsupportedCapabilityError):
        MaskedMeanPooling()(pooled_only)
    with pytest.raises(UnsupportedCapabilityError):
        AttentionPoolingClassificationHead(8, 2)(pooled_only)


def test_classification_baseline_values() -> None:
    logits = torch.tensor([[0.0, math.log(3.0)]])
    targets = torch.tensor([[1.0, 0.0]])
    expected = torch.tensor((math.log(2.0) + math.log(4.0)) / 2)
    assert torch.allclose(BinaryCrossEntropyWithLogitsLoss()(logits, targets), expected, atol=1e-6)
    multiclass = torch.tensor([[math.log(2.0), 0.0]])
    assert torch.allclose(
        CrossEntropyClassificationLoss()(multiclass, torch.tensor([0])),
        torch.tensor(math.log(3.0) - math.log(2.0)),
        atol=1e-6,
    )


def test_optional_classification_losses_are_finite() -> None:
    logits = torch.tensor([[0.2, -0.7], [1.0, -1.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = [
        FocalLoss()(logits, target),
        LabelSmoothingCrossEntropy(smoothing=0.1)(logits, torch.tensor([0, 1])),
        AsymmetricMultilabelLoss()(logits, target),
        OrdinalCumulativeLinkLoss()(logits, torch.tensor([0, 2])),
    ]
    total = sum(values)
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_empty_token_rows_are_finite() -> None:
    output = EncoderOutput(
        pooled_embedding=torch.zeros(1, 4),
        spatial_tokens=torch.randn(1, 3, 4),
        token_mask=torch.zeros(1, 3, dtype=torch.bool),
    )
    assert torch.isfinite(MaskedMeanPooling()(output)).all()
    assert torch.isfinite(TopKPooling(k=2)(output)).all()
