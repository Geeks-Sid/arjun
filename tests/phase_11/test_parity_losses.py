from __future__ import annotations

import pytest
import torch
from monai.losses import (
    DeepSupervisionLoss as MONAIDeepSupervisionLoss,
)
from monai.losses import (
    DiceCELoss as MONAIDiceCELoss,
)
from monai.losses import (
    DiceLoss as MONAIDiceLoss,
)
from monai.losses import (
    FocalLoss as MONAIFocalLoss,
)
from monai.losses import (
    TverskyLoss as MONAITverskyLoss,
)

from medfm.tasks.losses import (
    DeepSupervisionLoss,
    DiceBCELoss,
    DiceCELoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    dice_loss,
)


def _multiclass_case(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2)
    logits = torch.randn(2, 3, 4, 5, dtype=dtype)
    labels = torch.randint(0, 3, (2, 4, 5))
    target = torch.nn.functional.one_hot(labels, num_classes=3).movedim(-1, 1).to(dtype)
    return logits, target


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_focal_does_not_match_monai_alpha_and_mode(dtype: torch.dtype) -> None:
    logits, target = _multiclass_case(dtype)
    ours = FocalLoss(gamma=2.0, alpha=0.25, multiclass=True)(logits, target.argmax(dim=1))
    monai = MONAIFocalLoss(gamma=2.0, alpha=0.25, use_softmax=True)(logits, target)

    # MONAI applies focal weighting to every softmax channel and uses the
    # standard class-balanced alpha convention; this loss uses CE focal math.
    assert not torch.allclose(ours.float(), monai.float(), atol=1e-4, rtol=1e-4)
    if dtype == torch.float16:
        assert ours.dtype == dtype
        assert monai.dtype == torch.float32


def test_dice_kernel_matches_monai_in_float32_but_drifts_in_float16() -> None:
    logits, target = _multiclass_case()
    ours32 = dice_loss(logits, target)
    monai32 = MONAIDiceLoss(softmax=True, smooth_nr=1.0, smooth_dr=1.0)(logits, target)
    assert torch.allclose(ours32, monai32, atol=1e-6, rtol=1e-6)

    logits16, target16 = _multiclass_case(torch.float16)
    ours16 = dice_loss(logits16, target16)
    monai16 = MONAIDiceLoss(softmax=True, smooth_nr=1.0, smooth_dr=1.0)(logits16, target16)
    assert (ours16.float() - monai16.float()).abs() > 1e-4
    assert ours16.dtype == torch.float16


def test_dice_compositions_do_not_match_monai_smoothing_contract() -> None:
    logits, target = _multiclass_case()
    ours = DiceCELoss()(logits, target)
    monai = MONAIDiceCELoss(softmax=True)(logits, target)
    assert not torch.allclose(ours, monai, atol=1e-4, rtol=1e-4)

    binary_logits = torch.randn(2, 1, 4, 5)
    binary_target = torch.randint(0, 2, binary_logits.shape).float()
    ours_bce = DiceBCELoss()(binary_logits, binary_target)
    monai_bce = MONAIDiceCELoss(sigmoid=True)(binary_logits, binary_target)
    assert not torch.allclose(ours_bce, monai_bce, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tversky_empty_contract_and_monai_dtype_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 7, 6, dtype=dtype)
    labels = torch.randint(0, 3, (2, 7, 6))
    target = torch.nn.functional.one_hot(labels, num_classes=3).movedim(-1, 1).to(dtype)
    ours = TverskyLoss(alpha=0.3, beta=0.7)(logits, target)
    monai = MONAITverskyLoss(
        alpha=0.3,
        beta=0.7,
        softmax=True,
        smooth_nr=1.0,
        smooth_dr=1.0,
    )(logits, target)
    if dtype == torch.float32:
        assert torch.allclose(ours, monai, atol=1e-6, rtol=1e-6)
    else:
        assert (ours.float() - monai.float()).abs() > 1e-4
        assert ours.dtype == dtype

    empty = torch.zeros_like(target)
    assert TverskyLoss()(logits, empty).isfinite()


def test_deep_supervision_keeps_explicit_weight_and_dtype_contract() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 1, 4, 5, dtype=torch.float16)
    target = torch.randint(0, 2, (2, 1, 4, 5)).to(dtype=logits.dtype)
    lowres = logits[..., ::2, ::2]
    ours = DeepSupervisionLoss(DiceLoss(), weights=(0.25, 0.75))((logits, lowres), target)
    monai = MONAIDeepSupervisionLoss(
        MONAIDiceLoss(sigmoid=True, smooth_nr=1.0, smooth_dr=1.0),
        weights=[0.25, 0.75],
    )((logits, lowres), target)
    assert ours.dtype == torch.float16
    assert monai.dtype == torch.float32
    assert torch.isfinite(ours)
