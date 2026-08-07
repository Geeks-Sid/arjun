"""Accelerator-neutral baseline and optional task losses.

The implementations use ordinary PyTorch tensor operations only.  Reduction is
explicit and stable for empty masks, padded samples, and mixed precision
inputs; numerically sensitive accumulations are promoted to FP32.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from medfm.core.errors import ShapeContractError


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ShapeContractError(f"unsupported reduction {reduction!r}")


def _masked_reduce(values: torch.Tensor, mask: torch.Tensor | None, reduction: str) -> torch.Tensor:
    _validate_reduction(reduction)
    if mask is not None:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == values.ndim + 1 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(1)
        try:
            mask = mask.expand_as(values)
        except RuntimeError as exc:
            raise ShapeContractError(
                f"valid mask shape {tuple(mask.shape)} cannot broadcast to loss shape {tuple(values.shape)}"
            ) from exc
        values = values * mask.to(dtype=values.dtype)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.float().sum().to(dtype=values.dtype)
    if mask is None:
        denominator = torch.as_tensor(values.numel(), device=values.device, dtype=torch.float32).clamp_min(1.0)
    else:
        denominator = mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    return values.float().sum().div(denominator).to(dtype=values.dtype)


def _broadcast_class_weight(weight: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
    if weight is None:
        return None
    if weight.ndim != 1:
        raise ShapeContractError("class_weight must be a one-dimensional tensor")
    if values.shape[-1] == weight.shape[0]:
        return weight.to(device=values.device, dtype=values.dtype)
    return weight.to(device=values.device, dtype=values.dtype)


class BinaryCrossEntropyWithLogitsLoss(nn.Module):
    """Mandatory binary/multi-label BCE-with-logits baseline."""

    def __init__(self, *, class_weight: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.detach().clone(), persistent=False)
        else:
            self.class_weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ShapeContractError(f"BCE logits {tuple(logits.shape)} != targets {tuple(targets.shape)}")
        values = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
        if self.class_weight is not None:
            weight = self.class_weight.to(device=logits.device, dtype=logits.dtype)
            if weight.numel() != logits.shape[-1]:
                raise ShapeContractError(
                    f"class_weight length {weight.numel()} must equal the final logits dimension {logits.shape[-1]}"
                )
            shape = [1] * logits.ndim
            shape[-1] = weight.numel()
            values = values * weight.reshape(shape)
        return _masked_reduce(values, valid_mask, self.reduction)


class CrossEntropyClassificationLoss(nn.Module):
    """Mandatory multiclass cross-entropy baseline."""

    def __init__(
        self,
        *,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        _validate_reduction(reduction)
        if not 0.0 <= label_smoothing < 1.0:
            raise ShapeContractError("label_smoothing must be in [0, 1)")
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.detach().clone(), persistent=False)
        else:
            self.class_weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim < 2 or targets.shape != logits.shape[:1] + logits.shape[2:]:
            # For the common [B, C] case this is [B]. Dense segmentation is
            # handled by the segmentation loss helpers below.
            if logits.ndim != 2 or targets.shape != logits.shape[:1]:
                raise ShapeContractError(
                    f"cross-entropy expects logits [B, C, ...] and targets [B, ...], got "
                    f"{tuple(logits.shape)} and {tuple(targets.shape)}"
                )
        weight = (
            self.class_weight.to(device=logits.device, dtype=logits.dtype) if self.class_weight is not None else None
        )
        target_long = targets.to(device=logits.device, dtype=torch.long)
        values = F.cross_entropy(
            logits,
            target_long,
            weight=weight,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        if self.reduction == "mean" and weight is not None:
            active = target_long != self.ignore_index
            denominator = weight[target_long.clamp_min(0).clamp_max(weight.numel() - 1)] * active.to(weight.dtype)
            if valid_mask is not None:
                valid = valid_mask.to(device=logits.device, dtype=torch.bool)
                denominator = denominator * valid.to(dtype=denominator.dtype)
            return values.float().sum().div(denominator.float().sum().clamp_min(1.0)).to(dtype=logits.dtype)
        return _masked_reduce(values, valid_mask, self.reduction)


class FocalLoss(nn.Module):
    """Focal loss for binary/multi-label or multiclass logits."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        multiclass: bool = False,
        class_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ShapeContractError("focal gamma must be non-negative")
        _validate_reduction(reduction)
        self.gamma = float(gamma)
        self.multiclass = bool(multiclass)
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.detach().clone(), persistent=False)
            self.alpha_value: float | None = None
        else:
            self.alpha = None  # type: ignore[assignment]
            self.alpha_value = float(alpha) if alpha is not None else None
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.detach().clone(), persistent=False)
        else:
            self.class_weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.multiclass:
            values = F.cross_entropy(
                logits,
                targets.to(device=logits.device, dtype=torch.long),
                weight=(
                    self.class_weight.to(device=logits.device, dtype=logits.dtype)
                    if self.class_weight is not None
                    else None
                ),
                reduction="none",
            )
            pt = torch.exp(-values)
            values = (1.0 - pt).pow(self.gamma) * values
        else:
            if logits.shape != targets.shape:
                raise ShapeContractError("binary focal logits and targets must have equal shapes")
            values = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
            pt = torch.exp(-values)
            values = (1.0 - pt).pow(self.gamma) * values
            if self.alpha is not None:
                alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
                shape = [1] * logits.ndim
                shape[-1] = alpha.numel()
                values = values * alpha.reshape(shape)
            elif self.alpha_value is not None:
                values = values * self.alpha_value
        return _masked_reduce(values, valid_mask, self.reduction)


class LabelSmoothingCrossEntropy(CrossEntropyClassificationLoss):
    """Named label-smoothing CE variant for configuration and logs."""

    def __init__(self, smoothing: float = 0.1, **kwargs: object) -> None:
        super().__init__(label_smoothing=smoothing, **kwargs)


class AsymmetricMultilabelLoss(nn.Module):
    """Asymmetric focal-style loss for imbalanced multi-label targets."""

    def __init__(
        self,
        *,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma_neg < 0 or gamma_pos < 0 or not 0.0 <= clip < 1.0 or eps <= 0:
            raise ShapeContractError("invalid asymmetric-loss configuration")
        _validate_reduction(reduction)
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ShapeContractError("asymmetric multilabel logits and targets must have equal shapes")
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(xs_pos.clamp_min(self.eps))
        loss = loss + (1.0 - targets) * torch.log(xs_neg.clamp_min(self.eps))
        if self.gamma_neg or self.gamma_pos:
            asym_prob = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * (1.0 - asym_prob).pow(gamma)
        return _masked_reduce(-loss, valid_mask, self.reduction)


class OrdinalCumulativeLinkLoss(nn.Module):
    """BCE over cumulative ordinal events ``y > threshold``."""

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ShapeContractError("ordinal loss expects logits [B, K-1] and integer targets [B]")
        classes_minus_one = int(logits.shape[1])
        target = targets.to(device=logits.device, dtype=torch.long).unsqueeze(-1)
        thresholds = torch.arange(classes_minus_one, device=logits.device).reshape(1, -1)
        cumulative = (target > thresholds).to(dtype=logits.dtype)
        values = F.binary_cross_entropy_with_logits(logits, cumulative, reduction="none")
        return _masked_reduce(values, valid_mask, self.reduction)


# Short, stable names used by YAML configs.
BCEWithLogitsLoss = BinaryCrossEntropyWithLogitsLoss
CrossEntropyLoss = CrossEntropyClassificationLoss
FocalClassificationLoss = FocalLoss
AsymmetricLoss = AsymmetricMultilabelLoss
OrdinalLoss = OrdinalCumulativeLinkLoss


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------


def _segmentation_target(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if logits.ndim < 4:
        raise ShapeContractError("segmentation logits must be [B, K, spatial...]")
    if target.shape[0] != logits.shape[0] or tuple(target.shape[-len(logits.shape[2:]) :]) != tuple(logits.shape[2:]):
        raise ShapeContractError(
            f"segmentation target spatial shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}"
        )
    classes = int(logits.shape[1])
    if classes == 1:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        if target.shape != logits.shape:
            raise ShapeContractError("binary segmentation target must be [B, 1, spatial...] or [B, spatial...]")
        return target.to(dtype=logits.dtype), None
    if target.ndim == logits.ndim:
        if int(target.shape[1]) != classes:
            raise ShapeContractError("one-hot segmentation target channel count does not match logits")
        one_hot = target.to(dtype=logits.dtype)
        labels = one_hot.argmax(dim=1)
    elif target.ndim == logits.ndim - 1:
        labels = target.to(dtype=torch.long)
        one_hot = F.one_hot(labels, num_classes=classes).movedim(-1, 1).to(dtype=logits.dtype)
    else:
        raise ShapeContractError("multiclass segmentation target must be labels or one-hot channels")
    return one_hot, labels


def _voxel_weights(mask: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.ndim == logits.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != logits.ndim or int(mask.shape[0]) != int(logits.shape[0]):
        raise ShapeContractError("segmentation valid_mask must have batch-aligned spatial dimensions")
    if any(
        mask_size not in (1, logit_size) for mask_size, logit_size in zip(mask.shape[1:], logits.shape[1:], strict=True)
    ):
        raise ShapeContractError("segmentation valid_mask dimensions cannot broadcast to logits")
    return mask.expand_as(logits)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
    include_background: bool = True,
    valid_mask: torch.Tensor | None = None,
    class_volume_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice loss that treats empty target/prediction as a perfect class."""

    if smooth <= 0:
        raise ShapeContractError("Dice smooth must be positive")
    one_hot, _ = _segmentation_target(logits, target)
    probabilities = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
    mask = _voxel_weights(valid_mask, logits)
    if mask is not None:
        probabilities = probabilities * mask.to(dtype=probabilities.dtype)
        one_hot = one_hot * mask.to(dtype=one_hot.dtype)
    dims = tuple(range(2, logits.ndim))
    probabilities_fp32 = probabilities.float()
    target_fp32 = one_hot.float()
    intersection = (probabilities_fp32 * target_fp32).sum(dim=dims)
    denominator = probabilities_fp32.sum(dim=dims) + target_fp32.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    classes = int(logits.shape[1])
    if not include_background and classes > 1:
        dice = dice[:, 1:]
    if class_volume_weight is not None:
        weights = class_volume_weight.to(device=dice.device, dtype=dice.dtype)
        if not include_background and classes > 1:
            weights = weights[1:]
        if weights.numel() != dice.shape[-1]:
            raise ShapeContractError("class_volume_weight length does not match segmentation classes")
        dice = dice * weights.reshape(1, -1)
        return 1.0 - dice.sum() / weights.sum().clamp_min(1e-8)
    return (1.0 - dice).mean().to(dtype=logits.dtype)


class DiceLoss(nn.Module):
    def __init__(self, *, smooth: float = 1.0, include_background: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return dice_loss(
            logits,
            target,
            smooth=self.smooth,
            include_background=self.include_background,
            valid_mask=valid_mask,
        )


def _multiclass_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    one_hot, labels = _segmentation_target(logits, target)
    del one_hot
    if logits.shape[1] == 1:
        target_for_bce = target if target.ndim == logits.ndim else target.unsqueeze(1)
        values = F.binary_cross_entropy_with_logits(logits, target_for_bce.to(dtype=logits.dtype), reduction="none")
        if values.ndim > logits.ndim - 1:
            values = values.squeeze(1)
    else:
        assert labels is not None
        values = F.cross_entropy(
            logits,
            labels,
            weight=(class_weight.to(device=logits.device, dtype=logits.dtype) if class_weight is not None else None),
            label_smoothing=label_smoothing,
            reduction="none",
        )
    return _masked_reduce(values, valid_mask, "mean")


class DiceCELoss(nn.Module):
    """Default multiclass segmentation loss: Dice plus cross entropy."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        include_background: bool = True,
        class_volume_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight <= 0:
            raise ShapeContractError("DiceCE weights must be non-negative and not both zero")
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.label_smoothing = float(label_smoothing)
        self.include_background = include_background
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.detach().clone(), persistent=False)
        else:
            self.class_weight = None  # type: ignore[assignment]
        if class_volume_weight is not None:
            self.register_buffer("class_volume_weight", class_volume_weight.detach().clone(), persistent=False)
        else:
            self.class_volume_weight = None  # type: ignore[assignment]

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        dice = dice_loss(
            logits,
            target,
            include_background=self.include_background,
            valid_mask=valid_mask,
            class_volume_weight=self.class_volume_weight,
        )
        ce = _multiclass_ce(
            logits,
            target,
            class_weight=self.class_weight,
            label_smoothing=self.label_smoothing,
            valid_mask=valid_mask,
        )
        return self.dice_weight * dice + self.ce_weight * ce


class DiceBCELoss(DiceCELoss):
    """Default binary segmentation loss: Dice plus BCE-with-logits."""

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if logits.shape[1] != 1:
            raise ShapeContractError("DiceBCELoss requires one binary logit channel")
        dice = dice_loss(logits, target, valid_mask=valid_mask, include_background=True)
        target_for_bce = target if target.ndim == logits.ndim else target.unsqueeze(1)
        bce = _masked_reduce(
            F.binary_cross_entropy_with_logits(logits, target_for_bce.to(dtype=logits.dtype), reduction="none"),
            valid_mask,
            "mean",
        )
        return self.dice_weight * dice + self.ce_weight * bce


class FocalSegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float | None = None,
        multiclass: bool | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.multiclass = multiclass

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        multiclass = self.multiclass if self.multiclass is not None else logits.shape[1] > 1
        focal = FocalLoss(gamma=self.gamma, alpha=self.alpha, multiclass=multiclass)
        if multiclass:
            labels = target.argmax(dim=1) if target.ndim == logits.ndim else target
            return focal(logits, labels, valid_mask=valid_mask)
        target_for_bce = target if target.ndim == logits.ndim else target.unsqueeze(1)
        return focal(logits, target_for_bce, valid_mask=valid_mask)


class TverskyLoss(nn.Module):
    def __init__(self, *, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0 or smooth <= 0:
            raise ShapeContractError("invalid Tversky configuration")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        one_hot, _ = _segmentation_target(logits, target)
        probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
        mask = _voxel_weights(valid_mask, logits)
        if mask is not None:
            probs = probs * mask.to(dtype=probs.dtype)
            one_hot = one_hot * mask.to(dtype=one_hot.dtype)
        dims = tuple(range(2, logits.ndim))
        probs, one_hot = probs.float(), one_hot.float()
        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1 - one_hot)).sum(dim=dims)
        fn = ((1 - probs) * one_hot).sum(dim=dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - score).mean().to(dtype=logits.dtype)


class BoundaryLoss(nn.Module):
    """Simple differentiable boundary surrogate using local finite differences."""

    def __init__(self, *, include_background: bool = True) -> None:
        super().__init__()
        self.include_background = include_background

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        one_hot, _ = _segmentation_target(logits, target)
        probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
        if not self.include_background and probs.shape[1] > 1:
            probs, one_hot = probs[:, 1:], one_hot[:, 1:]
        mask = _voxel_weights(valid_mask, logits)
        if mask is not None and not self.include_background and mask.shape[1] > 1:
            mask = mask[:, 1:]
        total = logits.new_zeros(())
        dimensions = tuple(range(2, logits.ndim))
        terms = 0
        for axis in dimensions:
            if int(logits.shape[axis]) <= 1:
                continue
            pred_diff = probs.diff(dim=axis).abs()
            true_diff = one_hot.diff(dim=axis).abs()
            error = (pred_diff - true_diff).abs()
            if mask is not None:
                pair_mask = mask.narrow(axis, 0, int(mask.shape[axis]) - 1)
                total = total + (error * pair_mask.to(dtype=error.dtype)).sum() / pair_mask.sum().clamp_min(1)
            else:
                total = total + error.float().mean().to(dtype=logits.dtype)
            terms += 1
        return total / max(1, terms)


class DeepSupervisionLoss(nn.Module):
    """Weighted loss over final logits and explicit deep-supervision outputs."""

    def __init__(self, base_loss: nn.Module, weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = tuple(float(w) for w in (weights or (1.0,)))
        if not self.weights or any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ShapeContractError("deep-supervision weights must be non-negative and not all zero")

    def forward(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        target: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = (logits,) if isinstance(logits, torch.Tensor) else tuple(logits)
        if len(outputs) != len(self.weights):
            raise ShapeContractError("deep-supervision output count must equal configured weight count")
        total = outputs[0].new_zeros(())
        normalizer = sum(self.weights)
        for output, weight in zip(outputs, self.weights, strict=True):
            resized_target = target
            resized_mask = valid_mask
            if tuple(output.shape[2:]) != tuple(target.shape[-len(output.shape[2:]) :]):
                if output.shape[1] == 1:
                    resized_target = F.interpolate(
                        target if target.ndim == output.ndim else target.unsqueeze(1).to(dtype=output.dtype),
                        size=output.shape[2:],
                        mode="nearest",
                    )
                else:
                    if target.ndim == output.ndim:
                        resized_target = F.interpolate(
                            target.to(dtype=output.dtype), size=output.shape[2:], mode="nearest"
                        )
                    else:
                        resized_target = F.interpolate(
                            target.unsqueeze(1).to(dtype=output.dtype), size=output.shape[2:], mode="nearest"
                        ).squeeze(1)
            if resized_mask is not None:
                if resized_mask.ndim == output.ndim - 1:
                    resized_mask = resized_mask.unsqueeze(1)
                if tuple(resized_mask.shape[2:]) != tuple(output.shape[2:]):
                    resized_mask = F.interpolate(
                        resized_mask.to(dtype=output.dtype), size=output.shape[2:], mode="nearest"
                    )
                resized_mask = resized_mask.to(dtype=torch.bool)
            total = total + weight * self.base_loss(output, resized_target, valid_mask=resized_mask)
        return total / normalizer


def binary_cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return BinaryCrossEntropyWithLogitsLoss(**kwargs)(logits, targets, valid_mask=valid_mask)


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return CrossEntropyClassificationLoss(**kwargs)(logits, targets, valid_mask=valid_mask)


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return FocalLoss(**kwargs)(logits, targets, valid_mask=valid_mask)


def asymmetric_multilabel_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return AsymmetricMultilabelLoss(**kwargs)(logits, targets, valid_mask=valid_mask)


def ordinal_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return OrdinalCumulativeLinkLoss(**kwargs)(logits, targets, valid_mask=valid_mask)


def dice_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return DiceCELoss(**kwargs)(logits, target, valid_mask=valid_mask)


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, *, valid_mask: torch.Tensor | None = None, **kwargs: object
) -> torch.Tensor:
    return DiceBCELoss(**kwargs)(logits, target, valid_mask=valid_mask)


__all__ = [
    "BinaryCrossEntropyWithLogitsLoss",
    "CrossEntropyClassificationLoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "FocalClassificationLoss",
    "LabelSmoothingCrossEntropy",
    "AsymmetricMultilabelLoss",
    "AsymmetricLoss",
    "OrdinalCumulativeLinkLoss",
    "OrdinalLoss",
    "dice_loss",
    "DiceLoss",
    "DiceCELoss",
    "DiceBCELoss",
    "FocalSegmentationLoss",
    "TverskyLoss",
    "BoundaryLoss",
    "DeepSupervisionLoss",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "focal_loss",
    "asymmetric_multilabel_loss",
    "ordinal_loss",
    "dice_ce_loss",
    "dice_bce_loss",
]
