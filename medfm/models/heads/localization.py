"""2D/3D box heads, coordinate conversion, and IoU-style losses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn
from torchvision.ops import box_iou as _torchvision_box_iou  # type: ignore[import-untyped]
from torchvision.ops import generalized_box_iou as _torchvision_generalized_box_iou

from medfm.core.encoder import EncoderOutput
from medfm.core.enums import CoordinateSystem
from medfm.core.errors import ShapeContractError, UnsupportedCapabilityError
from medfm.core.sample import SpatialMetadata

from .classification import _as_pooled


@dataclass(frozen=True, eq=False)
class BoxOutput:
    """Box logits decoded into ``min...max`` coordinates."""

    boxes: torch.Tensor
    coordinate_system: CoordinateSystem
    normalized: bool = True
    objectness: torch.Tensor | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.boxes.ndim not in (2, 3) or self.boxes.shape[-1] not in (4, 6):
            raise ShapeContractError("boxes must be [B, 4|6] or [B, N, 4|6]")
        if self.objectness is not None and tuple(self.objectness.shape) != tuple(self.boxes.shape[:-1]):
            raise ShapeContractError("objectness shape must align with box batch/object dimensions")


class _BoxHead(nn.Module):
    dimensions: int

    def __init__(self, input_dim: int, num_boxes: int = 1, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        if input_dim <= 0 or num_boxes <= 0:
            raise ShapeContractError("box head dimensions must be positive")
        self.input_dim = int(input_dim)
        self.num_boxes = int(num_boxes)
        box_dim = self.dimensions * 2
        hidden = int(hidden_dim or max(input_dim, box_dim))
        self.regressor = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, num_boxes * box_dim))
        self.objectness = nn.Linear(input_dim, num_boxes)

    def forward(self, value: EncoderOutput | torch.Tensor) -> BoxOutput:
        pooled = _as_pooled(value)
        if pooled.shape[-1] != self.input_dim:
            raise ShapeContractError(f"box head expects pooled dimension {self.input_dim}, got {pooled.shape[-1]}")
        raw = self.regressor(pooled).reshape(pooled.shape[0], self.num_boxes, self.dimensions * 2)
        # Each coordinate is independent and normalized; ordering is enforced
        # by decoding the second half as a positive extent from the first half.
        starts = raw[..., : self.dimensions].sigmoid()
        extents = raw[..., self.dimensions :].sigmoid() * (1.0 - starts)
        boxes = torch.cat([starts, starts + extents], dim=-1)
        objectness = self.objectness(pooled)
        if self.num_boxes == 1:
            boxes = boxes[:, 0]
            objectness = objectness[:, 0]
        return BoxOutput(
            boxes=boxes,
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
            normalized=True,
            objectness=objectness,
        )


class BoxHead2D(_BoxHead):
    dimensions = 2


class BoxHead3D(_BoxHead):
    dimensions = 3


class SpatialBoxHead(nn.Module):
    """Predict boxes from spatial tokens with an optional fixed query count."""

    def __init__(self, input_dim: int, dimensions: int = 2, num_boxes: int = 1) -> None:
        super().__init__()
        if dimensions not in (2, 3):
            raise ShapeContractError("SpatialBoxHead dimensions must be 2 or 3")
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.num_boxes = num_boxes
        self.query = nn.Parameter(torch.zeros(num_boxes, input_dim))
        self.regressor = nn.Linear(input_dim, dimensions * 2)

    def forward(self, output: EncoderOutput) -> BoxOutput:
        if output.spatial_tokens is None:
            raise UnsupportedCapabilityError("SpatialBoxHead requires spatial tokens")
        tokens = output.spatial_tokens
        if tokens.shape[-1] != self.input_dim:
            raise ShapeContractError("spatial token dimension does not match SpatialBoxHead")
        mask = output.token_mask
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raw = self.regressor(pooled.unsqueeze(1) + self.query.to(device=pooled.device, dtype=pooled.dtype))
        starts = raw[..., : self.dimensions].sigmoid()
        ends = starts + raw[..., self.dimensions :].sigmoid() * (1.0 - starts)
        return BoxOutput(
            boxes=torch.cat([starts, ends], dim=-1),
            coordinate_system=CoordinateSystem.NORMALIZED_IMAGE,
            normalized=True,
        )


def _shape_and_dimensions(spatial_shape: Sequence[int], dimensions: int) -> tuple[torch.Tensor, int]:
    shape = tuple(int(v) for v in spatial_shape)
    if len(shape) != dimensions or any(v <= 0 for v in shape):
        raise ShapeContractError(f"spatial_shape must contain {dimensions} positive dimensions")
    # Public box coordinates are x/y(/z); spatial shape is accepted in the
    # conventional tensor order (H,W) or (D,H,W), hence reversal.
    sizes = tuple(reversed(shape))
    return torch.tensor(sizes, dtype=torch.float32), dimensions


def _affine_corners(boxes: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    dimensions = boxes.shape[-1] // 2
    starts, ends = boxes[..., :dimensions], boxes[..., dimensions:]
    corner_count = 2**dimensions
    bits = torch.arange(corner_count, device=boxes.device).unsqueeze(-1)
    choices = ((bits >> torch.arange(dimensions, device=boxes.device)) & 1).to(dtype=boxes.dtype)
    corners = starts.unsqueeze(-2) * (1.0 - choices) + ends.unsqueeze(-2) * choices
    matrix = affine.to(device=boxes.device, dtype=boxes.dtype)
    if matrix.shape[-2:] not in {(dimensions + 1, dimensions + 1), (dimensions, dimensions)}:
        raise ShapeContractError("affine rank does not match box dimensionality")
    if matrix.shape[-2:] == (dimensions, dimensions):
        mapped = torch.einsum("...ij,...cj->...ci", matrix, corners)
    else:
        homogeneous = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)
        mapped = torch.einsum("...ij,...cj->...ci", matrix, homogeneous)[..., :dimensions]
    return torch.cat([mapped.amin(dim=-2), mapped.amax(dim=-2)], dim=-1)


def normalized_to_physical_boxes(
    boxes: torch.Tensor,
    *,
    spatial_shape: Sequence[int] | None = None,
    spacing: Sequence[float] | None = None,
    affine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert normalized image boxes to physical coordinates.

    If an affine is supplied, all box corners are transformed so rotations and
    translations produce a conservative axis-aligned physical box.  Otherwise
    spacing (or unit spacing) is applied in x/y(/z) order.
    """

    if boxes.shape[-1] not in (4, 6):
        raise ShapeContractError("boxes must end in 4 or 6 coordinates")
    dimensions = boxes.shape[-1] // 2
    if affine is not None:
        if spatial_shape is None:
            raise ShapeContractError("spatial_shape is required when applying an affine to normalized boxes")
        sizes, _ = _shape_and_dimensions(spatial_shape, dimensions)
        voxel = boxes * torch.cat([sizes, sizes]).to(device=boxes.device, dtype=boxes.dtype)
        return _affine_corners(voxel, affine)
    if spatial_shape is not None:
        sizes, _ = _shape_and_dimensions(spatial_shape, dimensions)
    else:
        sizes = torch.ones(dimensions, dtype=torch.float32)
    if spacing is None:
        scale = sizes
    else:
        spacing_tuple = tuple(float(v) for v in spacing)
        if len(spacing_tuple) != dimensions or any(v <= 0 for v in spacing_tuple):
            raise ShapeContractError("spacing must match box dimensionality and be positive")
        scale = torch.tensor(tuple(reversed(spacing_tuple)), dtype=torch.float32) * sizes
    return boxes * torch.cat([scale, scale]).to(device=boxes.device, dtype=boxes.dtype)


def physical_to_normalized_boxes(
    boxes: torch.Tensor,
    *,
    spatial_shape: Sequence[int],
    spacing: Sequence[float] | None = None,
    affine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert axis-aligned physical boxes back to normalized image space."""

    if boxes.shape[-1] not in (4, 6):
        raise ShapeContractError("boxes must end in 4 or 6 coordinates")
    dimensions = boxes.shape[-1] // 2
    sizes, _ = _shape_and_dimensions(spatial_shape, dimensions)
    if affine is not None:
        inverse = torch.linalg.inv(affine.to(device=boxes.device, dtype=boxes.dtype))
        voxel = _affine_corners(boxes, inverse)
    else:
        voxel = boxes
    if spacing is not None:
        spacing_tuple = tuple(float(v) for v in spacing)
        if len(spacing_tuple) != dimensions or any(v <= 0 for v in spacing_tuple):
            raise ShapeContractError("spacing must match box dimensionality and be positive")
        scale = torch.tensor(tuple(reversed(spacing_tuple)), device=boxes.device, dtype=boxes.dtype)
        voxel = voxel / torch.cat([scale, scale])
    return voxel / torch.cat([sizes.to(device=boxes.device, dtype=boxes.dtype)] * 2)


def boxes_from_spatial_metadata(
    boxes: torch.Tensor, metadata: SpatialMetadata, *, physical: bool = True
) -> torch.Tensor:
    """Convert using the metadata's current shape, affine, and spacing."""

    if not physical:
        return boxes
    return normalized_to_physical_boxes(
        boxes,
        spatial_shape=metadata.current_shape,
        spacing=metadata.spacing_mm,
        affine=metadata.affine,
    )


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for aligned axis-aligned 2D or 3D boxes.

    The torchvision kernel is used for the common float32, batched 2D case.
    Its pairwise matrix is reduced to the aligned diagonal required by this
    module's contract; other dtypes, dimensions, and custom epsilons stay on
    the native implementation to preserve their output dtype and semantics.
    """

    if boxes_a.shape[-1] not in (4, 6) or boxes_b.shape[-1] != boxes_a.shape[-1]:
        raise ShapeContractError("box_iou expects matching 2D/3D box dimensions")
    if (
        boxes_a.ndim == boxes_b.ndim == 2
        and boxes_a.shape == boxes_b.shape
        and boxes_a.shape[-1] == 4
        and boxes_a.dtype == boxes_b.dtype == torch.float32
        and eps == 1e-7
        and bool(torch.all(boxes_a[:, :2] < boxes_a[:, 2:]))
        and bool(torch.all(boxes_b[:, :2] < boxes_b[:, 2:]))
    ):
        starts_a, ends_a = boxes_a[..., :2], boxes_a[..., 2:]
        starts_b, ends_b = boxes_b[..., :2], boxes_b[..., 2:]
        xyxy_a = torch.cat((starts_a, ends_a), dim=-1)
        xyxy_b = torch.cat((starts_b, ends_b), dim=-1)
        return cast(torch.Tensor, _torchvision_box_iou(xyxy_a, xyxy_b).diagonal())

    dimensions = boxes_a.shape[-1] // 2
    a0, a1 = boxes_a[..., :dimensions], boxes_a[..., dimensions:]
    b0, b1 = boxes_b[..., :dimensions], boxes_b[..., dimensions:]
    intersection_start = torch.maximum(a0, b0)
    intersection_end = torch.minimum(a1, b1)
    intersection = (intersection_end - intersection_start).clamp_min(0).prod(dim=-1)
    area_a = (a1 - a0).clamp_min(0).prod(dim=-1)
    area_b = (b1 - b0).clamp_min(0).prod(dim=-1)
    return intersection / (area_a + area_b - intersection).clamp_min(eps)


def generalized_box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    if (
        boxes_a.shape[-1] == boxes_b.shape[-1] == 4
        and boxes_a.ndim == boxes_b.ndim == 2
        and boxes_a.shape == boxes_b.shape
        and boxes_a.dtype == boxes_b.dtype == torch.float32
        and eps == 1e-7
        and bool(torch.all(boxes_a[:, :2] < boxes_a[:, 2:]))
        and bool(torch.all(boxes_b[:, :2] < boxes_b[:, 2:]))
    ):
        starts_a, ends_a = boxes_a[..., :2], boxes_a[..., 2:]
        starts_b, ends_b = boxes_b[..., :2], boxes_b[..., 2:]
        xyxy_a = torch.cat((starts_a, ends_a), dim=-1)
        xyxy_b = torch.cat((starts_b, ends_b), dim=-1)
        return cast(torch.Tensor, _torchvision_generalized_box_iou(xyxy_a, xyxy_b).diagonal())

    iou = box_iou(boxes_a, boxes_b, eps=eps)
    dimensions = boxes_a.shape[-1] // 2
    a0, a1 = boxes_a[..., :dimensions], boxes_a[..., dimensions:]
    b0, b1 = boxes_b[..., :dimensions], boxes_b[..., dimensions:]
    enclosing_start = torch.minimum(a0, b0)
    enclosing_end = torch.maximum(a1, b1)
    enclosing = (enclosing_end - enclosing_start).clamp_min(0).prod(dim=-1)
    intersection_start = torch.maximum(a0, b0)
    intersection_end = torch.minimum(a1, b1)
    intersection = (intersection_end - intersection_start).clamp_min(0).prod(dim=-1)
    union = (a1 - a0).clamp_min(0).prod(dim=-1) + (b1 - b0).clamp_min(0).prod(dim=-1) - intersection
    return iou - (enclosing - union) / enclosing.clamp_min(eps)


class BoxL1Loss(nn.Module):
    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ShapeContractError("invalid box loss reduction")
        self.reduction = reduction

    def forward(
        self,
        predicted: torch.Tensor | BoxOutput,
        target: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted_tensor = predicted.boxes if isinstance(predicted, BoxOutput) else predicted
        if predicted_tensor.shape != target.shape:
            raise ShapeContractError("predicted and target box shapes must match")
        values = torch.nn.functional.l1_loss(
            predicted_tensor, target.to(device=predicted_tensor.device), reduction="none"
        )
        mask_for_values: torch.Tensor | None = None
        if valid_mask is not None:
            mask = valid_mask.to(device=values.device, dtype=values.dtype)
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(-1)
            try:
                mask_for_values = mask.expand_as(values)
            except RuntimeError as exc:
                raise ShapeContractError("valid box mask must broadcast to box predictions") from exc
            values = values * mask_for_values
        if self.reduction == "none":
            return values
        if self.reduction == "sum":
            return values.sum()
        denominator = (
            torch.as_tensor(values.numel(), device=values.device, dtype=values.dtype)
            if mask_for_values is None
            else mask_for_values.sum()
        )
        return values.sum() / denominator.clamp_min(1.0)


class IoUBoxLoss(nn.Module):
    def __init__(self, *, generalized: bool = True) -> None:
        super().__init__()
        self.generalized = bool(generalized)

    def forward(
        self,
        predicted: torch.Tensor | BoxOutput,
        target: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted_tensor = predicted.boxes if isinstance(predicted, BoxOutput) else predicted
        target_tensor = target.to(device=predicted_tensor.device, dtype=predicted_tensor.dtype)
        if predicted_tensor.shape != target_tensor.shape:
            raise ShapeContractError("predicted and target box shapes must match")
        if predicted_tensor.numel() == 0:
            return predicted_tensor.sum() * 0.0
        score = (
            generalized_box_iou(predicted_tensor, target_tensor)
            if self.generalized
            else box_iou(predicted_tensor, target_tensor)
        )
        if valid_mask is not None:
            mask = valid_mask.to(device=score.device, dtype=score.dtype)
            while mask.ndim < score.ndim:
                mask = mask.unsqueeze(-1)
            if mask.shape != score.shape:
                try:
                    mask = mask.expand_as(score)
                except RuntimeError as exc:
                    raise ShapeContractError("valid box mask must broadcast to box predictions") from exc
            score = score * mask
            denominator = mask.sum().clamp_min(1.0)
            return (1.0 - score).sum() / denominator
        return (1.0 - score).mean().to(dtype=predicted_tensor.dtype)


GIoULoss = IoUBoxLoss
convert_normalized_boxes_to_physical = normalized_to_physical_boxes
convert_physical_boxes_to_normalized = physical_to_normalized_boxes
box_iou_loss = IoUBoxLoss
giou_loss = IoUBoxLoss


__all__ = [
    "BoxOutput",
    "BoxHead2D",
    "BoxHead3D",
    "SpatialBoxHead",
    "normalized_to_physical_boxes",
    "physical_to_normalized_boxes",
    "BoxL1Loss",
    "IoUBoxLoss",
    "GIoULoss",
    "convert_normalized_boxes_to_physical",
    "convert_physical_boxes_to_normalized",
    "box_iou_loss",
    "giou_loss",
]
