from __future__ import annotations

import torch
from torchvision.ops import box_iou as torchvision_box_iou
from torchvision.ops import generalized_box_iou as torchvision_generalized_box_iou

from medfm.models.heads.localization import box_iou, generalized_box_iou


def _random_valid_boxes(count: int, *, generator: torch.Generator) -> torch.Tensor:
    starts = torch.rand((count, 2), generator=generator)
    ends = starts + torch.rand((count, 2), generator=generator)
    return torch.cat((starts, ends), dim=-1)


def _formula_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    starts_a, ends_a = boxes_a[..., :2], boxes_a[..., 2:]
    starts_b, ends_b = boxes_b[..., :2], boxes_b[..., 2:]
    intersection = (torch.minimum(ends_a, ends_b) - torch.maximum(starts_a, starts_b)).clamp_min(0).prod(-1)
    area_a = (ends_a - starts_a).prod(-1)
    area_b = (ends_b - starts_b).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-7)


def _formula_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    starts_a, ends_a = boxes_a[..., :2], boxes_a[..., 2:]
    starts_b, ends_b = boxes_b[..., :2], boxes_b[..., 2:]
    iou = _formula_iou(boxes_a, boxes_b)
    intersection = (torch.minimum(ends_a, ends_b) - torch.maximum(starts_a, starts_b)).clamp_min(0).prod(-1)
    union = (ends_a - starts_a).prod(-1) + (ends_b - starts_b).prod(-1) - intersection
    enclosing = (torch.maximum(ends_a, ends_b) - torch.minimum(starts_a, starts_b)).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-7)


def test_float32_2d_box_iou_matches_formula_and_torchvision() -> None:
    generator = torch.Generator().manual_seed(17)
    boxes_a = _random_valid_boxes(64, generator=generator)
    boxes_b = _random_valid_boxes(64, generator=generator)

    expected = _formula_iou(boxes_a, boxes_b)
    torchvision = torchvision_box_iou(boxes_a, boxes_b).diagonal()

    assert torch.allclose(torchvision, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(box_iou(boxes_a, boxes_b), expected, atol=1e-6, rtol=1e-6)


def test_float32_2d_generalized_box_iou_matches_formula_and_torchvision() -> None:
    generator = torch.Generator().manual_seed(23)
    boxes_a = _random_valid_boxes(64, generator=generator)
    boxes_b = _random_valid_boxes(64, generator=generator)

    expected = _formula_giou(boxes_a, boxes_b)
    torchvision = torchvision_generalized_box_iou(boxes_a, boxes_b).diagonal()

    assert torch.allclose(torchvision, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(generalized_box_iou(boxes_a, boxes_b), expected, atol=1e-6, rtol=1e-6)


def test_non_float32_and_3d_box_paths_preserve_native_contract() -> None:
    boxes_a = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 4]], dtype=torch.float16)
    boxes_b = torch.tensor([[0, 0, 2, 2], [0, 1, 4, 5]], dtype=torch.float16)
    assert box_iou(boxes_a, boxes_b).dtype == torch.float16

    boxes_3d_a = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    boxes_3d_b = torch.tensor([[1.0, 0.0, 0.0, 3.0, 2.0, 2.0]])
    assert torch.allclose(box_iou(boxes_3d_a, boxes_3d_b), torch.tensor([1.0 / 3.0]))


def test_degenerate_float32_boxes_stay_on_native_kernel() -> None:
    boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    assert torch.equal(box_iou(boxes, boxes), torch.zeros(1))
    assert torch.equal(generalized_box_iou(boxes, boxes), torch.zeros(1))
