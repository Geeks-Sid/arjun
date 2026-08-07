"""Bounding-box task import surface."""

from medfm.models.heads.localization import (
    BoxHead2D,
    BoxHead3D,
    BoxL1Loss,
    BoxOutput,
    GIoULoss,
    IoUBoxLoss,
    SpatialBoxHead,
    box_iou,
    boxes_from_spatial_metadata,
    generalized_box_iou,
    normalized_to_physical_boxes,
    physical_to_normalized_boxes,
)

from .localization import LocalizationTask

__all__ = [
    "LocalizationTask",
    "BoxOutput",
    "BoxHead2D",
    "BoxHead3D",
    "SpatialBoxHead",
    "BoxL1Loss",
    "IoUBoxLoss",
    "GIoULoss",
    "box_iou",
    "generalized_box_iou",
    "normalized_to_physical_boxes",
    "physical_to_normalized_boxes",
    "boxes_from_spatial_metadata",
]
