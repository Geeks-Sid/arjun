"""Segmentation collators: dense 2D and 3D mask targets.

Stacks ``image`` plus a required ``mask`` (``[K, H, W]`` or ``[K, D, H, W]``)
so ``task_targets["segmentation"]`` matches the :class:`MedicalBatch` rules —
``[B, K, H, W]`` for 2D modalities, ``[B, K, D, H, W]`` for native 3D, with
spatial dims exactly matching ``pixel_values``. Padded regions are zeroed in
both image and mask; with a static bucket plan the spatial dims land on the
assigned ``IMAGE_2D``/``VOLUME_3D`` bucket.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from medfm.core.batch import BucketKind, MedicalBatch
from medfm.core.enums import Modality
from medfm.data.collators.base import (
    Collator,
    Example,
    fit_tensor,
    is_padded_example,
    require_tensor,
    spatial_metadata_of,
)
from medfm.data.errors import CollatorError


class _SegmentationCollatorBase(Collator):
    """Shared image+mask stacking for a fixed spatial rank."""

    _volumetric: ClassVar[bool]
    _kind: ClassVar[BucketKind]

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError(f"{type(self).__name__} requires a modality")
        spatial_rank = 3 if self._volumetric else 2
        images = [require_tensor(example, "image") for example in examples]
        masks = [require_tensor(example, "mask") for example in examples]
        for example, image, mask in zip(examples, images, masks, strict=True):
            if image.ndim != spatial_rank + 1:
                raise CollatorError(
                    f"sample {example['sample_id']!r} image must have rank {spatial_rank + 1} "
                    f"([C, *spatial]) for {self.modality.value}; got {tuple(image.shape)}"
                )
            if mask.ndim != spatial_rank + 1:
                raise CollatorError(
                    f"sample {example['sample_id']!r} mask must have rank {spatial_rank + 1} "
                    f"([K, *spatial]); got {tuple(mask.shape)}"
                )
            if tuple(image.shape[1:]) != tuple(mask.shape[1:]):
                raise CollatorError(
                    f"sample {example['sample_id']!r} mask spatial dims {tuple(mask.shape[1:])} must match "
                    f"image spatial dims {tuple(image.shape[1:])}"
                )
        channels = {int(image.shape[0]) for image in images}
        classes = {int(mask.shape[0]) for mask in masks}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent image channel counts across the batch: {sorted(channels)}")
        if len(classes) != 1:
            raise CollatorError(f"inconsistent mask class counts (K) across the batch: {sorted(classes)}")
        target, bucket = self._unified_target(self._kind, [tuple(int(d) for d in image.shape[1:]) for image in images])
        pixel_values = torch.stack([fit_tensor(image, target) for image in images])
        segmentation = torch.stack([fit_tensor(mask, target) for mask in masks])
        image_mask = torch.tensor([not is_padded_example(e) for e in examples], dtype=torch.bool)
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            spatial_metadata=spatial_metadata_of(examples),
            task_targets={"segmentation": segmentation},
            bucket=bucket,
        )


class Segmentation2DCollator(_SegmentationCollatorBase):
    """Collates 2D segmentation examples; targets are ``[B, K, H, W]``."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE, Modality.PATHOLOGY_TILE}
    )
    _volumetric: ClassVar[bool] = False
    _kind: ClassVar[BucketKind] = BucketKind.IMAGE_2D


class Segmentation3DCollator(_SegmentationCollatorBase):
    """Collates native-3D segmentation examples; targets are ``[B, K, D, H, W]``."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.CT_3D, Modality.MRI_3D})
    _volumetric: ClassVar[bool] = True
    _kind: ClassVar[BucketKind] = BucketKind.VOLUME_3D
