"""Classification collator: single-image 2D and native-3D modalities.

Stacks per-example ``image`` tensors (``[C, H, W]`` or ``[C, D, H, W]``) into
``pixel_values`` and optional scalar/vector ``label`` tensors into ``labels``
(``[B]`` or ``[B, K]``). With a static :class:`BucketPlan`, spatial dims are
padded to the assigned ``IMAGE_2D``/``VOLUME_3D`` bucket and the batch carries
its :class:`BucketId`; dynamic mode pads to the per-batch maximum.
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

_2D_MODALITIES = frozenset({Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE, Modality.PATHOLOGY_TILE})
_3D_MODALITIES = frozenset({Modality.CT_3D, Modality.MRI_3D})


class ClassificationCollator(Collator):
    """Collates classification examples for single-image 2D/3D modalities."""

    supported_modalities: ClassVar[frozenset[Modality]] = _2D_MODALITIES | _3D_MODALITIES

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError("ClassificationCollator requires a modality")
        volumetric = self.modality in _3D_MODALITIES
        image_ndim = 4 if volumetric else 3  # [C, D, H, W] or [C, H, W]
        kind = BucketKind.VOLUME_3D if volumetric else BucketKind.IMAGE_2D
        images = [require_tensor(example, "image") for example in examples]
        for example, image in zip(examples, images, strict=True):
            if image.ndim != image_ndim:
                raise CollatorError(
                    f"sample {example['sample_id']!r} image must have rank {image_ndim} for {self.modality.value}; "
                    f"got shape {tuple(image.shape)}"
                )
        channels = {int(image.shape[0]) for image in images}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent channel counts across the batch: {sorted(channels)}")
        target, bucket = self._unified_target(kind, [tuple(int(d) for d in image.shape[1:]) for image in images])
        pixel_values = torch.stack([fit_tensor(image, target) for image in images])
        labels = self._collate_labels(examples)
        image_mask = torch.tensor([not is_padded_example(e) for e in examples], dtype=torch.bool)
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            labels=labels,
            spatial_metadata=spatial_metadata_of(examples),
            bucket=bucket,
        )

    def _collate_labels(self, examples: list[Example]) -> torch.Tensor | None:
        if not any("label" in example for example in examples):
            return None
        labels: list[torch.Tensor] = []
        for example in examples:
            label = require_tensor(example, "label")
            if label.ndim not in (0, 1):
                raise CollatorError(
                    f"sample {example['sample_id']!r} label must be a scalar or 1-D [K] tensor; "
                    f"got shape {tuple(label.shape)}"
                )
            labels.append(label)
        widths = {int(label.shape[0]) for label in labels if label.ndim == 1}
        if len(widths) > 1 or (widths and any(label.ndim == 0 for label in labels)):
            raise CollatorError("inconsistent label shapes across the batch; use all scalars or all [K] vectors")
        return torch.stack(labels)
