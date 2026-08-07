"""Contrastive image-text collator.

Pairs a single 2D image with a tokenized text (tokenization happens upstream;
examples carry 1-D ``input_ids`` and optional ``attention_mask``). Text is
padded to the ``TEXT_TOKENS`` bucket in static mode (or the per-batch max in
dynamic mode) and the batch bucket reports the text bucket when the plan
declares one, else the image bucket. Images are padded to the ``IMAGE_2D``
bucket / per-batch max exactly like classification.
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


class ContrastiveCollator(Collator):
    """Collates image-text pairs for contrastive alignment (2D modalities)."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.XRAY_2D, Modality.CT_2D_SLICE, Modality.MRI_2D_SLICE, Modality.PATHOLOGY_TILE}
    )

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError("ContrastiveCollator requires a modality")
        images = [require_tensor(example, "image") for example in examples]
        for example, image in zip(examples, images, strict=True):
            if image.ndim != 3:
                raise CollatorError(
                    f"sample {example['sample_id']!r} image must be [C, H, W] for {self.modality.value}; "
                    f"got {tuple(image.shape)}"
                )
        channels = {int(image.shape[0]) for image in images}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent channel counts across the batch: {sorted(channels)}")
        target, image_bucket = self._unified_target(
            BucketKind.IMAGE_2D, [tuple(int(d) for d in image.shape[1:]) for image in images]
        )
        pixel_values = torch.stack([fit_tensor(image, target) for image in images])
        text = self._collate_text(examples)
        task_targets = {"lm_labels": text.lm_labels} if text.lm_labels is not None else {}
        image_mask = torch.tensor([not is_padded_example(e) for e in examples], dtype=torch.bool)
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            spatial_metadata=spatial_metadata_of(examples),
            task_targets=task_targets,
            bucket=text.bucket if text.bucket is not None else image_bucket,
        )
