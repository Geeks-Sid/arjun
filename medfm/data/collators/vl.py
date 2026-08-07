"""Vision-language collators: multi-image, volume/multi-series, and WSI.

Each VL collator pairs a variable-count visual side with tokenized text
(tokenization upstream; examples carry 1-D ``input_ids``). Variable visual
counts are padded to declared buckets with explicit masks:

- ``MultiImageVLCollator`` (``MULTI_IMAGE_2D``): ``images`` lists of
  ``[C, H, W]`` padded to the ``MULTI_IMAGE`` count bucket with
  ``image_mask [B, I]``; per-image spatial dims land on ``IMAGE_2D`` buckets.
- ``VolumeVLCollator`` (``CT_3D``/``MRI_3D``/``MULTI_SERIES_3D``): single
  volumes padded to ``VOLUME_3D`` buckets, or ``volumes`` lists padded to a
  series-count maximum with ``image_mask [B, S]``.
- ``WSIVLCollator`` (``PATHOLOGY_WSI``): ``tiles`` lists padded to the
  ``WSI_TILES`` bucket with ``image_mask [B, T]`` and zero-padded
  ``tile_coordinates [B, T, 2]``; precomputed tile embeddings
  (``visual_tokens [N, Dv]``) pad to the ``VISUAL_TOKENS`` bucket with
  ``visual_token_mask [B, N]`` instead of pixels.

Fixed limits are validated by the bucket plan: an example whose visual-token
or text-token count exceeds every declared bucket raises (policy ``error``)
rather than padding to an unplanned shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from medfm.core.batch import BucketId, BucketKind, MedicalBatch
from medfm.core.enums import Modality
from medfm.data.collators.base import (
    Collator,
    Example,
    TextCollate,
    fit_tensor,
    is_padded_example,
    require_tensor,
    spatial_metadata_of,
)
from medfm.data.errors import CollatorError


def _image_list(example: Example, key: str, expected_rank: int) -> list[torch.Tensor]:
    """A sample's variable-length visual payload (images/slices/volumes/tiles)."""
    value = example.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise CollatorError(f"sample {example.get('sample_id')!r} requires a non-empty '{key}' list")
    tensors: list[torch.Tensor] = []
    for item in value:
        if not isinstance(item, torch.Tensor) or item.ndim != expected_rank:
            shape = tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__
            raise CollatorError(
                f"sample {example.get('sample_id')!r} '{key}' entries must be rank-{expected_rank} tensors; got {shape}"
            )
        tensors.append(item)
    return tensors


@dataclass(frozen=True)
class _VisualTokensCollate:
    tokens: torch.Tensor  # [B, N, Dv]
    mask: torch.Tensor  # [B, N] bool
    bucket: BucketId | None


class _VLCollatorBase(Collator):
    """Shared text + padded-example handling for vision-language collators."""

    def _text_targets(self, examples: list[Example]) -> tuple[TextCollate, dict[str, torch.Tensor]]:
        """Collated text block plus LM-label task targets when present."""
        text = self._collate_text(examples)
        targets = {"lm_labels": text.lm_labels} if text.lm_labels is not None else {}
        return text, targets

    def _collate_visual_tokens(self, examples: list[Example]) -> _VisualTokensCollate | None:
        """Pad precomputed ``visual_tokens [N, Dv]`` to the visual-token bucket.

        Returns ``None`` when no example carries visual tokens. Every example
        must agree on the embedding width ``Dv``; padded rows are fully masked
        via ``visual_token_mask`` and final-batch padding replicas contribute
        all-False rows.
        """
        if not any("visual_tokens" in example for example in examples):
            return None
        tokens_list = [require_tensor(example, "visual_tokens") for example in examples]
        for example, tokens in zip(examples, tokens_list, strict=True):
            if tokens.ndim != 2:
                raise CollatorError(
                    f"sample {example['sample_id']!r} visual_tokens must be [N, Dv]; got {tuple(tokens.shape)}"
                )
        widths = {int(tokens.shape[1]) for tokens in tokens_list}
        if len(widths) != 1:
            raise CollatorError(f"inconsistent visual-token embedding widths across the batch: {sorted(widths)}")
        target, bucket = self._unified_target(
            BucketKind.VISUAL_TOKENS, [(int(tokens.shape[0]),) for tokens in tokens_list]
        )
        count = target[0]
        tokens_out = torch.zeros(len(examples), count, widths.pop(), dtype=tokens_list[0].dtype)
        mask_out = torch.zeros(len(examples), count, dtype=torch.bool)
        for row, (example, tokens) in enumerate(zip(examples, tokens_list, strict=True)):
            real = min(int(tokens.shape[0]), count)
            tokens_out[row, :real] = tokens[:real]
            if not is_padded_example(example):
                raw = example.get("visual_token_mask")
                if raw is None:
                    mask_out[row, :real] = True
                else:
                    provided = torch.as_tensor(raw)
                    if provided.ndim != 1 or int(provided.shape[0]) != int(tokens.shape[0]):
                        raise CollatorError(
                            f"sample {example['sample_id']!r} visual_token_mask must be 1-D with length "
                            f"{int(tokens.shape[0])}"
                        )
                    mask_out[row, :real] = provided[:real].to(torch.bool)
        return _VisualTokensCollate(tokens=tokens_out, mask=mask_out, bucket=bucket)


class MultiImageVLCollator(_VLCollatorBase):
    """Collates multi-image 2D VL examples (``MULTI_IMAGE_2D``)."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.MULTI_IMAGE_2D})

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError("MultiImageVLCollator requires a modality")
        image_sets = [_image_list(example, "images", 3) for example in examples]
        channels = {int(image.shape[0]) for images in image_sets for image in images}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent channel counts across the batch: {sorted(channels)}")
        spatial_target, _ = self._unified_target(
            BucketKind.IMAGE_2D,
            [tuple(int(d) for d in image.shape[1:]) for images in image_sets for image in images],
        )
        count_target, count_bucket = self._unified_target(
            BucketKind.MULTI_IMAGE, [(len(images),) for images in image_sets]
        )
        count = count_target[0]
        batch_size = len(examples)
        channel_count = channels.pop()
        pixel_values = torch.zeros(batch_size, count, channel_count, *spatial_target)
        image_mask = torch.zeros(batch_size, count, dtype=torch.bool)
        for row, (example, images) in enumerate(zip(examples, image_sets, strict=True)):
            real = min(len(images), count)
            for index, image in enumerate(images[:real]):
                pixel_values[row, index] = fit_tensor(image, spatial_target)
            if not is_padded_example(example):
                image_mask[row, :real] = True
        text, task_targets = self._text_targets(examples)
        visual = self._collate_visual_tokens(examples)
        if visual is not None:
            task_targets["visual_tokens"] = visual.tokens
            task_targets["visual_token_mask"] = visual.mask
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            spatial_metadata=spatial_metadata_of(examples),
            task_targets=task_targets,
            bucket=count_bucket,
        )


class VolumeVLCollator(_VLCollatorBase):
    """Collates volume VL examples: single volumes or multi-series sets."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.CT_3D, Modality.MRI_3D, Modality.MULTI_SERIES_3D}
    )

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError("VolumeVLCollator requires a modality")
        if self.modality is Modality.MULTI_SERIES_3D:
            return self._collate_multi_series(examples)
        return self._collate_single_volume(examples)

    def _collate_single_volume(self, examples: list[Example]) -> MedicalBatch:
        assert self.modality is not None
        images = [require_tensor(example, "image") for example in examples]
        for example, image in zip(examples, images, strict=True):
            if image.ndim != 4:
                raise CollatorError(
                    f"sample {example['sample_id']!r} image must be [C, D, H, W] for {self.modality.value}; "
                    f"got {tuple(image.shape)}"
                )
        channels = {int(image.shape[0]) for image in images}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent channel counts across the batch: {sorted(channels)}")
        target, bucket = self._unified_target(
            BucketKind.VOLUME_3D, [tuple(int(d) for d in image.shape[1:]) for image in images]
        )
        pixel_values = torch.stack([fit_tensor(image, target) for image in images])
        text, task_targets = self._text_targets(examples)
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
            bucket=bucket,
        )

    def _collate_multi_series(self, examples: list[Example]) -> MedicalBatch:
        assert self.modality is not None
        series_sets = [_image_list(example, "volumes", 4) for example in examples]
        channels = {int(volume.shape[0]) for series in series_sets for volume in series}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent channel counts across the batch: {sorted(channels)}")
        spatial_target, _ = self._unified_target(
            BucketKind.VOLUME_3D,
            [tuple(int(d) for d in volume.shape[1:]) for series in series_sets for volume in series],
        )
        # Series count is padded to the per-batch max (or a MULTI_IMAGE-style
        # count bucket when one is declared for the plan); there is no dedicated
        # BucketKind for series counts.
        if self.static and self.bucket_plan is not None and self.bucket_plan.has_kind(BucketKind.MULTI_IMAGE):
            count_target, _ = self._unified_target(BucketKind.MULTI_IMAGE, [(len(series),) for series in series_sets])
            count = count_target[0]
        else:
            count = max(len(series) for series in series_sets)
        batch_size = len(examples)
        pixel_values = torch.zeros(batch_size, count, channels.pop(), *spatial_target)
        image_mask = torch.zeros(batch_size, count, dtype=torch.bool)
        for row, (example, series) in enumerate(zip(examples, series_sets, strict=True)):
            real = min(len(series), count)
            for index, volume in enumerate(series[:real]):
                pixel_values[row, index] = fit_tensor(volume, spatial_target)
            if not is_padded_example(example):
                image_mask[row, :real] = True
        text, task_targets = self._text_targets(examples)
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            spatial_metadata=spatial_metadata_of(examples),
            task_targets=task_targets,
        )


class WSIVLCollator(_VLCollatorBase):
    """Collates WSI VL examples: tile pixels or precomputed tile embeddings."""

    supported_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.PATHOLOGY_WSI})

    def _collate(self, examples: list[Example]) -> MedicalBatch:
        if self.modality is None:
            raise CollatorError("WSIVLCollator requires a modality")
        visual = self._collate_visual_tokens(examples)
        if visual is not None:
            return self._collate_embeddings(examples, visual)
        return self._collate_tiles(examples)

    def _collate_tiles(self, examples: list[Example]) -> MedicalBatch:
        assert self.modality is not None
        tile_sets = [_image_list(example, "tiles", 3) for example in examples]
        channels = {int(tile.shape[0]) for tiles in tile_sets for tile in tiles}
        if len(channels) != 1:
            raise CollatorError(f"inconsistent tile channel counts across the batch: {sorted(channels)}")
        spatial_target, _ = self._unified_target(
            BucketKind.IMAGE_2D,
            [tuple(int(d) for d in tile.shape[1:]) for tiles in tile_sets for tile in tiles],
        )
        count_target, count_bucket = self._unified_target(BucketKind.WSI_TILES, [(len(tiles),) for tiles in tile_sets])
        count = count_target[0]
        batch_size = len(examples)
        pixel_values = torch.zeros(batch_size, count, channels.pop(), *spatial_target)
        image_mask = torch.zeros(batch_size, count, dtype=torch.bool)
        tile_coordinates = torch.zeros(batch_size, count, 2, dtype=torch.int64)
        for row, (example, tiles) in enumerate(zip(examples, tile_sets, strict=True)):
            real = min(len(tiles), count)
            coords = require_tensor(example, "tile_coordinates")
            if coords.ndim != 2 or int(coords.shape[0]) != len(tiles) or int(coords.shape[1]) not in (2, 4):
                raise CollatorError(
                    f"sample {example['sample_id']!r} tile_coordinates must be [T, 2|4] with T={len(tiles)}; "
                    f"got {tuple(coords.shape)}"
                )
            for index, tile in enumerate(tiles[:real]):
                pixel_values[row, index] = fit_tensor(tile, spatial_target)
            tile_coordinates[row, :real, : coords.shape[1]] = coords[:real].to(torch.int64)
            if not is_padded_example(example):
                image_mask[row, :real] = True
        text, task_targets = self._text_targets(examples)
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            pixel_values=pixel_values,
            image_mask=image_mask,
            tile_coordinates=tile_coordinates,
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            task_targets=task_targets,
            bucket=count_bucket,
        )

    def _collate_embeddings(self, examples: list[Example], visual: _VisualTokensCollate) -> MedicalBatch:
        assert self.modality is not None
        text, task_targets = self._text_targets(examples)
        task_targets["visual_tokens"] = visual.tokens
        task_targets["visual_token_mask"] = visual.mask
        return MedicalBatch(
            modality=self.modality,
            sample_ids=[str(e["sample_id"]) for e in examples],
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            task_targets=task_targets,
            bucket=visual.bucket,
        )
