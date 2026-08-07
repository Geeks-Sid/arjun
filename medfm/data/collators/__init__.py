"""Batch collators with static-shape bucket support (Phase 04, ADR 0008).

Every collator consumes per-example dicts and returns a validated
:class:`medfm.core.batch.MedicalBatch`; mixed modalities fail everywhere
except :class:`MultitaskCollator`. Static mode pads variable dims to
:class:`BucketPlan` buckets with explicit masks and reports bucket ids.
"""

from medfm.data.collators.base import (
    Collator,
    Example,
    FinalBatchPolicy,
    TextCollate,
    example_modality,
    fit_tensor,
    is_padded_example,
    require_tensor,
    spatial_metadata_of,
)
from medfm.data.collators.buckets import BucketPlan, pad_to_shape
from medfm.data.collators.classification import ClassificationCollator
from medfm.data.collators.contrastive import ContrastiveCollator
from medfm.data.collators.multitask import MultitaskBatch, MultitaskCollator
from medfm.data.collators.segmentation import Segmentation2DCollator, Segmentation3DCollator
from medfm.data.collators.vl import MultiImageVLCollator, VolumeVLCollator, WSIVLCollator

__all__ = [
    "BucketPlan",
    "ClassificationCollator",
    "Collator",
    "ContrastiveCollator",
    "Example",
    "FinalBatchPolicy",
    "MultiImageVLCollator",
    "MultitaskBatch",
    "MultitaskCollator",
    "Segmentation2DCollator",
    "Segmentation3DCollator",
    "TextCollate",
    "VolumeVLCollator",
    "WSIVLCollator",
    "example_modality",
    "fit_tensor",
    "is_padded_example",
    "pad_to_shape",
    "require_tensor",
    "spatial_metadata_of",
]
