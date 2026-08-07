"""Preprocessing, augmentation, and collation support (Phase 04).

Subpackages:

- :mod:`medfm.data.transforms` — deterministic/stochastic transform stages,
  model-aware :class:`PreprocessSpec` validation, per-sample seeding, and
  spatial-transform inversion history.
- :mod:`medfm.data.samplers` — group-aware distributed sampling (Phase 03)
  plus 3D patch sampling (Phase 04).
- :mod:`medfm.data.collators` — batch collation with static-shape buckets.
- :mod:`medfm.data.textprep` — text/VLM preparation and loss masking.
"""

from medfm.data.transforms.base import (
    InversionMode,
    Transform,
    TransformContext,
    TransformData,
    TransformRecord,
    derive_seed,
    invert_history,
    make_generator,
    register_inverter,
)
from medfm.data.transforms.ct import ClipHU, ToHounsfieldUnits, WindowChannels
from medfm.data.transforms.mri import (
    ForegroundZScoreNormalize,
    RobustPercentileNormalize,
    SequenceResolver,
    apply_n4_bias_field_correction,
    select_sequences,
    stack_sequences,
)
from medfm.data.transforms.pathology import (
    ReinhardStainNormalize,
    StainAugment,
    TileRecord,
    artifact_score,
    blur_score,
    compute_tissue_mask,
    make_thumbnail,
    make_tile_id,
    plan_tiles,
    tissue_fraction,
)
from medfm.data.transforms.pipeline import TransformPipeline
from medfm.data.transforms.radiology2d import (
    BodyRegionCrop,
    DecodeGrayscale,
    LetterboxResize,
    NormalizeImage,
    RandomFlip2D,
    RandomGaussianNoise,
    RandomIntensityShift,
    RandomRotate2D,
    RandomScale2D,
    RandomTranslate2D,
    RescaleIntensity,
    ToChannels,
)
from medfm.data.transforms.spatial3d import (
    CanonicalizeOrientation,
    ForegroundCrop3D,
    ResampleToSpacing,
)
from medfm.data.transforms.specs import NormalizationSpec, PreprocessSpec
from medfm.data.transforms.timing import (
    PhaseTiming,
    PreprocessTimer,
    PreprocessTimingReport,
    time_host_preprocessing,
)

__all__ = [
    "BodyRegionCrop",
    "CanonicalizeOrientation",
    "ClipHU",
    "DecodeGrayscale",
    "ForegroundCrop3D",
    "ForegroundZScoreNormalize",
    "InversionMode",
    "LetterboxResize",
    "NormalizationSpec",
    "NormalizeImage",
    "PhaseTiming",
    "PreprocessSpec",
    "PreprocessTimer",
    "PreprocessTimingReport",
    "RandomFlip2D",
    "RandomGaussianNoise",
    "RandomIntensityShift",
    "RandomRotate2D",
    "RandomScale2D",
    "RandomTranslate2D",
    "ReinhardStainNormalize",
    "ResampleToSpacing",
    "RescaleIntensity",
    "RobustPercentileNormalize",
    "SequenceResolver",
    "StainAugment",
    "TileRecord",
    "ToChannels",
    "ToHounsfieldUnits",
    "Transform",
    "TransformContext",
    "TransformData",
    "TransformPipeline",
    "TransformRecord",
    "WindowChannels",
    "apply_n4_bias_field_correction",
    "artifact_score",
    "blur_score",
    "compute_tissue_mask",
    "derive_seed",
    "invert_history",
    "make_generator",
    "make_thumbnail",
    "make_tile_id",
    "plan_tiles",
    "register_inverter",
    "select_sequences",
    "stack_sequences",
    "time_host_preprocessing",
    "tissue_fraction",
]
