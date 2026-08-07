"""Sampling: group-aware distributed shards and deterministic 3D patch sampling.

- :mod:`medfm.data.samplers.distributed` — group-aware distributed sampling
  with no leakage across ranks (Phase 03 contract, re-exported unchanged).
- :mod:`medfm.data.samplers.patches` — deterministic 3D patch samplers
  (random, foreground, class-balanced, box, lesion-centred, grid) with
  explicit origin, padding, physical-box, and positivity metadata
  (Phase 04).
"""

from __future__ import annotations

from medfm.data.samplers.distributed import (
    PADDING_INDEX,
    GroupAwareDistributedSampler,
    ResolvedSamples,
    SamplerShard,
    combine_shards_for_metrics,
    resolve_samples_before_collective,
    worker_init_fn,
    worker_seed,
)
from medfm.data.samplers.patches import (
    BoxPatchSampler,
    ClassBalancedPatchSampler,
    ForegroundPatchSampler,
    GridPatchSampler,
    LesionCenteredPatchSampler,
    Patch,
    PatchInfo,
    PatchSampler,
    RandomPatchSampler,
    extract_patch,
)

__all__ = [
    "PADDING_INDEX",
    "BoxPatchSampler",
    "ClassBalancedPatchSampler",
    "ForegroundPatchSampler",
    "GridPatchSampler",
    "GroupAwareDistributedSampler",
    "LesionCenteredPatchSampler",
    "Patch",
    "PatchInfo",
    "PatchSampler",
    "RandomPatchSampler",
    "ResolvedSamples",
    "SamplerShard",
    "combine_shards_for_metrics",
    "extract_patch",
    "resolve_samples_before_collective",
    "worker_init_fn",
    "worker_seed",
]
