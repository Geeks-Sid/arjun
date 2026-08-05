"""Dataset manifests, ingestion, and provenance (Phase 03).

Subpackages/modules:

- :mod:`medfm.data.manifests` — canonical Parquet manifests, validation, IO.
- :mod:`medfm.data.readers` — radiology/pathology payload readers.
- :mod:`medfm.data.caching` — cache keys, disk store, typed caches.
- :mod:`medfm.data.splits` — split generation + leakage checks (ADR 0004).
- :mod:`medfm.data.fingerprint` — deterministic dataset fingerprint reports.
- :mod:`medfm.data.samplers` — group-aware distributed sampling.
- :mod:`medfm.data.errors` — typed, privacy-safe data-layer errors.
"""

from __future__ import annotations

from medfm.data.errors import (
    CacheError,
    CorruptCacheEntryError,
    CorruptSampleError,
    DataError,
    ManifestError,
    ManifestSecurityError,
    ManifestVersionError,
    ReaderError,
    SplitLeakageError,
    UnsupportedFormatError,
)
from medfm.data.fingerprint import fingerprint_manifest, recommend_shape_buckets
from medfm.data.samplers import (
    PADDING_INDEX,
    GroupAwareDistributedSampler,
    ResolvedSamples,
    SamplerShard,
    combine_shards_for_metrics,
    resolve_samples_before_collective,
    worker_init_fn,
    worker_seed,
)
from medfm.data.splits import (
    DEFAULT_SITE_RATIOS,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEMPORAL_RATIOS,
    LeakageReport,
    LeakageViolation,
    ResearchOverride,
    SplitPolicy,
    SplitReport,
    assert_no_split_leakage,
    build_split_report,
    check_split_leakage,
    generate_split_assignment,
)

__all__ = [
    "DEFAULT_SITE_RATIOS",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TEMPORAL_RATIOS",
    "PADDING_INDEX",
    "CacheError",
    "CorruptCacheEntryError",
    "CorruptSampleError",
    "DataError",
    "GroupAwareDistributedSampler",
    "LeakageReport",
    "LeakageViolation",
    "ManifestError",
    "ManifestSecurityError",
    "ManifestVersionError",
    "ReaderError",
    "ResearchOverride",
    "ResolvedSamples",
    "SamplerShard",
    "SplitLeakageError",
    "SplitPolicy",
    "SplitReport",
    "UnsupportedFormatError",
    "assert_no_split_leakage",
    "build_split_report",
    "check_split_leakage",
    "combine_shards_for_metrics",
    "fingerprint_manifest",
    "generate_split_assignment",
    "recommend_shape_buckets",
    "resolve_samples_before_collective",
    "worker_init_fn",
    "worker_seed",
]
