"""Typed errors for the data layer (manifests, readers, splits, caching).

Privacy rule (docs/data_governance.md): messages raised from this layer may
contain sample_ids, identifier *hashes*, file paths, and reason codes — never
raw patient identifiers (MRNs, DICOM UIDs, accession numbers) and never report
text. Callers rely on these types to distinguish actionable, recoverable, and
policy-fatal conditions.
"""

from __future__ import annotations


class DataError(Exception):
    """Base class for all data-layer errors."""


class ManifestError(DataError):
    """A manifest violates the canonical schema or validation rules."""


class ManifestSecurityError(ManifestError):
    """A manifest URI uses a disallowed scheme or attempts path traversal."""


class ManifestVersionError(ManifestError):
    """A manifest declares a schema version this code cannot read or migrate."""


class ReaderError(DataError):
    """An input is unreadable, ambiguous, or inconsistent.

    Messages must be actionable: state what was found and what to fix.
    """


class UnsupportedFormatError(ReaderError):
    """The input uses a variant this reader deliberately does not support
    (e.g. multiframe DICOM, unsupported pixel data, missing optional backend)."""


class CorruptSampleError(ReaderError):
    """A single sample/tile/region is corrupt; recoverable at sample scope."""


class SplitLeakageError(DataError):
    """A split assignment violates the patient/site/temporal leakage policy."""


class CacheError(DataError):
    """Cache key, write, or eviction failure."""


class CorruptCacheEntryError(CacheError):
    """A cache entry failed integrity verification (partial/corrupt payload)."""
