"""Modality-aware payload readers (Phase 03).

Radiology readers preserve affine/spacing/dtype/orientation; pathology
readers preserve MPP, pyramid geometry, and level-0 tile coordinates. All
readers fail with actionable :class:`medfm.data.errors.ReaderError`
subclasses and never surface raw identifiers (UIDs are hashed at the reader
boundary; ``medfm/data/errors.py`` privacy rule).
"""

from __future__ import annotations

from medfm.data.readers.base import (
    READER_CONTRACT_VERSION,
    PayloadRead,
    Reader,
    hash_identifier,
    resolve_local_path,
    sample_from_manifest_row,
)
from medfm.data.readers.dicom import DICOMSeriesReader, discover_dicom_series
from medfm.data.readers.pathology import (
    CuCIMSlideReader,
    EmbeddingStoreReader,
    OpenSlideReader,
    PreExtractedTileReader,
    SlideReader,
    TiffSlideReader,
    convert_level_coords,
    validate_level_coords,
)
from medfm.data.readers.radiology import (
    MHAReader,
    NiftiReader,
    NumpyVolumeReader,
    PngJpegReader,
)

__all__ = [
    "DICOMSeriesReader",
    "MHAReader",
    "NiftiReader",
    "NumpyVolumeReader",
    "PngJpegReader",
    "CuCIMSlideReader",
    "EmbeddingStoreReader",
    "OpenSlideReader",
    "PreExtractedTileReader",
    "SlideReader",
    "TiffSlideReader",
    "PayloadRead",
    "Reader",
    "READER_CONTRACT_VERSION",
    "convert_level_coords",
    "discover_dicom_series",
    "hash_identifier",
    "resolve_local_path",
    "sample_from_manifest_row",
    "validate_level_coords",
]
