"""Compatibility exports for pathology embedding stores."""

from medfm.models.pathology.pipeline import (
    STORE_SCHEMA_VERSION,
    EmbeddingStore,
    ExtractionStats,
    StoredEmbeddings,
    TileEmbeddingMetadata,
    extract_slide_embeddings,
)

__all__ = [
    "EmbeddingStore",
    "ExtractionStats",
    "STORE_SCHEMA_VERSION",
    "StoredEmbeddings",
    "TileEmbeddingMetadata",
    "extract_slide_embeddings",
]
