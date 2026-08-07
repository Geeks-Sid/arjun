"""Contrastive alignment task import surface."""

from medfm.models.heads.retrieval import (
    DistributedNegativeProvider,
    ImageTextProjectionHead,
    ImageTextRetrievalHead,
    RetrievalOutput,
    SymmetricContrastiveLoss,
    symmetric_contrastive_loss,
)

from .retrieval import RetrievalTask

__all__ = [
    "RetrievalTask",
    "DistributedNegativeProvider",
    "ImageTextProjectionHead",
    "ImageTextRetrievalHead",
    "RetrievalOutput",
    "SymmetricContrastiveLoss",
    "symmetric_contrastive_loss",
]
