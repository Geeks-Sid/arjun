"""Compatibility exports for pathology slide aggregators."""

from medfm.models.pathology.aggregation import (
    AttentionMILAggregator,
    GigaPathFlashAggregator,
    MeanPoolingAggregator,
    SlideAggregation,
    SlideAggregator,
    TITANAggregator,
)

__all__ = [
    "AttentionMILAggregator",
    "GigaPathFlashAggregator",
    "MeanPoolingAggregator",
    "SlideAggregation",
    "SlideAggregator",
    "TITANAggregator",
]
