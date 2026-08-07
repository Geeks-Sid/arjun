"""Compatibility import surface for generic segmentation decoders."""

from .base import DecoderOutput, SegmentationOutput
from .fpn import FPNDecoder2D, FPNDecoder3D
from .language import LanguageConditionedMaskDecoder
from .masks import NativeModelDecoderWrapper, PromptableMaskDecoder, TransformerMaskDecoder
from .unet import UNetDecoder2D, UNetDecoder3D

__all__ = [
    "SegmentationOutput",
    "DecoderOutput",
    "UNetDecoder2D",
    "UNetDecoder3D",
    "FPNDecoder2D",
    "FPNDecoder3D",
    "TransformerMaskDecoder",
    "PromptableMaskDecoder",
    "LanguageConditionedMaskDecoder",
    "NativeModelDecoderWrapper",
]
