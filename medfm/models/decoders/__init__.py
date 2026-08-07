"""Encoder-independent segmentation decoders."""

from .base import SegmentationOutput
from .fpn import FPNDecoder2D, FPNDecoder3D
from .language import LanguageConditionedMaskDecoder
from .masks import NativeMaskDecoder, NativeModelDecoderWrapper, PromptableMaskDecoder, TransformerMaskDecoder
from .unet import UNetDecoder2D, UNetDecoder3D

DecoderOutput = SegmentationOutput

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
    "NativeMaskDecoder",
    "NativeModelDecoderWrapper",
]
