"""Model adapters, bridges, and encoder-independent task modules.

Concrete encoders remain in their modality-specific packages.  The ``heads``
and ``decoders`` packages consume only shared ``EncoderOutput`` semantics.
"""

__all__ = ["bridges", "decoders", "heads", "language", "pathology", "visual"]
