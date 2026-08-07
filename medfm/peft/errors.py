"""Typed failures for the PEFT and quantization subsystem.

The PEFT package is deliberately fail-closed: a caller can distinguish a
malformed configuration, an unknown architecture, an unavailable accelerator
capability, and an incompatible checkpoint without matching error strings.
"""

from __future__ import annotations

from medfm.core.errors import ContractError, SchemaValidationError


class PeftError(ContractError):
    """Base class for Phase 10 contract failures."""


class PeftConfigError(PeftError, SchemaValidationError):
    """A PEFT or quantization configuration is malformed."""


class BackendCapabilityError(PeftError):
    """The selected backend cannot provide the requested adaptation path."""


class QuantizationCapabilityError(BackendCapabilityError):
    """The requested quantization implementation is unavailable or unsupported."""


class BitsAndBytesCapabilityError(QuantizationCapabilityError):
    """bitsandbytes/CUDA NF4 support is unavailable for the selected runtime."""


class UnsupportedQuantizationError(QuantizationCapabilityError):
    """A quantization method is not accepted for a backend or model family."""


class UnknownArchitectureError(PeftError):
    """An architecture has no reviewed target policy and was not confirmed."""


class TargetResolutionError(PeftError):
    """Configured LoRA target patterns do not resolve safely."""


class TargetMatchError(TargetResolutionError):
    """A configured target pattern matched zero modules."""


class BroadTargetError(TargetResolutionError):
    """A configured target selection is unexpectedly broad."""


class TrainabilityError(PeftError):
    """The trainable/frozen parameter contract is violated."""


class QuantizedParameterError(TrainabilityError):
    """A quantized base parameter would receive gradients or optimizer state."""


class AdapterCheckpointError(PeftError):
    """An adapter checkpoint is malformed or cannot be serialized safely."""


class CheckpointCompatibilityError(AdapterCheckpointError):
    """A checkpoint does not belong to the requested base model/configuration."""


class OptionalDependencyError(PeftError):
    """An optional PEFT/quantization dependency is required by an operation."""


# Compatibility aliases used by callers that prefer a schema-oriented name.
PeftValidationError = PeftConfigError
QuantizationError = QuantizationCapabilityError


__all__ = [
    "AdapterCheckpointError",
    "BackendCapabilityError",
    "BitsAndBytesCapabilityError",
    "BroadTargetError",
    "CheckpointCompatibilityError",
    "OptionalDependencyError",
    "PeftConfigError",
    "PeftError",
    "PeftValidationError",
    "QuantizationCapabilityError",
    "QuantizationError",
    "QuantizedParameterError",
    "TargetMatchError",
    "TargetResolutionError",
    "TrainabilityError",
    "UnknownArchitectureError",
    "UnsupportedQuantizationError",
]
