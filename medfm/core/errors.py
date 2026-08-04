"""Typed errors for the core contract layer.

Every contract violation raises one of these (never a bare ``ValueError``) so
callers can distinguish unsupported combinations from malformed data and act
on the actionable messages these errors carry.
"""

from __future__ import annotations


class ContractError(Exception):
    """Base class for all core contract violations."""


class UnknownEnumValueError(ContractError):
    """A string did not match any canonical enum value and no migration applied."""


class SchemaVersionError(ContractError):
    """A payload declares a schema version this code cannot read or migrate."""


class SchemaValidationError(ContractError):
    """A sample/batch/metadata object violates its schema."""


class IdentifierError(SchemaValidationError):
    """A typed ID field carries (something that looks like) a raw identifier."""


class ShapeContractError(SchemaValidationError):
    """Tensor rank/shape is inconsistent with the declared modality or batch."""


class BucketError(SchemaValidationError):
    """A static-shape bucket is malformed or its padding lacks masks."""


class UnsupportedModalityError(ContractError):
    """The component does not support the requested modality."""


class UnsupportedTaskError(ContractError):
    """The component does not support the requested task."""


class UnsupportedCapabilityError(ContractError):
    """The component lacks a requested capability (e.g. spatial tokens)."""


class SerializationError(ContractError):
    """Canonical serialization or deserialization failed."""
