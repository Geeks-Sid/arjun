"""Typed, privacy-safe errors for export and inference.

Error messages are intentionally bounded and must never include report text,
image payloads, raw identifiers, or arbitrary file contents.  Callers can use
``code`` for machine routing while exposing only ``public_message`` to clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InferenceError(RuntimeError):
    """Base class for errors that are safe to classify at the API boundary."""

    code = "INFERENCE_ERROR"
    public_message = "inference request failed"
    retryable = False

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None) -> None:
        # Subclasses pass only static or already-redacted messages.
        super().__init__(message or self.public_message)
        self.details = dict(details or {})

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.public_message,
            "retryable": bool(self.retryable),
            "details": dict(self.details),
        }


class RequestValidationError(InferenceError):
    """Request is malformed or violates a declared input/task contract."""

    code = "INVALID_REQUEST"
    public_message = "request validation failed"


class RequestLimitError(RequestValidationError):
    """A request exceeds an explicit safety or memory limit."""

    code = "REQUEST_LIMIT_EXCEEDED"
    public_message = "request exceeds configured inference limits"


class UnsupportedTaskError(RequestValidationError):
    """The task is not available in the selected bundle/catalog."""

    code = "UNSUPPORTED_TASK"
    public_message = "requested task is not supported"


class BundleError(RuntimeError):
    """Base class for malformed or unsafe deployment bundles."""


class BundleValidationError(BundleError):
    """A bundle does not satisfy the versioned layout contract."""


class BundleChecksumError(BundleValidationError):
    """A required file is missing or its checksum does not match."""


class BundleCompatibilityError(BundleValidationError):
    """The requested base/backend is incompatible with a bundle."""


class BundleDependencyError(BundleValidationError):
    """The runtime/dependency contract cannot be satisfied."""


class StructuredOutputError(InferenceError):
    """Generated output is not valid for the declared task schema."""

    code = "INVALID_STRUCTURED_OUTPUT"
    public_message = "generated output did not satisfy the task schema"


class InferenceTimeoutError(InferenceError):
    """A request exceeded the service timeout."""

    code = "INFERENCE_TIMEOUT"
    public_message = "inference request timed out"
    retryable = True


class ServiceBusyError(InferenceError):
    """Backpressure rejected a request before model execution."""

    code = "SERVICE_BUSY"
    public_message = "inference service is busy"
    retryable = True


class BackendInferenceError(InferenceError):
    """Backend allocation/execution failed without exposing backend internals."""

    code = "BACKEND_ERROR"
    public_message = "inference backend failed"
    retryable = True


class OptionalDependencyError(InferenceError):
    """An explicitly requested optional medical-I/O dependency is unavailable."""

    code = "OPTIONAL_DEPENDENCY_MISSING"
    public_message = "requested medical output format is unavailable"


@dataclass(frozen=True)
class ErrorStatus:
    """Compact status retained in audit records and API responses."""

    code: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.code is None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "retryable": self.retryable}


__all__ = [
    "BackendInferenceError",
    "BundleChecksumError",
    "BundleCompatibilityError",
    "BundleDependencyError",
    "BundleError",
    "BundleValidationError",
    "ErrorStatus",
    "InferenceError",
    "InferenceTimeoutError",
    "OptionalDependencyError",
    "RequestLimitError",
    "RequestValidationError",
    "ServiceBusyError",
    "StructuredOutputError",
    "UnsupportedTaskError",
]
