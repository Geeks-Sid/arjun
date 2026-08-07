"""Versioned structured-findings validation and generation scoring helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from medfm.core.errors import ContractError

STRUCTURED_FINDINGS_SCHEMA_VERSION = 1
STRUCTURED_FINDINGS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "medfm://schemas/structured-findings/v1",
    "title": "MedFM structured findings",
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "impression"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "status", "anatomy", "laterality", "severity", "location"],
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "status": {"enum": ["present", "absent", "uncertain"]},
                    "anatomy": {"type": "string", "minLength": 1},
                    "laterality": {"enum": ["left", "right", "bilateral", "midline", "none"]},
                    "severity": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                },
            },
        },
        "impression": {"type": "string"},
    },
}


class StructuredFindingsError(ContractError):
    """Malformed or unsupported structured generation payload."""


@dataclass(frozen=True)
class StructuredValidationResult:
    valid: bool
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    schema_errors: tuple[str, ...] = ()
    sample_index: int | None = None

    @property
    def error_count(self) -> int:
        return int(self.parse_error is not None) + len(self.schema_errors)


@dataclass(frozen=True)
class StructuredValidationReport:
    total: int
    valid: int
    invalid: int
    parse_errors: int
    schema_errors: int
    results: tuple[StructuredValidationResult, ...] = ()

    @property
    def valid_fraction(self) -> float:
        return self.valid / self.total if self.total else 0.0


class StructuredFindingsValidator:
    """Validate generated JSON without retaining invalid raw text by default."""

    version = STRUCTURED_FINDINGS_SCHEMA_VERSION

    def __init__(
        self,
        *,
        debug_sink: Callable[[int, str], None] | None = None,
        debug_access_controlled: bool = False,
    ) -> None:
        if debug_sink is not None and not debug_access_controlled:
            raise StructuredFindingsError(
                "invalid raw generation can only be retained through an access-controlled debug sink"
            )
        self._validator = Draft202012Validator(STRUCTURED_FINDINGS_SCHEMA)
        self._debug_sink = debug_sink

    def validate(self, value: str | dict[str, Any], *, sample_index: int | None = None) -> StructuredValidationResult:
        parsed: dict[str, Any]
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError) as exc:
                result = StructuredValidationResult(
                    valid=False,
                    parse_error=f"invalid_json:{type(exc).__name__}",
                    sample_index=sample_index,
                )
                self._debug_invalid(sample_index, value)
                return result
            if not isinstance(decoded, dict):
                result = StructuredValidationResult(
                    valid=False,
                    parse_error="json_root_must_be_object",
                    sample_index=sample_index,
                )
                self._debug_invalid(sample_index, value)
                return result
            parsed = decoded
        elif isinstance(value, dict):
            parsed = value
        else:
            result = StructuredValidationResult(
                valid=False,
                parse_error=f"unsupported_value_type:{type(value).__name__}",
                sample_index=sample_index,
            )
            self._debug_invalid(sample_index, str(value))
            return result
        errors = tuple(sorted(self._format_error(error) for error in self._validator.iter_errors(parsed)))
        result = StructuredValidationResult(
            valid=not errors,
            parsed=parsed if not errors else None,
            schema_errors=errors,
            sample_index=sample_index,
        )
        if errors:
            raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
            self._debug_invalid(sample_index, raw)
        return result

    def validate_batch(self, values: Sequence[str | dict[str, Any]]) -> StructuredValidationReport:
        results = tuple(self.validate(value, sample_index=index) for index, value in enumerate(values))
        parse_errors = sum(result.parse_error is not None for result in results)
        schema_errors = sum(bool(result.schema_errors) for result in results)
        valid = sum(result.valid for result in results)
        return StructuredValidationReport(
            total=len(results),
            valid=valid,
            invalid=len(results) - valid,
            parse_errors=parse_errors,
            schema_errors=schema_errors,
            results=results,
        )

    @staticmethod
    def _format_error(error: Any) -> str:
        path = ".".join(str(part) for part in error.absolute_path)
        return f"{path or '<root>'}:{error.validator}:{error.message}"

    def _debug_invalid(self, sample_index: int | None, raw: str) -> None:
        if self._debug_sink is not None:
            self._debug_sink(-1 if sample_index is None else sample_index, raw)


def validate_structured_findings(value: str | dict[str, Any]) -> StructuredValidationResult:
    """Pure convenience wrapper for one payload."""

    return StructuredFindingsValidator().validate(value)


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Scoring result that keeps invalid-output counts visible."""

    report: StructuredValidationReport
    score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_generation_before_scoring(
    generated: Sequence[str | dict[str, Any]],
    scorer: Callable[[Sequence[dict[str, Any]]], float] | None = None,
    *,
    validator: StructuredFindingsValidator | None = None,
) -> StructuredGenerationResult:
    """Validate all generations, then score only valid parsed structures.

    Invalid samples are counted in the returned report and are never silently
    removed from metrics.  If no valid sample remains, scoring returns ``None``.
    """

    active_validator = validator or StructuredFindingsValidator()
    report = active_validator.validate_batch(generated)
    parsed = tuple(result.parsed for result in report.results if result.valid and result.parsed is not None)
    score = scorer(parsed) if scorer is not None and parsed else None
    return StructuredGenerationResult(
        report=report,
        score=score,
        diagnostics={
            "structured_schema_version": STRUCTURED_FINDINGS_SCHEMA_VERSION,
            "invalid_count": report.invalid,
            "parse_error_count": report.parse_errors,
            "schema_error_count": report.schema_errors,
        },
    )


__all__ = [
    "STRUCTURED_FINDINGS_SCHEMA_VERSION",
    "STRUCTURED_FINDINGS_SCHEMA",
    "StructuredFindingsError",
    "StructuredValidationResult",
    "StructuredValidationReport",
    "StructuredFindingsValidator",
    "validate_structured_findings",
    "StructuredGenerationResult",
    "validate_generation_before_scoring",
]
