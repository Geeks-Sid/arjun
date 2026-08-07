"""Bounded VLM decoding, prompt isolation, and structured-output validation."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from medfm.inference.errors import RequestLimitError, RequestValidationError, StructuredOutputError
from medfm.inference.schemas import InferenceLimits


@dataclass(frozen=True)
class GenerationConfig:
    """Generation controls with deterministic evaluation defaults."""

    mode: str = "greedy"
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    stop_tokens: tuple[str, ...] = ()
    prompt_version: str = "v1"
    output_schema: Mapping[str, Any] | None = None
    visual_token_limit: int = 8192
    research_sampling: bool = False
    clinical_style: bool = True
    tpu_length_buckets: tuple[int, ...] = ()
    return_uncertainty: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in {"greedy", "beam", "sampling"}:
            raise ValueError("generation mode must be greedy, beam, or sampling")
        if self.max_new_tokens <= 0 or self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ValueError("generation token limits are invalid")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 < float(self.top_p) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if mode == "beam" and self.num_beams < 2:
            raise ValueError("beam mode requires num_beams >= 2")
        if mode == "sampling" and not self.research_sampling:
            raise ValueError("sampling is research-only; set research_sampling=true explicitly")
        if self.clinical_style and mode == "sampling":
            raise ValueError("clinical-style generation must be deterministic")
        if self.visual_token_limit <= 0:
            raise ValueError("visual_token_limit must be positive")
        if self.tpu_length_buckets and tuple(sorted(set(self.tpu_length_buckets))) != self.tpu_length_buckets:
            raise ValueError("tpu_length_buckets must be sorted and unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GenerationConfig:
        raw = dict(data or {})
        if "stop_tokens" in raw:
            raw["stop_tokens"] = tuple(str(value) for value in raw["stop_tokens"])
        if "tpu_length_buckets" in raw:
            raw["tpu_length_buckets"] = tuple(int(value) for value in raw["tpu_length_buckets"])
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in ((field.name, getattr(self, field.name)) for field in self.__dataclass_fields__.values())
        }


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_count: int
    stopped_on: str | None
    schema_valid: bool | None
    parsed: Any | None
    uncertainty: Mapping[str, Any] | None = None
    invalid_output: bool = False
    compile_bucket: int | None = None
    prompt_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_count": self.token_count,
            "stopped_on": self.stopped_on,
            "schema_valid": self.schema_valid,
            "parsed": self.parsed,
            "uncertainty": dict(self.uncertainty) if self.uncertainty else None,
            "invalid_output": self.invalid_output,
            "compile_bucket": self.compile_bucket,
            "prompt_version": self.prompt_version,
        }


def build_safe_prompt(system_prompt: str, user_content: str, *, report_text: str | None = None) -> str:
    """Separate system instructions from untrusted report content.

    The report is data delimited by fixed markers; it is never interpolated into
    the system instruction and the output is never executed as a command.
    """

    if not isinstance(system_prompt, str) or not isinstance(user_content, str):
        raise RequestValidationError(details={"field": "prompt"})
    report_block = ""
    if report_text is not None:
        if not isinstance(report_text, str):
            raise RequestValidationError(details={"field": "report"})
        report_block = f"\n<untrusted_report>\n{report_text}\n</untrusted_report>"
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_content}{report_block}\n</user>"


def select_length_bucket(length: int, buckets: Sequence[int], *, allow_pad: bool = True) -> int:
    if length < 0:
        raise RequestValidationError(details={"field": "length"})
    ordered = tuple(sorted(set(int(value) for value in buckets if int(value) > 0)))
    if not ordered:
        return length
    for bucket in ordered:
        if length <= bucket:
            return bucket
    if allow_pad:
        raise RequestLimitError(details={"limit": "tpu_length_bucket", "maximum": ordered[-1]})
    raise RequestValidationError(details={"field": "length", "reason": "out of TPU bucket"})


def validate_json_output(text: str, schema: Mapping[str, Any]) -> tuple[bool, Any | None, tuple[str, ...]]:
    """Validate JSON output without returning the report/input in errors."""

    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False, None, ("invalid_json",)
    try:
        from jsonschema import Draft202012Validator

        validator = Draft202012Validator(dict(schema))
        errors = tuple(sorted(error.validator for error in validator.iter_errors(parsed)))
    except Exception:
        return False, None, ("invalid_schema",)
    return not errors, parsed, errors


def _decode_output(value: Any, *, tokenizer: Any | None = None) -> tuple[str, int]:
    if isinstance(value, str):
        return value, len(value.split())
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            value = value[0]
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            text = str(tokenizer.decode(value.detach().to("cpu").tolist(), skip_special_tokens=True))
        else:
            text = " ".join(str(int(token)) for token in value.detach().to("cpu").flatten().tolist())
        return text, int(value.numel())
    if isinstance(value, Mapping):
        for key in ("text", "sequences", "output", "generated_text"):
            if key in value:
                return _decode_output(value[key], tokenizer=tokenizer)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        if value and isinstance(value[0], str):
            text = str(value[0])
            return text, len(text.split())
        if value and isinstance(value[0], int | float) and not isinstance(value[0], bool):
            return _decode_output(torch.as_tensor(value), tokenizer=tokenizer)
        if value and isinstance(value[0], torch.Tensor):
            return _decode_output(value[0], tokenizer=tokenizer)
    raise RequestValidationError(details={"field": "generation_output", "reason": "unsupported model output"})


def _call_generator(model: Any, kwargs: Mapping[str, Any]) -> Any:
    generator = model.generate
    try:
        signature = inspect.signature(generator)
    except (TypeError, ValueError):
        return generator(**dict(kwargs))
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return generator(**dict(kwargs))
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return generator(**accepted)


def generate(
    model: Any,
    *,
    input_ids: torch.Tensor | None = None,
    visual_tokens: torch.Tensor | None = None,
    prompt: str | None = None,
    config: GenerationConfig | Mapping[str, Any] | None = None,
    limits: InferenceLimits | None = None,
    tokenizer: Any | None = None,
    constrained_decoder: Any | None = None,
) -> GenerationResult:
    """Generate bounded output with deterministic defaults and schema status."""

    generation = config if isinstance(config, GenerationConfig) else GenerationConfig.from_dict(config)
    max_tokens = min(generation.max_new_tokens, limits.max_output_tokens if limits else generation.max_new_tokens)
    if input_ids is not None:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RequestValidationError(details={"field": "input_ids"})
        if limits:
            limits.validate_tokens(
                int(input_ids.shape[1]),
                visual_tokens=int(visual_tokens.shape[1])
                if visual_tokens is not None and visual_tokens.ndim >= 2
                else 0,
            )
        elif int(input_ids.shape[1]) <= 0:
            raise RequestValidationError(details={"field": "input_ids"})
    visual_count = (
        int(visual_tokens.shape[1]) if isinstance(visual_tokens, torch.Tensor) and visual_tokens.ndim >= 2 else 0
    )
    if visual_count > generation.visual_token_limit:
        raise RequestLimitError(details={"limit": "visual_token_limit", "maximum": generation.visual_token_limit})
    compile_bucket = (
        select_length_bucket(max_tokens, generation.tpu_length_buckets) if generation.tpu_length_buckets else None
    )
    min_tokens = min(generation.min_new_tokens, max_tokens)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "min_new_tokens": min_tokens,
        "do_sample": generation.mode == "sampling",
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "num_beams": generation.num_beams,
    }
    if constrained_decoder is not None:
        kwargs["logits_processor"] = constrained_decoder
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    if visual_tokens is not None:
        kwargs["visual_tokens"] = visual_tokens
    if prompt is not None:
        kwargs["prompt"] = prompt
    with torch.inference_mode():
        if hasattr(model, "generate"):
            generated = _call_generator(model, kwargs)
        elif callable(model):
            generated = model(input_ids if input_ids is not None else prompt)
        else:
            raise RequestValidationError(details={"field": "model"})
    text, token_count = _decode_output(generated, tokenizer=tokenizer)
    if token_count > max_tokens:
        if isinstance(generated, torch.Tensor):
            clipped = generated[..., :max_tokens]
            text, token_count = _decode_output(clipped, tokenizer=tokenizer)
        else:
            text = " ".join(text.split()[:max_tokens])
            token_count = max_tokens
    stopped_on: str | None = None
    for stop in generation.stop_tokens:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
            stopped_on = stop
            token_count = min(token_count, len(text.split()))
            break
    schema_valid: bool | None = None
    parsed: Any | None = None
    invalid = False
    if generation.output_schema is not None:
        schema_valid, parsed, _errors = validate_json_output(text, generation.output_schema)
        invalid = not schema_valid
    uncertainty = {"status": "not_available"} if generation.return_uncertainty else None
    return GenerationResult(
        text=text,
        token_count=token_count,
        stopped_on=stopped_on,
        schema_valid=schema_valid,
        parsed=parsed,
        uncertainty=uncertainty,
        invalid_output=invalid,
        compile_bucket=compile_bucket,
        prompt_version=generation.prompt_version,
    )


def require_valid_output(result: GenerationResult) -> Any:
    if result.invalid_output or result.schema_valid is False:
        raise StructuredOutputError()
    return result.parsed if result.parsed is not None else result.text


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "build_safe_prompt",
    "generate",
    "require_valid_output",
    "select_length_bucket",
    "validate_json_output",
]
