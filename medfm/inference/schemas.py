"""Versioned request/response schemas and bounded inference limits."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from medfm.core.enums import Modality
from medfm.inference.errors import RequestLimitError, RequestValidationError

REQUEST_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1


class InferenceTask(StrEnum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    RETRIEVAL = "retrieval"
    VLM = "vlm"
    WSI = "wsi"

    @classmethod
    def parse(cls, value: str | InferenceTask) -> InferenceTask:
        raw = str(value).strip().lower()
        aliases = {"classify": cls.CLASSIFICATION, "segment": cls.SEGMENTATION, "generation": cls.VLM}
        try:
            if raw in aliases:
                return aliases[raw]
            return cls(raw)
        except ValueError as exc:
            raise RequestValidationError(details={"field": "task"}) from exc


@dataclass(frozen=True)
class InferenceLimits:
    """Hard request and runtime caps; every pipeline validates before execution."""

    max_batch_size: int = 8
    max_image_pixels: int = 4096 * 4096
    max_volume_voxels: int = 256 * 256 * 256
    max_images_per_sample: int = 32
    max_tiles: int = 4096
    max_tokens: int = 4096
    max_visual_tokens: int = 8192
    max_output_tokens: int = 1024
    max_memory_bytes: int | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        integer_fields = (
            "max_batch_size",
            "max_image_pixels",
            "max_volume_voxels",
            "max_images_per_sample",
            "max_tiles",
            "max_tokens",
            "max_visual_tokens",
            "max_output_tokens",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_memory_bytes is not None and int(self.max_memory_bytes) <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> InferenceLimits:
        raw = dict(data or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def validate_batch(self, batch_size: int) -> None:
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise RequestLimitError(details={"limit": "max_batch_size", "maximum": self.max_batch_size})

    def validate_tensor(self, tensor: torch.Tensor, *, modality: str | Modality, role: str = "input") -> None:
        if not isinstance(tensor, torch.Tensor):
            raise RequestValidationError(details={"field": role, "reason": "tensor required"})
        if tensor.ndim < 1:
            raise RequestValidationError(details={"field": role, "reason": "tensor rank must be positive"})
        self.validate_batch(int(tensor.shape[0]))
        modality_value = str(modality)
        if modality_value in {
            Modality.XRAY_2D.value,
            Modality.CT_2D_SLICE.value,
            Modality.MRI_2D_SLICE.value,
            Modality.PATHOLOGY_TILE.value,
        }:
            pixels = int(math.prod(int(v) for v in tensor.shape[-2:]))
            if pixels > self.max_image_pixels:
                raise RequestLimitError(details={"limit": "max_image_pixels", "maximum": self.max_image_pixels})
        if modality_value in {Modality.CT_3D.value, Modality.MRI_3D.value, Modality.MULTI_SERIES_3D.value}:
            spatial = tensor.shape[-3:]
            voxels = int(math.prod(int(v) for v in spatial))
            if voxels > self.max_volume_voxels:
                raise RequestLimitError(details={"limit": "max_volume_voxels", "maximum": self.max_volume_voxels})
        if modality_value == Modality.MULTI_IMAGE_2D.value:
            if int(tensor.shape[1]) > self.max_images_per_sample:
                raise RequestLimitError(
                    details={"limit": "max_images_per_sample", "maximum": self.max_images_per_sample}
                )
        if modality_value == Modality.PATHOLOGY_WSI.value:
            if int(tensor.shape[1]) > self.max_tiles:
                raise RequestLimitError(details={"limit": "max_tiles", "maximum": self.max_tiles})

    def validate_tokens(self, token_count: int, *, visual_tokens: int = 0) -> None:
        if token_count < 0 or token_count > self.max_tokens:
            raise RequestLimitError(details={"limit": "max_tokens", "maximum": self.max_tokens})
        if visual_tokens < 0 or visual_tokens > self.max_visual_tokens:
            raise RequestLimitError(details={"limit": "max_visual_tokens", "maximum": self.max_visual_tokens})


@dataclass(frozen=True)
class InferenceRequest:
    """Public request envelope; payload is validated by the routed pipeline."""

    task: InferenceTask
    modality: str
    payload: Mapping[str, Any]
    request_id: str | None = None
    adapter: str | None = None
    prompt_version: str | None = None
    schema_version: int = REQUEST_SCHEMA_VERSION
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise RequestValidationError(details={"field": "schema_version"})
        try:
            object.__setattr__(self, "task", InferenceTask.parse(self.task))
            object.__setattr__(self, "modality", str(Modality.from_value(self.modality)))
        except Exception as exc:
            raise RequestValidationError(details={"field": "task_or_modality"}) from exc
        if not isinstance(self.payload, Mapping):
            raise RequestValidationError(details={"field": "payload"})
        if not isinstance(self.options, Mapping):
            raise RequestValidationError(details={"field": "options"})
        if self.request_id is not None and (
            not isinstance(self.request_id, str)
            or len(self.request_id) > 128
            or any(ord(c) < 32 for c in self.request_id)
        ):
            raise RequestValidationError(details={"field": "request_id"})
        if self.adapter is not None and (
            not isinstance(self.adapter, str) or len(self.adapter) > 128 or "/" in self.adapter or "\\" in self.adapter
        ):
            raise RequestValidationError(details={"field": "adapter"})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InferenceRequest:
        raw = dict(data)
        if "payload" not in raw and "input" in raw:
            raw["payload"] = raw.pop("input")
        task = InferenceTask.parse(str(raw.get("task", "")))
        modality = str(raw.get("modality", ""))
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            raise RequestValidationError(details={"field": "payload"})
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise RequestValidationError(details={"field": "options"})
        return cls(
            task=task,
            modality=modality,
            payload=dict(payload),
            request_id=(str(raw["request_id"]) if raw.get("request_id") is not None else None),
            adapter=(str(raw["adapter"]) if raw.get("adapter") is not None else None),
            prompt_version=(str(raw["prompt_version"]) if raw.get("prompt_version") is not None else None),
            schema_version=int(raw.get("schema_version", REQUEST_SCHEMA_VERSION)),
            options=dict(options),
        )

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task": self.task.value,
            "modality": self.modality,
            "request_id": self.request_id,
            "adapter": self.adapter,
            "prompt_version": self.prompt_version,
            "options": dict(self.options),
        }
        if include_payload:
            data["payload"] = dict(self.payload)
        return data


@dataclass(frozen=True)
class InferenceResult:
    """Versioned result envelope with no implicit serialization of raw inputs."""

    task: str
    data: Mapping[str, Any]
    request_id: str | None = None
    schema_version: int = RESPONSE_SCHEMA_VERSION
    warnings: tuple[str, ...] = ()
    uncertainty: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_SCHEMA_VERSION:
            raise ValueError("unsupported inference response schema version")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self, *, serialize_tensors: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "request_id": self.request_id,
            "data": _serialize(self.data) if serialize_tensors else dict(self.data),
            "warnings": list(self.warnings),
            "uncertainty": _serialize(self.uncertainty) if serialize_tensors else self.uncertainty,
        }


@dataclass(frozen=True)
class InferenceResponse:
    """API response; failures remain structured and privacy-safe."""

    ok: bool
    request_id: str | None
    result: InferenceResult | None = None
    error: Mapping[str, Any] | None = None
    audit: Mapping[str, Any] | None = None
    schema_version: int = RESPONSE_SCHEMA_VERSION

    def to_dict(self, *, serialize_tensors: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "request_id": self.request_id,
            "result": self.result.to_dict(serialize_tensors=serialize_tensors) if self.result else None,
            "error": dict(self.error) if self.error else None,
            "audit": dict(self.audit) if self.audit else None,
        }


def payload_hash(value: Any) -> str:
    """Deterministically hash payload structure/content without logging it."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to("cpu").contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"{")
            for original_key in sorted(item, key=lambda key: str(key)):
                digest.update(str(original_key).encode("utf-8"))
                update(item[original_key])
            digest.update(b"}")
        elif isinstance(item, list | tuple):
            digest.update(b"[")
            for value in item:
                update(value)
            digest.update(b"]")
        elif isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
        elif item is None or isinstance(item, str | int | float | bool):
            digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        else:
            digest.update(f"opaque:{type(item).__module__}.{type(item).__qualname__}".encode())

    update(value)
    return digest.hexdigest()


def validate_request(
    request: InferenceRequest | Mapping[str, Any], limits: InferenceLimits | None = None
) -> InferenceRequest:
    """Validate only envelope-level fields before a model/adapter is allocated."""

    parsed = request if isinstance(request, InferenceRequest) else InferenceRequest.from_dict(request)
    if limits is not None:
        for value in parsed.payload.values():
            if isinstance(value, torch.Tensor) and value.ndim:
                limits.validate_batch(int(value.shape[0]))
                break
    return parsed


def _serialize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").tolist()
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "InferenceLimits",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceResult",
    "InferenceTask",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "payload_hash",
    "validate_request",
]
