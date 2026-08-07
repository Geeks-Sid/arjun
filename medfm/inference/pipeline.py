"""Task pipelines that share the training backend abstraction.

Pipelines validate modality, task, tensor shapes, and limits before calling a
model.  Models are supplied by reviewed application code; this module does not
resolve arbitrary Python paths from request data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.core.enums import Modality
from medfm.inference.errors import (
    BackendInferenceError,
    RequestLimitError,
    RequestValidationError,
    UnsupportedTaskError,
)
from medfm.inference.generation import GenerationConfig, generate
from medfm.inference.schemas import (
    InferenceLimits,
    InferenceRequest,
    InferenceResult,
    InferenceTask,
    validate_request,
)
from medfm.inference.sliding_window import sliding_window_inference
from medfm.training.backend import AcceleratorBackend, BackendError, create_backend
from medfm.training.config import AcceleratorConfig


@dataclass(frozen=True)
class BucketPolicy:
    """Predeclared static buckets for TPU variable dimensions."""

    name: str
    buckets: tuple[tuple[int, ...], ...]
    pad: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.buckets:
            raise ValueError("bucket policy requires a name and at least one bucket")
        normalized = tuple(tuple(int(value) for value in shape) for shape in self.buckets)
        if any(not shape or any(value <= 0 for value in shape) for shape in normalized):
            raise ValueError("bucket shapes must be positive")
        if normalized != tuple(sorted(normalized, key=lambda shape: math.prod(shape))):
            raise ValueError("bucket shapes must be ordered by capacity")
        object.__setattr__(self, "buckets", normalized)

    def select(self, shape: Sequence[int]) -> tuple[int, ...]:
        requested = tuple(int(value) for value in shape)
        for bucket in self.buckets:
            if len(bucket) == len(requested) and all(
                actual <= limit for actual, limit in zip(requested, bucket, strict=True)
            ):
                return bucket
        raise RequestLimitError(details={"limit": f"{self.name}_bucket", "maximum": list(self.buckets[-1])})

    def pad_tensor(
        self, tensor: torch.Tensor, *, dim_offset: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        shape = tuple(int(value) for value in tensor.shape[dim_offset:])
        bucket = self.select(shape)
        if shape == bucket:
            mask = torch.ones(shape, device=tensor.device, dtype=torch.bool)
            return tensor, mask, bucket
        if not self.pad:
            raise RequestValidationError(details={"field": self.name, "reason": "input is outside exact bucket"})
        pad_shape = (*tensor.shape[:dim_offset], *bucket)
        padded = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        mask = torch.zeros(bucket, device=tensor.device, dtype=torch.bool)
        slices: tuple[slice, ...] = tuple(slice(0, value) for value in shape)
        index: tuple[slice, ...] = (slice(None),) * (tensor.ndim - len(shape)) + slices
        padded[index] = tensor
        mask[slices] = True
        return padded, mask, bucket


class InferenceRuntime:
    """Owns backend placement, autocast, synchronization, and memory policy."""

    def __init__(
        self,
        backend: AcceleratorBackend | AcceleratorConfig | str = "cpu",
        *,
        use_accelerate: bool = False,
        max_memory_bytes: int | None = None,
    ) -> None:
        try:
            self.backend = (
                backend
                if isinstance(backend, AcceleratorBackend)
                else create_backend(backend, use_accelerate=use_accelerate)
            )
        except Exception as exc:
            raise BackendInferenceError(details={"backend": str(backend)}) from exc
        self.max_memory_bytes = max_memory_bytes

    @property
    def device(self) -> torch.device:
        return self.backend.device

    @property
    def name(self) -> str:
        return self.backend.name

    def prepare(self, value: Any) -> Any:
        return self.backend.prepare_batch(value)

    def memory_snapshot(self) -> Any:
        return self.backend.memory_snapshot()

    def execute(self, model: Any, model_input: Any) -> Any:
        if isinstance(model, nn.Module):
            model.eval()
        prepared = self.prepare(model_input)
        try:
            with torch.inference_mode(), self.backend.autocast():
                output = model(prepared)
            self.backend.mark_step()
        except (BackendError, RuntimeError) as exc:
            raise BackendInferenceError(details={"backend": self.name}) from exc
        self._check_memory()
        return output

    def warmup(
        self, buckets: Mapping[str, Sequence[Sequence[int]]], runner: Callable[[str, tuple[int, ...]], Any]
    ) -> dict[str, tuple[int, ...]]:
        """Warm every declared bucket before readiness/latency measurement."""

        warmed: dict[str, tuple[int, ...]] = {}
        for name, shapes in buckets.items():
            for shape in shapes:
                normalized = tuple(int(value) for value in shape)
                runner(str(name), normalized)
                warmed[str(name)] = normalized
        self.backend.synchronize()
        return warmed

    def _check_memory(self) -> None:
        if self.max_memory_bytes is None:
            return
        snapshot = self.memory_snapshot()
        value = getattr(snapshot, "peak_allocated_bytes", None)
        if value is not None and int(value) > self.max_memory_bytes:
            raise RequestLimitError(details={"limit": "max_memory_bytes", "maximum": self.max_memory_bytes})


class InferencePipeline:
    task: InferenceTask

    def __init__(
        self,
        model: Any,
        *,
        runtime: InferenceRuntime | None = None,
        backend: AcceleratorBackend | AcceleratorConfig | str | None = None,
        limits: InferenceLimits | None = None,
        preprocess: Callable[[Any], Any] | None = None,
        postprocess: Callable[[Any], Any] | None = None,
        input_builder: Callable[[torch.Tensor, str], Any] | None = None,
        model_id: str = "unknown-model",
        model_revision: str = "unknown-revision",
        preprocess_hash: str = "unknown-preprocess",
    ) -> None:
        self.model = model
        self.runtime = runtime or InferenceRuntime(
            backend or "cpu", max_memory_bytes=(limits.max_memory_bytes if limits else None)
        )
        self.limits = limits or InferenceLimits()
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.input_builder = input_builder
        self.model_id = model_id
        self.model_revision = model_revision
        self.preprocess_hash = preprocess_hash

    def preflight(self, request: InferenceRequest | Mapping[str, Any]) -> InferenceRequest:
        """Validate task/modality/payload limits before adapter/model loading."""

        parsed = validate_request(request, self.limits)
        if parsed.task is not self.task:
            raise UnsupportedTaskError(details={"expected": self.task.value, "received": parsed.task.value})
        self._preflight_request(parsed)
        return parsed

    def _preflight_request(self, request: InferenceRequest) -> None:
        del request

    def run(self, request: InferenceRequest | Mapping[str, Any]) -> InferenceResult:
        parsed = self.preflight(request)
        return self._run_request(parsed)

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError

    def _tensor(self, payload: Mapping[str, Any], *keys: str) -> torch.Tensor:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise RequestValidationError(details={"field": keys[0], "reason": "tensor required"})

    def _preprocess(self, value: Any) -> Any:
        return self.preprocess(value) if self.preprocess is not None else value

    def _input(self, tensor: torch.Tensor, modality: str) -> Any:
        return self.input_builder(tensor, modality) if self.input_builder is not None else tensor

    def _call(self, tensor: torch.Tensor, modality: str) -> Any:
        return self.runtime.execute(self.model, self._input(tensor, modality))

    @staticmethod
    def _mapping_output(output: Any, *, primary: str = "logits") -> dict[str, Any]:
        if isinstance(output, Mapping):
            return dict(output)
        if isinstance(output, torch.Tensor):
            return {primary: output}
        if isinstance(output, list | tuple) and output and isinstance(output[0], torch.Tensor):
            return {primary: output[0]}
        raise RequestValidationError(details={"field": "model_output", "reason": "tensor or mapping required"})

    def _finish(
        self, task: str, data: Mapping[str, Any], request: InferenceRequest, *, warnings: Sequence[str] = ()
    ) -> InferenceResult:
        output = self.postprocess(dict(data)) if self.postprocess is not None else dict(data)
        if not isinstance(output, Mapping):
            raise RequestValidationError(details={"field": "postprocess", "reason": "mapping required"})
        return InferenceResult(task=task, data=dict(output), request_id=request.request_id, warnings=tuple(warnings))


class ClassificationPipeline(InferencePipeline):
    task = InferenceTask.CLASSIFICATION

    def predict(
        self,
        pixel_values: torch.Tensor,
        *,
        modality: str | Modality,
        request_id: str | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> InferenceResult:
        request = InferenceRequest(
            task=self.task,
            modality=str(modality),
            payload={"pixel_values": pixel_values, "sample_ids": list(sample_ids or ())},
            request_id=request_id,
        )
        return self.run(request)

    def _preflight_request(self, request: InferenceRequest) -> None:
        modality = _modality(request.modality)
        tensor = self._tensor(request.payload, "pixel_values", "image", "input")
        _validate_modality_tensor(tensor, modality, self.limits)

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        modality = _modality(request.modality)
        tensor = self._tensor(request.payload, "pixel_values", "image", "input")
        _validate_modality_tensor(tensor, modality, self.limits)
        tensor = self._preprocess(tensor)
        if not isinstance(tensor, torch.Tensor):
            raise RequestValidationError(details={"field": "preprocess", "reason": "tensor required"})
        _validate_modality_tensor(tensor, modality, self.limits)
        outputs = self._mapping_output(self._call(tensor, modality.value))
        logits = outputs.get("logits", outputs.get("scores"))
        if not isinstance(logits, torch.Tensor):
            raise RequestValidationError(details={"field": "model_output.logits"})
        if logits.ndim == 1:
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
        elif logits.shape[-1] == 1:
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
        else:
            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
        data = {**outputs, "logits": logits, "probabilities": probabilities, "predictions": predictions}
        if request.payload.get("sample_ids"):
            data["sample_ids"] = tuple(str(value) for value in request.payload["sample_ids"])
        return self._finish(self.task.value, data, request)


class SegmentationPipeline(InferencePipeline):
    task = InferenceTask.SEGMENTATION

    def __init__(
        self,
        *args: Any,
        window_shape: tuple[int, int, int] | None = None,
        overlap: float = 0.25,
        sigma_scale: float = 0.125,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.window_shape = window_shape
        self.overlap = overlap
        self.sigma_scale = sigma_scale

    def _preflight_request(self, request: InferenceRequest) -> None:
        modality = _modality(request.modality)
        tensor = self._tensor(request.payload, "pixel_values", "volume", "image")
        _validate_modality_tensor(tensor, modality, self.limits)

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        modality = _modality(request.modality)
        tensor = self._tensor(request.payload, "pixel_values", "volume", "image")
        _validate_modality_tensor(tensor, modality, self.limits)
        tensor = self._preprocess(tensor)
        if not isinstance(tensor, torch.Tensor):
            raise RequestValidationError(details={"field": "preprocess"})
        metadata = request.payload.get("spatial_metadata")
        if metadata is not None and (not isinstance(metadata, Sequence) or len(metadata) != int(tensor.shape[0])):
            raise RequestValidationError(details={"field": "spatial_metadata"})
        metas = list(metadata) if metadata is not None else None
        if tensor.ndim == 5 and self.window_shape is not None:
            logits: torch.Tensor | None = sliding_window_inference(
                tensor,
                lambda crop, _metadata=None: self._mapping_output(self._call(crop, modality.value)).get("logits"),
                window_shape=self.window_shape,
                overlap=self.overlap,
                sigma_scale=self.sigma_scale,
                metadata=metas,
            )
        else:
            logits = self._mapping_output(self._call(tensor, modality.value)).get("logits")
        if not isinstance(logits, torch.Tensor):
            raise RequestValidationError(details={"field": "model_output.logits"})
        threshold = float(request.options.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise RequestValidationError(details={"field": "threshold"})
        if logits.shape[1] == 1:
            probabilities = torch.sigmoid(logits)
            mask = probabilities >= threshold
        else:
            probabilities = torch.softmax(logits, dim=1)
            mask = probabilities.argmax(dim=1, keepdim=True)
        if request.payload.get("restore") and metas:
            from medfm.inference.export_nifti import restore_mask_to_original

            restored_masks = [
                restore_mask_to_original(mask[index], metas[index], history=request.payload.get("history"))
                for index in range(int(mask.shape[0]))
            ]
            mask = torch.stack(restored_masks, dim=0)
        return self._finish(self.task.value, {"logits": logits, "probabilities": probabilities, "mask": mask}, request)


class RetrievalPipeline(InferencePipeline):
    task = InferenceTask.RETRIEVAL

    def _preflight_request(self, request: InferenceRequest) -> None:
        image = request.payload.get("image_embeddings", request.payload.get("pixel_values"))
        text = request.payload.get("text_embeddings", request.payload.get("input_ids"))
        for value, field_name in ((image, "image_embeddings"), (text, "text_embeddings")):
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.ndim < 2:
                    raise RequestValidationError(details={"field": field_name})
                self.limits.validate_batch(int(value.shape[0]))

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        image = request.payload.get("image_embeddings")
        text = request.payload.get("text_embeddings")
        if image is None and text is None:
            image = request.payload.get("pixel_values")
            text = request.payload.get("input_ids")
        if image is not None and not isinstance(image, torch.Tensor):
            raise RequestValidationError(details={"field": "image_embeddings"})
        if text is not None and not isinstance(text, torch.Tensor):
            raise RequestValidationError(details={"field": "text_embeddings"})
        if image is None and text is None:
            raise RequestValidationError(details={"field": "embeddings", "reason": "image or text input required"})
        for value in (image, text):
            if isinstance(value, torch.Tensor):
                if value.ndim < 2:
                    raise RequestValidationError(details={"field": "embeddings"})
                self.limits.validate_batch(int(value.shape[0]))
        if request.payload.get("image_embeddings") is not None or request.payload.get("text_embeddings") is not None:
            raw = {"image_embeddings": image, "text_embeddings": text}
        else:
            if isinstance(image, torch.Tensor):
                model_input = image
            elif isinstance(text, torch.Tensor):
                model_input = text
            else:
                raise RequestValidationError(details={"field": "embeddings", "reason": "image or text input required"})
            raw = self._call(model_input, request.modality)
        outputs = self._mapping_output(raw, primary="image_embeddings")
        image_embedding = outputs.get("image_embeddings", image)
        text_embedding = outputs.get("text_embeddings", text)
        data: dict[str, Any] = dict(outputs)
        if isinstance(image_embedding, torch.Tensor):
            image_embedding = torch.nn.functional.normalize(image_embedding.float(), dim=-1)
            data["image_embeddings"] = image_embedding
        if isinstance(text_embedding, torch.Tensor):
            text_embedding = torch.nn.functional.normalize(text_embedding.float(), dim=-1)
            data["text_embeddings"] = text_embedding
        if isinstance(image_embedding, torch.Tensor) and isinstance(text_embedding, torch.Tensor):
            data["similarity"] = image_embedding @ text_embedding.transpose(-1, -2)
        return self._finish(self.task.value, data, request)


class VLMPipeline(InferencePipeline):
    task = InferenceTask.VLM

    def _preflight_request(self, request: InferenceRequest) -> None:
        input_ids = request.payload.get("input_ids")
        visual_tokens = request.payload.get("visual_tokens")
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
                raise RequestValidationError(details={"field": "input_ids"})
            self.limits.validate_tokens(int(input_ids.shape[1]))
        visual_count = 0
        if visual_tokens is not None:
            if not isinstance(visual_tokens, torch.Tensor) or visual_tokens.ndim != 3:
                raise RequestValidationError(details={"field": "visual_tokens"})
            visual_count = int(visual_tokens.shape[1])
        self.limits.validate_tokens(0, visual_tokens=visual_count)

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        input_ids = request.payload.get("input_ids")
        visual_tokens = request.payload.get("visual_tokens")
        config = GenerationConfig.from_dict(request.options.get("generation"))
        result = generate(
            self.model,
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            prompt=request.payload.get("prompt"),
            config=config,
            limits=self.limits,
        )
        return self._finish(self.task.value, {"text": result.text, **result.to_dict()}, request)


class WSIPipeline(InferencePipeline):
    task = InferenceTask.WSI

    def _preflight_request(self, request: InferenceRequest) -> None:
        tiles = self._tensor(request.payload, "tiles", "pixel_values")
        if tiles.ndim not in (4, 5):
            raise RequestValidationError(details={"field": "tiles"})
        tile_count = int(tiles.shape[0] if tiles.ndim == 4 else tiles.shape[1])
        self.limits.validate_batch(1 if tiles.ndim == 4 else int(tiles.shape[0]))
        if tile_count > self.limits.max_tiles:
            raise RequestLimitError(details={"limit": "max_tiles", "maximum": self.limits.max_tiles})
        coords = request.payload.get("coordinates")
        expected_batch = 1 if tiles.ndim == 4 else int(tiles.shape[0])
        if coords is not None and (not isinstance(coords, torch.Tensor) or int(coords.shape[0]) != expected_batch):
            raise RequestValidationError(details={"field": "coordinates"})

    def _run_request(self, request: InferenceRequest) -> InferenceResult:
        tiles = self._tensor(request.payload, "tiles", "pixel_values")
        if tiles.ndim == 4:
            tiles = tiles.unsqueeze(0)
        if int(tiles.shape[1]) > self.limits.max_tiles:
            raise RequestLimitError(details={"limit": "max_tiles", "maximum": self.limits.max_tiles})
        coords = request.payload.get("coordinates")
        if coords is not None and (
            not isinstance(coords, torch.Tensor) or tuple(coords.shape[:2]) != tuple(tiles.shape[:2])
        ):
            raise RequestValidationError(details={"field": "coordinates"})
        tiles = self._preprocess(tiles)
        output = self._call(tiles, request.modality)
        outputs = self._mapping_output(output, primary="tile_embeddings")
        if isinstance(outputs.get("tile_embeddings"), torch.Tensor):
            tile_embeddings = outputs["tile_embeddings"]
            outputs["slide_embedding"] = tile_embeddings.mean(dim=1)
        if coords is not None:
            outputs["coordinates"] = coords
        return self._finish(self.task.value, outputs, request)


def _modality(value: str | Modality) -> Modality:
    try:
        return value if isinstance(value, Modality) else Modality.from_value(str(value))
    except Exception as exc:
        raise RequestValidationError(details={"field": "modality"}) from exc


def _validate_modality_tensor(tensor: torch.Tensor, modality: Modality, limits: InferenceLimits) -> None:
    expected_rank = modality.expected_pixel_rank
    if expected_rank is None:
        raise RequestValidationError(details={"field": "modality", "reason": "text modality requires token inputs"})
    if tensor.ndim != expected_rank:
        raise RequestValidationError(
            details={"field": "pixel_values", "reason": f"{modality.value} expects rank {expected_rank}"}
        )
    limits.validate_tensor(tensor, modality=modality.value)


__all__ = [
    "BucketPolicy",
    "ClassificationPipeline",
    "InferencePipeline",
    "InferenceRuntime",
    "RetrievalPipeline",
    "SegmentationPipeline",
    "VLMPipeline",
    "WSIPipeline",
]
