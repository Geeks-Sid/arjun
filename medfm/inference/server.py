"""Adapter-aware inference service with validation, backpressure, and audit."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medfm.inference.audit import AuditLogger, memory_peak_bytes
from medfm.inference.bundle import ModelBundle, load_bundle
from medfm.inference.errors import (
    BundleError,
    InferenceError,
    InferenceTimeoutError,
    ServiceBusyError,
)
from medfm.inference.pipeline import InferencePipeline, InferenceRuntime
from medfm.inference.schemas import (
    InferenceLimits,
    InferenceRequest,
    InferenceResponse,
    InferenceResult,
    validate_request,
)


@dataclass(frozen=True)
class AdapterRegistration:
    name: str
    bundle: ModelBundle | Path
    revision: str


class AdapterManager:
    """Load only requested adapters and evict them with a bounded LRU cache."""

    def __init__(
        self,
        *,
        base_model: Any | None = None,
        base_model_loader: Callable[[Any], Any] | None = None,
        adapter_loader: Callable[[Any, ModelBundle, str], Any] | None = None,
        adapter_unloader: Callable[[Any, str], None] | None = None,
        max_loaded: int = 2,
        allowed_bundle_root: str | Path | None = None,
    ) -> None:
        if max_loaded <= 0:
            raise ValueError("max_loaded must be positive")
        self.base_model = base_model
        self.base_model_loader = base_model_loader
        self.adapter_loader = adapter_loader
        self.adapter_unloader = adapter_unloader
        self.max_loaded = int(max_loaded)
        self.allowed_bundle_root = Path(allowed_bundle_root).expanduser().resolve() if allowed_bundle_root else None
        self._registrations: dict[str, AdapterRegistration] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()
        self._bundles: dict[str, ModelBundle] = {}
        self._active: str | None = None
        self._lock = threading.RLock()

    @property
    def active_adapter(self) -> str | None:
        return self._active

    @property
    def loaded_adapters(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._loaded)

    def register(self, name: str, bundle: ModelBundle | str | Path, *, revision: str | None = None) -> None:
        if not name or "/" in name or "\\" in name:
            raise ValueError("adapter names must be simple identifiers")
        if isinstance(bundle, ModelBundle):
            validated = bundle
        else:
            path = Path(bundle).expanduser().resolve()
            if self.allowed_bundle_root is not None and not path.is_relative_to(self.allowed_bundle_root):
                raise BundleError("adapter bundle path is outside the approved bundle root")
            validated = load_bundle(path)
        if validated.adapter_names and name not in validated.adapter_names:
            raise BundleError("registered adapter is not present in the bundle")
        self._registrations[name] = AdapterRegistration(name, validated, revision or validated.manifest.model_revision)
        self._bundles[name] = validated

    def load(self, name: str) -> Any:
        with self._lock:
            if name not in self._registrations:
                raise BundleError("requested adapter is not registered")
            if name in self._loaded:
                value = self._loaded.pop(name)
                self._loaded[name] = value
                self._activate(name, value)
                return value
            bundle = self._bundles[name]
            if self.adapter_loader is not None:
                value = self.adapter_loader(self.base_model, bundle, name)
            elif self.base_model_loader is not None:
                value = self.base_model_loader(bundle.load_adapter(name))
            elif self.base_model is not None:
                value = self.base_model
            else:
                raise BundleError("no reviewed adapter loader is configured")
            self._loaded[name] = value
            self._activate(name, value)
            while len(self._loaded) > self.max_loaded:
                evicted, old_value = self._loaded.popitem(last=False)
                if evicted == self._active:
                    self._active = None
                if self.adapter_unloader is not None:
                    self.adapter_unloader(old_value, evicted)
                _reset_adapter_state(old_value)
            return value

    def load_adapter(self, name: str) -> Any:
        return self.load(name)

    def switch(self, name: str) -> Any:
        return self.load(name)

    def unload(self, name: str) -> bool:
        with self._lock:
            value = self._loaded.pop(name, None)
            if value is None:
                return False
            if self.adapter_unloader is not None:
                self.adapter_unloader(value, name)
            _reset_adapter_state(value)
            if self._active == name:
                self._active = None
            return True

    def _activate(self, name: str, value: Any) -> None:
        if self._active == name:
            return
        if self._active is not None and self._active in self._loaded:
            _reset_adapter_state(self._loaded[self._active])
        setter = getattr(value, "set_active_adapter", None)
        if callable(setter):
            setter(name)
        else:
            setter = getattr(value, "set_adapter", None)
            if callable(setter):
                setter(name)
        self._active = name


class InferenceService:
    """Common request/result service for CPU, CUDA, and TPU worker pools."""

    def __init__(
        self,
        pipelines: Mapping[str, InferencePipeline],
        *,
        adapter_manager: AdapterManager | None = None,
        audit_logger: AuditLogger | None = None,
        limits: InferenceLimits | None = None,
        runtime: InferenceRuntime | None = None,
        max_concurrency: int = 1,
        queue_capacity: int = 16,
        timeout_seconds: float | None = None,
    ) -> None:
        if max_concurrency <= 0 or queue_capacity < 0:
            raise ValueError("max_concurrency must be positive and queue_capacity non-negative")
        self.pipelines = {str(key).lower(): value for key, value in pipelines.items()}
        self.adapter_manager = adapter_manager
        self.audit_logger = audit_logger or AuditLogger()
        self.limits = limits or InferenceLimits()
        self.runtime = runtime or (next(iter(self.pipelines.values())).runtime if self.pipelines else None)
        self.timeout_seconds = float(timeout_seconds or self.limits.timeout_seconds)
        self._semaphore = threading.BoundedSemaphore(int(max_concurrency) + int(queue_capacity))
        self._executor = ThreadPoolExecutor(max_workers=int(max_concurrency), thread_name_prefix="medfm-infer")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> InferenceService:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def handle(self, request: InferenceRequest | Mapping[str, Any]) -> InferenceResponse:
        started = time.perf_counter()
        parsed: InferenceRequest | None = None
        try:
            # This is intentionally before adapter lookup/model allocation.
            parsed = validate_request(request, self.limits)
            pipeline = self.pipelines.get(parsed.task.value)
            if pipeline is None:
                raise InferenceError(details={"task": parsed.task.value})
            if not self._semaphore.acquire(blocking=False):
                raise ServiceBusyError()
            try:
                future = self._executor.submit(self._execute, pipeline, parsed)
                try:
                    result = future.result(timeout=self.timeout_seconds)
                except TimeoutError as exc:
                    future.cancel()
                    raise InferenceTimeoutError() from exc
            finally:
                self._semaphore.release()
            audit = self._audit(parsed, result, None, started)
            return InferenceResponse(ok=True, request_id=parsed.request_id, result=result, audit=audit)
        except Exception as exc:
            safe_error = _safe_error(exc)
            audit = self._audit(parsed, None, safe_error.get("code"), started) if parsed is not None else None
            return InferenceResponse(
                ok=False,
                request_id=parsed.request_id if parsed else None,
                error=safe_error,
                audit=audit,
            )

    def predict(self, request: InferenceRequest | Mapping[str, Any]) -> InferenceResponse:
        return self.handle(request)

    def _execute(self, pipeline: InferencePipeline, request: InferenceRequest) -> InferenceResult:
        parsed = pipeline.preflight(request)
        if parsed.adapter is not None and self.adapter_manager is not None:
            self.adapter_manager.load(parsed.adapter)
        return pipeline.run(parsed)

    def _audit(
        self,
        request: InferenceRequest | None,
        result: InferenceResult | None,
        error_status: str | None,
        started: float,
    ) -> dict[str, Any] | None:
        if request is None:
            return None
        pipeline = self.pipelines.get(request.task.value)
        if pipeline is None:
            return None
        adapter_id = request.adapter
        adapter_revision = None
        if self.adapter_manager is not None and adapter_id in self.adapter_manager._registrations:
            adapter_revision = self.adapter_manager._registrations[adapter_id].revision
        event = self.audit_logger.create_event(
            model_id=pipeline.model_id,
            model_revision=pipeline.model_revision,
            adapter_id=adapter_id,
            adapter_revision=adapter_revision,
            preprocess_hash=pipeline.preprocess_hash,
            prompt_version=request.prompt_version,
            input_value=request.payload,
            output_value=result.data if result else None,
            runtime=pipeline.runtime.name,
            peak_vram_bytes=memory_peak_bytes(pipeline.runtime),
            error_status=error_status,
            request_id=request.request_id,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"task": request.task.value, "modality": request.modality},
        )
        return event.to_dict()


def _safe_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, InferenceError):
        return exc.to_error()
    return {"code": "INFERENCE_ERROR", "message": "inference request failed", "retryable": False, "details": {}}


def _reset_adapter_state(model: Any) -> None:
    for name in ("reset_adapter_state", "clear_sharding_state", "clear_cache"):
        callback = getattr(model, name, None)
        if callable(callback):
            callback()


__all__ = ["AdapterManager", "AdapterRegistration", "InferenceService"]
