"""Operational and access-controlled audit records for inference.

Operational records contain hashes and bounded metadata only.  Raw images,
reports, DICOM UIDs, and identifiers are never written by :class:`AuditLogger`.
A separate :class:`ClinicalAuditStore` is intentionally explicit and requires a
role for reads and deletion.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch

from medfm.inference.schemas import payload_hash

AUDIT_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _safe_scalar(value: Any, *, default: str = "unknown") -> str:
    if value is None:
        return default
    raw = str(value)
    if len(raw) > 128 or any(ord(char) < 32 for char in raw):
        return default
    return raw


@dataclass(frozen=True)
class AuditEvent:
    """Required fields for one operational inference event."""

    event_id: str
    timestamp: str
    model_id: str
    model_revision: str
    adapter_id: str | None
    adapter_revision: str | None
    preprocess_hash: str
    prompt_version: str | None
    input_hash: str
    output_schema_version: int
    runtime: str
    peak_vram_bytes: int | None
    error_status: str | None
    output_hash: str | None = None
    duration_ms: float | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "preprocess_hash": self.preprocess_hash,
            "prompt_version": self.prompt_version,
            "input_hash": self.input_hash,
            "output_schema_version": self.output_schema_version,
            "runtime": self.runtime,
            "peak_vram_bytes": self.peak_vram_bytes,
            "error_status": self.error_status,
            "output_hash": self.output_hash,
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
            "metadata": dict(self.metadata),
        }


class AuditLogger:
    """Append-only operational logger with bounded, redacted fields."""

    def __init__(self, path: str | Path | None = None, *, retain_days: int | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.retain_days = retain_days
        if retain_days is not None and retain_days < 0:
            raise ValueError("retain_days must be non-negative")
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def create_event(
        self,
        *,
        model_id: str,
        model_revision: str,
        adapter_id: str | None,
        adapter_revision: str | None,
        preprocess_hash: str,
        prompt_version: str | None,
        input_value: Any,
        output_value: Any | None,
        runtime: str,
        peak_vram_bytes: int | None,
        error_status: str | None,
        request_id: str | None = None,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            timestamp=_timestamp(),
            model_id=_safe_scalar(model_id),
            model_revision=_safe_scalar(model_revision),
            adapter_id=_safe_scalar(adapter_id, default="") if adapter_id is not None else None,
            adapter_revision=_safe_scalar(adapter_revision, default="") if adapter_revision is not None else None,
            preprocess_hash=_safe_scalar(preprocess_hash),
            prompt_version=_safe_scalar(prompt_version, default="") if prompt_version is not None else None,
            input_hash=payload_hash(input_value),
            output_schema_version=1,
            runtime=_safe_scalar(runtime),
            peak_vram_bytes=int(peak_vram_bytes) if peak_vram_bytes is not None else None,
            error_status=_safe_scalar(error_status, default="") if error_status else None,
            output_hash=payload_hash(output_value) if output_value is not None else None,
            duration_ms=float(duration_ms) if duration_ms is not None else None,
            request_id=_safe_scalar(request_id, default="") if request_id is not None else None,
            metadata=_redact_metadata(metadata or {}),
        )
        self.log(event)
        return event

    def log(self, event: AuditEvent) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        if self.retain_days is not None:
            self.purge_older_than(self.retain_days)

    def read(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
        return records

    def purge_older_than(self, days: int) -> int:
        if self.path is None or not self.path.exists():
            return 0
        cutoff = datetime.now(UTC) - timedelta(days=int(days))
        keep: list[str] = []
        removed = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                timestamp = datetime.fromisoformat(str(record["timestamp"]))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                if timestamp < cutoff:
                    removed += 1
                    continue
            except Exception:
                # Keep malformed operational records; deletion is never an
                # excuse to silently lose audit evidence.
                pass
            keep.append(line)
        self.path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        return removed


class ClinicalAuditStore:
    """Separate protected store for explicitly authorized clinical audit data."""

    def __init__(self, root: str | Path, *, allowed_roles: tuple[str, ...] = ("clinical_auditor",)) -> None:
        self.root = Path(root)
        self.allowed_roles = frozenset(allowed_roles)
        self.root.mkdir(parents=True, exist_ok=True)

    def _check_role(self, role: str) -> None:
        if role not in self.allowed_roles:
            raise PermissionError("clinical audit access is restricted")

    def put(self, event_id: str, payload: Mapping[str, Any], *, role: str) -> Path:
        self._check_role(role)
        if not event_id or "/" in event_id or "\\" in event_id or ".." in event_id:
            raise ValueError("invalid audit event id")
        # Encrypting/escrowing the payload is deployment-specific; this store
        # remains separate and access controlled rather than pretending JSONL is
        # encryption.  Values are never mirrored into operational logs.
        path = self.root / f"{event_id}.json"
        path.write_text(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def get(self, event_id: str, *, role: str) -> dict[str, Any]:
        self._check_role(role)
        path = self.root / f"{event_id}.json"
        if not path.exists():
            raise KeyError(event_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("clinical audit payload is malformed")
        return value

    def delete(self, event_id: str, *, role: str) -> bool:
        self._check_role(role)
        path = self.root / f"{event_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def purge(self, *, before: datetime, role: str) -> int:
        self._check_role(role)
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                timestamp = datetime.fromisoformat(str(value.get("timestamp")))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                if timestamp < before:
                    path.unlink()
                    removed += 1
            except (OSError, ValueError, TypeError):
                continue
        return removed


def memory_peak_bytes(runtime: Any | None = None) -> int | None:
    """Read backend-owned memory snapshots without importing CUDA/XLA modules."""

    if runtime is not None:
        try:
            snapshot = runtime.memory_snapshot()
            value = getattr(snapshot, "peak_allocated_bytes", None)
            if value is not None:
                return int(value)
            if isinstance(snapshot, Mapping) and snapshot.get("peak_allocated_bytes") is not None:
                return int(snapshot["peak_allocated_bytes"])
        except Exception:
            return None
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None
    return None


def _redact_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    blocked = {"image", "images", "pixels", "pixel_values", "report", "text", "prompt", "uid", "mrn", "identifier"}
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if any(token in normalized for token in blocked):
            result[str(key)] = "<redacted>"
        elif isinstance(item, Mapping):
            result[str(key)] = _redact_metadata(item)
        elif isinstance(item, str | int | float | bool) or item is None:
            result[str(key)] = item
        else:
            result[str(key)] = f"<{type(item).__name__}>"
    return result


__all__ = ["AUDIT_SCHEMA_VERSION", "AuditEvent", "AuditLogger", "ClinicalAuditStore", "memory_peak_bytes"]
