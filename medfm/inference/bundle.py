"""Portable, checksum-verified deployment bundles.

The deployment bundle is deliberately independent from a training run
 directory.  Adapter/bridge/head tensors are canonical CPU safetensors; a
merged artifact, when explicitly requested, is recorded as a secondary
convenience and is never used as the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml

from medfm.core.serialization import config_hash
from medfm.inference.errors import (
    BundleChecksumError,
    BundleCompatibilityError,
    BundleValidationError,
)

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_LAYOUT = (
    "bundle.json",
    "model_card.md",
    "license_summary.md",
    "base_models.json",
    "preprocessing.yaml",
    "postprocessing.yaml",
    "task_schema.json",
    "inference_config.yaml",
    "adapters",
    "bridge",
    "heads",
    "calibration",
    "examples",
    "checksums.json",
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Deployment bundles must not silently become resumable training checkpoints.
_FORBIDDEN_RESUME_PARTS = {
    "optimizer",
    "scheduler",
    "scaler",
    "rng_state",
    "distributed",
    "shards",
    "shard",
    "checkpoints",
}
_FORBIDDEN_RESUME_SUFFIXES = (".distcp", ".metadata", ".pkl", ".pickle")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_relative(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise BundleValidationError(f"bundle path is not relative and safe: {raw!r}")
    normalized = candidate.as_posix()
    if normalized in {".", ""} or normalized.startswith("./"):
        raise BundleValidationError(f"bundle path is not safe: {raw!r}")
    return normalized


def _reject_resume_artifact(relative: str) -> None:
    lower = relative.lower()
    parts = set(PurePosixPath(lower).parts)
    if parts & _FORBIDDEN_RESUME_PARTS or lower.endswith(_FORBIDDEN_RESUME_SUFFIXES):
        raise BundleValidationError(
            "resumable/sharded training checkpoint artifacts are not deployment inputs; "
            "convert explicitly to canonical adapter safetensors first"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"tensor group value {name!r} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise BundleValidationError(f"tensor {name!r} must use a dense strided layout")
    if value.is_quantized:
        raise BundleValidationError(f"tensor {name!r} is quantized; export dequantized CPU tensors")
    # Detach first so export never retains a training graph.  CPU is the
    # canonical device and contiguous storage makes cross-backend loads stable.
    return value.detach().to(device="cpu").contiguous()


_SAFE_DTYPES = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_DTYPES_FROM_SAFE = {value: key for key, value in _SAFE_DTYPES.items()}


def _write_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    if not tensors:
        raise BundleValidationError(f"cannot write empty tensor artifact: {path}")
    canonical = {_safe_tensor_name(name): _canonical_tensor(value, str(name)) for name, value in tensors.items()}
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in sorted(canonical.items()):
        safe_dtype = _SAFE_DTYPES.get(tensor.dtype)
        if safe_dtype is None:
            raise BundleValidationError(f"unsupported safetensors dtype {tensor.dtype} for {name!r}")
        raw = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": safe_dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"medfm.canonical_device": "cpu", "medfm.format": "safetensors"}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-((8 + len(encoded)) % 8)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = path.read_bytes()
        header_length = struct.unpack("<Q", payload[:8])[0]
        header_end = 8 + header_length
        header = json.loads(payload[8:header_end].decode("utf-8"))
        output: dict[str, torch.Tensor] = {}
        for name, record in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES_FROM_SAFE[record["dtype"]]
            start, end = (int(value) for value in record["data_offsets"])
            raw = memoryview(bytearray(payload[header_end + start : header_end + end]))
            tensor = (
                torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(tuple(int(v) for v in record["shape"]))
            )
            output[str(name)] = tensor.contiguous()
        return output
    except Exception as exc:
        raise BundleValidationError(f"could not load tensor artifact {path.name}") from exc


def _safe_tensor_name(name: str) -> str:
    value = str(name)
    if not value or value.startswith("__") or ".." in value.split("."):
        raise BundleValidationError(f"unsafe tensor name {value!r}")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleValidationError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise BundleValidationError(f"{label} must be a JSON object")
    return value


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return _read_safetensors(path)


@dataclass(frozen=True)
class BaseModelReference:
    """Pinned base model identity required to apply an adapter safely."""

    model_id: str
    revision: str
    architecture: str | None = None
    config_hash: str | None = None
    license_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise BundleValidationError("base model references require model_id and revision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "config_hash": self.config_hash,
            "license_id": self.license_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BaseModelReference:
        return cls(
            model_id=str(value.get("model_id", "")),
            revision=str(value.get("revision", "")),
            architecture=(str(value["architecture"]) if value.get("architecture") is not None else None),
            config_hash=(str(value["config_hash"]) if value.get("config_hash") is not None else None),
            license_id=(str(value["license_id"]) if value.get("license_id") is not None else None),
        )


@dataclass(frozen=True)
class RuntimeSupport:
    """Certified runtime matrix and explicit prohibited combinations."""

    minimum_python: str = ">=3.11,<3.14"
    minimum_torch: str = ">=2.9"
    backends: Mapping[str, str] = field(
        default_factory=lambda: {"cpu": "untested", "cuda": "untested", "xla_tpu": "untested"}
    )
    hardware_notes: tuple[str, ...] = ()
    prohibited_combinations: tuple[str, ...] = ()
    tpu_buckets: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_python": self.minimum_python,
            "minimum_torch": self.minimum_torch,
            "backends": dict(self.backends),
            "hardware_notes": list(self.hardware_notes),
            "prohibited_combinations": list(self.prohibited_combinations),
            "tpu_buckets": {key: [list(shape) for shape in value] for key, value in self.tpu_buckets.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> RuntimeSupport:
        raw = dict(value or {})
        buckets: dict[str, tuple[tuple[int, ...], ...]] = {}
        for kind, shapes in dict(raw.get("tpu_buckets", {})).items():
            if not isinstance(shapes, Sequence) or isinstance(shapes, str | bytes):
                raise BundleValidationError("runtime tpu_buckets values must be shape lists")
            buckets[str(kind)] = tuple(tuple(int(v) for v in shape) for shape in shapes)
        return cls(
            minimum_python=str(raw.get("minimum_python", ">=3.11,<3.14")),
            minimum_torch=str(raw.get("minimum_torch", ">=2.9")),
            backends={str(k): str(v) for k, v in dict(raw.get("backends", {})).items()},
            hardware_notes=tuple(str(v) for v in raw.get("hardware_notes", ())),
            prohibited_combinations=tuple(str(v) for v in raw.get("prohibited_combinations", ())),
            tpu_buckets=buckets,
        )


@dataclass(frozen=True)
class BundleManifest:
    """Versioned deployment metadata serialized as ``bundle.json``."""

    bundle_id: str
    model_id: str
    model_revision: str
    task: str
    base_models: tuple[BaseModelReference, ...]
    schema_version: int = BUNDLE_SCHEMA_VERSION
    adapter_only: bool = True
    canonical_device: str = "cpu"
    canonical_format: str = "safetensors"
    modalities: tuple[str, ...] = ()
    runtime: RuntimeSupport = field(default_factory=RuntimeSupport)
    preprocess_hash: str = ""
    postprocess_hash: str = ""
    task_schema_hash: str = ""
    tensor_groups: tuple[str, ...] = ()
    merged_artifacts: tuple[str, ...] = ()
    created_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise BundleValidationError(f"unsupported bundle schema version {self.schema_version}")
        if not self.bundle_id or not self.model_id or not self.model_revision or not self.task:
            raise BundleValidationError("bundle_id, model_id, model_revision, and task are required")
        if not self.base_models:
            raise BundleValidationError("deployment bundle must pin at least one base model")
        if self.adapter_only is not True or self.canonical_device != "cpu" or self.canonical_format != "safetensors":
            raise BundleValidationError("canonical deployment artifacts must be adapter-only CPU safetensors")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "task": self.task,
            "modalities": list(self.modalities),
            "base_models": [base.to_dict() for base in self.base_models],
            "adapter_only": self.adapter_only,
            "canonical_device": self.canonical_device,
            "canonical_format": self.canonical_format,
            "runtime": self.runtime.to_dict(),
            "preprocess_hash": self.preprocess_hash,
            "postprocess_hash": self.postprocess_hash,
            "task_schema_hash": self.task_schema_hash,
            "tensor_groups": list(self.tensor_groups),
            "merged_artifacts": list(self.merged_artifacts),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleManifest:
        raw = dict(value)
        bases = raw.get("base_models")
        if not isinstance(bases, Sequence) or isinstance(bases, str | bytes):
            raise BundleValidationError("bundle.json base_models must be a list")
        return cls(
            bundle_id=str(raw.get("bundle_id", "")),
            model_id=str(raw.get("model_id", "")),
            model_revision=str(raw.get("model_revision", "")),
            task=str(raw.get("task", "")),
            base_models=tuple(BaseModelReference.from_dict(item) for item in bases),
            schema_version=int(raw.get("schema_version", -1)),
            adapter_only=bool(raw.get("adapter_only", False)),
            canonical_device=str(raw.get("canonical_device", "")),
            canonical_format=str(raw.get("canonical_format", "")),
            modalities=tuple(str(v) for v in raw.get("modalities", ())),
            runtime=RuntimeSupport.from_dict(raw.get("runtime")),
            preprocess_hash=str(raw.get("preprocess_hash", "")),
            postprocess_hash=str(raw.get("postprocess_hash", "")),
            task_schema_hash=str(raw.get("task_schema_hash", "")),
            tensor_groups=tuple(str(v) for v in raw.get("tensor_groups", ())),
            merged_artifacts=tuple(str(v) for v in raw.get("merged_artifacts", ())),
            created_at=str(raw.get("created_at", "")),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class ModelBundle:
    """A validated bundle handle; no source training directory is retained."""

    root: Path
    manifest: BundleManifest
    checksums: Mapping[str, str]

    @property
    def bundle_id(self) -> str:
        return self.manifest.bundle_id

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(
            PurePosixPath(group).parts[-1] for group in self.manifest.tensor_groups if group.startswith("adapters/")
        )

    def path(self, relative: str | Path) -> Path:
        safe = _safe_relative(relative)
        path = (self.root / safe).resolve()
        root = self.root.resolve()
        if not path.is_relative_to(root):
            raise BundleValidationError("bundle path escapes the bundle root")
        if path.is_symlink():
            raise BundleValidationError("symlinks are not allowed in deployment bundles")
        return path

    def load_tensor_group(self, group: str) -> dict[str, torch.Tensor]:
        safe_group = _safe_relative(group).removesuffix("/")
        if safe_group not in self.manifest.tensor_groups and safe_group not in self.manifest.merged_artifacts:
            raise BundleValidationError(f"tensor group {group!r} is not declared by this bundle")
        relative = safe_group if safe_group.endswith("/model.safetensors") else f"{safe_group}/model.safetensors"
        return _load_safetensors(self.path(relative))

    def metadata_json(self, name: str) -> dict[str, Any]:
        """Read a small JSON metadata artifact after bundle validation."""

        relative = _safe_relative(name)
        if relative.endswith(".json") is False:
            raise BundleValidationError("metadata_json accepts only JSON artifacts")
        return _read_json(self.path(relative), label=relative)

    def load_adapter(self, adapter_name: str) -> dict[str, torch.Tensor]:
        if adapter_name not in self.adapter_names:
            raise BundleCompatibilityError(f"requested adapter is not present: {adapter_name!r}")
        return self.load_tensor_group(f"adapters/{adapter_name}")

    def load_model(
        self,
        base_model_loader: Callable[[BaseModelReference], Any],
        *,
        adapter_name: str | None = None,
        adapter_applier: Callable[[Any, Mapping[str, torch.Tensor], str | None], Any] | None = None,
    ) -> Any:
        """Build a base model through an explicit loader and apply canonical tensors."""

        base = self.manifest.base_models[0]
        model = base_model_loader(base)
        groups: dict[str, torch.Tensor] = {}
        if adapter_name is None:
            for name in self.adapter_names:
                groups.update({f"adapters/{name}/{key}": value for key, value in self.load_adapter(name).items()})
        else:
            groups.update(
                {f"adapters/{adapter_name}/{key}": value for key, value in self.load_adapter(adapter_name).items()}
            )
        for group in ("bridge", "heads"):
            if group in self.manifest.tensor_groups:
                groups.update({f"{group}/{key}": value for key, value in self.load_tensor_group(group).items()})
        if adapter_applier is None:
            if groups:
                raise BundleCompatibilityError(
                    "bundle contains trained tensors but no reviewed adapter_applier was provided"
                )
            return model
        return adapter_applier(model, groups, adapter_name)


def _base_matches(base: BaseModelReference, expected: Mapping[str, Any]) -> list[str]:
    aliases = {"base_model_id": "model_id", "base_revision": "revision"}
    normalized = {aliases.get(str(key), str(key)): value for key, value in expected.items()}
    errors: list[str] = []
    for field_name in ("model_id", "revision", "architecture", "config_hash"):
        if field_name in normalized and normalized[field_name] is not None:
            actual = getattr(base, field_name)
            if actual != str(normalized[field_name]):
                errors.append(f"base {field_name} mismatch")
    return errors


class BundleLoader:
    """Reusable validator that never allocates a base model during validation."""

    def __init__(self, directory: str | Path, **validation: Any) -> None:
        self.directory = Path(directory)
        self.validation = dict(validation)

    def load(self) -> ModelBundle:
        return validate_bundle(self.directory, **self.validation)


DeploymentBundle = ModelBundle


def _validate_checksums(root: Path) -> dict[str, str]:
    checksum_path = root / "checksums.json"
    if not checksum_path.exists():
        raise BundleChecksumError("bundle is missing checksums.json")
    data = _read_json(checksum_path, label="checksums.json")
    raw = data.get("files")
    if not isinstance(raw, Mapping):
        raise BundleChecksumError("checksums.json files must be an object")
    checksums: dict[str, str] = {}
    for relative, expected in raw.items():
        safe = _safe_relative(str(relative))
        _reject_resume_artifact(safe)
        if not isinstance(expected, str) or not _HEX_SHA256.fullmatch(expected):
            raise BundleChecksumError(f"invalid checksum entry for {safe}")
        path = root / safe
        if not path.exists() or not path.is_file() or path.is_symlink():
            raise BundleChecksumError(f"checksum file is missing or unsafe: {safe}")
        actual = _sha256(path)
        if actual != expected:
            raise BundleChecksumError(f"checksum mismatch for {safe}")
        checksums[safe] = expected
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    if actual_files != set(checksums):
        missing = sorted(actual_files - set(checksums))
        extra = sorted(set(checksums) - actual_files)
        detail = ", ".join([f"missing={missing[:4]}" if missing else "", f"extra={extra[:4]}" if extra else ""]).strip(
            ", "
        )
        raise BundleChecksumError(f"checksums do not cover exactly the bundle files ({detail})")
    return checksums


def validate_bundle(
    directory: str | Path,
    *,
    expected_base: Mapping[str, Any] | None = None,
    base_model_id: str | None = None,
    base_revision: str | None = None,
    backend: str | None = None,
) -> ModelBundle:
    """Validate layout, schema, checksums, base compatibility, and backend policy."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise BundleValidationError("bundle path must be a real directory")
    bundle_path = root / "bundle.json"
    if not bundle_path.exists():
        raise BundleValidationError("bundle is missing bundle.json")
    manifest = BundleManifest.from_dict(_read_json(bundle_path, label="bundle.json"))
    required = {
        "bundle.json",
        "model_card.md",
        "license_summary.md",
        "base_models.json",
        "preprocessing.yaml",
        "postprocessing.yaml",
        "task_schema.json",
        "inference_config.yaml",
        "checksums.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise BundleValidationError(f"bundle is missing required files: {', '.join(missing)}")
    if not (root / "model_card.md").read_text(encoding="utf-8").strip():
        raise BundleValidationError("model_card.md must not be empty")
    if not (root / "license_summary.md").read_text(encoding="utf-8").strip():
        raise BundleValidationError("license_summary.md must not be empty")
    base_data = _read_json(root / "base_models.json", label="base_models.json")
    declared_bases = base_data.get("base_models", base_data.get("models"))
    if not isinstance(declared_bases, Sequence) or isinstance(declared_bases, str | bytes):
        raise BundleValidationError("base_models.json must declare a base_models list")
    if tuple(BaseModelReference.from_dict(item) for item in declared_bases) != manifest.base_models:
        raise BundleValidationError("base_models.json does not match bundle.json")
    try:
        preprocessing = yaml.safe_load((root / "preprocessing.yaml").read_text(encoding="utf-8")) or {}
        postprocessing = yaml.safe_load((root / "postprocessing.yaml").read_text(encoding="utf-8")) or {}
        inference_config = yaml.safe_load((root / "inference_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise BundleValidationError("invalid YAML metadata in bundle") from exc
    task_schema = _read_json(root / "task_schema.json", label="task_schema.json")
    if manifest.preprocess_hash and config_hash(preprocessing) != manifest.preprocess_hash:
        raise BundleValidationError("preprocessing hash does not match bundle.json")
    if manifest.postprocess_hash and config_hash(postprocessing) != manifest.postprocess_hash:
        raise BundleValidationError("postprocessing hash does not match bundle.json")
    if manifest.task_schema_hash and config_hash(task_schema) != manifest.task_schema_hash:
        raise BundleValidationError("task schema hash does not match bundle.json")
    del inference_config  # parsed above to fail closed on malformed YAML
    expected = dict(expected_base or {})
    if base_model_id is not None:
        expected["model_id"] = base_model_id
    if base_revision is not None:
        expected["revision"] = base_revision
    if expected:
        errors = _base_matches(manifest.base_models[0], expected)
        if errors:
            raise BundleCompatibilityError("; ".join(errors))
    if backend is not None:
        status = str(manifest.runtime.backends.get(backend, "unsupported")).lower()
        if status in {"unsupported", "blocked", "prohibited", "untested"}:
            raise BundleCompatibilityError(f"bundle backend status is not certified for {backend}")
        for prohibited in manifest.runtime.prohibited_combinations:
            if backend.lower() in prohibited.lower():
                raise BundleCompatibilityError("requested backend is prohibited by the bundle runtime policy")
    checksums = _validate_checksums(root)
    declared_groups = set(manifest.tensor_groups) | set(manifest.merged_artifacts)
    for group in declared_groups:
        safe_group = _safe_relative(group)
        if safe_group.endswith("/model.safetensors"):
            artifact = root / safe_group
        else:
            artifact = root / safe_group / "model.safetensors"
        if not artifact.exists() or not artifact.is_file():
            raise BundleValidationError(f"declared tensor group is missing: {group}")
        _load_safetensors(artifact)
    for group in manifest.merged_artifacts:
        if group != "merged" and not group.startswith("merged/"):
            raise BundleValidationError("merged artifacts must live under merged/")
    # Reject any symlink, unknown executable, or unsafe path before a loader can
    # inspect user-controlled metadata.
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BundleValidationError("symlinks are not allowed in deployment bundles")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _reject_resume_artifact(relative)
            if path.stat().st_mode & 0o111:
                raise BundleValidationError(f"executable files are not allowed in bundles: {relative}")
    return ModelBundle(root=root, manifest=manifest, checksums=checksums)


class BundleBuilder:
    """Build an independent bundle from metadata and trained tensor groups."""

    def __init__(
        self,
        directory: str | Path,
        *,
        bundle_id: str,
        model_id: str,
        model_revision: str,
        task: str,
        base_models: Sequence[BaseModelReference | Mapping[str, Any]],
        model_card: str,
        license_summary: str,
        preprocessing: Mapping[str, Any],
        postprocessing: Mapping[str, Any],
        task_schema: Mapping[str, Any],
        inference_config: Mapping[str, Any],
        modalities: Sequence[str] = (),
        runtime: RuntimeSupport | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.bundle_id = str(bundle_id)
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.task = str(task)
        self.base_models = tuple(
            value if isinstance(value, BaseModelReference) else BaseModelReference.from_dict(value)
            for value in base_models
        )
        self.model_card = str(model_card)
        self.license_summary = str(license_summary)
        self.preprocessing = dict(preprocessing)
        self.postprocessing = dict(postprocessing)
        self.task_schema = dict(task_schema)
        self.inference_config = dict(inference_config)
        self.modalities = tuple(str(value) for value in modalities)
        self.runtime = runtime if isinstance(runtime, RuntimeSupport) else RuntimeSupport.from_dict(runtime)
        self.metadata = dict(metadata or {})
        self._groups: dict[str, Mapping[str, torch.Tensor]] = {}
        self._examples: dict[str, bytes] = {}
        self._calibration: dict[str, Any] | None = None
        self._merged: Mapping[str, torch.Tensor] | None = None
        self._allow_merged = False

    def add_tensor_group(self, group: str, tensors: Mapping[str, torch.Tensor]) -> BundleBuilder:
        safe = _safe_relative(group).removesuffix("/")
        if safe.startswith("merged/"):
            raise BundleValidationError("use add_merged_artifact for secondary merged exports")
        if safe not in {"bridge", "heads"} and not safe.startswith("adapters/"):
            raise BundleValidationError("canonical tensor groups must be bridge, heads, or adapters/<name>")
        if safe.startswith("adapters/") and len(PurePosixPath(safe).parts) != 2:
            raise BundleValidationError("adapter groups must be adapters/<adapter_name>")
        self._groups[safe] = dict(tensors)
        return self

    def add_adapter(self, name: str, tensors: Mapping[str, torch.Tensor]) -> BundleBuilder:
        return self.add_tensor_group(f"adapters/{_safe_relative(name)}", tensors)

    def add_example(self, relative: str, content: str | bytes) -> BundleBuilder:
        safe = _safe_relative(Path("examples") / relative)
        if not safe.startswith("examples/"):
            raise BundleValidationError("examples must live under examples/")
        self._examples[safe] = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        return self

    def set_calibration(self, calibration: Mapping[str, Any]) -> BundleBuilder:
        self._calibration = dict(calibration)
        return self

    def add_merged_artifact(self, tensors: Mapping[str, torch.Tensor], *, explicit_conversion: str) -> BundleBuilder:
        if not explicit_conversion.strip():
            raise BundleValidationError("merged artifacts require a documented explicit conversion step")
        self._merged = dict(tensors)
        self._allow_merged = True
        self.metadata = {**self.metadata, "merged_conversion": explicit_conversion}
        return self

    def build(self, *, overwrite: bool = False) -> ModelBundle:
        if self.directory.exists():
            if not overwrite and any(self.directory.iterdir()):
                raise BundleValidationError(f"bundle directory is not empty: {self.directory}")
            if overwrite:
                shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "adapters").mkdir()
        (self.directory / "bridge").mkdir()
        (self.directory / "heads").mkdir()
        (self.directory / "calibration").mkdir()
        (self.directory / "examples").mkdir()
        (self.directory / "merged").mkdir()
        if not self.model_card.strip() or not self.license_summary.strip():
            raise BundleValidationError("model card and license summary must be non-empty")
        (self.directory / "model_card.md").write_text(self.model_card, encoding="utf-8")
        (self.directory / "license_summary.md").write_text(self.license_summary, encoding="utf-8")
        _json_dump(
            self.directory / "base_models.json",
            {"schema_version": 1, "base_models": [b.to_dict() for b in self.base_models]},
        )
        (self.directory / "preprocessing.yaml").write_text(
            yaml.safe_dump(self.preprocessing, sort_keys=True), encoding="utf-8"
        )
        (self.directory / "postprocessing.yaml").write_text(
            yaml.safe_dump(self.postprocessing, sort_keys=True), encoding="utf-8"
        )
        _json_dump(self.directory / "task_schema.json", self.task_schema)
        (self.directory / "inference_config.yaml").write_text(
            yaml.safe_dump(self.inference_config, sort_keys=True), encoding="utf-8"
        )
        groups: list[str] = []
        for group, tensors in sorted(self._groups.items()):
            _write_safetensors(self.directory / group / "model.safetensors", tensors)
            groups.append(group)
        if self._calibration is not None:
            _json_dump(self.directory / "calibration" / "calibration.json", self._calibration)
        for relative, content in self._examples.items():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        merged: list[str] = []
        if self._merged is not None:
            if not self._allow_merged:
                raise BundleValidationError("merged artifact conversion was not explicitly documented")
            _write_safetensors(self.directory / "merged" / "model.safetensors", self._merged)
            merged.append("merged")
        manifest = BundleManifest(
            bundle_id=self.bundle_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            task=self.task,
            base_models=self.base_models,
            modalities=self.modalities,
            runtime=self.runtime,
            preprocess_hash=config_hash(self.preprocessing),
            postprocess_hash=config_hash(self.postprocessing),
            task_schema_hash=config_hash(self.task_schema),
            tensor_groups=tuple(groups),
            merged_artifacts=tuple(merged),
            created_at=datetime.now(UTC).isoformat(),
            metadata=self.metadata,
        )
        _json_dump(self.directory / "bundle.json", manifest.to_dict())
        files = {
            path.relative_to(self.directory).as_posix(): _sha256(path)
            for path in self.directory.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        }
        _json_dump(self.directory / "checksums.json", {"schema_version": 1, "algorithm": "sha256", "files": files})
        return validate_bundle(self.directory)


def load_bundle(
    directory: str | Path,
    *,
    expected_base: Mapping[str, Any] | None = None,
    base_model_id: str | None = None,
    base_revision: str | None = None,
    backend: str | None = None,
) -> ModelBundle:
    """Alias emphasizing that validation happens before any model allocation."""

    return validate_bundle(
        directory,
        expected_base=expected_base,
        base_model_id=base_model_id,
        base_revision=base_revision,
        backend=backend,
    )


__all__ = [
    "BUNDLE_LAYOUT",
    "BUNDLE_SCHEMA_VERSION",
    "BaseModelReference",
    "BundleBuilder",
    "BundleLoader",
    "BundleManifest",
    "DeploymentBundle",
    "ModelBundle",
    "RuntimeSupport",
    "load_bundle",
    "validate_bundle",
]
