"""Canonical CPU safetensor adapter exports and merge/compatibility checks."""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medfm.core.serialization import config_hash, materialize_cpu
from medfm.peft.config import LoRAConfig
from medfm.peft.errors import AdapterCheckpointError, CheckpointCompatibilityError, QuantizedParameterError
from medfm.peft.lora import (
    inject_lora,
    is_quantized_parameter,
    merge_lora_adapters,
    unmerge_lora_adapters,
)

CHECKPOINT_SCHEMA_VERSION = 1

# Safetensors dtype names for the scalar types used by model/adapters.
_DTYPE_TO_SAFE = {
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
_SAFE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_SAFE.items()}


def _write_safetensors(
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write the small, dependency-free subset of the safetensors format.

    ``safetensors`` is a core project dependency in normal installations, but
    the fallback keeps CPU contract tests runnable in a minimal environment.
    The emitted files are standard safetensors and can be read by the upstream
    package when it is installed.
    """
    ordered = sorted(tensors.items())
    raw_chunks: list[bytes] = []
    header: dict[str, Any] = {}
    offset = 0
    for name, tensor in ordered:
        if not isinstance(tensor, torch.Tensor):
            raise AdapterCheckpointError(f"checkpoint tensor {name!r} is not a torch.Tensor")
        cpu = materialize_cpu(tensor).contiguous()
        safe_dtype = _DTYPE_TO_SAFE.get(cpu.dtype)
        if safe_dtype is None:
            raise AdapterCheckpointError(f"unsupported safetensors dtype {cpu.dtype} for {name!r}")
        raw = cpu.view(torch.uint8).numpy().tobytes()
        end = offset + len(raw)
        header[name] = {
            "dtype": safe_dtype,
            "shape": list(cpu.shape),
            "data_offsets": [offset, end],
        }
        raw_chunks.append(raw)
        offset = end
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-((8 + len(encoded)) % 8)) % 8
    encoded += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(raw_chunks))


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = path.read_bytes()
        if len(payload) < 8:
            raise ValueError("file is shorter than a safetensors header")
        header_length = struct.unpack("<Q", payload[:8])[0]
        header_start = 8
        header_end = header_start + header_length
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("header must be an object")
        output: dict[str, torch.Tensor] = {}
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(record, dict):
                raise ValueError(f"tensor record {name!r} is not an object")
            dtype = _SAFE_TO_DTYPE[record["dtype"]]
            start, end = (int(value) for value in record["data_offsets"])
            view_start = header_end + start
            view_end = header_end + end
            if view_start < header_end or view_end > len(payload) or view_start > view_end:
                raise ValueError(f"tensor {name!r} has invalid data offsets")
            raw = memoryview(bytearray(payload[view_start:view_end]))
            tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype)
            output[name] = tensor.reshape(tuple(int(value) for value in record["shape"]))
        return output
    except Exception as exc:
        raise AdapterCheckpointError(f"could not read safetensors file {path}: {exc}") from exc


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adapter_names(model: nn.Module) -> tuple[str, ...]:
    metadata = getattr(model, "_medfm_peft_adapters", None)
    if isinstance(metadata, dict) and metadata:
        return tuple(sorted(str(name) for name in metadata))
    names: set[str] = set()
    for name, _ in model.named_parameters():
        match = re.search(r"lora_[AB]\.([^.]+)\.", name)
        if match:
            names.add(match.group(1))
        match = re.search(r"dora_magnitude_([^\.]+)$", name)
        if match:
            names.add(match.group(1))
    return tuple(sorted(names))


def _adapter_parameter(name: str, adapter_name: str) -> bool:
    return (
        f"lora_A.{adapter_name}." in name
        or f"lora_B.{adapter_name}." in name
        or name.endswith(f"dora_magnitude_{adapter_name}")
    )


def _role_for_name(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("bridge", "projector", "boundary")):
        return "bridge"
    if any(token in lowered for token in ("decoder", "segmentation_decoder")):
        return "decoder"
    if any(token in lowered for token in ("head", "classifier", "lm_head")):
        return "head"
    return None


def _collect_groups(
    model: nn.Module,
    *,
    adapter_names: Iterable[str],
    component_prefixes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    names = tuple(adapter_names)
    groups: dict[str, dict[str, torch.Tensor]] = {f"adapter:{name}": {} for name in names}
    groups.update({"bridge": {}, "head": {}, "decoder": {}})
    prefix_items = tuple((str(prefix), str(role)) for role, prefix in (component_prefixes or {}).items())
    for name, parameter in model.named_parameters():
        adapter_group = next((adapter for adapter in names if _adapter_parameter(name, adapter)), None)
        if adapter_group is not None:
            if is_quantized_parameter(parameter):
                raise QuantizedParameterError(f"quantized parameter {name!r} cannot be exported as trainable")
            groups[f"adapter:{adapter_group}"][name] = materialize_cpu(parameter)
            continue
        if not parameter.requires_grad:
            continue
        if is_quantized_parameter(parameter):
            raise QuantizedParameterError(f"quantized parameter {name!r} cannot be exported as trainable")
        role = None
        for prefix, mapped_role in prefix_items:
            if name == prefix or name.startswith(prefix + "."):
                role = mapped_role
                break
        role = role or _role_for_name(name)
        if role in {"bridge", "head", "decoder"}:
            groups[role][name] = materialize_cpu(parameter)
        else:
            raise AdapterCheckpointError(
                f"trainable parameter {name!r} is not an adapter/bridge/head/decoder; "
                "canonical export refuses full base weights"
            )
    return {group: tensors for group, tensors in groups.items() if tensors}


def _manifest_config(model: nn.Module, peft_config: LoRAConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if peft_config is not None:
        return peft_config.to_dict() if isinstance(peft_config, LoRAConfig) else dict(peft_config)
    for module in model.modules():
        metadata = getattr(module, "_medfm_peft_adapters", None)
        if isinstance(metadata, dict) and metadata:
            return {"adapters": {str(name): dict(value) for name, value in metadata.items()}}
    return {}


def save_adapter_checkpoint(
    directory: str | Path,
    model: nn.Module,
    *,
    base_model_id: str,
    base_revision: str,
    architecture: str,
    peft_config: LoRAConfig | Mapping[str, Any] | None = None,
    component_prefixes: Mapping[str, str] | None = None,
    adapter_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save only trainable adapter/component tensors as CPU safetensors."""
    if not base_model_id or not base_revision or not architecture:
        raise AdapterCheckpointError("base_model_id, base_revision, and architecture are required for export")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    directory = Path(directory)
    all_names = _adapter_names(model)
    if adapter_name is not None:
        if adapter_name not in all_names:
            raise AdapterCheckpointError(f"adapter {adapter_name!r} is not present; available={list(all_names)}")
        all_names = (adapter_name,)
    groups = _collect_groups(model, adapter_names=all_names, component_prefixes=component_prefixes)
    if not groups:
        raise AdapterCheckpointError("nothing trained to export: no adapter, bridge, head, or decoder parameters")
    config = _manifest_config(model, peft_config)
    if isinstance(peft_config, LoRAConfig):
        adapter_configs = {peft_config.adapter_name: peft_config.to_dict()}
    elif isinstance(config.get("adapters"), dict):
        adapter_configs = {str(name): dict(value) for name, value in config["adapters"].items()}
    elif all_names and isinstance(config, dict) and "method" in config:
        adapter_configs = {all_names[0]: config}
    else:
        adapter_configs = {}
    tensor_files: dict[str, str] = {}
    aggregate: dict[str, torch.Tensor] = {}
    for group, tensors in groups.items():
        if group.startswith("adapter:"):
            name = group.split(":", 1)[1]
            relative = Path("adapters") / name / "model.safetensors"
        else:
            relative = Path(group) / "model.safetensors"
        _write_safetensors(tensors, directory / relative, metadata={"base_model_id": base_model_id})
        tensor_files[group] = relative.as_posix()
        aggregate.update(tensors)
    _write_safetensors(aggregate, directory / "adapter.safetensors", metadata={"base_model_id": base_model_id})
    manifest: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "adapter_only",
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "architecture": architecture,
        "adapter_names": list(all_names),
        "adapter_configs": adapter_configs,
        "configuration": config,
        "config_hash": config_hash(config),
        "tensor_files": tensor_files,
        "aggregate_tensor_file": "adapter.safetensors",
        "canonical_device": "cpu",
        "canonical_format": "safetensors",
        "created_at": datetime.now(UTC).isoformat(),
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    _write_json(directory / "manifest.json", manifest)
    return directory


def load_checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise AdapterCheckpointError(f"no manifest.json in adapter checkpoint {directory}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterCheckpointError(f"invalid adapter manifest {path}: {exc}") from exc
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise AdapterCheckpointError(
            f"unsupported checkpoint schema {manifest.get('checkpoint_schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if manifest.get("canonical_format") != "safetensors":
        raise AdapterCheckpointError("canonical adapter exports must use safetensors")
    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise AdapterCheckpointError("adapter manifest configuration must be an object")
    if config_hash(dict(configuration)) != manifest["config_hash"]:
        raise AdapterCheckpointError("adapter manifest config_hash does not match configuration")
    return manifest


def _validate_base(
    manifest: Mapping[str, Any],
    *,
    base_model_id: str,
    base_revision: str,
    architecture: str | None,
    configuration: Mapping[str, Any] | None,
) -> None:
    if manifest["base_model_id"] != base_model_id:
        raise CheckpointCompatibilityError(
            f"wrong base model: checkpoint={manifest['base_model_id']!r}, requested={base_model_id!r}"
        )
    if manifest["base_revision"] != base_revision:
        raise CheckpointCompatibilityError(
            f"wrong base revision: checkpoint={manifest['base_revision']!r}, requested={base_revision!r}"
        )
    if architecture is not None and manifest["architecture"] != architecture:
        raise CheckpointCompatibilityError(
            f"wrong base architecture: checkpoint={manifest['architecture']!r}, requested={architecture!r}"
        )
    if configuration is not None and config_hash(dict(configuration)) != manifest["config_hash"]:
        raise CheckpointCompatibilityError("adapter configuration hash does not match checkpoint")


def _ensure_manifest_adapters(
    model: nn.Module,
    manifest: Mapping[str, Any],
    *,
    adapter_name: str | None = None,
) -> None:
    configs = manifest.get("adapter_configs") or {}
    if not isinstance(configs, Mapping):
        return
    existing = set(_adapter_names(model))
    for name, raw in configs.items():
        if adapter_name is not None and str(name) != adapter_name:
            continue
        if name in existing:
            continue
        try:
            config = LoRAConfig.from_dict(raw)
        except Exception as exc:
            raise AdapterCheckpointError(f"invalid adapter config for {name!r}: {exc}") from exc
        if config.adapter_name != name:
            config = LoRAConfig.from_dict({**config.to_dict(), "adapter_name": name})
        inject_lora(
            model,
            config,
            architecture=str(manifest["architecture"]),
            confirm_unknown=True,
        )


def load_adapter_checkpoint(
    directory: str | Path,
    model: nn.Module,
    *,
    base_model_id: str,
    base_revision: str,
    architecture: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    adapter_name: str | None = None,
) -> nn.Module:
    """Validate provenance, inject missing named adapters, and load CPU tensors."""
    manifest = load_checkpoint_manifest(directory)
    _validate_base(
        manifest,
        base_model_id=base_model_id,
        base_revision=base_revision,
        architecture=architecture,
        configuration=configuration,
    )
    _ensure_manifest_adapters(model, manifest, adapter_name=adapter_name)
    files = manifest["tensor_files"]
    selected_files = {
        group: relative
        for group, relative in files.items()
        if adapter_name is None or group in {f"adapter:{adapter_name}", "bridge", "head", "decoder"}
    }
    if adapter_name is not None and f"adapter:{adapter_name}" not in selected_files:
        raise AdapterCheckpointError(f"adapter {adapter_name!r} is not present in checkpoint")
    loaded: dict[str, torch.Tensor] = {}
    for relative in selected_files.values():
        loaded.update(_read_safetensors(Path(directory) / relative))
    current = model.state_dict()
    missing = sorted(name for name in loaded if name not in current)
    if missing:
        raise CheckpointCompatibilityError(
            "checkpoint tensors do not match the base architecture; missing keys: " + ", ".join(missing[:8])
        )
    incompatible = model.load_state_dict(loaded, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected:
        raise CheckpointCompatibilityError("unexpected adapter keys: " + ", ".join(unexpected[:8]))
    if adapter_name is not None:
        from medfm.peft.lora import set_active_adapter

        set_active_adapter(model, adapter_name)
    return model


def merge_for_inference(model: nn.Module, adapter_name: str | None = None) -> nn.Module:
    return merge_lora_adapters(model, adapter_name)


def compare_merged_unmerged(
    model: nn.Module,
    forward: Callable[[], Any],
    *,
    adapter_name: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Compare a callable's output before/after merge and restore unmerged state."""
    model.eval()
    with torch.no_grad():
        unmerged = forward()
        merge_lora_adapters(model, adapter_name)
        try:
            merged = forward()
        finally:
            unmerge_lora_adapters(model, adapter_name)
    return _nested_allclose(unmerged, merged, atol=atol, rtol=rtol)


def _nested_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, tuple | list):
        return len(left) == len(right) and all(
            _nested_allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _nested_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    try:
        return bool(left == right)
    except Exception:
        return False


# Handoff-friendly aliases.
export_adapter_checkpoint = save_adapter_checkpoint
import_adapter_checkpoint = load_adapter_checkpoint
merge_adapter = merge_for_inference


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "compare_merged_unmerged",
    "export_adapter_checkpoint",
    "import_adapter_checkpoint",
    "load_adapter_checkpoint",
    "load_checkpoint_manifest",
    "merge_adapter",
    "merge_for_inference",
    "save_adapter_checkpoint",
]
