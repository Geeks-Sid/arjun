"""Deterministic canonical serialization for contract objects.

Rules enforced here:

- Non-tensor schema fields serialize to deterministic JSON/YAML (sorted keys,
  fixed separators) so hashes are stable across runs and machines.
- Tensor *metadata* (shape + accelerator-neutral dtype name) serializes
  separately from tensor *payloads*. Device locations are never serialized:
  canonical artifacts are accelerator-neutral and must be materialized on CPU
  before export (:func:`materialize_cpu`).
- Small metadata tensors (affine matrices, tile coordinates, spacing vectors)
  may serialize inline as nested lists via :func:`tensor_to_data`; large
  payloads must go through a tensor store, never through JSON.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
import yaml

from medfm.core.errors import SerializationError

#: Maximum number of elements a tensor may have to be serialized inline as
#: nested lists (metadata such as affines, coordinates, spacing). Larger
#: tensors are payloads and must go through a tensor store (e.g. safetensors).
MAX_INLINE_TENSOR_ELEMENTS = 4096

#: Accelerator-neutral canonical dtype names <-> torch dtypes. Device-specific
#: dtypes (e.g. quantized packed layouts) have no canonical name on purpose.
_CANONICAL_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_DTYPE_TO_CANONICAL: dict[torch.dtype, str] = {v: k for k, v in _CANONICAL_DTYPES.items()}


def canonical_dtype_name(dtype: torch.dtype) -> str:
    """Return the accelerator-neutral canonical name for a torch dtype."""
    try:
        return _DTYPE_TO_CANONICAL[dtype]
    except KeyError:
        raise SerializationError(f"no accelerator-neutral canonical name for torch dtype {dtype}") from None


def dtype_from_canonical(name: str) -> torch.dtype:
    """Resolve a canonical dtype name to a torch dtype."""
    try:
        return _CANONICAL_DTYPES[name]
    except KeyError:
        raise SerializationError(
            f"unknown canonical dtype name {name!r}; legal values: {sorted(_CANONICAL_DTYPES)}"
        ) from None


@dataclass(frozen=True)
class TensorMeta:
    """Serializable description of a tensor payload: shape + dtype only.

    Never carries a device or storage location; payloads are restored on CPU
    and moved by the caller's backend.
    """

    shape: tuple[int, ...]
    dtype: str  # canonical dtype name

    @classmethod
    def of(cls, tensor: torch.Tensor) -> TensorMeta:
        return cls(shape=tuple(tensor.shape), dtype=canonical_dtype_name(tensor.dtype))

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMeta:
        # Validates the dtype name and stores it back in canonical form.
        dtype_name = canonical_dtype_name(dtype_from_canonical(str(data["dtype"])))
        return cls(shape=tuple(int(d) for d in data["shape"]), dtype=dtype_name)


def tensor_to_data(tensor: torch.Tensor) -> Any:
    """Serialize a small metadata tensor to nested Python lists (lossless).

    Integers round-trip exactly; floats round-trip through Python floats
    (float32/float64 values are represented exactly). Refuses large tensors:
    those are payloads, not metadata.
    """
    if tensor.numel() > MAX_INLINE_TENSOR_ELEMENTS:
        raise SerializationError(
            f"tensor with {tensor.numel()} elements exceeds the inline metadata limit "
            f"({MAX_INLINE_TENSOR_ELEMENTS}); store it as a payload, not in JSON/YAML"
        )
    return tensor.detach().cpu().tolist()


def tensor_from_data(data: Any, dtype: torch.dtype) -> torch.Tensor:
    """Rebuild a CPU tensor from :func:`tensor_to_data` output."""
    return torch.tensor(data, dtype=dtype, device="cpu")


def materialize_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor and move it to CPU for canonical export.

    Works for CPU, CUDA, and XLA tensors (XLA tensors transfer on ``.cpu()``).
    """
    return tensor.detach().cpu()


def canonical_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, fixed separators, no float noise."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_yaml(data: Any) -> str:
    """Deterministic YAML: sorted keys, block style."""
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=False)


def config_hash(data: Any) -> str:
    """SHA-256 over the canonical JSON of a configuration-like mapping."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
