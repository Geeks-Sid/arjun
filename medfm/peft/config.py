"""Versioned LoRA/QLoRA configuration and pre-allocation policy checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from medfm.core.serialization import canonical_dtype_name, config_hash
from medfm.peft.errors import (
    BackendCapabilityError,
    PeftConfigError,
    QuantizationCapabilityError,
    UnsupportedQuantizationError,
)

PEFT_SCHEMA_VERSION = 1


class PeftMethod(StrEnum):
    LORA = "lora"


class BiasMode(StrEnum):
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class TargetPolicy(StrEnum):
    ARCHITECTURE_DEFAULT = "architecture_default"
    EXPLICIT = "explicit"


class QuantizationMethod(StrEnum):
    NONE = "none"
    BITSANDBYTES_NF4 = "bitsandbytes_nf4"
    EXPERIMENTAL_XLA = "experimental_xla"


class QuantType(StrEnum):
    NF4 = "nf4"
    FP4 = "fp4"


class BackendKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    XLA_TPU = "xla_tpu"


_CUDA_BACKENDS = frozenset({"cuda", "cuda_single", "cuda_distributed"})
_XLA_BACKENDS = frozenset({"xla_tpu", "tpu_single_host", "tpu_multi_host"})
_CPU_BACKENDS = frozenset({"cpu"})


def normalize_backend(backend: str | BackendKind) -> BackendKind:
    """Map registry and runtime backend names to one policy vocabulary."""
    value = str(backend).lower()
    if value in _CUDA_BACKENDS:
        return BackendKind.CUDA
    if value in _XLA_BACKENDS:
        return BackendKind.XLA_TPU
    if value in _CPU_BACKENDS:
        return BackendKind.CPU
    raise BackendCapabilityError(f"unknown PEFT backend {backend!r}; expected cpu, cuda/cuda_single, or xla_tpu/tpu_*")


def _dtype_name(value: str | torch.dtype) -> str:
    if isinstance(value, torch.dtype):
        try:
            return canonical_dtype_name(value)
        except Exception as exc:  # pragma: no cover - defensive for new torch dtypes
            raise PeftConfigError(f"unsupported compute dtype {value}") from exc
    normalized = str(value).lower().replace("torch.", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise PeftConfigError(f"unsupported compute dtype {value!r}; expected bfloat16, float16, or float32") from exc


@dataclass(frozen=True)
class LoRAConfig:
    """Serializable LoRA configuration shared by language and visual paths.

    Quantization is intentionally absent.  A LoRA config is valid on CPU,
    CUDA, or XLA; a separate :class:`QuantizationConfig` selects QLoRA.
    """

    method: str = PeftMethod.LORA.value
    enabled: bool = True
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    bias: str = BiasMode.NONE.value
    target_policy: str = TargetPolicy.ARCHITECTURE_DEFAULT.value
    target_modules: tuple[str, ...] | None = None
    modules_to_save: tuple[str, ...] = ()
    adapter_name: str = "default"
    use_rslora: bool = False
    use_dora: bool = False
    architecture: str | None = None
    confirm_target_modules: bool = False
    max_target_modules: int = 512
    schema_version: int = PEFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PEFT_SCHEMA_VERSION:
            raise PeftConfigError(
                f"unsupported PEFT schema_version={self.schema_version}; expected {PEFT_SCHEMA_VERSION}"
            )
        method = str(self.method).lower()
        if method != PeftMethod.LORA.value:
            raise PeftConfigError(f"unsupported PEFT method {self.method!r}; only 'lora' is accepted")
        object.__setattr__(self, "method", method)
        bias = str(self.bias).lower()
        try:
            BiasMode(bias)
        except ValueError as exc:
            raise PeftConfigError("bias must be one of none, all, or lora_only") from exc
        object.__setattr__(self, "bias", bias)
        policy = str(self.target_policy).lower()
        try:
            TargetPolicy(policy)
        except ValueError as exc:
            raise PeftConfigError("target_policy must be architecture_default or explicit") from exc
        object.__setattr__(self, "target_policy", policy)
        if self.rank <= 0:
            raise PeftConfigError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise PeftConfigError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise PeftConfigError("LoRA dropout must be in [0, 1)")
        if policy == TargetPolicy.EXPLICIT.value and not self.target_modules:
            raise PeftConfigError("explicit target_policy requires at least one target_modules pattern")
        if self.target_modules is not None:
            patterns = tuple(str(pattern) for pattern in self.target_modules)
            if not patterns or any(not pattern for pattern in patterns):
                raise PeftConfigError("target_modules cannot contain empty patterns")
            if len(set(patterns)) != len(patterns):
                raise PeftConfigError("target_modules must be unique")
            object.__setattr__(self, "target_modules", patterns)
        modules = tuple(str(name) for name in self.modules_to_save)
        if any(not name for name in modules):
            raise PeftConfigError("modules_to_save cannot contain empty module names")
        if len(set(modules)) != len(modules):
            raise PeftConfigError("modules_to_save must be unique")
        object.__setattr__(self, "modules_to_save", modules)
        if not self.adapter_name or "." in self.adapter_name or " " in self.adapter_name:
            raise PeftConfigError("adapter_name must be a non-empty dotted-path-safe name")
        if self.max_target_modules <= 0:
            raise PeftConfigError("max_target_modules must be positive")

    @property
    def r(self) -> int:
        """PEFT-compatible alias for ``rank``."""
        return self.rank

    @property
    def lora_alpha(self) -> float:
        """PEFT-compatible alias for ``alpha``."""
        return self.alpha

    @property
    def scaling(self) -> float:
        return float(self.alpha) / (self.rank**0.5 if self.use_rslora else self.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "enabled": self.enabled,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "target_policy": self.target_policy,
            "target_modules": None if self.target_modules is None else list(self.target_modules),
            "modules_to_save": list(self.modules_to_save),
            "adapter_name": self.adapter_name,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "architecture": self.architecture,
            "confirm_target_modules": self.confirm_target_modules,
            "max_target_modules": self.max_target_modules,
        }

    def config_hash(self) -> str:
        return config_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoRAConfig:
        raw = dict(data)
        if "r" in raw and "rank" not in raw:
            raw["rank"] = raw.pop("r")
        if "lora_alpha" in raw and "alpha" not in raw:
            raw["alpha"] = raw.pop("lora_alpha")
        if "target_modules" in raw and raw["target_modules"] is not None:
            raw["target_modules"] = tuple(str(v) for v in raw["target_modules"])
        if "modules_to_save" in raw and raw["modules_to_save"] is not None:
            raw["modules_to_save"] = tuple(str(v) for v in raw["modules_to_save"])
        allowed = {
            "method",
            "enabled",
            "rank",
            "alpha",
            "dropout",
            "bias",
            "target_policy",
            "target_modules",
            "modules_to_save",
            "adapter_name",
            "use_rslora",
            "use_dora",
            "architecture",
            "confirm_target_modules",
            "max_target_modules",
            "schema_version",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PeftConfigError(f"unknown LoRA configuration fields: {unknown}")
        return cls(**raw)


PeftConfig = LoRAConfig


@dataclass(frozen=True)
class QuantizationConfig:
    """Independent storage/compute configuration for QLoRA or experimental XLA."""

    enabled: bool = False
    method: str = QuantizationMethod.NONE.value
    load_in_4bit: bool = False
    quant_type: str = QuantType.NF4.value
    double_quant: bool = False
    compute_dtype: str | torch.dtype = "bfloat16"
    experimental_xla_quantization: bool = False
    device: str | None = None
    schema_version: int = PEFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PEFT_SCHEMA_VERSION:
            raise PeftConfigError(
                f"unsupported quantization schema_version={self.schema_version}; expected {PEFT_SCHEMA_VERSION}"
            )
        try:
            method = QuantizationMethod(str(self.method).lower()).value
        except ValueError as exc:
            raise PeftConfigError(f"unsupported quantization method {self.method!r}") from exc
        object.__setattr__(self, "method", method)
        dtype = _dtype_name(self.compute_dtype)
        object.__setattr__(self, "compute_dtype", dtype)
        quant_type = str(self.quant_type).lower()
        if quant_type not in {QuantType.NF4.value, QuantType.FP4.value}:
            raise PeftConfigError("quant_type must be nf4 or fp4")
        object.__setattr__(self, "quant_type", quant_type)
        if not self.enabled:
            if method != QuantizationMethod.NONE.value or self.load_in_4bit:
                raise PeftConfigError("disabled quantization must use method='none' and load_in_4bit=false")
            if self.experimental_xla_quantization:
                raise PeftConfigError("experimental_xla_quantization requires enabled quantization")
            return
        if method == QuantizationMethod.NONE.value:
            raise PeftConfigError("enabled quantization requires a method")
        if method == QuantizationMethod.BITSANDBYTES_NF4.value:
            if not self.load_in_4bit:
                raise PeftConfigError("bitsandbytes NF4 requires load_in_4bit=true")
            if quant_type != QuantType.NF4.value:
                raise PeftConfigError("the accepted QLoRA path uses quant_type='nf4'")
            if dtype not in {"bfloat16", "float16"}:
                raise PeftConfigError("bitsandbytes NF4 compute_dtype must be bfloat16 or float16")
            if self.experimental_xla_quantization:
                raise PeftConfigError("bitsandbytes NF4 cannot be marked experimental XLA quantization")
        elif method == QuantizationMethod.EXPERIMENTAL_XLA.value:
            if not self.experimental_xla_quantization:
                raise PeftConfigError("experimental_xla quantization requires experimental_xla_quantization=true")
            if self.load_in_4bit:
                raise PeftConfigError("experimental XLA quantization cannot use bitsandbytes load_in_4bit")

    @property
    def is_qlora(self) -> bool:
        return self.enabled and self.method == QuantizationMethod.BITSANDBYTES_NF4.value

    @property
    def is_experimental_xla(self) -> bool:
        return self.enabled and self.method == QuantizationMethod.EXPERIMENTAL_XLA.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "method": self.method,
            "load_in_4bit": self.load_in_4bit,
            "quant_type": self.quant_type,
            "double_quant": self.double_quant,
            "compute_dtype": self.compute_dtype,
            "experimental_xla_quantization": self.experimental_xla_quantization,
            "device": self.device,
        }

    def config_hash(self) -> str:
        return config_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantizationConfig:
        raw = dict(data)
        allowed = {
            "schema_version",
            "enabled",
            "method",
            "load_in_4bit",
            "quant_type",
            "double_quant",
            "compute_dtype",
            "experimental_xla_quantization",
            "device",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PeftConfigError(f"unknown quantization configuration fields: {unknown}")
        return cls(**raw)


@dataclass(frozen=True)
class BackendPeftPlan:
    """Resolved policy, recorded in manifests and run metadata."""

    backend: BackendKind
    peft: LoRAConfig
    quantization: QuantizationConfig
    mode: str
    quantization_equivalence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "mode": self.mode,
            "quantization_equivalence": self.quantization_equivalence,
            "peft": self.peft.to_dict(),
            "quantization": self.quantization.to_dict(),
        }

    def config_hash(self) -> str:
        return config_hash(self.to_dict())


def validate_backend_combination(
    peft: LoRAConfig,
    quantization: QuantizationConfig,
    backend: str | BackendKind,
    *,
    model_family: str | None = None,
    require_cuda_runtime: bool = False,
) -> BackendPeftPlan:
    """Validate the complete PEFT/backend matrix before model allocation."""
    if not isinstance(peft, LoRAConfig):
        raise PeftConfigError("peft must be a LoRAConfig")
    if not isinstance(quantization, QuantizationConfig):
        raise PeftConfigError("quantization must be a QuantizationConfig")
    selected_backend = normalize_backend(backend)
    family = None if model_family is None else str(model_family).lower()
    if quantization.is_qlora:
        if selected_backend is not BackendKind.CUDA:
            raise UnsupportedQuantizationError(
                "bitsandbytes NF4 QLoRA is CUDA-only; TPU uses first-class BF16 LoRA and is never labeled TPU QLoRA"
            )
        if family is not None and family not in {"language", "llm", "vlm", "causal_lm"}:
            raise UnsupportedQuantizationError(
                f"NF4 QLoRA is restricted to language-model families, got {model_family!r}"
            )
        if require_cuda_runtime:
            if not torch.cuda.is_available():
                raise QuantizationCapabilityError("NF4 QLoRA requires a CUDA runtime, but torch.cuda is unavailable")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise QuantizationCapabilityError(
                    "NF4 QLoRA requires bitsandbytes; install the CUDA extra (medfm[cuda]) before loading the model"
                ) from exc
        return BackendPeftPlan(
            selected_backend,
            peft,
            quantization,
            mode="qlora_nf4",
            quantization_equivalence="cuda_bitsandbytes_nf4",
        )
    if quantization.is_experimental_xla:
        if selected_backend is not BackendKind.XLA_TPU:
            raise UnsupportedQuantizationError("experimental XLA quantization is only valid for xla_tpu")
        return BackendPeftPlan(
            selected_backend,
            peft,
            quantization,
            mode="experimental_xla_quantization",
            quantization_equivalence="experimental_not_qlora_equivalent",
        )
    if selected_backend is BackendKind.XLA_TPU:
        return BackendPeftPlan(
            selected_backend,
            peft,
            quantization,
            mode="bf16_lora",
            quantization_equivalence="tpu_bf16_lora_not_qlora",
        )
    return BackendPeftPlan(
        selected_backend,
        peft,
        quantization,
        mode="lora" if peft.enabled else "frozen",
        quantization_equivalence="unquantized",
    )


# Compatibility spellings used by Hugging Face examples.
LoraConfig = LoRAConfig
PEFTConfig = LoRAConfig
QLoRAConfig = QuantizationConfig


__all__ = [
    "BackendKind",
    "BackendPeftPlan",
    "BiasMode",
    "LoRAConfig",
    "LoraConfig",
    "PEFTConfig",
    "PEFT_SCHEMA_VERSION",
    "PeftConfig",
    "PeftMethod",
    "QLoRAConfig",
    "QuantType",
    "QuantizationConfig",
    "QuantizationMethod",
    "TargetPolicy",
    "normalize_backend",
    "validate_backend_combination",
]
