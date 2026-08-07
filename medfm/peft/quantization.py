"""CUDA NF4 and TPU BF16 adaptation policy.

The module validates policy before weight construction, keeps bitsandbytes
imports lazy, and makes the TPU baseline explicit instead of silently treating
BF16 LoRA as QLoRA.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from medfm.peft.config import (
    BackendKind,
    BackendPeftPlan,
    LoRAConfig,
    QuantizationConfig,
    normalize_backend,
    validate_backend_combination,
)
from medfm.peft.errors import (
    BitsAndBytesCapabilityError,
    QuantizationCapabilityError,
    QuantizedParameterError,
    UnsupportedQuantizationError,
)
from medfm.peft.lora import audit_trainable_parameters, is_quantized_parameter


@dataclass(frozen=True)
class QuantizationRuntime:
    bitsandbytes_available: bool
    cuda_available: bool
    transformers_available: bool

    @property
    def nf4_available(self) -> bool:
        return self.bitsandbytes_available and self.cuda_available and self.transformers_available


def runtime_capabilities() -> QuantizationRuntime:
    return QuantizationRuntime(
        bitsandbytes_available=importlib.util.find_spec("bitsandbytes") is not None,
        cuda_available=bool(torch.cuda.is_available()),
        transformers_available=importlib.util.find_spec("transformers") is not None,
    )


def validate_quantization(
    config: QuantizationConfig,
    *,
    backend: str | BackendKind,
    model_family: str | None = None,
    check_runtime: bool = False,
) -> BackendPeftPlan:
    """Validate quantization and backend capability before model allocation."""
    plan = validate_backend_combination(
        LoRAConfig(),
        config,
        backend,
        model_family=model_family,
        require_cuda_runtime=False,
    )
    runtime = runtime_capabilities()
    if config.is_qlora and check_runtime:
        if plan.backend is not BackendKind.CUDA:
            raise UnsupportedQuantizationError("bitsandbytes NF4 is not a TPU/CPU backend")
        if not runtime.cuda_available:
            raise BitsAndBytesCapabilityError("bitsandbytes NF4 requires torch.cuda.is_available() == True")
        if not runtime.bitsandbytes_available:
            raise BitsAndBytesCapabilityError(
                "bitsandbytes is unavailable; install the CUDA extra before constructing a QLoRA model"
            )
        if not runtime.transformers_available:
            raise BitsAndBytesCapabilityError(
                "transformers is unavailable; BitsAndBytesConfig is required for supported 4-bit loading"
            )
    return plan


def build_bitsandbytes_config(config: QuantizationConfig) -> Any:
    """Build the upstream ``BitsAndBytesConfig`` lazily and fail typed."""
    if not config.is_qlora:
        raise UnsupportedQuantizationError("BitsAndBytesConfig requires an enabled bitsandbytes NF4 config")
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise BitsAndBytesCapabilityError(
            "transformers is required to construct a supported NF4 loading config"
        ) from exc
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config.compute_dtype)]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(config.double_quant),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def mark_quantized_parameters(model: nn.Module) -> int:
    """Mark base parameters so optimizer audits remain independent of bnb types."""
    count = 0
    for parameter in model.parameters():
        parameter._medfm_quantized = True  # type: ignore[attr-defined]
        parameter.requires_grad_(False)
        count += int(parameter.numel())
    return count


def disable_training_kv_cache(model: nn.Module) -> bool:
    """Disable the Transformers training KV cache when the config exposes it."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "use_cache"):
        return False
    config.use_cache = False
    return True


def prepare_model_for_kbit_training(
    model: nn.Module,
    config: QuantizationConfig,
    *,
    backend: str | BackendKind = BackendKind.CUDA,
    model_family: str = "language",
) -> nn.Module:
    """Prepare an already 4-bit-loaded model through supported APIs.

    Weight loading itself must happen with the returned
    :func:`build_bitsandbytes_config` before constructing the Transformers
    model.  This function is the post-load safety gate and never converts an
    arbitrary FP model while pretending it is NF4.
    """
    plan = validate_quantization(config, backend=backend, model_family=model_family, check_runtime=True)
    if not config.is_qlora or plan.backend is not BackendKind.CUDA:
        raise UnsupportedQuantizationError("prepare_model_for_kbit_training only accepts CUDA NF4 QLoRA")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    model_device = next(model.parameters(), None)
    if model_device is None or model_device.device.type != "cuda":
        raise QuantizationCapabilityError("a bitsandbytes NF4 model must be placed on CUDA before k-bit preparation")
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        # Transformers sets this marker for a supported 4-bit load.  Refusing
        # to infer it prevents a full-precision model from being mislabeled.
        raise QuantizationCapabilityError(
            "model is not marked is_loaded_in_4bit; load it with the upstream BitsAndBytesConfig first"
        )
    try:
        from peft import prepare_model_for_kbit_training as prepare
    except ImportError as exc:
        raise BitsAndBytesCapabilityError(
            "PEFT is required for supported k-bit training preparation; install medfm[hf]"
        ) from exc
    prepared = prepare(model)
    mark_quantized_parameters(prepared)
    disable_training_kv_cache(prepared)
    for parameter in prepared.parameters():
        if not is_quantized_parameter(parameter):
            # PEFT may leave LayerNorm/output parameters usable for a caller's
            # modules_to_save, but the base model remains frozen until explicit
            # modules-to-save handling.
            parameter.requires_grad_(False)
    prepared._medfm_quantization_config = config.to_dict()  # type: ignore[attr-defined]
    prepared._medfm_quantization_mode = plan.mode  # type: ignore[attr-defined]
    return prepared


def prepare_bf16_lora_model(
    model: nn.Module,
    *,
    backend: str | BackendKind = BackendKind.XLA_TPU,
    peft_config: LoRAConfig | None = None,
) -> nn.Module:
    """Prepare the accepted TPU frozen-base/BF16-LoRA path."""
    selected = normalize_backend(backend)
    if selected is BackendKind.XLA_TPU and peft_config is None:
        peft_config = LoRAConfig()
    if selected is BackendKind.XLA_TPU:
        validate_backend_combination(peft_config or LoRAConfig(), QuantizationConfig(), selected)
    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    disable_training_kv_cache(model)
    model._medfm_quantization_mode = "bf16_lora"  # type: ignore[attr-defined]
    model._medfm_compute_dtype = "bfloat16"  # type: ignore[attr-defined]
    return model


def verify_compute_dtype(model: nn.Module, expected: str | torch.dtype) -> None:
    expected_dtype = expected
    if isinstance(expected, str):
        expected_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(expected.lower().replace("torch.", ""))
    if not isinstance(expected_dtype, torch.dtype):
        raise QuantizationCapabilityError(f"unsupported expected compute dtype {expected!r}")
    observed = next(model.parameters(), None)
    if observed is not None and observed.dtype != expected_dtype:
        raise QuantizationCapabilityError(
            f"compute dtype mismatch: expected {expected_dtype}, observed {observed.dtype}"
        )


def verify_quantized_optimizer_exclusion(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Reject quantized parameters in every optimizer group."""
    offending: list[str] = []
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if is_quantized_parameter(parameter):
                offending.append(parameter_names.get(id(parameter), "<unnamed>"))
    if offending:
        raise QuantizedParameterError("quantized base parameters entered the optimizer: " + ", ".join(offending[:8]))


def assert_qlora_trainability(model: nn.Module) -> None:
    """Fail if a QLoRA model exposes full base weights or no adapter."""
    audit = audit_trainable_parameters(model, qlora=True)
    if audit.adapter_parameters <= 0:
        raise QuantizedParameterError("QLoRA model has no trainable adapter parameters")


__all__ = [
    "QuantizationRuntime",
    "assert_qlora_trainability",
    "build_bitsandbytes_config",
    "disable_training_kv_cache",
    "mark_quantized_parameters",
    "prepare_bf16_lora_model",
    "prepare_model_for_kbit_training",
    "runtime_capabilities",
    "validate_quantization",
    "verify_compute_dtype",
    "verify_quantized_optimizer_exclusion",
]
