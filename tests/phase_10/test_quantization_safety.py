from __future__ import annotations

import pytest
import torch
from torch import nn

from medfm.peft import (
    LoRAConfig,
    QuantizedParameterError,
    audit_trainable_parameters,
    inject_lora,
    mark_quantized_parameters,
    optimizer_parameter_groups,
    verify_quantized_optimizer_exclusion,
)


def _wrapped_linear() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(4, 3))
    inject_lora(
        model,
        LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_policy="explicit",
            target_modules=("0",),
            architecture="vision",
        ),
        architecture="vision",
    )
    return model


def test_quantized_base_is_frozen_and_never_enters_optimizer() -> None:
    model = _wrapped_linear()
    for name, parameter in model.named_parameters():
        if "base_layer" in name:
            parameter._medfm_quantized = True  # type: ignore[attr-defined]
            parameter.requires_grad_(False)
    audit = audit_trainable_parameters(model, qlora=True)
    assert audit.adapter_parameters > 0
    assert audit.quantized_parameters > 0
    groups = optimizer_parameter_groups(model)
    optimizer = torch.optim.AdamW(groups)
    verify_quantized_optimizer_exclusion(model, optimizer)
    assert all(not any("base_layer" in name for name in group["parameter_names"]) for group in groups)


def test_qlora_rejects_full_base_trainability_and_quantized_optimizer_groups() -> None:
    model = _wrapped_linear()
    model[0].base_layer.weight.requires_grad_(True)
    model[0].base_layer.weight._medfm_quantized = True  # type: ignore[attr-defined]
    with pytest.raises(QuantizedParameterError):
        audit_trainable_parameters(model, qlora=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(QuantizedParameterError):
        verify_quantized_optimizer_exclusion(model, optimizer)


def test_zero_trainable_parameters_fails_audit() -> None:
    model = nn.Linear(2, 2)
    model.requires_grad_(False)
    with pytest.raises(Exception, match="zero trainable"):
        audit_trainable_parameters(model)


def test_no_accidental_quantization_on_cpu_lora_path() -> None:
    model = _wrapped_linear()
    assert all(not getattr(parameter, "_medfm_quantized", False) for parameter in model.parameters())
    assert mark_quantized_parameters(nn.Linear(2, 2)) == 6
