from __future__ import annotations

import pytest
import torch
from torch import nn

from medfm.peft import (
    BackendKind,
    LoRAConfig,
    PeftConfigError,
    QuantizationCapabilityError,
    QuantizationConfig,
    UnsupportedQuantizationError,
    validate_backend_combination,
)
from medfm.peft.quantization import (
    disable_training_kv_cache,
    prepare_bf16_lora_model,
    validate_quantization,
)


def test_lora_and_quantization_configs_are_separate_and_versioned() -> None:
    config = LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.1,
        bias="lora_only",
        modules_to_save=("classifier", "vl_bridge"),
        adapter_name="vision",
        use_rslora=True,
        use_dora=True,
    )
    payload = config.to_dict()
    assert payload["schema_version"] == 1
    assert payload["modules_to_save"] == ["classifier", "vl_bridge"]
    assert config.scaling == pytest.approx(16 / (8**0.5))
    assert LoRAConfig.from_dict(payload) == config

    quant = QuantizationConfig(
        enabled=True,
        method="bitsandbytes_nf4",
        load_in_4bit=True,
        quant_type="NF4",
        double_quant=True,
        compute_dtype=torch.bfloat16,
    )
    assert quant.is_qlora
    assert quant.to_dict()["compute_dtype"] == "bfloat16"
    assert not QuantizationConfig().enabled


def test_invalid_configuration_fails_before_injection() -> None:
    with pytest.raises(PeftConfigError, match="explicit target_policy"):
        LoRAConfig(target_policy="explicit")
    with pytest.raises(PeftConfigError, match="compute_dtype"):
        QuantizationConfig(
            enabled=True,
            method="bitsandbytes_nf4",
            load_in_4bit=True,
            compute_dtype="float32",
        )


def test_cuda_nf4_and_tpu_bf16_matrix_is_explicit() -> None:
    peft = LoRAConfig()
    qlora = QuantizationConfig(enabled=True, method="bitsandbytes_nf4", load_in_4bit=True)
    cuda_plan = validate_backend_combination(peft, qlora, "cuda", model_family="language")
    assert cuda_plan.backend is BackendKind.CUDA
    assert cuda_plan.mode == "qlora_nf4"
    with pytest.raises(UnsupportedQuantizationError, match="CUDA-only"):
        validate_backend_combination(peft, qlora, "xla_tpu", model_family="language")
    tpu_plan = validate_backend_combination(peft, QuantizationConfig(), "xla_tpu")
    assert tpu_plan.mode == "bf16_lora"
    assert tpu_plan.quantization_equivalence == "tpu_bf16_lora_not_qlora"


def test_bf16_lora_preparation_disables_cache_and_never_claims_qlora() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.config = type("Config", (), {"use_cache": True})()

    model = Model()
    prepared = prepare_bf16_lora_model(model, backend="xla_tpu")
    assert prepared._medfm_quantization_mode == "bf16_lora"
    assert prepared._medfm_compute_dtype == "bfloat16"
    assert next(prepared.parameters()).dtype is torch.bfloat16
    assert not prepared.config.use_cache
    assert all(not parameter.requires_grad for parameter in prepared.parameters())
    assert disable_training_kv_cache(prepared)


def test_runtime_missing_bitsandbytes_is_typed_when_requested() -> None:
    qlora = QuantizationConfig(enabled=True, method="bitsandbytes_nf4", load_in_4bit=True)
    try:
        plan = validate_quantization(qlora, backend="cuda", model_family="language", check_runtime=True)
    except QuantizationCapabilityError as exc:
        assert "bitsandbytes" in str(exc).lower() or "transformers" in str(exc).lower()
    else:
        # Protected GPU environments may provide both optional packages.
        assert plan.mode == "qlora_nf4"
