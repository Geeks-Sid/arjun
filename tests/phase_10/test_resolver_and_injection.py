from __future__ import annotations

import pytest
import torch
from torch import nn

from medfm.models.language import GenericHFCausalLMAdapter
from medfm.models.visual.native_3d import GenericMONAI3DAdapter
from medfm.peft import (
    LoRAConfig,
    TargetResolutionError,
    UnknownArchitectureError,
    audit_trainable_parameters,
    build_optimizer_parameter_groups,
    inject_language_lora,
    inject_lora,
    inject_visual_lora,
    inspect_modules,
)


class TinyVisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.self_attn = nn.Module()
            block.self_attn.q_proj = nn.Linear(8, 8)
            block.self_attn.k_proj = nn.Linear(8, 8)
            block.self_attn.v_proj = nn.Linear(8, 8)
            block.self_attn.out_proj = nn.Linear(8, 8)
            block.mlp = nn.Module()
            block.mlp.fc1 = nn.Linear(8, 16)
            block.mlp.fc2 = nn.Linear(16, 8)
            self.encoder.layers.append(block)
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.encoder.layers:
            x = block.self_attn.out_proj(x) + block.mlp.fc2(torch.tanh(block.mlp.fc1(x)))
        return self.norm(x).mean(dim=1)


def test_2d_vit_resolver_excludes_stem_norm_and_selects_attention_mlp() -> None:
    model = TinyVisionTransformer()
    resolution = inspect_modules(model, architecture="vision", config=LoRAConfig(architecture="vision"))
    assert resolution.selected_count == 12
    assert all("patch_embed" not in name for name in resolution.selected_names)
    assert all("norm" not in name for name in resolution.selected_names)
    assert all(record.reason for record in resolution.records if record.selected)

    result = inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=0.0, architecture="vision"), architecture="vision")
    output = model(torch.randn(2, 3, 8, 8))
    output.sum().backward()
    assert result.selected_modules
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if "lora_" in name)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "patch_embed" in name or "norm" in name
    )
    audit = audit_trainable_parameters(model)
    assert audit.adapter_parameters > 0
    assert audit.other_trainable_parameters == 0
    assert build_optimizer_parameter_groups(model)[0]["params"]


def test_3d_default_targets_only_late_transformer_stage() -> None:
    adapter = GenericMONAI3DAdapter.build_tiny()
    result = inject_visual_lora(
        adapter,
        LoRAConfig(rank=2, alpha=4, dropout=0.0, architecture="3d_transformer"),
    )
    assert result.selected_modules
    assert all("blocks.layers.1." in name for name in result.selected_modules)
    assert all("patch_embed" not in name for name in result.selected_modules)


def test_language_lora_freezes_full_base_and_keeps_boundary_bridge_trainable() -> None:
    adapter = GenericHFCausalLMAdapter.build_tiny(
        hidden_size=16,
        vocab_size=32,
        max_text_tokens=32,
        visual_token_buckets=(4,),
        construction_seed=9,
    )
    result = inject_language_lora(
        adapter,
        LoRAConfig(rank=2, alpha=4, dropout=0.0, architecture="llm", adapter_name="language"),
    )
    assert result.selected_modules
    assert all(
        not parameter.requires_grad
        for name, parameter in adapter.model.named_parameters()
        if "lora_" not in name and "dora_magnitude_" not in name
    )
    assert any(parameter.requires_grad for parameter in adapter.boundary_embeddings.parameters())
    text = adapter.tokenize(["prompt answer"])
    labels = torch.full_like(text.input_ids, -100)
    labels[:, -1] = 3
    output = adapter.forward_with_visual_tokens(text, None, labels)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for name, parameter in adapter.named_parameters() if "lora_" in name)
    assert adapter.trainable_module_declarations()["language_lora"]
    audit = audit_trainable_parameters(adapter)
    assert audit.adapter_parameters > 0
    assert audit.bridge_parameters > 0


def test_unknown_architecture_requires_explicit_confirmation_and_target() -> None:
    model = nn.Sequential(nn.Linear(4, 3))
    with pytest.raises(UnknownArchitectureError):
        inject_lora(model, LoRAConfig(architecture="unreviewed"), architecture="unreviewed")
    config = LoRAConfig(
        rank=2,
        alpha=4,
        target_policy="explicit",
        target_modules=("0",),
        architecture="unreviewed",
        confirm_target_modules=True,
    )
    assert inject_lora(model, config, architecture="unreviewed").selected_modules == ("0",)


def test_zero_targets_and_overbroad_patterns_fail_closed() -> None:
    with pytest.raises(TargetResolutionError):
        inspect_modules(
            TinyVisionTransformer(),
            architecture="vision",
            config=LoRAConfig(target_policy="explicit", target_modules=("does.not.exist",)),
        )
    with pytest.raises(TargetResolutionError):
        inject_lora(
            TinyVisionTransformer(),
            LoRAConfig(
                rank=2,
                target_policy="explicit",
                target_modules=(r".*",),
                max_target_modules=2,
                confirm_target_modules=True,
                architecture="vision",
            ),
            architecture="vision",
        )
