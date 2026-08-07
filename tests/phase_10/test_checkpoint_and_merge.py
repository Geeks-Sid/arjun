from __future__ import annotations

import json
import tempfile

import pytest
import torch
from torch import nn

from medfm.peft import (
    CheckpointCompatibilityError,
    LoRAConfig,
    add_named_adapter,
    compare_merged_unmerged,
    inject_lora,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    set_active_adapter,
)


def _model_with_adapters() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(4, 3))
    model.bridge = nn.Linear(3, 3)  # type: ignore[attr-defined]
    vision = LoRAConfig(
        rank=2,
        alpha=4,
        dropout=0.0,
        target_policy="explicit",
        target_modules=("0",),
        architecture="vision",
        adapter_name="vision",
    )
    inject_lora(model, vision, architecture="vision")
    model.bridge.requires_grad_(True)  # type: ignore[attr-defined]
    add_named_adapter(
        model,
        LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_policy="explicit",
            target_modules=("0",),
            architecture="vision",
            adapter_name="site",
        ),
        architecture="vision",
    )
    model.bridge.requires_grad_(True)  # type: ignore[attr-defined]
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_(0.0, 0.1)
    return model


def test_separate_named_adapters_round_trip_without_base_weights() -> None:
    source = _model_with_adapters()
    set_active_adapter(source, "vision")
    with tempfile.TemporaryDirectory() as directory:
        save_adapter_checkpoint(
            directory,
            source,
            base_model_id="tiny-vision",
            base_revision="revision-a",
            architecture="vision",
        )
        manifest = json.loads(open(f"{directory}/manifest.json", encoding="utf-8").read())
        assert manifest["canonical_format"] == "safetensors"
        assert "adapter:vision" in manifest["tensor_files"]
        assert "adapter:site" in manifest["tensor_files"]
        assert all("base_layer" not in path for path in manifest["tensor_files"].values())

        restored = nn.Sequential(nn.Linear(4, 3))
        restored.bridge = nn.Linear(3, 3)  # type: ignore[attr-defined]
        load_adapter_checkpoint(
            directory,
            restored,
            base_model_id="tiny-vision",
            base_revision="revision-a",
            architecture="vision",
            adapter_name="vision",
        )
        assert set(restored[0].lora_A) == {"vision"}
        assert restored._medfm_active_adapter == "vision"
        assert torch.equal(
            source[0].lora_B["vision"].weight,
            restored[0].lora_B["vision"].weight,
        )

        restored_site = nn.Sequential(nn.Linear(4, 3))
        restored_site.bridge = nn.Linear(3, 3)  # type: ignore[attr-defined]
        load_adapter_checkpoint(
            directory,
            restored_site,
            base_model_id="tiny-vision",
            base_revision="revision-a",
            architecture="vision",
            adapter_name="site",
        )
        assert set(restored_site[0].lora_A) == {"site"}


def test_wrong_base_revision_and_architecture_are_rejected() -> None:
    source = _model_with_adapters()
    with tempfile.TemporaryDirectory() as directory:
        save_adapter_checkpoint(
            directory,
            source,
            base_model_id="tiny-vision",
            base_revision="revision-a",
            architecture="vision",
        )
        with pytest.raises(CheckpointCompatibilityError, match="wrong base revision"):
            load_adapter_checkpoint(
                directory,
                nn.Sequential(nn.Linear(4, 3)),
                base_model_id="tiny-vision",
                base_revision="revision-b",
                architecture="vision",
            )
        with pytest.raises(CheckpointCompatibilityError, match="wrong base model"):
            load_adapter_checkpoint(
                directory,
                nn.Sequential(nn.Linear(4, 3)),
                base_model_id="other-model",
                base_revision="revision-a",
                architecture="vision",
            )


def test_merged_and_unmerged_outputs_are_equivalent_and_artifact_stays_unmerged() -> None:
    model = _model_with_adapters()
    set_active_adapter(model, "vision")
    x = torch.randn(3, 4)
    assert compare_merged_unmerged(model, lambda: model(x), adapter_name="vision", atol=2e-5)
    assert not model[0].merged
    assert any("lora_A.vision" in name for name, _ in model.named_parameters())
