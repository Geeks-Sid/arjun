"""Minimal LoRA optimization step on a tiny local model (CPU, no downloads)."""

from __future__ import annotations

import torch


def _build_model() -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"])
    return get_peft_model(base, config)


def test_lora_freezes_base_and_trains_adapters():
    model = _build_model()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    assert trainable and frozen
    assert all("lora_" in name for name in trainable)
    assert any("base_model" in name for name in frozen)


def test_one_lora_optimization_step_updates_only_adapters():
    model = _build_model()
    params = dict(model.named_parameters())
    frozen_before = {name: p.detach().clone() for name, p in params.items() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in params.items() if p.requires_grad}

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    inputs = torch.randn(8, 16)
    targets = torch.randint(0, 4, (8,))

    model.train()
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

    for name, before in frozen_before.items():
        assert torch.equal(before, params[name].detach()), f"frozen param changed: {name}"
    changed = [name for name, before in trainable_before.items() if not torch.equal(before, params[name].detach())]
    assert changed, "no LoRA parameter was updated"
