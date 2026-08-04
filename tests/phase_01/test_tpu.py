"""Protected TPU tests. Enable with MEDFM_RUN_TPU_TESTS=1 (make test-tpu).

Run on a TPU VM after scripts/tpu_vm_bootstrap.sh. When enabled, a missing
XLA runtime is a FAILURE, not a skip.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.tpu


@pytest.fixture(scope="module")
def xla():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    except ImportError as exc:
        pytest.fail(f"TPU tests enabled but torch_xla is not importable: {exc}")
    return xm, xr


def test_tpu_runtime_present(xla):
    _xm, xr = xla
    count = xr.global_runtime_device_count()
    assert count >= 1, "no TPU devices visible to the PJRT runtime"


def test_bf16_linear_step_on_every_local_device(xla):
    xm, xr = xla
    device_count = xr.global_runtime_device_count()
    assert device_count >= 1, "no TPU devices visible"

    for ordinal in range(device_count):
        device = xm.xla_device(ordinal)
        torch.manual_seed(0)
        model = torch.nn.Linear(32, 8).to(device=device, dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(4, 32, dtype=torch.bfloat16, device=device)
        targets = torch.randn(4, 8, dtype=torch.bfloat16, device=device)

        before = model.weight.detach().float().sum().item()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()
        after = model.weight.detach().float().sum().item()

        assert torch.isfinite(loss.float()).item()
        assert before != after, f"no parameter update on TPU device {ordinal}"


@pytest.mark.distributed
def test_multi_device_all_reduce(xla):
    xm, xr = xla
    device_count = xr.global_runtime_device_count()
    if device_count < 2:
        pytest.skip(f"multi-device reduction needs >= 2 TPU devices, found {device_count}")

    device = xm.xla_device()
    value = torch.ones(16, dtype=torch.float32, device=device)
    reduced = xm.all_reduce(xm.REDUCE_SUM, value)
    xm.mark_step()
    assert reduced.cpu().eq(float(device_count)).all()
