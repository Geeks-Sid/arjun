"""Protected CUDA tests. Enable with MEDFM_RUN_GPU_TESTS=1 (make test-gpu).

When enabled, missing CUDA hardware is a FAILURE, not a skip.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_cuda_hardware_present_when_enabled():
    assert torch.cuda.is_available(), "GPU tests enabled but no CUDA device is visible"
    assert torch.cuda.is_bf16_supported(), "target GPU must support BF16"


def test_bf16_allocation_and_peak_memory_reporting():
    assert torch.cuda.is_available(), "GPU tests enabled but no CUDA device is visible"
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device)
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device)
    out = a @ b
    torch.cuda.synchronize(device)

    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"
    peak = torch.cuda.max_memory_allocated(device)
    assert peak >= a.numel() * a.element_size() * 2
    print(f"\npeak allocated VRAM: {peak / 2**20:.1f} MiB on {torch.cuda.get_device_name(0)}")


def test_multi_device_reduction():
    device_count = torch.cuda.device_count()
    if device_count < 2:
        pytest.skip(f"multi-device reduction needs >= 2 GPUs, found {device_count}")
    totals = []
    for idx in range(device_count):
        device = torch.device(f"cuda:{idx}")
        value = torch.ones(8, dtype=torch.float32, device=device) * (idx + 1)
        reduced = value.to("cpu").sum()
        totals.append(reduced.item())
    expected = sum((i + 1) * 8 for i in range(device_count))
    assert sum(totals) == pytest.approx(expected)
