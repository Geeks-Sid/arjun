"""Phase 17 CPU-safe fixtures and protected hardware guards."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def tiny_volume() -> object:
    import torch

    return torch.arange(1 * 1 * 7 * 6 * 5, dtype=torch.float32).reshape(1, 1, 7, 6, 5)


@pytest.fixture
def bundle_builder(tmp_path):
    from medfm.inference import BaseModelReference, BundleBuilder, RuntimeSupport

    return BundleBuilder(
        tmp_path / "bundle",
        bundle_id="phase17-bundle",
        model_id="phase17-model",
        model_revision="model-rev",
        task="classification",
        base_models=[BaseModelReference("base-model", "base-rev", architecture="tiny")],
        model_card="# Phase 17 test model\n",
        license_summary="Test-only license; not for clinical deployment.\n",
        preprocessing={"name": "identity", "channels": 1},
        postprocessing={"name": "softmax"},
        task_schema={"type": "object", "required": ["predictions"]},
        inference_config={"backend": "cpu", "limits": {"max_batch_size": 2}},
        modalities=["XRAY_2D"],
        runtime=RuntimeSupport(backends={"cpu": "tested", "cuda": "untested", "xla_tpu": "untested"}),
    )


@pytest.fixture(autouse=True)
def _guard_hardware_tests(request: pytest.FixtureRequest) -> None:
    for marker, variable in {
        "gpu": "MEDFM_RUN_GPU_TESTS",
        "tpu": "MEDFM_RUN_TPU_TESTS",
        "distributed": "MEDFM_RUN_DISTRIBUTED_TESTS",
    }.items():
        if request.node.get_closest_marker(marker) and os.environ.get(variable) != "1":
            pytest.skip(f"protected {marker} test; set {variable}=1 to enable")
