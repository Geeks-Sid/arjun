from __future__ import annotations

import pytest

from medfm.training.backend import CpuBackend, resolve_attention
from medfm.training.config import RunConfig, RunConfigError
from medfm.training.pipeline import ComponentBuilders, TrainingPipeline


def test_run_config_is_typed_and_canonical(tiny_config) -> None:
    first = tiny_config.config_hash()
    second = RunConfig.from_dict(tiny_config.to_dict())
    assert first == second.config_hash()
    assert second.global_batch_size == 2
    assert '"backend":"cpu"' in second.canonical_json()


def test_global_batch_mismatch_fails_before_build() -> None:
    with pytest.raises(RunConfigError, match="global_batch_size"):
        RunConfig.from_dict(
            {
                "accelerator": {"backend": "cpu", "world_size": 2},
                "batch": {
                    "microbatch_per_device": 2,
                    "gradient_accumulation_steps": 2,
                    "global_batch_size": 3,
                },
            }
        )


def test_cpu_backend_has_no_accelerator_runtime_dependency() -> None:
    backend = CpuBackend(use_accelerate=False)
    assert backend.device.type == "cpu"
    assert backend.topology.world_size == 1
    assert resolve_attention("sdpa", backend="cpu").selected in {"sdpa", "eager"}


def test_pipeline_preflight_happens_before_model_builder(tiny_config) -> None:
    order: list[str] = []

    def model(config):
        order.append("model")
        raise AssertionError("model builder must not run during preflight")

    pipeline = TrainingPipeline(
        tiny_config,
        builders=ComponentBuilders(model=model),
        dry_run=True,
    )
    result = pipeline.dry_run_summary()
    assert result.model_summary is not None
    assert result.model_summary.allocated is False
    assert order == []
