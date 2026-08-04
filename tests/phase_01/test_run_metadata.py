"""Run-metadata capture: completeness, determinism, validation."""

from __future__ import annotations

import json

import pytest
import torch

from medfm.training.run_metadata import RunMetadata, capture_run_metadata


def _capture(**overrides):
    kwargs = {
        "accelerator_backend": "cpu",
        "seed": 1234,
        "precision": "bf16",
        "microbatch_per_device": 2,
        "gradient_accumulation_steps": 4,
        "world_size": 8,
        "adapter_config": {"r": 8, "lora_alpha": 16},
        "preprocessing_config": {"spacing": [1.0, 1.0, 1.0]},
        "shape_buckets": {"text_tokens": [256, 512]},
        "compiler_flags": {"PJRT_DEVICE": "TPU"},
    }
    kwargs.update(overrides)
    return capture_run_metadata(**kwargs)


def test_required_fields_present():
    meta = _capture()
    assert meta.git_commit  # repo is a git checkout
    assert meta.git_dirty is not None
    assert meta.lockfile_sha256  # uv.lock exists and is hashed
    assert meta.python_version
    assert meta.torch_version
    assert meta.accelerator_backend == "cpu"
    assert meta.seed == 1234
    assert meta.precision == "bf16"


def test_effective_batch_size_formula():
    meta = _capture(microbatch_per_device=2, gradient_accumulation_steps=4, world_size=8)
    assert meta.effective_batch_size == 2 * 8 * 4


def test_placeholder_hashes_recorded():
    meta = _capture(dataset_manifest_sha256="abc123", base_model_revision="rev-1")
    assert meta.dataset_manifest_sha256 == "abc123"
    assert meta.base_model_revision == "rev-1"
    assert meta.adapter_config_sha256
    assert meta.preprocessing_config_sha256


def test_trainable_parameter_counts():
    model = torch.nn.Linear(4, 4)
    for param in model.parameters():
        param.requires_grad = False
    model.weight.requires_grad = True
    meta = _capture(model=model)
    assert meta.total_parameters == 4 * 4 + 4
    assert meta.trainable_parameters == 4 * 4


def test_serialization_is_deterministic():
    first = _capture()
    second = _capture()
    assert first.to_canonical_json() == second.to_canonical_json()
    parsed = json.loads(first.to_canonical_json())
    assert parsed == first.to_dict()
    assert first.config_hash() == second.config_hash()


def test_config_hash_excludes_measured_fields():
    base = _capture()
    measured_variant = RunMetadata(**{**base.to_dict(), "peak_memory_bytes": 999, "xla_metrics_report": "x"})
    assert base.config_hash() == measured_variant.config_hash()


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="accelerator backend"):
        _capture(accelerator_backend="tpu_v5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        _capture(microbatch_per_device=0)
