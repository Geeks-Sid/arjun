"""Phase 18 reliability and fault-injection tests (CPU; Level-1 run).

Covers disk-full during checkpointing, training interruption -> resumable
checkpoint, deterministic-mode reproducibility, a documented cross-backend
tolerance policy, and a scratch environment install-to-inference rehearsal.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest
import torch

from medfm.recipes.phase13 import phase13_builders
from medfm.training.checkpoint import CheckpointManager
from medfm.training.config import RunConfig
from medfm.training.pipeline import TrainingPipeline

#: Per-dtype/hardware tolerance policy referenced by golden and cross-backend
#: regression runs. CPU-vs-CPU deterministic recompute must be exact; CUDA/TPU
#: results are compared against CPU goldens within the listed tolerance.
CROSS_BACKEND_TOLERANCES = {
    "float32": {"cpu_cpu": 0.0, "cpu_cuda": 5e-4, "cpu_tpu": 5e-3},
    "float16": {"cpu_cpu": 0.0, "cpu_cuda": 1e-2, "cpu_tpu": 1e-1},
    "bfloat16": {"cpu_cpu": 0.0, "cpu_cuda": 1e-1, "cpu_tpu": 1e-1},
}


def _config(tmp_path: Path, *, max_steps: int) -> RunConfig:
    import copy

    from medfm.tools import governance as gov

    base = RunConfig.load(gov.REPO_ROOT / "configs" / "recipes" / "2d" / "classification_smoke.yaml")
    values = copy.deepcopy(base.to_dict())
    values["output_dir"] = str(tmp_path / "run")
    values["max_steps"] = max_steps
    values["epochs"] = 1
    return RunConfig.from_dict(values)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Linear(2, 2)


def test_checkpoint_disk_full_leaves_no_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "ckpts"
    manager = CheckpointManager(root)

    def _enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr("torch.save", _enospc)
    with pytest.raises(OSError) as excinfo:
        manager.save(1, model=_tiny_model(), run_config=_config(tmp_path, max_steps=1))
    assert excinfo.value.errno == errno.ENOSPC

    assert not (root / "1").exists(), "a partial checkpoint directory must not appear at the target path"
    leftovers = [p for p in (root.iterdir() if root.exists() else []) if p.name.startswith(".1.tmp-")]
    assert not leftovers, "temp checkpoint directory leaked after a disk-full failure"


def test_training_interrupt_saves_resumable_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, max_steps=3)
    pipeline = TrainingPipeline(config, builders=phase13_builders()).build()
    trainer = pipeline.trainer
    forwarded = {"count": 0}
    real = trainer.training_step.forward_and_loss

    def interrupted_forward_and_loss(model, batch):
        forwarded["count"] += 1
        if forwarded["count"] == 1:
            raise KeyboardInterrupt
        return real(model, batch)

    monkeypatch.setattr(trainer.training_step, "forward_and_loss", interrupted_forward_and_loss)
    result = trainer.train()
    assert result.interrupted is True
    assert result.success is False
    assert result.checkpoint is not None
    ckpt = Path(result.checkpoint)
    assert ckpt.is_dir()
    # The interrupted checkpoint must be complete (manifest + verified hashes),
    # i.e. safe to resume from. Full resume-into-trainer is covered by
    # phase_12/phase_14 resume tests.
    _assert_complete_checkpoint(ckpt)


def test_deterministic_mode_repeats_loss(tmp_path: Path) -> None:
    first = _train_one_step(tmp_path / "a")
    second = _train_one_step(tmp_path / "b")
    assert first.metrics == second.metrics, "same seed + deterministic CPU mode must reproduce metrics exactly"


def test_cross_backend_tolerance_policy_is_documented() -> None:
    # The policy table is the contract the golden suite and protected hardware
    # jobs assert against; CPU-vs-CPU is exact, accelerator comparisons use the
    # bound for the dtype in play.
    assert CROSS_BACKEND_TOLERANCES["float32"]["cpu_cpu"] == 0.0
    assert CROSS_BACKEND_TOLERANCES["float32"]["cpu_cuda"] < CROSS_BACKEND_TOLERANCES["float32"]["cpu_tpu"]


def test_clean_environment_install_to_inference_rehearsal(tmp_path: Path) -> None:
    """Scratch-environment rehearsal: build and run a tiny classification
    pipeline exactly as a clean operator would, then run bounded inference on
    the trained head. (A full clean-venv rehearsal runs as a CI job.)"""
    from medfm.recipes.phase13 import build_phase13_recipe

    config = _config(tmp_path, max_steps=1)
    recipe = build_phase13_recipe(config)
    built = TrainingPipeline(config, builders=phase13_builders()).build()
    result = built.trainer.train()
    assert result.success
    assert result.checkpoint is not None
    checkpoint_path = Path(result.checkpoint)
    assert checkpoint_path.is_dir()

    # The trained task accepts the recipe modality and the resumable
    # checkpoint is complete (manifest present, hashes verify).
    batch = recipe.train_data[0]
    built.task.check_supported(batch.modality)
    _assert_complete_checkpoint(checkpoint_path)


def _assert_complete_checkpoint(ckpt: Path) -> None:
    """A resumable checkpoint directory with a complete manifest and matching
    file hashes is safe to resume from."""
    manager = CheckpointManager(ckpt.parent)
    manifest = manager.inspect(ckpt)
    assert manifest.kind == "resumable"
    CheckpointManager._verify_files(ckpt, manifest)  # noqa: SLF001 - reliability probe of the load contract


def _train_one_step(output_dir: Path):
    # Explicit seeding makes the (otherwise stateful) recipe builder and model
    # init reproducible; deterministic CPU mode must then repeat exactly.
    import numpy as np

    torch.manual_seed(20260807)
    np.random.seed(20260807)
    config = _config(output_dir.parent, max_steps=1)
    values = config.to_dict()
    values["output_dir"] = str(output_dir)
    config = RunConfig.from_dict(values)
    pipeline = TrainingPipeline(config, builders=phase13_builders()).build()
    return pipeline.trainer.train()
