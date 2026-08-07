from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from medfm.training.checkpoint import IncompleteCheckpointError
from medfm.training.config import FreezeStageConfig
from medfm.training.memory import CUDA_OOM_SUGGESTIONS, CompilationMonitor, TpuMemoryPlanner, diagnose_oom
from medfm.training.optimizer import build_optimizer
from medfm.training.steps import ClassificationTrainingStep
from medfm.training.trainer import Trainer


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = torch.nn.Linear(1, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, batch):
        values = batch.pixel_values.flatten(1).mean(dim=1, keepdim=True)
        return {"logits": self.classifier(torch.tanh(self.bridge(values)))}


def _trainer(tiny_config, tiny_batch, tiny_task, *, steps: int = 1, accumulation: int = 1) -> Trainer:
    config = replace(
        tiny_config,
        max_steps=steps,
        batch=replace(tiny_config.batch, gradient_accumulation_steps=accumulation),
    )
    model = TinyClassifier()
    optimizer = build_optimizer(model, config.optimizer, backend="cpu")
    return Trainer(
        model,
        optimizer,
        tiny_task,
        [tiny_batch, tiny_batch, tiny_batch],
        config,
        training_step=ClassificationTrainingStep(tiny_task),
    )


def test_accumulation_counts_optimizer_steps(tiny_config, tiny_batch, tiny_task) -> None:
    trainer = _trainer(tiny_config, tiny_batch, tiny_task, steps=2, accumulation=2)
    result = trainer.train()
    assert result.success
    assert result.optimizer_steps == 2
    assert result.effective_batch_size == 4


def test_freeze_schedule_keeps_classifier_frozen_at_boundary(tiny_config, tiny_batch, tiny_task) -> None:
    config = replace(
        tiny_config,
        max_steps=1,
        freeze_schedule=(
            FreezeStageConfig(until_step=1, train=("bridge",)),
            FreezeStageConfig(until_step=None, train=("bridge", "task_head")),
        ),
    )
    model = TinyClassifier()
    classifier_before = model.classifier.weight.detach().clone()
    optimizer = build_optimizer(model, config.optimizer, backend="cpu")
    trainer = Trainer(
        model, optimizer, tiny_task, [tiny_batch], config, training_step=ClassificationTrainingStep(tiny_task)
    )
    trainer.train()
    assert torch.equal(classifier_before, model.classifier.weight.detach())


def test_oom_diagnostic_is_ordered_and_does_not_mutate_config(tiny_config) -> None:
    before = tiny_config.config_hash()
    diagnostic = diagnose_oom(RuntimeError("CUDA out of memory"), backend="cuda", run_config=tiny_config)
    assert diagnostic.suggestions == CUDA_OOM_SUGGESTIONS
    assert not diagnostic.scientific_configuration_mutated
    assert tiny_config.config_hash() == before


def test_tpu_planner_never_uses_cuda_memory_api() -> None:
    planner = TpuMemoryPlanner()
    assert planner.backend_name == "xla_tpu"
    assert planner.budget_bytes(None) is None


def test_recompilation_gate_names_bucket_and_sample() -> None:
    monitor = CompilationMonitor(warmup_steps=1, max_steady_state_compilations=0, fail=True)
    monitor.observe(step=0, metrics={"compilation_count": 1}, bucket="small", sample="warmup")
    with pytest.raises(Exception, match="bucket='large'.*sample='sample-7'"):
        monitor.observe(step=1, metrics={"compilation_count": 2}, bucket="large", sample="sample-7")


def test_checkpoint_corruption_is_rejected(tmp_path, tiny_config, tiny_batch, tiny_task) -> None:
    trainer = _trainer(tiny_config, tiny_batch, tiny_task, steps=1)
    trainer.train()
    checkpoint = trainer.checkpoint_manager.path_for("last")
    model_file = checkpoint / "model.pt"
    model_file.write_bytes(model_file.read_bytes() + b"corrupt")
    with pytest.raises(IncompleteCheckpointError, match="corrupt"):
        trainer.checkpoint_manager.inspect(checkpoint)


def test_adapter_export_is_cpu_safetensors(tmp_path, tiny_config, tiny_batch, tiny_task) -> None:
    trainer = _trainer(tiny_config, tiny_batch, tiny_task, steps=1)
    trainer.train()
    exported = trainer.export_adapter("portable")
    assert (exported / "adapter.safetensors").exists()
    assert trainer.checkpoint_manager.inspect(exported).kind == "adapter_only"


def test_resuming_completed_checkpoint_is_a_noop(tiny_config, tiny_batch, tiny_task) -> None:
    first = _trainer(tiny_config, tiny_batch, tiny_task, steps=1)
    completed = first.train()
    assert completed.success

    resumed = _trainer(tiny_config, tiny_batch, tiny_task, steps=1)
    resumed.resume(first.checkpoint_manager.path_for("last"))
    result = resumed.train()

    assert result.success
    assert result.global_step == 1
    assert result.optimizer_steps == 1
