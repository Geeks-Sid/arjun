"""Unified resumable trainer with task-specific steps and explicit resource policy."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from medfm.core.task import TaskModule
from medfm.training.backend import AcceleratorBackend, create_backend
from medfm.training.checkpoint import CheckpointManager, CheckpointState
from medfm.training.config import RunConfig
from medfm.training.data import bucket_name
from medfm.training.evaluation import EvaluationResult, Evaluator
from medfm.training.memory import (
    CompilationMonitor,
    MemoryPlan,
    MemoryPlanner,
    OOMDiagnostic,
    diagnose_oom,
    enforce_memory_plan,
    planner_for_backend,
)
from medfm.training.optimizer import (
    FreezeSchedule,
    OptimizerBundle,
    apply_freeze_schedule,
    audit_gradients,
    audit_trainable_parameters,
    build_optimizer,
    rebuild_optimizer,
)
from medfm.training.run_metadata import RunMetadata, capture_run_metadata
from medfm.training.steps import TrainingStep, make_training_step
from medfm.training.tracking import (
    FailureReporter,
    LocalJSONTracker,
    Tracker,
    assert_finite_loss,
)


class TrainingError(RuntimeError):
    """Base trainer failure."""


class TrainingOOMError(TrainingError):
    """OOM with an ordered diagnostic; configuration was not mutated."""

    def __init__(self, diagnostic: OOMDiagnostic) -> None:
        super().__init__(diagnostic.render())
        self.diagnostic = diagnostic


@dataclass
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    micro_step: int = 0
    batch_in_epoch: int = 0
    optimizer_steps: int = 0
    best_metric: float | None = None
    best_criterion: str | None = None
    interrupted: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    stage: tuple[str, ...] = ()

    def to_checkpoint(self) -> CheckpointState:
        return CheckpointState(
            global_step=self.global_step,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            micro_step=self.micro_step,
            best_metric=self.best_metric,
            best_criterion=self.best_criterion,
            interrupted=self.interrupted,
            metrics=dict(self.metrics),
            stage=self.stage,
        )

    def load(self, state: CheckpointState) -> None:
        self.global_step = state.global_step
        self.epoch = state.epoch
        self.batch_in_epoch = state.batch_in_epoch
        self.micro_step = state.micro_step
        self.optimizer_steps = state.global_step
        self.best_metric = state.best_metric
        self.best_criterion = state.best_criterion
        self.interrupted = state.interrupted
        self.stage = tuple(state.stage)
        self.metrics = dict(state.metrics)


@dataclass(frozen=True)
class TrainingResult:
    success: bool
    interrupted: bool
    global_step: int
    optimizer_steps: int
    epoch: int
    metrics: dict[str, float]
    backend: str
    effective_batch_size: int
    peak_memory: dict[str, Any]
    checkpoint: str | None = None
    oom_diagnostic: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "interrupted": self.interrupted,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "epoch": self.epoch,
            "metrics": dict(self.metrics),
            "backend": self.backend,
            "effective_batch_size": self.effective_batch_size,
            "peak_memory": dict(self.peak_memory),
            "checkpoint": self.checkpoint,
            "oom_diagnostic": self.oom_diagnostic,
            "metadata": dict(self.metadata),
        }


class Trainer:
    """One loop for CPU, CUDA, DDP/FSDP, and replicated XLA runs."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizerBundle | torch.optim.Optimizer,
        task: TaskModule,
        train_dataloader: Any,
        config: RunConfig | None = None,
        *,
        backend: AcceleratorBackend | None = None,
        training_step: TrainingStep | None = None,
        validation_dataloader: Any | None = None,
        evaluator: Evaluator | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        tracker: Tracker | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        memory_planner: MemoryPlanner | None = None,
        run_metadata: RunMetadata | None = None,
        command: list[str] | None = None,
        role_map: Mapping[str, str] | None = None,
        require_all_gradients: bool = False,
    ) -> None:
        self.config = config or RunConfig()
        self.backend = backend or create_backend(
            self.config.accelerator,
            gradient_accumulation_steps=self.config.batch.gradient_accumulation_steps,
        )
        if (
            self.backend.uses_accelerate
            and self.backend.gradient_accumulation_steps != self.config.batch.gradient_accumulation_steps
        ):
            raise TrainingError(
                "backend and RunConfig disagree on gradient_accumulation_steps; "
                "construct the backend from the resolved RunConfig"
            )
        self.model = model
        self.task = task
        self.optimizer_components = {"task": task} if isinstance(task, nn.Module) else {}
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.training_step = training_step or make_training_step(task)
        self.role_map = dict(role_map or {})
        self.require_all_gradients = require_all_gradients
        self.state = TrainingState()
        self.freeze_schedule = FreezeSchedule.from_config(self.config.freeze_schedule)
        self.optimizer_bundle: OptimizerBundle | None
        self.scaler = scaler
        if isinstance(optimizer, OptimizerBundle):
            self.optimizer_bundle = optimizer
            self.optimizer = optimizer.optimizer
            self.scheduler = scheduler if scheduler is not None else optimizer.scheduler
        else:
            self.optimizer_bundle = None
            self.optimizer = optimizer
            self.scheduler = scheduler
        self._prepared = False
        self._last_safe_checkpoint: Path | None = None
        self._oom_diagnostic: OOMDiagnostic | None = None
        self._last_metrics: dict[str, float] = {}
        self.memory_planner = memory_planner or planner_for_backend(self.backend.name, self.config.memory)
        self.compilation_monitor = (
            CompilationMonitor(
                warmup_steps=self.config.accelerator.warmup_steps,
                max_steady_state_compilations=self.config.accelerator.max_steady_state_compilations,
                fail=self.config.accelerator.fail_on_recompilation_after_warmup,
            )
            if self.backend.name == "xla_tpu"
            else None
        )
        output_dir = Path(self.config.output_dir)
        self.run_dir = output_dir / self.config.config_hash()[:16]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker or LocalJSONTracker(self.run_dir / "tracking")
        self.failure_reporter = FailureReporter(self.run_dir)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            self.run_dir / "checkpoints",
            backend=self.backend,
        )
        self._prepare_components()
        self.memory_plan: MemoryPlan = self.memory_planner.plan(
            self.model,
            optimizer=self.optimizer,
        )
        enforce_memory_plan(self.memory_plan)
        self._prepare_runtime_components()
        self.command = list(command or sys.argv)
        self.run_metadata = run_metadata or capture_run_metadata(
            accelerator_backend=self.backend.name,
            seed=self.config.seed,
            precision=self.config.accelerator.precision,
            microbatch_per_device=self.config.batch.microbatch_per_device,
            gradient_accumulation_steps=self.config.batch.gradient_accumulation_steps,
            model=self.model,
            components=self.optimizer_components,
            world_size=self.backend.topology.world_size,
            base_model_revision=self.config.base_model_revision,
            adapter_config=self.config.peft.to_dict(),
            dataset_manifest_sha256=self.config.dataset_hash,
            preprocessing_config={"hash": self.config.preprocessing_hash},
            shape_buckets=self.config.extensions.get("shape_buckets", {}),
            rank=self.backend.topology.rank,
            local_rank=self.backend.topology.local_rank,
            host_index=self.backend.topology.host_index,
            sharding_mesh=self.backend.topology.sharding_mesh,
            extra={"topology": self.backend.topology.to_dict(), "config_hash": self.config.config_hash()},
        )
        self.tracker.log_params(
            {
                "run_config": self.config.to_dict(),
                "config_hash": self.config.config_hash(),
                "run_metadata": self.run_metadata.to_dict(),
                "backend_capabilities": self.backend.capabilities.to_dict(),
                "optimizer_groups": self.optimizer_bundle.group_summary() if self.optimizer_bundle else [],
                "memory_plan": self.memory_plan.to_dict(),
            }
        )

    def _prepare_components(self) -> None:
        if self._prepared:
            return
        self.model = self.backend.configure_model_for_training(
            self.model,
            use_cache=self.config.memory.use_cache_during_training,
            gradient_checkpointing=any(self.config.memory.gradient_checkpointing.values()),
        )
        # Apply stage zero before any optimizer state can be allocated.
        self._apply_stage(0, force=True)
        component_parameter_ids = {
            id(parameter)
            for component in self.optimizer_components.values()
            for parameter in component.parameters()
            if parameter.requires_grad
        }
        optimizer_parameter_ids = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        if component_parameter_ids - optimizer_parameter_ids:
            if self.optimizer_bundle is not None:
                self.optimizer_bundle = rebuild_optimizer(
                    self.model,
                    self.optimizer_bundle,
                    backend=self.backend.name,
                    role_map=self.role_map,
                    components=self.optimizer_components,
                )
            else:
                self.optimizer_bundle = build_optimizer(
                    self.model,
                    self.config.optimizer,
                    backend=self.backend.name,
                    role_map=self.role_map,
                    components=self.optimizer_components,
                )
            self.optimizer = self.optimizer_bundle.optimizer
            self.scheduler = self.optimizer_bundle.scheduler
        self._prepared = True

    def _prepare_runtime_components(self) -> None:
        """Place coupled runtime objects after freeze/optimizer construction."""
        if isinstance(self.task, nn.Module):
            self.task = self.task.to(self.backend.device)
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.validation_dataloader,
            self.scheduler,
        ) = self.backend.prepare_training(
            self.model,
            self.optimizer,
            self.train_dataloader,
            scheduler=self.scheduler,
            validation_dataloader=self.validation_dataloader,
        )
        if self.optimizer_bundle is not None:
            self.optimizer_bundle.optimizer = self.optimizer
            self.optimizer_bundle.scheduler = self.scheduler

    def _apply_stage(self, step: int, *, force: bool = False, rebuild: bool = True) -> None:
        active = self.freeze_schedule.trainable_roles(step)
        if not active:
            return
        if not force and tuple(active) == self.state.stage:
            return
        apply_freeze_schedule(
            self.model,
            self.freeze_schedule,
            step,
            role_map=self.role_map,
            components=self.optimizer_components,
        )
        audit_trainable_parameters(
            self.model,
            expected_roles=active,
            role_map=self.role_map,
            components=self.optimizer_components,
        )
        if rebuild and self.optimizer_bundle is not None:
            self.optimizer_bundle = rebuild_optimizer(
                self.model,
                self.optimizer_bundle,
                backend=self.backend.name,
                role_map=self.role_map,
                components=self.optimizer_components,
            )
            self.optimizer = self.optimizer_bundle.optimizer
            self.scheduler = self.optimizer_bundle.scheduler
        elif rebuild and self.optimizer_bundle is None:
            self.optimizer_bundle = build_optimizer(
                self.model,
                self.config.optimizer,
                backend=self.backend.name,
                role_map=self.role_map,
                components=self.optimizer_components,
            )
            self.optimizer = self.optimizer_bundle.optimizer
            self.scheduler = self.optimizer_bundle.scheduler
        if rebuild and self._prepared:
            self.optimizer, self.scheduler = self.backend.prepare_optimizer_scheduler(
                self.optimizer,
                self.scheduler,
            )
            if self.optimizer_bundle is not None:
                self.optimizer_bundle.optimizer = self.optimizer
                self.optimizer_bundle.scheduler = self.scheduler
        self.state.stage = tuple(active)

    def _loss_for_backward(self, loss_output: Any) -> torch.Tensor:
        loss_total = cast(torch.Tensor, loss_output.total)
        total = loss_total.float() if loss_total.dtype in (torch.float16, torch.bfloat16) else loss_total
        count_value = (
            loss_output.diagnostics.get("valid_count") if isinstance(loss_output.diagnostics, Mapping) else None
        )
        if not isinstance(count_value, torch.Tensor):
            count_value = torch.as_tensor(max(0, int(loss_output.sample_count)), dtype=total.dtype, device=total.device)
        count_value = count_value.to(device=total.device, dtype=total.dtype)
        global_count = self.backend.reduce_sum(count_value)
        # DDP/XLA gradient reductions average replicas. Reweight each local
        # mean by world size so the resulting gradient is the true global
        # supervised-example mean rather than a second world-size average.
        world_size = self.backend.topology.world_size if self.backend.topology.world_size > 1 else 1
        divisor = (
            self.backend.gradient_accumulation_steps
            if self.backend.uses_accelerate
            else self.config.batch.gradient_accumulation_steps
        )
        normalized = total * count_value / global_count.clamp_min(1.0)
        return normalized * float(world_size) / float(divisor)

    def _optimizer_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.model.parameters())
        for component in self.optimizer_components.values():
            parameters.extend(component.parameters())
        return parameters

    def _optimizer_boundary(self) -> None:
        gradient_report = audit_gradients(
            self.model,
            role_map=self.role_map,
            require_all_gradients=self.require_all_gradients,
            components=self.optimizer_components,
        )
        clip_norm: float | None = None
        if self.config.optimizer.max_grad_norm is not None:
            value = self.backend.clip_grad_norm(
                self._optimizer_parameters(),
                self.config.optimizer.max_grad_norm,
                optimizer=self.optimizer,
            )
            clip_norm = float(value.detach().cpu())
        self.backend.optimizer_step(self.optimizer)
        if self.scheduler is not None:
            self.scheduler.step()
        self.backend.zero_grad(self.optimizer)
        self.backend.mark_step()
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        metrics = {f"grad_norm/{name}": value for name, value in gradient_report.component_norms.items()}
        if clip_norm is not None:
            metrics["grad_norm/global_pre_clip"] = clip_norm
        self.tracker.log_metrics(metrics, self.state.global_step)

    def _record_loss(self, loss_output: Any) -> None:
        total = float(loss_output.total.detach().float().cpu())
        self._last_metrics["train/loss"] = total
        for name, value in loss_output.component_dict().items():
            self._last_metrics[f"train/loss/{name}"] = float(value)
        self.state.metrics.update(self._last_metrics)

    def _reset_task_metrics(self) -> None:
        reset = getattr(self.task, "reset_metrics", None)
        if callable(reset):
            reset()

    def _update_task_metrics(self, batch: Any) -> None:
        update = getattr(self.task, "update_metrics", None)
        output = getattr(self.training_step, "last_model_output", None)
        if callable(update) and output is not None:
            update(output, batch)

    def _finish_task_metrics(self) -> None:
        compute = getattr(self.task, "compute_metrics", None)
        if callable(compute):
            metrics = compute()
            if isinstance(metrics, Mapping):
                self.state.metrics.update({f"train/{key}": float(value) for key, value in metrics.items()})

    def _observe_compilation(self, batch: Any) -> None:
        if self.compilation_monitor is None:
            return
        sample_ids = getattr(batch, "sample_ids", ())
        sample = str(sample_ids[0]) if sample_ids else None
        self.compilation_monitor.observe(
            step=self.state.global_step,
            metrics=self.backend.runtime_metrics(),
            bucket=bucket_name(batch),
            sample=sample,
        )

    def train(self, *, max_steps: int | None = None) -> TrainingResult:
        target_steps = max_steps if max_steps is not None else self.config.max_steps
        if target_steps is not None and target_steps < self.state.global_step:
            raise TrainingError("max_steps is smaller than the resumed global step")
        self.model.train()
        self.backend.zero_grad(self.optimizer)
        pending = 0
        supported = False
        last_batch_seen = target_steps is not None and self.state.global_step >= target_steps
        try:
            for epoch in range(self.state.epoch, self.config.epochs):
                _set_epoch(self.train_dataloader, epoch)
                _set_epoch(self.validation_dataloader, epoch)
                self._reset_task_metrics()
                epoch_complete = True
                for batch_index, raw_batch in enumerate(self.train_dataloader):
                    if batch_index < self.state.batch_in_epoch:
                        continue
                    last_batch_seen = True
                    if target_steps is not None and self.state.global_step >= target_steps:
                        epoch_complete = False
                        break
                    self._apply_stage(self.state.global_step)
                    batch = self.backend.prepare_batch(raw_batch)
                    if not supported:
                        check_supported = getattr(self.task, "check_supported", None)
                        if callable(check_supported):
                            check_supported(batch.modality)
                    with self.backend.autocast():
                        loss_output = self.training_step.forward_and_loss(self.model, batch)
                    # Every rank participates in this guard before the true-count
                    # loss collective, so one bad replica cannot strand peers.
                    local_bad = (
                        not isinstance(loss_output.total, torch.Tensor)
                        or loss_output.total.ndim != 0
                        or not bool(torch.isfinite(loss_output.total.detach()).all())
                    )
                    if self.backend.topology.world_size > 1:
                        from medfm.training.distributed import synchronize_failure

                        any_bad = synchronize_failure(self.backend, local_bad)
                        if any_bad and not local_bad:
                            raise TrainingError("another rank reported a non-finite loss")
                    # This check deliberately precedes the true-count collective
                    # in _loss_for_backward.
                    assert_finite_loss(loss_output.total, step=self.state.global_step)
                    backward_loss = self._loss_for_backward(loss_output)
                    self.backend.backward(backward_loss)
                    pending += 1
                    self.state.micro_step += 1
                    self.state.batch_in_epoch = batch_index + 1
                    self._record_loss(loss_output)
                    self._update_task_metrics(batch)
                    self.backend.mark_step()
                    self._observe_compilation(batch)
                    if pending >= self.config.batch.gradient_accumulation_steps:
                        self._optimizer_boundary()
                        pending = 0
                        if self.config.save_every_steps and self.state.global_step % self.config.save_every_steps == 0:
                            self._last_safe_checkpoint = self.save_checkpoint("last")
                        if target_steps is not None and self.state.global_step >= target_steps:
                            epoch_complete = False
                            break
                self._finish_task_metrics()
                # A final partial accumulation is an explicit flush, not a
                # silent drop of gradients or an extra scientific batch.
                if pending and (target_steps is None or self.state.global_step < target_steps):
                    self._optimizer_boundary()
                    pending = 0
                if epoch_complete:
                    self.state.epoch = epoch + 1
                    self.state.batch_in_epoch = 0
                if target_steps is not None and self.state.global_step >= target_steps:
                    break
            if not last_batch_seen and target_steps not in (None, 0):
                raise TrainingError("training dataloader produced no batches")
            self._last_safe_checkpoint = self.save_checkpoint("last")
            if self.validation_dataloader is not None:
                evaluation = self.validate()
                self.state.metrics.update({f"val/{k}": v for k, v in evaluation.metrics.items()})
            self.tracker.log_metrics(self.state.metrics, self.state.global_step)
            return self._result(success=True)
        except KeyboardInterrupt:
            self.state.interrupted = True
            try:
                self._last_safe_checkpoint = self.save_checkpoint("interrupted")
            except Exception:
                pass
            return self._result(success=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _looks_like_oom(exc):
                self._report_failure(exc)
                raise
            diagnostic = diagnose_oom(
                exc,
                backend=self.backend.name,
                run_config=self.config,
                snapshot=self.backend.memory_snapshot(),
            )
            self._oom_diagnostic = diagnostic
            self._report_failure(exc, extra={"oom_diagnostic": diagnostic.to_dict()})
            raise TrainingOOMError(diagnostic) from exc
        except BaseException as exc:
            self._report_failure(exc)
            raise
        finally:
            self.tracker.close()

    def validate(self, *, max_batches: int | None = None) -> EvaluationResult:
        if self.validation_dataloader is None:
            raise TrainingError("validation dataloader was not configured")
        evaluator = self.evaluator
        if evaluator is None:
            evaluator = Evaluator(
                model=self.model,
                task=self.task,
                step=self.training_step,
                backend=self.backend,
            )
        result = evaluator.evaluate(
            self.validation_dataloader,
            global_step=self.state.global_step,
            max_batches=max_batches,
        )
        self.tracker.log_metrics(
            {f"val/{key}": value for key, value in result.metrics.items()},
            self.state.global_step,
        )
        return result

    @property
    def evaluator(self) -> Evaluator | None:
        return getattr(self, "_evaluator", None)

    @evaluator.setter
    def evaluator(self, value: Evaluator | None) -> None:
        self._evaluator = value

    def save_checkpoint(self, name: str | int = "last") -> Path:
        backend_info = {
            "backend": self.backend.name,
            "world_size": self.backend.topology.world_size,
            "topology": self.backend.topology.to_dict(),
            "sharding_mesh": self.backend.topology.sharding_mesh,
            "compiler_runtime": {
                **self.backend.runtime_metrics(),
                "compilation_monitor": (
                    self.compilation_monitor.to_dict() if self.compilation_monitor is not None else {}
                ),
            },
            "memory_plan": self.memory_plan.to_dict(),
        }
        path = self.checkpoint_manager.save(
            name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            sampler_state=_loader_state(self.train_dataloader),
            state=self.state.to_checkpoint(),
            run_config=self.config,
            metrics=self.state.metrics,
            best_criterion=self.state.best_criterion,
            backend_metadata=backend_info,
            static_bucket_schema=self.config.extensions.get("shape_buckets", {}),
            components={"task": self.task} if isinstance(self.task, nn.Module) else None,
        )
        return path

    def resume(self, checkpoint: str | Path, *, allow_topology_change: bool = False) -> TrainingState:
        state = self.checkpoint_manager.load(
            checkpoint,
            model=self.model,
            run_config=self.config,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            components={"task": self.task} if isinstance(self.task, nn.Module) else None,
            sampler=_loader_state_target(self.train_dataloader),
            allow_topology_change=allow_topology_change,
        )
        self.state.load(state)
        self._apply_stage(self.state.global_step, force=True, rebuild=False)
        return self.state

    def export_adapter(self, name: str | int = "adapter") -> Path:
        return self.checkpoint_manager.export_adapter(
            name,
            model=self.model,
            run_config=self.config,
            components={"task": self.task} if isinstance(self.task, nn.Module) else None,
        )

    def _report_failure(self, error: BaseException, *, extra: dict[str, Any] | None = None) -> None:
        try:
            self.failure_reporter.write(
                error,
                config=self.config.to_dict(),
                command=self.command,
            )
        except Exception:
            pass

    def _result(self, *, success: bool) -> TrainingResult:
        snapshot = self.backend.memory_snapshot().to_dict()
        metadata = self.run_metadata.to_dict()
        metadata["runtime_backend"] = self.backend.runtime_metrics()
        metadata["memory_plan"] = self.memory_plan.to_dict()
        return TrainingResult(
            success=success,
            interrupted=self.state.interrupted,
            global_step=self.state.global_step,
            optimizer_steps=self.state.optimizer_steps,
            epoch=self.state.epoch,
            metrics=dict(self.state.metrics),
            backend=self.backend.name,
            effective_batch_size=self.config.batch.resolved_global_batch(self.backend.topology.world_size),
            peak_memory=snapshot,
            checkpoint=str(self._last_safe_checkpoint) if self._last_safe_checkpoint is not None else None,
            oom_diagnostic=self._oom_diagnostic.to_dict() if self._oom_diagnostic is not None else None,
            metadata=metadata,
        )


def _set_epoch(loader: Any, epoch: int) -> None:
    for candidate in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None), loader):
        setter = getattr(candidate, "set_epoch", None)
        if callable(setter):
            setter(epoch)


def _loader_state(loader: Any) -> dict[str, Any] | None:
    for candidate in (loader, getattr(loader, "sampler", None)):
        state_dict = getattr(candidate, "state_dict", None)
        if callable(state_dict):
            return dict(state_dict())
    return None


def _loader_state_target(loader: Any) -> Any | None:
    if callable(getattr(loader, "load_state_dict", None)):
        return loader
    sampler = getattr(loader, "sampler", None)
    return sampler if callable(getattr(sampler, "load_state_dict", None)) else None


def _looks_like_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message or "resource exhausted" in message


__all__ = [
    "Trainer",
    "TrainingError",
    "TrainingOOMError",
    "TrainingResult",
    "TrainingState",
]
