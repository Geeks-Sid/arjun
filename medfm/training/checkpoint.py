"""Atomic resumable checkpoints and accelerator-neutral adapter exports."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medfm.peft.checkpoint import _write_safetensors
from medfm.training.config import RunConfig

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base checkpoint failure."""


class IncompleteCheckpointError(CheckpointError):
    """A checkpoint is missing its completion marker or required files."""


class CheckpointCompatibilityError(CheckpointError):
    """A checkpoint belongs to a different scientific/runtime contract."""


@dataclass(frozen=True)
class CheckpointState:
    global_step: int = 0
    epoch: int = 0
    micro_step: int = 0
    batch_in_epoch: int = 0
    best_metric: float | None = None
    best_criterion: str | None = None
    interrupted: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    stage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "micro_step": self.micro_step,
            "batch_in_epoch": self.batch_in_epoch,
            "best_metric": self.best_metric,
            "best_criterion": self.best_criterion,
            "interrupted": self.interrupted,
            "metrics": dict(self.metrics),
            "stage": list(self.stage),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointState:
        return cls(
            global_step=int(data.get("global_step", 0)),
            epoch=int(data.get("epoch", 0)),
            micro_step=int(data.get("micro_step", 0)),
            batch_in_epoch=int(data.get("batch_in_epoch", 0)),
            best_metric=(float(data["best_metric"]) if data.get("best_metric") is not None else None),
            best_criterion=(str(data["best_criterion"]) if data.get("best_criterion") is not None else None),
            interrupted=bool(data.get("interrupted", False)),
            metrics={str(key): float(value) for key, value in dict(data.get("metrics", {})).items()},
            stage=tuple(str(value) for value in data.get("stage", ())),
        )


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    kind: str
    complete: bool
    run_config_hash: str
    model_id: str
    base_model_revision: str | None
    backend: str
    precision: str
    distribution: str
    world_size: int
    topology: dict[str, Any]
    sharding_mesh: dict[str, Any]
    compiler_runtime: dict[str, Any]
    static_bucket_schema: dict[str, Any]
    files: dict[str, str]
    metrics: dict[str, float]
    best_criterion: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "complete": self.complete,
            "run_config_hash": self.run_config_hash,
            "model_id": self.model_id,
            "base_model_revision": self.base_model_revision,
            "backend": self.backend,
            "precision": self.precision,
            "distribution": self.distribution,
            "world_size": self.world_size,
            "topology": dict(self.topology),
            "sharding_mesh": dict(self.sharding_mesh),
            "compiler_runtime": dict(self.compiler_runtime),
            "static_bucket_schema": dict(self.static_bucket_schema),
            "files": dict(self.files),
            "metrics": dict(self.metrics),
            "best_criterion": self.best_criterion,
            "extra": dict(self.extra),
        }


class CheckpointManager:
    """Write complete checkpoints via temp-directory + atomic rename."""

    def __init__(self, root: str | Path, *, backend: Any | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def path_for(self, name: str | int) -> Path:
        value = str(name)
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise CheckpointError(f"unsafe checkpoint name {name!r}")
        return self.root / value

    def save(
        self,
        name: str | int,
        *,
        model: nn.Module,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        state: CheckpointState | Mapping[str, Any] | None = None,
        run_config: RunConfig,
        metrics: Mapping[str, float] | None = None,
        best_criterion: str | None = None,
        sampler_state: Mapping[str, Any] | None = None,
        base_references: Mapping[str, Any] | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
        static_bucket_schema: Mapping[str, Any] | None = None,
        components: Mapping[str, nn.Module] | None = None,
        adapter_only: bool = False,
        overwrite: bool = True,
    ) -> Path:
        target = self.path_for(name)
        if target.exists() and not overwrite:
            raise CheckpointError(f"checkpoint already exists: {target}")
        if self._skip_non_coordinator(run_config):
            self._synchronize()
            return target
        state_obj = state if isinstance(state, CheckpointState) else CheckpointState.from_dict(state or {})
        if metrics:
            state_obj = CheckpointState(
                **{**state_obj.to_dict(), "metrics": {str(k): float(v) for k, v in metrics.items()}},
            )
        use_distributed_storage = _requires_distributed_checkpoint(run_config, adapter_only=adapter_only)
        storage_format = "torch_distributed_checkpoint" if use_distributed_storage else "torch"
        tmp = (
            self._shared_distributed_tmp(target, run_config)
            if use_distributed_storage
            else Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(self.root)))
        )
        try:
            files: dict[str, str] = {}
            state_dict = _combined_state_dict(model, components)
            if adapter_only:
                model_file = tmp / "adapter.safetensors"
                tensors = _trainable_tensors(model, components)
                if not tensors:
                    raise CheckpointError("adapter-only export has no trainable adapter/component tensors")
                _write_safetensors(tensors, model_file, metadata={"model_id": run_config.model_id})
                files[model_file.name] = _sha256(model_file)
            elif use_distributed_storage:
                dcp_root = tmp / "dcp"
                dcp_root.mkdir(parents=True, exist_ok=True)
                _save_distributed_checkpoint(
                    _distributed_state_dict(
                        model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        components=components,
                    ),
                    dcp_root,
                    xla=run_config.accelerator.backend == "xla_tpu",
                )
                files.update(_file_hashes(dcp_root, relative_to=tmp))
                self._synchronize_distributed()
                if not self._is_distributed_coordinator(run_config):
                    self._synchronize_distributed()
                    return target
            else:
                model_file = tmp / "model.pt"
                torch.save(_cpu_state_dict(state_dict), model_file)
                files[model_file.name] = _sha256(model_file)

            if not adapter_only:
                if not use_distributed_storage and optimizer is not None:
                    optimizer_state = optimizer.state_dict() if hasattr(optimizer, "state_dict") else optimizer
                    path = tmp / "optimizer.pt"
                    torch.save(_cpu_nested(optimizer_state), path)
                    files[path.name] = _sha256(path)
                if scheduler is not None and hasattr(scheduler, "state_dict"):
                    path = tmp / "scheduler.pt"
                    torch.save(_cpu_nested(scheduler.state_dict()), path)
                    files[path.name] = _sha256(path)
                if scaler is not None and hasattr(scaler, "state_dict"):
                    path = tmp / "scaler.pt"
                    torch.save(_cpu_nested(scaler.state_dict()), path)
                    files[path.name] = _sha256(path)
                path = tmp / "rng.pt"
                torch.save(capture_rng_state(backend=self.backend), path)
                files[path.name] = _sha256(path)
                if sampler_state is not None:
                    path = tmp / "sampler.pt"
                    torch.save(_cpu_nested(dict(sampler_state)), path)
                    files[path.name] = _sha256(path)

            state_path = tmp / "training_state.json"
            state_path.write_text(json.dumps(state_obj.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            files[state_path.name] = _sha256(state_path)

            backend_info = dict(backend_metadata or {})
            topology = dict(backend_info.get("topology", {}))
            backend_name = str(backend_info.get("backend", run_config.accelerator.backend))
            world_size = int(backend_info.get("world_size", run_config.accelerator.world_size))
            manifest = CheckpointManifest(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                kind="adapter_only" if adapter_only else "resumable",
                complete=False,
                run_config_hash=run_config.config_hash(),
                model_id=run_config.model_id,
                base_model_revision=run_config.base_model_revision,
                backend=backend_name,
                precision=run_config.accelerator.precision,
                distribution=run_config.accelerator.distribution,
                world_size=world_size,
                topology=topology,
                sharding_mesh=dict(backend_info.get("sharding_mesh", run_config.accelerator.sharding_mesh)),
                compiler_runtime=dict(backend_info.get("compiler_runtime", {})),
                static_bucket_schema=dict(static_bucket_schema or {}),
                files=files,
                metrics={str(k): float(v) for k, v in state_obj.metrics.items()},
                best_criterion=best_criterion or state_obj.best_criterion,
                extra={
                    "adapter_components": sorted(_component_names(_trainable_tensors(model, components))),
                    "base_references": dict(base_references or {}),
                    "run_config": run_config.to_dict(),
                    "storage_format": storage_format,
                },
            )
            # The complete marker is written last in the temp tree.  Readers
            # never accept a directory without it and matching file hashes.
            manifest_path = tmp / "manifest.json"
            complete = CheckpointManifest(**{**manifest.__dict__, "complete": True})
            manifest_path.write_text(json.dumps(complete.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
            if use_distributed_storage or self._is_xla_replicated(run_config):
                self._synchronize_distributed() if use_distributed_storage else self._synchronize()
            return target
        except BaseException:
            if not use_distributed_storage or self._is_distributed_coordinator(run_config):
                shutil.rmtree(tmp, ignore_errors=True)
            raise

    save_checkpoint = save

    def _is_xla_replicated(self, run_config: RunConfig) -> bool:
        return run_config.accelerator.backend == "xla_tpu" and run_config.accelerator.distribution == "replicated"

    def _skip_non_coordinator(self, run_config: RunConfig) -> bool:
        if not self._is_xla_replicated(run_config):
            return False
        topology = getattr(self.backend, "topology", None)
        rank = getattr(topology, "rank", run_config.accelerator.rank)
        return int(rank) != 0

    def _is_distributed_coordinator(self, run_config: RunConfig) -> bool:
        topology = getattr(self.backend, "topology", None)
        if topology is not None:
            return int(getattr(topology, "rank", 0)) == 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return run_config.accelerator.rank == 0

    def _shared_distributed_tmp(self, target: Path, run_config: RunConfig) -> Path:
        temporary = self.root / f".{target.name}.dcp-tmp"
        if self._is_distributed_coordinator(run_config):
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=True)
        self._synchronize_distributed()
        if not temporary.exists():
            raise CheckpointError(f"distributed checkpoint temporary directory was not created: {temporary}")
        return temporary

    def _synchronize_distributed(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        else:
            self._synchronize()

    def _synchronize(self) -> None:
        synchronize = getattr(self.backend, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def inspect(self, path: str | Path) -> CheckpointManifest:
        root = Path(path)
        manifest = self._read_manifest(root)
        self._verify_files(root, manifest)
        return manifest

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        run_config: RunConfig,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        sampler: Any | None = None,
        components: Mapping[str, nn.Module] | None = None,
        strict: bool = True,
        allow_topology_change: bool = False,
    ) -> CheckpointState:
        root = Path(path)
        manifest = self.inspect(root)
        self._validate_compatibility(
            manifest,
            run_config,
            allow_topology_change=allow_topology_change,
        )
        if manifest.kind != "resumable":
            raise CheckpointCompatibilityError("adapter-only artifact cannot restore optimizer/training state")
        use_distributed_storage = manifest.extra.get("storage_format") == "torch_distributed_checkpoint"
        if use_distributed_storage:
            if optimizer is not None:
                _prime_optimizer_for_checkpoint(optimizer)
            distributed_state = _distributed_state_dict(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                components=components,
            )
            _load_distributed_checkpoint(
                distributed_state,
                root / "dcp",
                xla=run_config.accelerator.backend == "xla_tpu",
            )
            model_state = _flatten_distributed_model_state(distributed_state, components)
        else:
            model_state = torch.load(
                root / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        result = _load_combined_state_dict(model, model_state, components, strict=strict)
        if not strict and (result["missing_keys"] or result["unexpected_keys"]):
            raise CheckpointCompatibilityError(
                f"checkpoint model mismatch: missing={result['missing_keys'][:4]}, "
                f"unexpected={result['unexpected_keys'][:4]}"
            )
        if not use_distributed_storage and optimizer is not None and "optimizer.pt" in manifest.files:
            optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True))
        if scheduler is not None and "scheduler.pt" in manifest.files:
            scheduler.load_state_dict(torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True))
        if scaler is not None and "scaler.pt" in manifest.files:
            scaler.load_state_dict(torch.load(root / "scaler.pt", map_location="cpu", weights_only=True))
        if use_distributed_storage and optimizer is not None and "optimizer" in distributed_state:
            optimizer.load_state_dict(distributed_state["optimizer"])
        if sampler is not None and "sampler.pt" in manifest.files:
            state = torch.load(root / "sampler.pt", map_location="cpu", weights_only=False)
            loader = getattr(sampler, "load_state_dict", None)
            if callable(loader):
                loader(state)
        if "rng.pt" in manifest.files:
            restore_rng_state(torch.load(root / "rng.pt", map_location="cpu", weights_only=False), backend=self.backend)
        state_data = json.loads((root / "training_state.json").read_text(encoding="utf-8"))
        return CheckpointState.from_dict(state_data)

    resume = load

    def export_adapter(
        self,
        name: str | int,
        *,
        model: nn.Module,
        run_config: RunConfig,
        base_references: Mapping[str, Any] | None = None,
        components: Mapping[str, nn.Module] | None = None,
        overwrite: bool = True,
    ) -> Path:
        return self.save(
            name,
            model=model,
            components=components,
            run_config=run_config,
            base_references=base_references,
            adapter_only=True,
            overwrite=overwrite,
        )

    export_adapter_checkpoint = export_adapter

    @staticmethod
    def _read_manifest(root: Path) -> CheckpointManifest:
        path = root / "manifest.json"
        if not path.exists():
            raise IncompleteCheckpointError(f"checkpoint has no manifest: {root}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IncompleteCheckpointError(f"checkpoint manifest is unreadable: {root}") from exc
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError("unsupported checkpoint schema version")
        if raw.get("complete") is not True:
            raise IncompleteCheckpointError(f"checkpoint is not complete: {root}")
        return CheckpointManifest(
            schema_version=int(raw["schema_version"]),
            kind=str(raw["kind"]),
            complete=True,
            run_config_hash=str(raw["run_config_hash"]),
            model_id=str(raw["model_id"]),
            base_model_revision=raw.get("base_model_revision"),
            backend=str(raw["backend"]),
            precision=str(raw["precision"]),
            distribution=str(raw["distribution"]),
            world_size=int(raw["world_size"]),
            topology=dict(raw.get("topology", {})),
            sharding_mesh=dict(raw.get("sharding_mesh", {})),
            compiler_runtime=dict(raw.get("compiler_runtime", {})),
            static_bucket_schema=dict(raw.get("static_bucket_schema", {})),
            files={str(key): str(value) for key, value in dict(raw.get("files", {})).items()},
            metrics={str(key): float(value) for key, value in dict(raw.get("metrics", {})).items()},
            best_criterion=raw.get("best_criterion"),
            extra=dict(raw.get("extra", {})),
        )

    @staticmethod
    def _verify_files(root: Path, manifest: CheckpointManifest) -> None:
        for filename, digest in manifest.files.items():
            path = root / filename
            if not path.exists() or _sha256(path) != digest:
                raise IncompleteCheckpointError(f"checkpoint file missing or corrupt: {path}")
        if "training_state.json" not in manifest.files:
            raise IncompleteCheckpointError("checkpoint has no training_state.json")

    @staticmethod
    def _validate_compatibility(
        manifest: CheckpointManifest, run_config: RunConfig, *, allow_topology_change: bool
    ) -> None:
        if manifest.run_config_hash != run_config.config_hash():
            raise CheckpointCompatibilityError(
                "run configuration hash mismatch; refusing to resume under a changed scientific configuration"
            )
        if manifest.model_id != run_config.model_id:
            raise CheckpointCompatibilityError("checkpoint model_id does not match current run")
        if manifest.base_model_revision != run_config.base_model_revision:
            raise CheckpointCompatibilityError("checkpoint base-model revision does not match current run")
        if not allow_topology_change and (
            manifest.backend != run_config.accelerator.backend
            or manifest.world_size != run_config.accelerator.world_size
            or manifest.distribution != run_config.accelerator.distribution
        ):
            raise CheckpointCompatibilityError(
                "checkpoint topology/backend does not match current run; reshard explicitly before changing topology"
            )


def _requires_distributed_checkpoint(run_config: RunConfig, *, adapter_only: bool) -> bool:
    if adapter_only:
        return False
    accelerator = run_config.accelerator
    return (accelerator.backend == "cuda" and accelerator.distribution == "fsdp") or (
        accelerator.backend == "xla_tpu" and accelerator.distribution == "spmd_fsdp"
    )


def _distributed_state_dict(
    model: nn.Module,
    *,
    optimizer: Any | None,
    scheduler: Any | None,
    scaler: Any | None,
    components: Mapping[str, nn.Module] | None,
) -> dict[str, Any]:
    del scheduler, scaler
    state: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        state["optimizer"] = optimizer.state_dict()
    for name, module in dict(components or {}).items():
        state[f"component:{name}"] = module.state_dict()
    return state


def _flatten_distributed_model_state(
    state: Mapping[str, Any],
    components: Mapping[str, nn.Module] | None,
) -> Mapping[str, Any]:
    if not components:
        return state["model"]
    combined: dict[str, Any] = {f"model.{key}": value for key, value in dict(state["model"]).items()}
    for name in components:
        values = dict(state.get(f"component:{name}", {}))
        combined.update({f"{name}.{key}": value for key, value in values.items()})
    return combined


def _dcp_module(*, xla: bool) -> Any:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise CheckpointError("distributed checkpoints require an initialized process group")
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        raise CheckpointError("this PyTorch build lacks torch.distributed.checkpoint") from exc
    return dcp


def _dcp_planner(*, xla: bool, saving: bool) -> Any | None:
    if not xla:
        return None
    try:
        from torch_xla.experimental import distributed_checkpoint as xla_checkpoint
    except ImportError as exc:
        raise CheckpointError(
            "TPU SPMD checkpoint planners require torch_xla.experimental.distributed_checkpoint"
        ) from exc
    planner_type = xla_checkpoint.SPMDSavePlanner if saving else xla_checkpoint.SPMDLoadPlanner
    return planner_type()


def _dcp_call(function: Any, payload: dict[str, Any], *, root: Path, xla: bool, saving: bool) -> Any:
    import torch.distributed.checkpoint as dcp

    import_kwargs: dict[str, Any] = {
        "storage_writer" if saving else "storage_reader": (
            dcp.FileSystemWriter(str(root)) if saving else dcp.FileSystemReader(str(root))
        ),
    }
    planner = _dcp_planner(xla=xla, saving=saving)
    if planner is not None:
        import_kwargs["planner"] = planner
    parameters = inspect.signature(function).parameters
    state_name = "state_dict" if "state_dict" in parameters else "state"
    import_kwargs[state_name] = payload
    return function(**import_kwargs)


def _save_distributed_checkpoint(state: dict[str, Any], root: Path, *, xla: bool) -> None:
    dcp = _dcp_module(xla=xla)
    _dcp_call(dcp.save, state, root=root, xla=xla, saving=True)


def _load_distributed_checkpoint(state: dict[str, Any], root: Path, *, xla: bool) -> None:
    dcp = _dcp_module(xla=xla)
    _dcp_call(dcp.load, state, root=root, xla=xla, saving=False)


def _prime_optimizer_for_checkpoint(optimizer: Any) -> None:
    if not hasattr(optimizer, "state") or optimizer.state:
        return
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
        if isinstance(parameter, torch.Tensor)
    ]
    if not parameters:
        return
    rng = capture_rng_state()
    values = [(parameter, parameter.detach().clone(), parameter.grad) for parameter in parameters]
    try:
        for parameter, _, _ in values:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    finally:
        with torch.no_grad():
            for parameter, value, grad in values:
                parameter.copy_(value)
                parameter.grad = grad
        restore_rng_state(rng)


def _file_hashes(root: Path, *, relative_to: Path) -> dict[str, str]:
    return {
        path.relative_to(relative_to).as_posix(): _sha256(path) for path in sorted(root.rglob("*")) if path.is_file()
    }


def capture_rng_state(*, backend: Any | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if backend is not None and getattr(backend, "name", None) == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], *, backend: Any | None = None) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if backend is not None and getattr(backend, "name", None) == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _combined_state_dict(
    model: nn.Module,
    components: Mapping[str, nn.Module] | None,
) -> dict[str, torch.Tensor]:
    if not components:
        return dict(model.state_dict())
    combined: dict[str, torch.Tensor] = {}
    for prefix, module in {"model": model, **dict(components)}.items():
        combined.update({f"{prefix}.{key}": value for key, value in module.state_dict().items()})
    return combined


def _load_combined_state_dict(
    model: nn.Module,
    state: Mapping[str, Any],
    components: Mapping[str, nn.Module] | None,
    *,
    strict: bool,
) -> dict[str, list[str]]:
    if not components:
        result = model.load_state_dict(state, strict=strict)
        return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
    modules = {"model": model, **dict(components)}
    missing: list[str] = []
    unexpected: list[str] = []
    split: dict[str, dict[str, Any]] = {name: {} for name in modules}
    for key, value in state.items():
        prefix, separator, child = str(key).partition(".")
        if not separator or prefix not in split:
            unexpected.append(str(key))
            continue
        split[prefix][child] = value
    for prefix, module in modules.items():
        result = module.load_state_dict(split[prefix], strict=strict)
        missing.extend(f"{prefix}.{key}" for key in result.missing_keys)
        unexpected.extend(f"{prefix}.{key}" for key in result.unexpected_keys)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _trainable_tensors(
    model: nn.Module,
    components: Mapping[str, nn.Module] | None = None,
) -> dict[str, torch.Tensor]:
    modules = {"model": model, **dict(components or {})}
    values: dict[str, torch.Tensor] = {}
    for prefix, module in modules.items():
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                values[f"{prefix}.{name}" if components else name] = parameter.detach().cpu().contiguous()
    return values


def _component_names(tensors: Mapping[str, torch.Tensor]) -> set[str]:
    components: set[str] = set()
    for name in tensors:
        lowered = name.lower()
        if "lora" in lowered or "adapter" in lowered:
            components.add("adapter")
        if any(token in lowered for token in ("bridge", "projector", "boundary")):
            components.add("bridge")
        if any(token in lowered for token in ("head", "classifier", "lm_head")):
            components.add("head")
        if any(token in lowered for token in ("decoder", "unet", "fpn")):
            components.add("decoder")
    return components


def _cpu_state_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _cpu_nested(item) for key, item in value.items()}


def _cpu_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointManifest",
    "CheckpointState",
    "IncompleteCheckpointError",
    "capture_rng_state",
    "restore_rng_state",
]
