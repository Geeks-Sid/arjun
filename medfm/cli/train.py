"""``python -m medfm.cli.train`` unified training entry point.

The CLI contains only an offline tiny recipe for contract/smoke runs.  Real
recipes inject their own builders through :class:`TrainingPipeline`; model
choices never live in the trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from medfm.core.batch import MedicalBatch
from medfm.core.enums import Modality, TaskType
from medfm.recipes.phase13 import phase13_builders
from medfm.recipes.phase14 import phase14_builders
from medfm.recipes.phase15 import phase15_builders
from medfm.tasks.classification import ClassificationTask
from medfm.tasks.segmentation import BinarySegmentationTask
from medfm.training.backend import BackendUnavailableError
from medfm.training.config import RunConfig, RunConfigError
from medfm.training.optimizer import build_optimizer
from medfm.training.pipeline import ComponentBuilders, TrainingPipeline
from medfm.training.steps import make_training_step
from medfm.training.trainer import Trainer


class TinyClassificationModel(nn.Module):
    """Small deterministic model used only by the smoke recipe."""

    def __init__(self, *, classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 8)
        self.classifier = nn.Linear(8, classes)

    def forward(self, batch: MedicalBatch) -> dict[str, torch.Tensor]:
        if batch.pixel_values is None:
            raise ValueError("tiny classification recipe requires pixel_values")
        values = batch.pixel_values.float()
        reduced = values.flatten(1).mean(dim=1, keepdim=True)
        hidden = torch.tanh(self.encoder(reduced))
        return {"logits": self.classifier(hidden)}


class TinySegmentationModel(nn.Module):
    """Tiny 2D/3D convolutional segmentation model for contract smoke."""

    def __init__(self, *, channels: int = 1) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv3d = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, batch: MedicalBatch) -> dict[str, torch.Tensor]:
        if batch.pixel_values is None:
            raise ValueError("tiny segmentation recipe requires pixel_values")
        values = batch.pixel_values.float()
        if values.ndim == 4:
            return {"segmentation": self.conv2d(values)}
        if values.ndim == 5:
            return {"segmentation": self.conv3d(values)}
        raise ValueError(f"unsupported tiny segmentation input rank {values.ndim}")


def _task_name(config: RunConfig) -> str:
    return str(config.task.get("type", config.task.get("name", "multiclass_classification"))).lower()


def _tiny_model(config: RunConfig) -> nn.Module:
    name = _task_name(config)
    return TinySegmentationModel() if "segmentation" in name else TinyClassificationModel()


def _tiny_task(config: RunConfig, model: nn.Module) -> nn.Module:
    name = _task_name(config)
    if "segmentation" in name:
        return BinarySegmentationTask(nn.Identity())
    task_type = TaskType.BINARY_CLASSIFICATION if "binary" in name else TaskType.MULTICLASS_CLASSIFICATION
    return ClassificationTask(nn.Identity(), task_type=task_type)


def _tiny_data(config: RunConfig) -> list[MedicalBatch]:
    name = _task_name(config)
    torch.manual_seed(config.seed)
    if "segmentation" in name:
        modality = Modality.CT_3D if "3d" in name else Modality.XRAY_2D
        shape = (2, 1, 3, 4, 4) if modality is Modality.CT_3D else (2, 1, 8, 8)
        values = torch.randn(shape)
        target_shape = (2, 1, *shape[2:])
        return [
            MedicalBatch(
                modality=modality,
                sample_ids=["tiny-0", "tiny-1"],
                pixel_values=values,
                task_targets={"segmentation": (torch.rand(target_shape) > 0.5).float()},
            )
        ]
    return [
        MedicalBatch(
            modality=Modality.XRAY_2D,
            sample_ids=["tiny-0", "tiny-1"],
            pixel_values=torch.randn(2, 1, 8, 8),
            task_targets={"classification": torch.tensor([0, 1])},
        )
    ]


def tiny_builders() -> ComponentBuilders:
    def dataset(config: RunConfig, *_args: Any) -> list[MedicalBatch]:
        return _tiny_data(config)

    def model(config: RunConfig, *_args: Any) -> nn.Module:
        return _tiny_model(config)

    def peft(model: nn.Module, *_args: Any) -> nn.Module:
        return model

    def optimizer(model: nn.Module, config: RunConfig, backend: Any) -> Any:
        return build_optimizer(model, config.optimizer, backend=backend.name)

    def task(config: RunConfig, model: nn.Module) -> nn.Module:
        return _tiny_task(config, model)

    def trainer(
        config: RunConfig,
        backend: Any,
        model: nn.Module,
        optimizer: Any,
        task_module: nn.Module,
        dataset: list[MedicalBatch],
    ) -> Trainer:
        return Trainer(
            model,
            optimizer,
            task_module,
            dataset,
            config,
            backend=backend,
            training_step=make_training_step(task_module),
        )

    return ComponentBuilders(dataset=dataset, model=model, peft=peft, optimizer=optimizer, task=task, trainer=trainer)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medfm train", description="run a resumable medfm training configuration")
    parser.add_argument("--config", required=True, help="YAML/JSON RunConfig")
    parser.add_argument("--backend", choices=["cpu", "cuda", "xla_tpu"], help="override configured backend")
    parser.add_argument(
        "--dry-run", "--model-summary", action="store_true", help="validate without allocating model weights"
    )
    parser.add_argument("--resume", help="resumable checkpoint directory")
    parser.add_argument("--max-steps", type=int, help="explicit optimizer-step limit")
    parser.add_argument("--export-adapter", action="store_true", help="write a CPU safetensors adapter artifact")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def _builders_for_config(config: RunConfig) -> ComponentBuilders:
    """Select a recipe-owned builder from the declared experiment family."""

    family = str(config.recipe.get("family", config.recipe.get("type", ""))).lower().replace("-", "_")
    phase15_families = {
        "tile_classification",
        "wsi_classification",
        "wsi_vlm",
        "pathology_segmentation",
        "tiled_segmentation",
    }
    if (
        family in phase15_families
        or str(config.recipe.get("phase", "")).lower() in {"15", "phase15"}
        or str(config.model.get("family", "")).lower().startswith("phase15")
        or str(config.recipe.get("modality", "")).upper() in {"PATHOLOGY_TILE", "PATHOLOGY_WSI"}
    ):
        return phase15_builders()

    phase14_families = {
        "native_3d_classification",
        "native_3d_segmentation",
        "native_3d_vlm",
        "slice_sequence_vlm",
        "language_conditioned_segmentation",
        "language_conditioned_3d_segmentation",
    }
    modality = str(config.recipe.get("modality", "")).upper()
    if (
        family in phase14_families
        or modality in {"CT_3D", "MRI_3D", "MULTI_SERIES_3D", "MULTI_IMAGE_2D"}
        or str(config.recipe.get("phase", "")).lower() in {"14", "phase14"}
    ):
        return phase14_builders()
    if family in {
        "classification",
        "segmentation",
        "promptable_segmentation",
        "native_vlm",
        "external_vlm",
        "native",
        "external",
        "seg",
    }:
        return phase13_builders()
    return tiny_builders()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = RunConfig.load(args.config)
        if args.backend:
            config = replace(config, accelerator=replace(config.accelerator, backend=args.backend))
        if args.max_steps is not None:
            config = replace(config, max_steps=args.max_steps)
        pipeline = TrainingPipeline(config, builders=_builders_for_config(config), dry_run=args.dry_run)
        if args.dry_run:
            payload = pipeline.dry_run_summary().model_summary.to_dict()  # type: ignore[union-attr]
        else:
            built = pipeline.build()
            if not isinstance(built.trainer, Trainer):
                raise RuntimeError("configured recipe did not build a Trainer")
            trainer = built.trainer
            if args.resume:
                trainer.resume(args.resume)
            result = trainer.train()
            if args.export_adapter:
                result = replace(result, metadata={**result.metadata, "adapter_export": str(trainer.export_adapter())})
            payload = result.to_dict()
    except (RunConfigError, BackendUnavailableError, RuntimeError, ValueError) as exc:
        print(f"medfm train: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        if args.dry_run:
            print(f"dry-run OK: {payload['model_id']} (weights not allocated)")
        else:
            print(
                f"training {'OK' if payload['success'] else 'INTERRUPTED'}: "
                f"backend={payload['backend']} optimizer_steps={payload['optimizer_steps']}"
            )
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
