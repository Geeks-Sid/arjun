"""CLI for bounded classification, segmentation, retrieval, VLM, and WSI inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from medfm.inference import (
    ClassificationPipeline,
    InferenceLimits,
    InferenceRequest,
    InferenceRuntime,
    InferenceTask,
    RetrievalPipeline,
    SegmentationPipeline,
    WSIPipeline,
)
from medfm.inference.errors import InferenceError


class _MeanClassifier(nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.classes = int(classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value.float().mean(dim=tuple(range(1, value.ndim)), keepdim=False)
        if self.classes == 1:
            return score.unsqueeze(-1)
        logits = torch.zeros((value.shape[0], self.classes), dtype=score.dtype, device=score.device)
        logits[:, 0] = -score
        logits[:, 1] = score
        return logits


class _IdentitySegmentation(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            return value[:, :1]
        return value[:, :1]


class _MeanEmbedding(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.width = int(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value.float().mean(dim=tuple(range(1, value.ndim)), keepdim=False).unsqueeze(-1)
        return score.expand(-1, self.width)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise InferenceError(details={"field": "config"}) from exc
    if not isinstance(value, Mapping):
        raise InferenceError(details={"field": "config"})
    return dict(value)


def _load_tensor(config: Mapping[str, Any], *, modality: str) -> torch.Tensor:
    input_config = config.get("input", {})
    if not isinstance(input_config, Mapping):
        raise InferenceError(details={"field": "input"})
    source = input_config.get("path")
    if source is not None:
        path = Path(str(source)).expanduser().resolve()
        if not path.is_file() or path.suffix not in {".pt", ".pth", ".npy"}:
            raise InferenceError(details={"field": "input.path", "reason": "only .pt/.pth/.npy are supported"})
        if path.suffix == ".npy":
            import numpy as np

            tensor = torch.from_numpy(np.load(path, allow_pickle=False))
        else:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise InferenceError(details={"field": "input.path", "reason": "file must contain one tensor"})
        return tensor
    synthetic = input_config.get("synthetic", config.get("synthetic"))
    if not isinstance(synthetic, Mapping):
        raise InferenceError(details={"field": "input.synthetic"})
    shape = tuple(int(value) for value in synthetic.get("shape", (1, 1, 32, 32)))
    if not shape or any(value <= 0 for value in shape):
        raise InferenceError(details={"field": "input.synthetic.shape"})
    expected_rank = {"CT_3D": 5, "MRI_3D": 5, "PATHOLOGY_WSI": 5, "MULTI_IMAGE_2D": 5}.get(modality, 4)
    if len(shape) != expected_rank:
        raise InferenceError(details={"field": "input.synthetic.shape", "reason": "shape rank does not match modality"})
    return torch.full(shape, float(synthetic.get("value", 0.25)), dtype=torch.float32)


def _model_for(task: InferenceTask, config: Mapping[str, Any]) -> nn.Module:
    model = config.get("model", {})
    model_type = str(model.get("type", "mean_classifier")) if isinstance(model, Mapping) else "mean_classifier"
    if model_type == "mean_classifier" and task is InferenceTask.CLASSIFICATION:
        classes = int(model.get("classes", 2)) if isinstance(model, Mapping) else 2
        return _MeanClassifier(classes)
    if model_type == "identity_segmentation" and task is InferenceTask.SEGMENTATION:
        return _IdentitySegmentation()
    if model_type == "mean_embedding" and task in {InferenceTask.RETRIEVAL, InferenceTask.WSI}:
        width = int(model.get("width", 8)) if isinstance(model, Mapping) else 8
        return _MeanEmbedding(width)
    raise InferenceError(details={"field": "model.type", "reason": "unapproved built-in model for task"})


def build_pipeline(config: Mapping[str, Any], *, task_override: str | None = None) -> tuple[Any, InferenceTask, str]:
    task = InferenceTask.parse(task_override or str(config.get("task", "classification")))
    modality = str(config.get("modality", "XRAY_2D"))
    limits = InferenceLimits.from_dict(config.get("limits"))
    runtime = InferenceRuntime(str(config.get("backend", "cpu")), max_memory_bytes=limits.max_memory_bytes)
    model = _model_for(task, config)
    common = {
        "runtime": runtime,
        "limits": limits,
        "model_id": str(config.get("model_id", f"smoke-{task.value}")),
        "model_revision": str(config.get("model_revision", "builtin")),
        "preprocess_hash": str(config.get("preprocess_hash", "builtin-identity")),
    }
    if task is InferenceTask.CLASSIFICATION:
        return ClassificationPipeline(model, **common), task, modality
    if task is InferenceTask.SEGMENTATION:
        window = config.get("window_shape")
        return (
            SegmentationPipeline(
                model, window_shape=tuple(int(value) for value in window) if window else None, **common
            ),
            task,
            modality,
        )
    if task is InferenceTask.RETRIEVAL:
        return RetrievalPipeline(model, **common), task, modality
    if task is InferenceTask.WSI:
        return WSIPipeline(model, **common), task, modality
    if task is InferenceTask.VLM:
        raise InferenceError(
            details={"field": "model.type", "reason": "CLI smoke VLM requires a reviewed model adapter"}
        )
    raise InferenceError(details={"field": "task"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medfm infer", description="Run bounded medical inference.")
    parser.add_argument("task", choices=[task.value for task in InferenceTask])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--format", choices=["json"], default="json")
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        pipeline, task, modality = build_pipeline(config, task_override=args.task)
        tensor = _load_tensor(config, modality=modality)
        request = InferenceRequest(task=task, modality=modality, payload={"pixel_values": tensor}, request_id="cli")
        result = pipeline.run(request)
        print(json.dumps(result.to_dict(serialize_tensors=True), sort_keys=True))
        return 0
    except Exception as exc:
        if isinstance(exc, InferenceError):
            error = exc.to_error()
        else:
            error = {
                "code": "INFERENCE_ERROR",
                "message": "inference request failed",
                "retryable": False,
                "details": {},
            }
        print(json.dumps({"schema_version": 1, "ok": False, "error": error}, sort_keys=True))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
