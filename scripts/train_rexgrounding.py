#!/usr/bin/env python
"""Train and run a CT-FM language-conditioned segmenter on ReXGroundingCT.

The released ReXGroundingCT repository contains masks and report text. CT
volumes are resolved from CT-RATE, either from a local snapshot or through an
explicit, one-volume-at-a-time Hugging Face download. The default training
route is patch-based so the 24 GiB workstation GPU never receives a full
512x512xD scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from medfm.data.anatomy_grounding import filter_prediction_by_region, resolve_report_region, select_report_prior
from medfm.data.totalsegmentator import DEFAULT_THORACIC_LABELS, load_total_segmentator_prior
from medfm.models.decoders import FPNDecoder3D
from medfm.tasks.losses import DiceBCELoss

DEFAULT_DATA_DIR = Path("../RexGroundingData")
DEFAULT_OUTPUT_DIR = Path("artifacts/runs/rexgroundingct/ctfm_fpn")
DEFAULT_VISUAL_MODEL = "project-lighter/ct_fm_feature_extractor"
DEFAULT_TEXT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DEFAULT_CT_REPO = "ibrahimhamamci/CT-RATE"


@dataclass(frozen=True)
class ChallengeEntry:
    split: str
    name: str
    findings: dict[str, str]
    categories: dict[str, str]
    shape: tuple[int, int, int]
    mask_path: Path | None


@dataclass(frozen=True)
class FindingRecord:
    entry: ChallengeEntry
    finding_key: str
    finding_index: int
    text: str


@dataclass(frozen=True)
class VolumeData:
    array_hwd: np.ndarray
    affine: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--volume-root", type=Path, default=None)
    parser.add_argument("--ct-repo", default=DEFAULT_CT_REPO)
    parser.add_argument(
        "--totalseg-prior-dir",
        type=Path,
        default=None,
        help="offline TotalSegmentator output root; each case is a subdirectory of binary NIfTI masks",
    )
    parser.add_argument(
        "--totalseg-labels",
        nargs="+",
        default=DEFAULT_THORACIC_LABELS,
        help="TotalSegmentator `total` task labels used as spatial prior channels",
    )
    parser.add_argument("--ct-cache-dir", type=Path, default=None)
    parser.add_argument("--allow-remote-download", action="store_true")
    parser.add_argument("--visual-model", default=DEFAULT_VISUAL_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--train-cases", type=int, default=0, help="0 means all official training cases")
    parser.add_argument("--predict-cases", type=int, default=0, help="0 means all cases in --split")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(96, 96, 96), metavar=("D", "H", "W"))
    parser.add_argument("--patches-per-finding", type=int, default=2)
    parser.add_argument("--positive-ratio", type=float, default=0.75)
    parser.add_argument(
        "--region-guidance-weight",
        type=float,
        default=0.02,
        help="weak penalty for predicted voxels outside a report-selected lobe region",
    )
    parser.add_argument("--region-postprocess-min-confidence", type=float, default=0.6)
    parser.add_argument("--region-postprocess-halo-mm", type=float, default=15.0)
    parser.add_argument("--region-postprocess-overlap", type=float, default=0.02)
    parser.add_argument("--inference-stride", type=int, nargs=3, default=(64, 64, 64), metavar=("D", "H", "W"))
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dev-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--visual-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--train-visual", action="store_true")
    parser.add_argument("--train-text", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cases-for-smoke", type=int, default=0)
    return parser.parse_args()


def read_hf_token() -> str | None:
    environment_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if environment_token:
        return environment_token.strip()
    for candidate in (Path.home() / ".secrets.txt", Path.home() / ".cache/huggingface/token"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                value = line.partition("=")[2].strip()
                if value:
                    return value
        value = candidate.read_text(encoding="utf-8").strip()
        if value.startswith("hf_"):
            return value
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finding_sort_key(key: str) -> tuple[int, str]:
    try:
        return (0, f"{int(key):08d}")
    except ValueError:
        return (1, key)


def load_entries(data_dir: Path, split: str, *, require_masks: bool) -> list[ChallengeEntry]:
    metadata_path = data_dir / "MICCAI_challenge_dataset.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_entries = payload.get(split)
    if not isinstance(raw_entries, list):
        raise ValueError(f"{metadata_path} has no list for split {split!r}")
    entries: list[ChallengeEntry] = []
    mask_dir = data_dir / "segmentations"
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid {split} entry: expected object")
        name = str(raw.get("name", "")).strip()
        findings_raw = raw.get("findings")
        shape_raw = raw.get("shape")
        if not name or not isinstance(findings_raw, dict) or not isinstance(shape_raw, list) or len(shape_raw) != 3:
            raise ValueError(f"invalid {split} entry identity/findings/shape for {name!r}")
        findings = {str(key): str(value).strip() for key, value in findings_raw.items() if str(value).strip()}
        if not findings:
            raise ValueError(f"{split} entry {name!r} has no findings")
        categories_raw = raw.get("categories", {})
        categories = (
            {str(key): str(value) for key, value in categories_raw.items()} if isinstance(categories_raw, dict) else {}
        )
        mask_path = mask_dir / name
        if require_masks and not mask_path.is_file():
            raise FileNotFoundError(f"missing released segmentation mask: {mask_path}")
        entries.append(
            ChallengeEntry(
                split=split,
                name=name,
                findings=findings,
                categories=categories,
                shape=tuple(int(value) for value in shape_raw),
                mask_path=mask_path if mask_path.is_file() else None,
            )
        )
    return sorted(entries, key=lambda entry: entry.name)


def records_from_entries(entries: Iterable[ChallengeEntry]) -> list[FindingRecord]:
    records: list[FindingRecord] = []
    for entry in entries:
        for key in sorted(entry.findings, key=_finding_sort_key):
            try:
                index = int(key)
            except ValueError as exc:
                raise ValueError(f"finding key {key!r} in {entry.name} is not an integer channel") from exc
            records.append(FindingRecord(entry, key, index, entry.findings[key]))
    return records


def select_cases(entries: list[ChallengeEntry], limit: int) -> list[ChallengeEntry]:
    if limit < 0:
        raise ValueError("case limits must be non-negative")
    return entries if limit == 0 else entries[:limit]


def split_train_cases(
    entries: list[ChallengeEntry], fraction: float, seed: int
) -> tuple[list[ChallengeEntry], list[ChallengeEntry]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--dev-fraction must be between zero and one")
    train: list[ChallengeEntry] = []
    dev: list[ChallengeEntry] = []
    for entry in entries:
        digest = hashlib.sha256(f"rexgroundingct-dev-v1:{seed}:{entry.name}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        (dev if value < fraction else train).append(entry)
    if not train or not dev:
        raise ValueError("deterministic train/dev split produced an empty partition")
    return train, dev


def ct_rate_path(name: str) -> str:
    if not name.endswith(".nii.gz"):
        raise ValueError(f"CT volume name must end in .nii.gz: {name}")
    stem = name[: -len(".nii.gz")]
    pieces = stem.split("_")
    if len(pieces) < 3:
        raise ValueError(f"cannot infer CT-RATE hierarchy from {name}")
    split = "valid_fixed" if pieces[0] == "valid" else "train_fixed"
    study = "_".join(pieces[:2])
    series = "_".join(pieces[:-1])
    return f"dataset/{split}/{study}/{series}/{name}"


def resolve_local_volume(root: Path, name: str) -> Path | None:
    relative = Path(ct_rate_path(name))
    candidates = (
        root / name,
        root / relative,
        root / relative.relative_to("dataset"),
        root / "dataset" / relative.relative_to("dataset"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class VolumeSource:
    def __init__(
        self,
        *,
        root: Path | None,
        repo_id: str,
        cache_dir: Path,
        allow_remote: bool,
        token: str | None,
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.allow_remote = allow_remote
        self.token = token

    def load(self, name: str) -> VolumeData:
        local_path = resolve_local_volume(self.root, name) if self.root is not None else None
        if local_path is not None:
            return read_nifti(local_path)
        if not self.allow_remote:
            expected = ct_rate_path(name)
            raise FileNotFoundError(
                f"CT volume {name} is not available under {self.root}; expected a CT-RATE fixed path "
                f"like {expected}. Use --allow-remote-download or stage the CT-RATE fixed volumes."
            )
        if self.token is None:
            raise RuntimeError("remote CT-RATE access requires HF_TOKEN or ~/.secrets.txt")
        from huggingface_hub import hf_hub_download

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rexgroundingct-", dir=self.cache_dir) as temp:
            temp_root = Path(temp)
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=ct_rate_path(name),
                token=self.token,
                cache_dir=temp_root / "hub",
                local_dir=temp_root / "local",
            )
            result = read_nifti(Path(downloaded))
        return result


def read_nifti(path: Path) -> VolumeData:
    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"CT volume {path} must be 3D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"CT volume {path} contains non-finite values")
    return VolumeData(array_hwd=np.ascontiguousarray(array), affine=np.asarray(image.affine, dtype=np.float64))


def read_mask(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    image = nib.load(str(path))
    mask = np.asarray(image.dataobj, dtype=np.uint8)
    if mask.ndim != 4:
        raise ValueError(f"segmentation mask {path} must be 4D [F,H,W,D], got {mask.shape}")
    if tuple(mask.shape[1:]) != expected_shape:
        raise ValueError(f"segmentation mask {path} shape {mask.shape[1:]} != metadata shape {expected_shape}")
    return np.ascontiguousarray(mask)


def extract_patch(
    array_dhw: np.ndarray,
    origin: Sequence[int],
    patch_shape: Sequence[int],
    fill: float = 0.0,
) -> np.ndarray:
    patch = np.full(tuple(int(v) for v in patch_shape), fill, dtype=array_dhw.dtype)
    source: list[slice] = []
    destination: list[slice] = []
    for _axis, (start, width, size) in enumerate(zip(origin, patch_shape, array_dhw.shape, strict=True)):
        src_start = max(0, int(start))
        src_end = min(size, int(start) + int(width))
        if src_end <= src_start:
            return patch
        dst_start = src_start - int(start)
        source.append(slice(src_start, src_end))
        destination.append(slice(dst_start, dst_start + src_end - src_start))
    patch[tuple(destination)] = array_dhw[tuple(source)]
    return patch


def random_origin(shape: Sequence[int], patch_shape: Sequence[int], rng: np.random.Generator) -> tuple[int, int, int]:
    return tuple(
        int(rng.integers(min(0, int(size) - int(width)), max(0, int(size) - int(width)) + 1))
        for size, width in zip(shape, patch_shape, strict=True)
    )


def lesion_origin(
    positive_voxels: np.ndarray,
    shape: Sequence[int],
    patch_shape: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if len(positive_voxels) == 0:
        return random_origin(shape, patch_shape, rng)
    center = positive_voxels[int(rng.integers(0, len(positive_voxels)))]
    return tuple(int(center[axis]) - int(patch_shape[axis]) // 2 for axis in range(3))


def normalize_ct(patch: np.ndarray) -> np.ndarray:
    return np.clip((patch.astype(np.float32, copy=False) + 1024.0) / 3072.0, 0.0, 1.0)


class RexPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[FindingRecord],
        *,
        source: VolumeSource,
        patch_shape: tuple[int, int, int],
        patches_per_finding: int,
        positive_ratio: float,
        tokenizer: Any,
        max_text_length: int,
        seed: int,
        training: bool,
        totalseg_prior_dir: Path | None = None,
        totalseg_labels: Sequence[str] = (),
        cache_items: int = 1,
    ) -> None:
        if not records:
            raise ValueError("patch dataset cannot be empty")
        if any(int(v) <= 0 for v in patch_shape):
            raise ValueError("patch shape must contain positive dimensions")
        if patches_per_finding <= 0:
            raise ValueError("patches_per_finding must be positive")
        if not 0.0 <= positive_ratio <= 1.0:
            raise ValueError("positive_ratio must be in [0,1]")
        self.records = records
        self.source = source
        self.patch_shape = patch_shape
        self.patches_per_finding = patches_per_finding
        self.positive_ratio = positive_ratio
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.seed = seed
        self.training = training
        self.totalseg_prior_dir = totalseg_prior_dir
        self.totalseg_labels = tuple(str(label) for label in totalseg_labels)
        if self.totalseg_prior_dir is not None and not self.totalseg_labels:
            raise ValueError("TotalSegmentator prior labels cannot be empty when a prior directory is set")
        self.epoch = 0
        self.cache_items = max(1, int(cache_items))
        self._cache: OrderedDict[str, tuple[VolumeData, np.ndarray, np.ndarray | None]] = OrderedDict()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_finding

    def _load_pair(self, record: FindingRecord) -> tuple[VolumeData, np.ndarray, np.ndarray | None]:
        key = record.entry.name
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if record.entry.mask_path is None:
            raise FileNotFoundError(f"no released mask for {key}")
        volume = self.source.load(key)
        mask = read_mask(record.entry.mask_path, record.entry.shape)
        if record.finding_index < 0 or record.finding_index >= mask.shape[0]:
            raise ValueError(
                f"finding channel {record.finding_index} for {key} is outside mask channels {mask.shape[0]}"
            )
        if tuple(volume.array_hwd.shape) != record.entry.shape:
            raise ValueError(
                f"CT volume {key} shape {volume.array_hwd.shape} does not match "
                f"released metadata {record.entry.shape}; use the CT-RATE fixed snapshot, "
                "not an unrelated resampling"
            )
        organ_prior = (
            load_total_segmentator_prior(
                self.totalseg_prior_dir,
                key,
                self.totalseg_labels,
                record.entry.shape,
                volume.affine,
            )
            if self.totalseg_prior_dir is not None
            else None
        )
        value = (volume, mask, organ_prior)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_items:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index // self.patches_per_finding]
        repeat = index % self.patches_per_finding
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97 + repeat)
        volume, masks, organ_prior = self._load_pair(record)
        region = resolve_report_region(record.text) if organ_prior is not None else None
        region_prior = (
            select_report_prior(organ_prior, self.totalseg_labels, region)
            if organ_prior is not None and region is not None
            else None
        )
        volume_dhw = np.moveaxis(volume.array_hwd, -1, 0)
        mask_dhw = np.moveaxis(masks[record.finding_index], -1, 0)
        positive_voxels = np.argwhere(mask_dhw > 0)
        choose_positive = bool(len(positive_voxels) and rng.random() < self.positive_ratio)
        origin = (
            lesion_origin(positive_voxels, volume_dhw.shape, self.patch_shape, rng)
            if choose_positive
            else random_origin(volume_dhw.shape, self.patch_shape, rng)
        )
        image_patch = normalize_ct(extract_patch(volume_dhw, origin, self.patch_shape, fill=-1024.0))
        target_patch = extract_patch(mask_dhw, origin, self.patch_shape, fill=0).astype(np.float32, copy=False)
        target_patch = (target_patch > 0).astype(np.float32, copy=False)
        prior_patch = (
            np.stack(
                [extract_patch(channel, origin, self.patch_shape, fill=0) for channel in region_prior],
                axis=0,
            ).astype(np.float32, copy=False)
            if region_prior is not None
            else None
        )
        region_confidence = 0.0 if region is None else region.confidence
        if self.training:
            gain = float(rng.uniform(0.95, 1.05))
            bias = float(rng.uniform(-0.025, 0.025))
            image_patch = np.clip(image_patch * gain + bias, 0.0, 1.0)
        result: dict[str, Any] = {
            "pixel_values": torch.from_numpy(image_patch[None].copy()),
            "target": torch.from_numpy(target_patch[None].copy()),
            "text": f"segment finding: {record.text}",
            "sample_id": f"{record.entry.name}:{record.finding_key}:{repeat}",
            "positive_patch": choose_positive,
            "region_confidence": region_confidence,
        }
        if prior_patch is not None:
            result["region_priors"] = torch.from_numpy(prior_patch.copy())
        return result


def make_collator(tokenizer: Any, max_text_length: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(row["text"]) for row in rows]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        result: dict[str, Any] = {
            "pixel_values": torch.stack([row["pixel_values"] for row in rows]),
            "target": torch.stack([row["target"] for row in rows]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"].to(dtype=torch.bool),
            "sample_ids": [str(row["sample_id"]) for row in rows],
            "positive_patch": torch.tensor([bool(row["positive_patch"]) for row in rows]),
            "region_confidence": torch.tensor([float(row["region_confidence"]) for row in rows], dtype=torch.float32),
        }
        has_priors = ["region_priors" in row for row in rows]
        if any(has_priors) and not all(has_priors):
            raise ValueError("a batch cannot mix samples with and without report-selected priors")
        if all(has_priors):
            result["region_priors"] = torch.stack([row["region_priors"] for row in rows])
        return result

    return collate


class TextConditionedCTFM(nn.Module):
    """CT-FM pyramid with text gates and optional TotalSegmentator fusion."""

    def __init__(
        self,
        *,
        visual_model: str,
        text_model: str,
        patch_shape: tuple[int, int, int],
        token: str | None,
        train_visual: bool,
        train_text: bool,
        prior_channels: int = 0,
    ) -> None:
        super().__init__()
        from lighter_zoo import SegResEncoder
        from transformers import AutoModel

        if prior_channels < 0:
            raise ValueError("prior_channels must be non-negative")
        self.visual = SegResEncoder.from_pretrained(visual_model, token=token)
        self.text = AutoModel.from_pretrained(text_model, token=token)
        self.visual_name = visual_model
        self.text_name = text_model
        self.train_visual = bool(train_visual)
        self.train_text = bool(train_text)
        self.prior_channels = int(prior_channels)
        self.text_hidden_size = int(self.text.config.hidden_size)
        self.text_projection = nn.Linear(self.text_hidden_size, 128)
        self._set_frozen_parameters()
        with torch.no_grad():
            was_training = self.visual.training
            self.visual.eval()
            dummy = torch.zeros((1, 1, *patch_shape), dtype=torch.float32)
            feature_maps = self.visual(dummy)
            if not isinstance(feature_maps, (tuple, list)) or not feature_maps:
                raise RuntimeError("CT-FM feature extractor must return a non-empty feature-map list")
            channels = tuple(int(value.shape[1]) for value in feature_maps)
            self.visual.train(was_training)
        self.feature_channels = channels
        self.text_gates = nn.ModuleList([nn.Linear(128, channels_i) for channels_i in channels])
        if self.prior_channels:
            self.prior_encoder: nn.Module | None = nn.Sequential(
                nn.Conv3d(self.prior_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
            self.prior_projections = nn.ModuleList([nn.Conv3d(64, channel, kernel_size=1) for channel in channels])
            for projection in self.prior_projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        else:
            self.prior_encoder = None
            self.prior_projections = nn.ModuleList()
        self.decoder = FPNDecoder3D(channels, 1, pyramid_channels=128, deep_supervision=False)

    def _set_frozen_parameters(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad = self.train_visual
        for parameter in self.text.parameters():
            parameter.requires_grad = self.train_text

    def train(self, mode: bool = True) -> TextConditionedCTFM:
        super().train(mode)
        if not self.train_visual:
            self.visual.eval()
        if not self.train_text:
            self.text.eval()
        return self

    @staticmethod
    def _pool_text(hidden: Tensor, attention_mask: Tensor) -> Tensor:
        weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _text_embedding(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        context = nullcontext() if self.train_text else torch.no_grad()
        with context:
            output = self.text(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_text(output.last_hidden_state, attention_mask)
        return self.text_projection(pooled)

    def _visual_features(self, pixel_values: Tensor) -> Sequence[Tensor]:
        context = nullcontext() if self.train_visual else torch.no_grad()
        with context:
            features = self.visual(pixel_values)
        if not isinstance(features, (tuple, list)):
            raise RuntimeError("CT-FM feature extractor returned a non-sequence output")
        return features

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        region_priors: Tensor | None = None,
    ) -> Tensor:
        text = self._text_embedding(input_ids, attention_mask)
        features = self._visual_features(pixel_values)
        if len(features) != len(self.text_gates):
            raise RuntimeError(f"CT-FM returned {len(features)} feature maps; expected {len(self.text_gates)}")
        if self.prior_channels:
            if region_priors is None:
                raise ValueError("report-selected TotalSegmentator priors are required by this fusion model")
            if region_priors.ndim != 5 or region_priors.shape[1] != self.prior_channels:
                raise ValueError(
                    f"expected region_priors with {self.prior_channels} channels and rank 5, "
                    f"got {tuple(region_priors.shape)}"
                )
            if self.prior_encoder is None:
                raise RuntimeError("prior encoder is missing despite prior_channels being enabled")
            prior_context = self.prior_encoder(region_priors)
            fused_features = [
                feature
                + projection(
                    F.interpolate(
                        prior_context,
                        size=tuple(int(value) for value in feature.shape[-3:]),
                        mode="trilinear",
                        align_corners=False,
                    )
                )
                for feature, projection in zip(features, self.prior_projections, strict=True)
            ]
        else:
            if region_priors is not None:
                raise ValueError("region_priors were supplied to a model without TotalSegmentator fusion")
            fused_features = list(features)
        conditioned: list[Tensor] = []
        for feature, gate in zip(fused_features, self.text_gates, strict=True):
            scale = 1.0 + 0.5 * torch.tanh(gate(text)).view(text.shape[0], -1, 1, 1, 1)
            conditioned.append(feature * scale)
        return self.decoder(conditioned, output_size=tuple(int(v) for v in pixel_values.shape[-3:])).logits


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()
    }


def autocast_context(device: torch.device, amp_dtype: torch.dtype):
    return torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()


def dice_score(logits: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    predicted = torch.sigmoid(logits) >= threshold
    target_bool = target >= 0.5
    intersection = (predicted & target_bool).flatten(1).sum(dim=1).float()
    denominator = predicted.flatten(1).sum(dim=1).float() + target_bool.flatten(1).sum(dim=1).float()
    return ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def build_optimizer(model: TextConditionedCTFM, args: argparse.Namespace) -> torch.optim.Optimizer:
    visual: list[Tensor] = []
    head: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (visual if name.startswith("visual.") else head).append(parameter)
    groups: list[dict[str, Any]] = []
    if head:
        groups.append({"params": head, "lr": args.learning_rate})
    if visual:
        groups.append({"params": visual, "lr": args.visual_learning_rate})
    if not groups:
        raise RuntimeError("no trainable parameters remain")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))


def region_guidance_penalty(
    logits: Tensor,
    region_priors: Tensor,
    region_confidence: Tensor,
    *,
    dilation_voxels: int = 3,
) -> Tensor:
    """Penalize confident predictions far from a report-selected region softly."""

    if region_priors.ndim != 5 or logits.ndim != 5 or region_priors.shape[0] != logits.shape[0]:
        raise ValueError("logits and region_priors must be batched 5D tensors with matching batch size")
    if dilation_voxels < 0:
        raise ValueError("dilation_voxels must be non-negative")
    confidence = region_confidence.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if confidence.numel() != logits.shape[0]:
        raise ValueError("region_confidence must contain one value per batch item")
    region = (region_priors.amax(dim=1, keepdim=True) > 0).to(dtype=logits.dtype)
    present = region.flatten(1).amax(dim=1)
    weights = confidence.clamp(0.0, 1.0) * present
    if not bool(torch.any(weights > 0)):
        return logits.new_zeros(())
    kernel = 2 * dilation_voxels + 1
    allowed = F.max_pool3d(region, kernel_size=kernel, stride=1, padding=dilation_voxels) if dilation_voxels else region
    outside = 1.0 - allowed.clamp(0.0, 1.0)
    weight_map = weights.view(-1, 1, 1, 1, 1)
    penalty = torch.sigmoid(logits) * outside * weight_map
    denominator = weight_map.sum() * float(logits.shape[-3] * logits.shape[-2] * logits.shape[-1])
    return penalty.sum() / denominator.clamp_min(1.0)


def evaluate_patch_loss(
    model: TextConditionedCTFM,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    criterion = DiceBCELoss()
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device)
            with autocast_context(device, amp_dtype):
                logits = model(
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("region_priors"),
                )
                loss = criterion(logits, batch["target"])
            losses.append(float(loss.float().cpu()))
            dices.append(float(dice_score(logits.float(), batch["target"].float()).cpu()))
    if not losses:
        raise ValueError("evaluation loader produced no batches")
    return {"loss": sum(losses) / len(losses), "dice": sum(dices) / len(dices), "batches": float(len(losses))}


def save_checkpoint(
    path: Path,
    *,
    model: TextConditionedCTFM,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "visual_model": model.visual_name,
            "text_model": model.text_name,
            "feature_channels": model.feature_channels,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: TextConditionedCTFM,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_model = state["model"]
    incompatible = model.load_state_dict(checkpoint_model, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    baseline_checkpoint = not any(str(key).startswith("prior_encoder.") for key in checkpoint_model)
    allowed_missing = (
        {key for key in missing if key.startswith(("prior_encoder.", "prior_projections."))}
        if model.prior_channels and baseline_checkpoint
        else set()
    )
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint architecture mismatch: missing={sorted(missing - allowed_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if allowed_missing:
        print("checkpoint_prior_initialized", json.dumps({"missing": sorted(allowed_missing)}, sort_keys=True))
    if optimizer is not None and "optimizer" in state:
        saved_optimizer = state["optimizer"]
        saved_groups = len(saved_optimizer.get("param_groups", []))
        current_groups = len(optimizer.param_groups)
        saved_sizes = [len(group.get("params", [])) for group in saved_optimizer.get("param_groups", [])]
        current_sizes = [len(group["params"]) for group in optimizer.param_groups]
        if saved_groups == current_groups and saved_sizes == current_sizes:
            optimizer.load_state_dict(saved_optimizer)
        else:
            print(
                "checkpoint_optimizer_reset",
                json.dumps(
                    {
                        "saved_param_groups": saved_groups,
                        "current_param_groups": current_groups,
                        "saved_group_sizes": saved_sizes,
                        "current_group_sizes": current_sizes,
                    },
                    sort_keys=True,
                ),
            )
    return state


def train(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or args.gradient_accumulation <= 0 or args.max_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch size, gradient accumulation, max steps, and epochs must be positive")
    if args.positive_ratio < 0.0 or args.positive_ratio > 1.0:
        raise ValueError("positive ratio must be in [0,1]")
    if args.region_guidance_weight < 0.0:
        raise ValueError("region guidance weight must be non-negative")
    seed_everything(args.seed)
    data_dir = args.data_dir.resolve()
    volume_root = args.volume_root.resolve() if args.volume_root is not None else data_dir / "ct_rate_fixed"
    cache_dir = args.ct_cache_dir.resolve() if args.ct_cache_dir is not None else data_dir / ".remote_volume_cache"
    totalseg_prior_dir = args.totalseg_prior_dir.resolve() if args.totalseg_prior_dir is not None else None
    totalseg_labels = tuple(str(label) for label in args.totalseg_labels)
    if totalseg_prior_dir is not None and not totalseg_prior_dir.is_dir():
        raise FileNotFoundError(f"TotalSegmentator prior directory does not exist: {totalseg_prior_dir}")
    train_cases = load_entries(data_dir, "train", require_masks=True)
    train_cases = select_cases(train_cases, args.train_cases)
    train_cases, dev_cases = split_train_cases(train_cases, args.dev_fraction, args.seed)
    if args.max_cases_for_smoke > 0:
        train_cases = select_cases(train_cases, args.max_cases_for_smoke)
        dev_cases = select_cases(dev_cases, max(1, min(args.max_cases_for_smoke, len(dev_cases))))
    source = VolumeSource(
        root=volume_root,
        repo_id=args.ct_repo,
        cache_dir=cache_dir,
        allow_remote=args.allow_remote_download,
        token=read_hf_token(),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, token=source.token)
    max_text_length = int(getattr(tokenizer.model_max_length, "__int__", lambda: 256)())
    max_text_length = min(max_text_length, 256)
    train_dataset = RexPatchDataset(
        records_from_entries(train_cases),
        source=source,
        patch_shape=tuple(args.patch_shape),
        patches_per_finding=args.patches_per_finding,
        positive_ratio=args.positive_ratio,
        tokenizer=tokenizer,
        totalseg_prior_dir=totalseg_prior_dir,
        totalseg_labels=totalseg_labels,
        max_text_length=max_text_length,
        seed=args.seed,
        training=True,
    )
    dev_dataset = RexPatchDataset(
        records_from_entries(dev_cases),
        source=source,
        patch_shape=tuple(args.patch_shape),
        patches_per_finding=1,
        positive_ratio=1.0,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        seed=args.seed + 11,
        totalseg_prior_dir=totalseg_prior_dir,
        totalseg_labels=totalseg_labels,
        training=False,
    )
    collator = make_collator(tokenizer, max_text_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for the tiny smoke path")
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = TextConditionedCTFM(
        prior_channels=len(totalseg_labels) if totalseg_prior_dir is not None else 0,
        visual_model=args.visual_model,
        text_model=args.text_model,
        patch_shape=tuple(args.patch_shape),
        token=source.token,
        train_visual=args.train_visual,
        train_text=args.train_text,
    ).to(device)
    optimizer = build_optimizer(model, args)
    start_step = 0
    start_epoch = 0
    if args.checkpoint is not None:
        state = load_checkpoint(args.checkpoint, model, optimizer)
        start_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = DiceBCELoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    microsteps = 0
    epoch = start_epoch
    best_dice = -math.inf
    history: list[dict[str, Any]] = []
    start_time = time.time()
    while epoch < args.epochs and step < args.max_steps:
        train_dataset.set_epoch(epoch)
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            with autocast_context(device, amp_dtype):
                logits = model(
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("region_priors"),
                )
                loss = criterion(logits, batch["target"])
                region_priors = batch.get("region_priors")
                if region_priors is not None and args.region_guidance_weight > 0.0:
                    loss = loss + args.region_guidance_weight * region_guidance_penalty(
                        logits,
                        region_priors,
                        batch["region_confidence"],
                    )
                loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}")
            loss.backward()
            microsteps += 1
            if microsteps % args.gradient_accumulation == 0:
                clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 10 == 0:
                    payload = {"step": step, "loss": float(loss.detach().float().cpu()) * args.gradient_accumulation}
                    if device.type == "cuda":
                        payload["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                    print("train_step", json.dumps(payload, sort_keys=True))
                if args.eval_every > 0 and step % args.eval_every == 0:
                    metrics = evaluate_patch_loss(
                        model, dev_loader, device=device, amp_dtype=amp_dtype, max_batches=args.dev_batches
                    )
                    history.append({"step": step, **metrics})
                    print("dev_patch", json.dumps({"step": step, **metrics}, sort_keys=True))
                    if metrics["dice"] > best_dice:
                        best_dice = metrics["dice"]
                        save_checkpoint(
                            output_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            epoch=epoch,
                            args=args,
                            metrics=metrics,
                        )
                if step >= args.max_steps:
                    break
        epoch += 1
    if microsteps % args.gradient_accumulation:
        clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        step += 1
    final_metrics = evaluate_patch_loss(
        model, dev_loader, device=device, amp_dtype=amp_dtype, max_batches=args.dev_batches
    )
    save_checkpoint(
        output_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        epoch=epoch,
        args=args,
        metrics=final_metrics,
    )
    summary = {
        "run_kind": "rexgroundingct_ctfm_text_conditioned_fpn",
        "seed": args.seed,
        "device": str(device),
        "visual_model": args.visual_model,
        "text_model": args.text_model,
        "visual_trainable": args.train_visual,
        "text_trainable": args.train_text,
        "feature_channels": model.feature_channels,
        "totalseg_prior_dir": None if totalseg_prior_dir is None else str(totalseg_prior_dir),
        "totalseg_labels": list(totalseg_labels),
        "report_region_guided": totalseg_prior_dir is not None,
        "region_guidance_weight": args.region_guidance_weight,
        "prior_channels": model.prior_channels,
        "data_dir": str(data_dir),
        "volume_root": str(volume_root),
        "metadata_sha256": sha256_file(data_dir / "MICCAI_challenge_dataset.json"),
        "train_cases": len(train_cases),
        "dev_cases": len(dev_cases),
        "train_findings": len(train_dataset.records),
        "dev_findings": len(dev_dataset.records),
        "patch_shape": list(args.patch_shape),
        "patches_per_finding": args.patches_per_finding,
        "optimizer_steps": step,
        "final_dev_patch": final_metrics,
        "best_dev_dice": None if best_dice == -math.inf else best_dice,
        "history": history,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("summary", json.dumps({key: value for key, value in summary.items() if key != "history"}, sort_keys=True))
    return 0


def starts_for_axis(size: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    final = size - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def flush_prediction_batch(
    model: TextConditionedCTFM,
    patch_batch: list[np.ndarray],
    locations: list[tuple[int, int, int]],
    encoded: dict[str, Tensor],
    accumulated: np.ndarray,
    weights: np.ndarray,
    patch_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
    device: torch.device,
    amp_dtype: torch.dtype,
    region_patch_batch: list[np.ndarray] | None = None,
) -> None:
    if not patch_batch:
        return
    if region_patch_batch is not None and len(region_patch_batch) != len(patch_batch):
        raise ValueError("prediction CT and report-region patch batches have different lengths")
    batch_array = torch.from_numpy(np.stack(patch_batch)[:, None]).to(device=device, dtype=torch.float32)
    region_array = (
        torch.from_numpy(np.stack(region_patch_batch)).to(device=device, dtype=torch.float32)
        if region_patch_batch is not None
        else None
    )
    text_ids = encoded["input_ids"].expand(len(patch_batch), -1)
    text_mask = encoded["attention_mask"].expand(len(patch_batch), -1).to(dtype=torch.bool)
    with torch.no_grad(), autocast_context(device, amp_dtype):
        logits = (
            model(batch_array, text_ids, text_mask)
            if region_array is None
            else model(batch_array, text_ids, text_mask, region_array)
        )
    probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
    for probability, (z, y, x) in zip(probabilities, locations, strict=True):
        dz, dy, dx = patch_shape
        z1 = min(z + dz, volume_shape[0])
        y1 = min(y + dy, volume_shape[1])
        x1 = min(x + dx, volume_shape[2])
        accumulated[z:z1, y:y1, x:x1] += probability[: z1 - z, : y1 - y, : x1 - x]
        weights[z:z1, y:y1, x:x1] += 1.0
    patch_batch.clear()
    locations.clear()
    if region_patch_batch is not None:
        region_patch_batch.clear()


def predict_case(
    model: TextConditionedCTFM,
    entry: ChallengeEntry,
    *,
    source: VolumeSource,
    patch_shape: tuple[int, int, int],
    stride: tuple[int, int, int],
    batch_size: int,
    threshold: float,
    tokenizer: Any,
    max_text_length: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    totalseg_prior_dir: Path | None = None,
    totalseg_labels: Sequence[str] = (),
    region_postprocess_min_confidence: float = 0.6,
    region_postprocess_halo_mm: float = 15.0,
    region_postprocess_overlap: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    volume = source.load(entry.name)
    volume_dhw = np.moveaxis(volume.array_hwd, -1, 0)
    organ_prior = (
        load_total_segmentator_prior(
            totalseg_prior_dir,
            entry.name,
            totalseg_labels,
            entry.shape,
            volume.affine,
        )
        if totalseg_prior_dir is not None
        else None
    )
    starts = [
        starts_for_axis(int(size), int(patch), int(step))
        for size, patch, step in zip(volume_dhw.shape, patch_shape, stride, strict=True)
    ]
    finding_texts = [f"segment finding: {entry.findings[key]}" for key in sorted(entry.findings, key=_finding_sort_key)]
    predictions: list[np.ndarray] = []
    model.eval()
    for text in finding_texts:
        encoded_raw = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded_raw.items()}
        selected_prior = (
            select_report_prior(organ_prior, totalseg_labels, resolve_report_region(text))
            if organ_prior is not None
            else None
        )
        accumulated = np.zeros(volume_dhw.shape, dtype=np.float32)
        weights = np.zeros(volume_dhw.shape, dtype=np.float32)
        patch_batch: list[np.ndarray] = []
        region_patch_batch: list[np.ndarray] | None = [] if selected_prior is not None else None
        locations: list[tuple[int, int, int]] = []
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    origin = (z, y, x)
                    patch = normalize_ct(extract_patch(volume_dhw, origin, patch_shape, fill=-1024.0))
                    patch_batch.append(patch)
                    if region_patch_batch is not None and selected_prior is not None:
                        region_patch_batch.append(
                            np.stack(
                                [extract_patch(channel, origin, patch_shape, fill=0) for channel in selected_prior],
                                axis=0,
                            ).astype(np.float32, copy=False)
                        )
                    locations.append(origin)
                    if len(patch_batch) >= batch_size:
                        flush_prediction_batch(
                            model,
                            patch_batch,
                            locations,
                            encoded,
                            accumulated,
                            weights,
                            patch_shape,
                            tuple(int(value) for value in volume_dhw.shape),
                            device,
                            amp_dtype,
                            region_patch_batch=region_patch_batch,
                        )
        flush_prediction_batch(
            model,
            patch_batch,
            locations,
            encoded,
            accumulated,
            weights,
            patch_shape,
            tuple(int(value) for value in volume_dhw.shape),
            device,
            amp_dtype,
            region_patch_batch=region_patch_batch,
        )
        predictions.append((accumulated / np.maximum(weights, 1.0) >= threshold).astype(np.uint8))
    output_dhw = np.stack(predictions, axis=0)
    output_hwd = np.transpose(output_dhw, (0, 2, 3, 1))
    if organ_prior is not None:
        output_hwd = filter_prediction_by_region(
            output_hwd,
            finding_texts,
            organ_prior,
            totalseg_labels,
            volume.affine,
            min_confidence=region_postprocess_min_confidence,
            halo_mm=region_postprocess_halo_mm,
            min_component_overlap=region_postprocess_overlap,
        )
    return output_hwd, volume.affine


def predict(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    data_dir = args.data_dir.resolve()
    require_masks = args.split != "test"
    entries = select_cases(load_entries(data_dir, args.split, require_masks=require_masks), args.predict_cases)
    if args.max_cases_for_smoke > 0:
        entries = select_cases(entries, args.max_cases_for_smoke)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in --mode predict")
    volume_root = args.volume_root.resolve() if args.volume_root is not None else data_dir / "ct_rate_fixed"
    cache_dir = args.ct_cache_dir.resolve() if args.ct_cache_dir is not None else data_dir / ".remote_volume_cache"
    totalseg_prior_dir = args.totalseg_prior_dir.resolve() if args.totalseg_prior_dir is not None else None
    totalseg_labels = tuple(str(label) for label in args.totalseg_labels)
    if totalseg_prior_dir is not None and not totalseg_prior_dir.is_dir():
        raise FileNotFoundError(f"TotalSegmentator prior directory does not exist: {totalseg_prior_dir}")
    source = VolumeSource(
        root=volume_root,
        repo_id=args.ct_repo,
        cache_dir=cache_dir,
        allow_remote=args.allow_remote_download,
        token=read_hf_token(),
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, token=source.token)
    max_text_length = min(int(getattr(tokenizer.model_max_length, "__int__", lambda: 256)()), 256)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = TextConditionedCTFM(
        visual_model=args.visual_model,
        text_model=args.text_model,
        patch_shape=tuple(args.patch_shape),
        token=source.token,
        prior_channels=len(totalseg_labels) if totalseg_prior_dir is not None else 0,
        train_visual=False,
        train_text=False,
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    output_dir = args.output_dir / "predictions" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries, 1):
        prediction, affine = predict_case(
            model,
            entry,
            source=source,
            patch_shape=tuple(args.patch_shape),
            stride=tuple(args.inference_stride),
            batch_size=args.inference_batch_size,
            threshold=args.threshold,
            totalseg_prior_dir=totalseg_prior_dir,
            totalseg_labels=totalseg_labels,
            region_postprocess_min_confidence=args.region_postprocess_min_confidence,
            region_postprocess_halo_mm=args.region_postprocess_halo_mm,
            region_postprocess_overlap=args.region_postprocess_overlap,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            device=device,
            amp_dtype=amp_dtype,
        )
        nib.save(nib.Nifti1Image(prediction, affine), str(output_dir / entry.name))
        print("predicted", json.dumps({"index": index, "total": len(entries), "file": entry.name}, sort_keys=True))
    manifest = {
        "split": args.split,
        "cases": [entry.name for entry in entries],
        "threshold": args.threshold,
        "totalseg_prior_dir": None if totalseg_prior_dir is None else str(totalseg_prior_dir),
        "totalseg_labels": list(totalseg_labels),
        "report_region_postprocess": totalseg_prior_dir is not None,
        "region_postprocess_min_confidence": args.region_postprocess_min_confidence,
        "region_postprocess_halo_mm": args.region_postprocess_halo_mm,
        "region_postprocess_overlap": args.region_postprocess_overlap,
        "patch_shape": list(args.patch_shape),
        "stride": list(args.inference_stride),
        "checkpoint": str(args.checkpoint),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> int:
    args = parse_args()
    return train(args) if args.mode == "train" else predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
