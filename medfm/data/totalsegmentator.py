"""Optional TotalSegmentator anatomical-prior integration for CT grounding."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

# These classes belong to TotalSegmentator's openly available ``total`` CT task.
# Keep the prior set thorax-focused and avoid licensed high-resolution subtasks.
DEFAULT_THORACIC_LABELS: tuple[str, ...] = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "heart",
    "aorta",
    "trachea",
    "esophagus",
    "pulmonary_vein",
)


class TotalSegmentatorPriorError(RuntimeError):
    """Raised when an anatomical-prior artifact violates the spatial contract."""


def prior_case_id(volume_name: str) -> str:
    if not volume_name.endswith(".nii.gz"):
        raise ValueError(f"volume name must end in .nii.gz: {volume_name!r}")
    return volume_name[: -len(".nii.gz")]


def prior_case_dir(root: Path, volume_name: str) -> Path:
    return root / prior_case_id(volume_name)


def prior_paths(root: Path, volume_name: str, labels: Sequence[str]) -> tuple[Path, ...]:
    case_dir = prior_case_dir(root, volume_name)
    return tuple(case_dir / f"{label}.nii.gz" for label in labels)


def load_total_segmentator_prior(
    root: Path,
    volume_name: str,
    labels: Sequence[str],
    expected_shape_hwd: Sequence[int],
    expected_affine: np.ndarray | None = None,
) -> np.ndarray:
    """Load binary TotalSegmentator masks as ``[C,D,H,W]`` float32 data.

    TotalSegmentator writes one NIfTI per requested class. The loader refuses
    missing files, shape changes, and affine changes instead of silently
    resampling an anatomical prior onto an unrelated voxel grid.
    """

    normalized_labels = tuple(str(label) for label in labels)
    if not normalized_labels:
        raise ValueError("at least one TotalSegmentator label is required")
    expected_shape = tuple(int(value) for value in expected_shape_hwd)
    if len(expected_shape) != 3 or any(value <= 0 for value in expected_shape):
        raise ValueError(f"expected_shape_hwd must contain three positive values, got {expected_shape}")
    arrays: list[np.ndarray] = []
    for label, path in zip(normalized_labels, prior_paths(root, volume_name, normalized_labels), strict=True):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing TotalSegmentator prior {path}; run prepare_totalsegmentator_priors.py for {volume_name}"
            )
        image = cast(Any, nib.load(str(path)))
        actual_shape = tuple(int(value) for value in image.shape)
        if actual_shape != expected_shape:
            raise TotalSegmentatorPriorError(
                f"TotalSegmentator prior {label!r} for {volume_name} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )
        if expected_affine is not None and not np.allclose(image.affine, expected_affine, rtol=0.0, atol=1e-4):
            raise TotalSegmentatorPriorError(
                f"TotalSegmentator prior {label!r} for {volume_name} has a mismatched affine"
            )
        arrays.append((np.asarray(image.dataobj) > 0).astype(np.float32, copy=False))
    stacked_hwd = np.stack(arrays, axis=0)
    return np.ascontiguousarray(np.moveaxis(stacked_hwd, -1, 1))


def build_totalsegmentator_command(
    input_path: Path,
    output_dir: Path,
    labels: Sequence[str],
    *,
    device: str = "gpu",
    fast: bool = False,
    executable: str = "TotalSegmentator",
) -> list[str]:
    """Build a deterministic CLI invocation without invoking a shell."""

    normalized_labels = tuple(str(label) for label in labels)
    if not normalized_labels:
        raise ValueError("at least one TotalSegmentator label is required")
    command = [
        executable,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-ta",
        "total",
        "--roi_subset",
        *normalized_labels,
        "--device",
        str(device),
    ]
    if fast:
        command.append("--fast")
    return command


def run_totalsegmentator(
    input_path: Path,
    output_dir: Path,
    labels: Sequence[str],
    *,
    device: str = "gpu",
    fast: bool = False,
    executable: str = "TotalSegmentator",
) -> tuple[Path, ...]:
    """Run TotalSegmentator once and verify all requested output files exist."""

    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        raise TotalSegmentatorPriorError(
            "TotalSegmentator is not installed; install the optional dependency with "
            "`uv sync --extra totalsegmentator` or `uv pip install TotalSegmentator==2.17.0`"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"CT input does not exist: {input_path}")
    normalized_labels = tuple(str(label) for label in labels)
    if not normalized_labels:
        raise ValueError("at least one TotalSegmentator label is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The caller passes the case directory, so output paths are explicit.
    paths = tuple(output_dir / f"{label}.nii.gz" for label in normalized_labels)
    if all(path.is_file() for path in paths):
        return paths
    if any(path.exists() for path in paths):
        raise TotalSegmentatorPriorError(
            f"partial TotalSegmentator output exists in {output_dir}; remove it before retrying"
        )
    command = build_totalsegmentator_command(
        input_path,
        output_dir,
        normalized_labels,
        device=device,
        fast=fast,
        executable=resolved_executable,
    )
    subprocess.run(command, check=True)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise TotalSegmentatorPriorError(f"TotalSegmentator completed without requested outputs: {missing}")
    return paths


__all__ = [
    "DEFAULT_THORACIC_LABELS",
    "TotalSegmentatorPriorError",
    "build_totalsegmentator_command",
    "load_total_segmentator_prior",
    "prior_case_dir",
    "prior_case_id",
    "prior_paths",
    "run_totalsegmentator",
]
