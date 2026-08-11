from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from medfm.data.anatomy_grounding import (
    filter_prediction_by_region,
    resolve_report_region,
    select_report_prior,
)
from medfm.data.totalsegmentator import (
    DEFAULT_THORACIC_LABELS,
    TotalSegmentatorPriorError,
    build_totalsegmentator_command,
    load_total_segmentator_prior,
)
from scripts.train_rexgrounding import (
    ChallengeEntry,
    FindingRecord,
    RexPatchDataset,
    VolumeSource,
    ct_rate_path,
    extract_patch,
    normalize_ct,
    predict_case,
    region_guidance_penalty,
    starts_for_axis,
)


def test_ct_rate_path_preserves_train_and_valid_fixed_namespaces() -> None:
    assert ct_rate_path("train_1741_b_2.nii.gz") == (
        "dataset/train_fixed/train_1741/train_1741_b/train_1741_b_2.nii.gz"
    )
    assert ct_rate_path("valid_827_a_1.nii.gz") == ("dataset/valid_fixed/valid_827/valid_827_a/valid_827_a_1.nii.gz")


def test_patch_extraction_pads_and_clips_without_changing_shape() -> None:
    source = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    patch = extract_patch(source, origin=(-1, 2, 4), patch_shape=(3, 4, 4), fill=-1.0)
    assert patch.shape == (3, 4, 4)
    assert patch[0, 0, 0] == -1.0
    assert normalize_ct(np.array([-1024.0, 2048.0])).tolist() == [0.0, 1.0]
    assert starts_for_axis(200, 96, 64) == [0, 64, 104]


def test_totalsegmentator_prior_loader_keeps_geometry_and_binary_channels() -> None:
    labels = DEFAULT_THORACIC_LABELS[:2]
    shape_hwd = (4, 5, 6)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        case_dir = root / "train_1_a_1"
        case_dir.mkdir()
        first = np.zeros(shape_hwd, dtype=np.uint8)
        first[1, 2, 3] = 1
        second = np.full(shape_hwd, 2, dtype=np.uint8)
        for label, array in zip(labels, (first, second), strict=True):
            nib.save(nib.Nifti1Image(array, affine), str(case_dir / f"{label}.nii.gz"))
        prior = load_total_segmentator_prior(root, "train_1_a_1.nii.gz", labels, shape_hwd, affine)
        assert prior.shape == (2, 6, 4, 5)
        assert prior.dtype == np.float32
        assert prior[0, 3, 1, 2] == 1.0
        assert set(np.unique(prior[1])) == {1.0}
        nib.save(nib.Nifti1Image(second, np.eye(4)), str(case_dir / f"{labels[1]}.nii.gz"))
        with pytest.raises(TotalSegmentatorPriorError, match="mismatched affine"):
            load_total_segmentator_prior(root, "train_1_a_1.nii.gz", labels, shape_hwd, affine)


def test_totalsegmentator_command_is_shell_free_and_uses_total_task() -> None:
    command = build_totalsegmentator_command(
        Path("input.nii.gz"),
        Path("priors/case"),
        DEFAULT_THORACIC_LABELS[:2],
        device="gpu",
        fast=True,
    )
    assert command[:8] == [
        "TotalSegmentator",
        "-i",
        "input.nii.gz",
        "-o",
        "priors/case",
        "-ta",
        "total",
        "--roi_subset",
    ]
    assert command[-3:] == ["--device", "gpu", "--fast"]
    assert "--fast" in command


def test_prediction_restores_released_hwd_mask_geometry() -> None:
    model = type(
        "ZeroModel",
        (nn.Module,),
        {"forward": lambda self, x, input_ids, attention_mask: torch.zeros((x.shape[0], 1, *x.shape[-3:]))},
    )()

    def tokenizer(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        volume_path = root / "dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz"
        volume_path.parent.mkdir(parents=True)
        volume = np.zeros((8, 10, 12), dtype=np.float32)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(volume_path))
        entry = ChallengeEntry("test", volume_path.name, {"0": "left lesion"}, {}, volume.shape, None)
        prediction, affine = predict_case(
            model,
            entry,
            source=VolumeSource(
                root=root,
                repo_id="unused",
                cache_dir=root / "cache",
                allow_remote=False,
                token=None,
            ),
            patch_shape=(4, 4, 4),
            stride=(3, 3, 3),
            batch_size=2,
            threshold=0.5,
            tokenizer=tokenizer,
            max_text_length=8,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
        )
    assert prediction.shape == (1, 8, 10, 12)
    assert affine.shape == (4, 4)


def test_prediction_passes_selected_region_and_postprocesses_components() -> None:
    class RegionAwareModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[torch.Tensor] = []

        def forward(
            self,
            x: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            region_priors: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del input_ids, attention_mask
            assert region_priors is not None
            self.seen.append(region_priors.detach().clone())
            logits = torch.full((x.shape[0], 1, *x.shape[-3:]), -20.0, device=x.device)
            logits[:, :, 1, 1, 1] = 20.0
            logits[:, :, 0, 0, 0] = 20.0
            return logits

    def tokenizer(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del args, kwargs
        return {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    model = RegionAwareModel()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        name = "train_1_a_1.nii.gz"
        volume_path = root / "dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz"
        volume_path.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(volume_path))
        prior_root = root / "priors"
        case_dir = prior_root / "train_1_a_1"
        case_dir.mkdir(parents=True)
        lower_index = DEFAULT_THORACIC_LABELS.index("lung_lower_lobe_right")
        upper_index = DEFAULT_THORACIC_LABELS.index("lung_upper_lobe_right")
        for index, label in enumerate(DEFAULT_THORACIC_LABELS):
            prior = np.zeros((4, 4, 4), dtype=np.uint8)
            if index in (lower_index, upper_index):
                prior[1, 1, 1] = 1
            nib.save(nib.Nifti1Image(prior, np.eye(4)), str(case_dir / f"{label}.nii.gz"))
        entry = ChallengeEntry("train", name, {"0": "right lower lobe lesion"}, {}, (4, 4, 4), None)
        prediction, _ = predict_case(
            model,
            entry,
            source=VolumeSource(
                root=root,
                repo_id="unused",
                cache_dir=root / "cache",
                allow_remote=False,
                token=None,
            ),
            patch_shape=(4, 4, 4),
            stride=(4, 4, 4),
            batch_size=1,
            threshold=0.5,
            tokenizer=tokenizer,
            max_text_length=8,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            totalseg_prior_dir=prior_root,
            totalseg_labels=DEFAULT_THORACIC_LABELS,
            region_postprocess_halo_mm=0.0,
            region_postprocess_overlap=1.0,
        )
    assert len(model.seen) == 1
    assert model.seen[0][0, lower_index].sum() == 1.0
    assert model.seen[0][0, upper_index].sum() == 0.0
    assert prediction[0, 1, 1, 1] == 1
    assert prediction[0, 0, 0, 0] == 0


def test_report_region_selects_only_text_named_lobes() -> None:
    region = resolve_report_region("A nodule in the RLL and lingula")
    assert region.labels == ("lung_upper_lobe_left", "lung_lower_lobe_right")
    assert region.confidence == 1.0
    prior = np.zeros((len(DEFAULT_THORACIC_LABELS), 4, 5, 6), dtype=np.float32)
    lower_index = DEFAULT_THORACIC_LABELS.index("lung_lower_lobe_right")
    upper_index = DEFAULT_THORACIC_LABELS.index("lung_upper_lobe_right")
    prior[lower_index, 1, 2, 3] = 1.0
    prior[upper_index, 1, 2, 3] = 1.0
    selected = select_report_prior(prior, DEFAULT_THORACIC_LABELS, region)
    assert selected[lower_index, 1, 2, 3] == 1.0
    assert selected[upper_index].sum() == 0.0


def test_report_region_uses_soft_broad_lung_fallback() -> None:
    region = resolve_report_region("Diffuse abnormalities in both lungs")
    assert region.labels == DEFAULT_THORACIC_LABELS[:5]
    assert region.confidence == 0.65
    assert region.mode == "lung"
    assert resolve_report_region("pleural thickening").labels == ()


def test_region_postprocessor_keeps_near_component_and_removes_far_component() -> None:
    prior = np.zeros((len(DEFAULT_THORACIC_LABELS), 4, 5, 6), dtype=np.float32)
    lower_index = DEFAULT_THORACIC_LABELS.index("lung_lower_lobe_right")
    prior[lower_index, 1, 2, 3] = 1.0
    prediction = np.zeros((1, 5, 6, 4), dtype=np.uint8)
    prediction[0, 2, 3, 1] = 1
    prediction[0, 0, 0, 0] = 1
    filtered = filter_prediction_by_region(
        prediction,
        ("right lower lobe",),
        prior,
        DEFAULT_THORACIC_LABELS,
        np.eye(4),
        halo_mm=0.0,
        min_component_overlap=1.0,
    )
    assert filtered[0, 2, 3, 1] == 1
    assert filtered[0, 0, 0, 0] == 0


def test_region_guidance_penalty_is_zero_without_confident_region() -> None:
    logits = torch.zeros((1, 1, 5, 5, 5))
    region_priors = torch.zeros((1, len(DEFAULT_THORACIC_LABELS), 5, 5, 5))
    region_priors[:, 0, 2, 2, 2] = 1.0
    assert region_guidance_penalty(logits, region_priors, torch.tensor([0.0]), dilation_voxels=0).item() == 0.0
    assert region_guidance_penalty(logits, region_priors, torch.tensor([1.0]), dilation_voxels=0).item() > 0.0


def test_patch_dataset_emits_report_selected_prior_channels() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        name = "train_1_a_1.nii.gz"
        shape = (8, 8, 8)
        volume_path = root / "dataset/train_fixed/train_1/train_1_a/train_1_a_1.nii.gz"
        volume_path.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), str(volume_path))
        mask_path = root / "segmentations" / name
        mask_path.parent.mkdir()
        mask = np.zeros((1, *shape), dtype=np.uint8)
        mask[0, 3, 4, 2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
        prior_root = root / "priors"
        case_dir = prior_root / "train_1_a_1"
        case_dir.mkdir(parents=True)
        lower_index = DEFAULT_THORACIC_LABELS.index("lung_lower_lobe_right")
        upper_index = DEFAULT_THORACIC_LABELS.index("lung_upper_lobe_right")
        for index, label in enumerate(DEFAULT_THORACIC_LABELS):
            prior = np.zeros(shape, dtype=np.uint8)
            if index == lower_index:
                prior[3, 4, 2] = 1
            if index == upper_index:
                prior[3, 4, 2] = 1
            nib.save(nib.Nifti1Image(prior, np.eye(4)), str(case_dir / f"{label}.nii.gz"))
        entry = ChallengeEntry("train", name, {"0": "right lower lobe lesion"}, {}, shape, mask_path)
        dataset = RexPatchDataset(
            [FindingRecord(entry, "0", 0, entry.findings["0"])],
            source=VolumeSource(
                root=root,
                repo_id="unused",
                cache_dir=root / "cache",
                allow_remote=False,
                token=None,
            ),
            patch_shape=(4, 4, 4),
            patches_per_finding=1,
            positive_ratio=1.0,
            tokenizer=None,
            max_text_length=8,
            seed=2026,
            training=False,
            totalseg_prior_dir=prior_root,
            totalseg_labels=DEFAULT_THORACIC_LABELS,
        )
        sample = dataset[0]
    assert sample["region_confidence"] == 1.0
    assert sample["region_priors"].shape == (len(DEFAULT_THORACIC_LABELS), 4, 4, 4)
    assert sample["region_priors"][lower_index].sum() == 1.0
    assert sample["region_priors"][upper_index].sum() == 0.0
