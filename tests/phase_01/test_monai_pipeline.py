"""Synthetic MONAI 3D load/crop pipeline (no downloads, CPU)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def test_synthetic_nifti_load_crop_preserves_affine(tmp_path: Path):
    from monai.transforms.compose import Compose
    from monai.transforms.io.array import LoadImage
    from monai.transforms.spatial.array import CenterSpatialCrop
    from monai.transforms.utility.array import EnsureChannelFirst

    rng = np.random.default_rng(42)
    volume = rng.normal(size=(32, 48, 40)).astype(np.float32)
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    nifti_path = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), nifti_path)

    pipeline = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            CenterSpatialCrop(roi_size=(16, 24, 20)),
        ]
    )
    image = pipeline(str(nifti_path))

    assert tuple(image.shape) == (1, 16, 24, 20)
    # Spacing/direction are preserved; the crop shifts the origin by the crop
    # offset (8, 12, 10 voxels) times spacing (1.5, 2.0, 2.5 mm), and the
    # pre-crop affine stays recorded under "original_affine".
    expected_affine = affine.copy()
    expected_affine[:3, 3] = [12.0, 24.0, 25.0]
    assert np.allclose(np.asarray(image.meta["affine"]), expected_affine, atol=1e-4)
    assert np.allclose(np.asarray(image.meta["original_affine"]), affine, atol=1e-4)
    # Center crop of a random volume must match the source data exactly.
    cropped_source = volume[8:24, 12:36, 10:30]
    assert np.allclose(np.asarray(image)[0], cropped_source, atol=1e-5)


def test_monai_pipeline_import_does_not_initialize_cuda():
    """Run in a subprocess: other in-process tests legitimately probe CUDA."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; import torch; import monai; "
        "assert not torch.cuda.is_initialized(), 'CUDA initialized on import'; "
        "assert 'torch_xla' not in sys.modules, 'torch_xla imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
