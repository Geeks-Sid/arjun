"""Device-transfer fixtures on real accelerator hardware (protected).

CPU coverage lives in test_batch.py; these run only on guarded runners
(make test-gpu / make test-tpu).
"""

import pytest
import torch
from contract_fixtures import make_spatial

from medfm.core import MedicalBatch, Modality
from medfm.core.serialization import materialize_cpu


def _assert_transfer_preserves_metadata(batch: MedicalBatch, device: torch.device) -> None:
    moved = batch.to(device)
    assert moved.device is not None and moved.device.type == device.type
    assert moved.modality is batch.modality
    assert moved.sample_ids == batch.sample_ids
    assert moved.bucket == batch.bucket
    for original, transferred in zip(batch.spatial_metadata, moved.spatial_metadata, strict=True):
        if original is None:
            assert transferred is None
        else:
            assert transferred.spacing_mm == original.spacing_mm
            assert torch.equal(materialize_cpu(transferred.affine), materialize_cpu(original.affine))
    # Round-trip back to CPU keeps tensor values.
    back = moved.to("cpu")
    assert torch.equal(back.pixel_values, batch.pixel_values)


@pytest.mark.gpu
def test_batch_transfer_on_cuda():
    if not torch.cuda.is_available():
        pytest.fail("MEDFM_RUN_GPU_TESTS=1 but no CUDA device is available")
    batch = MedicalBatch(
        modality=Modality.CT_3D,
        pixel_values=torch.randn(2, 1, 8, 16, 16),
        image_mask=torch.ones(2, dtype=torch.bool),
        spatial_metadata=[make_spatial((8, 16, 16)), None],
        sample_ids=["a", "b"],
    )
    _assert_transfer_preserves_metadata(batch, torch.device("cuda"))


@pytest.mark.tpu
def test_batch_transfer_on_xla():
    import torch_xla  # noqa: F401 — lazy import; TPU-only dependency
    import torch_xla.runtime as xr

    if xr.device_type() != "TPU":
        pytest.fail("MEDFM_RUN_TPU_TESTS=1 but PJRT device is not TPU")
    import torch_xla.core.xla_model as xm

    batch = MedicalBatch(
        modality=Modality.CT_3D,
        pixel_values=torch.randn(2, 1, 8, 16, 16),
        image_mask=torch.ones(2, dtype=torch.bool),
        spatial_metadata=[make_spatial((8, 16, 16)), None],
        sample_ids=["a", "b"],
    )
    _assert_transfer_preserves_metadata(batch, xm.xla_device())
