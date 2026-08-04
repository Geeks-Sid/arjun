"""Phase smoke runner: ``python -m medfm.tools.smoke --phase <NN>``.

Phase 01 smoke (CPU-safe, no weight downloads):
  1. doctor JSON validates against medfm/tools/doctor_schema.json,
  2. a synthetic NIfTI volume round-trips through a MONAI load/crop pipeline,
  3. a tiny local LoRA-wrapped model performs one optimization step.

Exits 0 on success, 1 on failure, 2 on usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from medfm.tools import governance as gov


def _check_doctor_json() -> None:
    from jsonschema import Draft202012Validator

    from medfm.tools import doctor

    report = doctor.collect(backend="cpu")
    schema = gov.load_json(Path(__file__).resolve().parent / "doctor_schema.json")
    errors = sorted(e.message for e in Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise RuntimeError(f"doctor report failed schema validation: {errors}")


def _check_monai_3d_pipeline() -> None:
    import nibabel as nib
    import numpy as np
    from monai.transforms.compose import Compose
    from monai.transforms.io.array import LoadImage
    from monai.transforms.spatial.array import CenterSpatialCrop
    from monai.transforms.utility.array import EnsureChannelFirst

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.nii.gz"
        rng = np.random.default_rng(0)
        volume = rng.normal(size=(32, 48, 40)).astype(np.float32)
        affine = np.diag([1.5, 2.0, 2.5, 1.0])
        nib.save(nib.Nifti1Image(volume, affine), path)

        pipeline = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                CenterSpatialCrop(roi_size=(16, 24, 20)),
            ]
        )
        image = pipeline(str(path))
        if tuple(image.shape) != (1, 16, 24, 20):
            raise RuntimeError(f"unexpected crop shape: {tuple(image.shape)}")
        # Center crop shifts the affine origin by offset (8, 12, 10) * spacing
        # (1.5, 2.0, 2.5); the pre-crop affine must stay recorded.
        expected = affine.copy()
        expected[:3, 3] = [12.0, 24.0, 25.0]
        recorded = np.asarray(image.meta["affine"])
        original = np.asarray(image.meta["original_affine"])
        if not (np.allclose(recorded, expected, atol=1e-4) and np.allclose(original, affine, atol=1e-4)):
            raise RuntimeError(f"affine not preserved: {recorded}")


def _check_tiny_lora_step() -> None:
    from typing import Any, cast

    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"])
    # get_peft_model's stubs expect PreTrainedModel, but any nn.Module works.
    model = get_peft_model(cast(Any, base), config)

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    if not trainable or not frozen:
        raise RuntimeError("LoRA wrap produced no trainable or no frozen parameters")

    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    inputs = torch.randn(8, 16)
    targets = torch.randint(0, 4, (8,))

    model.train()
    before = model(inputs).sum().item()
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable):
        raise RuntimeError("no gradients reached the LoRA parameters")
    optimizer.step()
    after = model(inputs).sum().item()
    if before == after:
        raise RuntimeError("model output unchanged after one optimization step")


def _check_core_sample_roundtrip() -> None:
    import torch

    from medfm.core import (
        ImageReference,
        MedicalSample,
        Modality,
        ProvenanceMetadata,
        SpatialMetadata,
        canonical_json,
    )

    sample = MedicalSample(
        sample_id="smoke-ct3d",
        patient_id_hash="ab12" * 16,
        modality=Modality.CT_3D,
        image_references=(ImageReference(uri="s3://bucket/volume.nii.gz"),),
        spatial=SpatialMetadata(
            original_shape=(8, 16, 16),
            current_shape=(8, 16, 16),
            affine=torch.eye(4, dtype=torch.float64),
            spacing_mm=(2.0, 2.0, 2.0),
        ),
        provenance=ProvenanceMetadata(dataset_name="synthetic", dataset_version="0.1"),
    )
    blob = canonical_json(sample.to_dict())
    import json

    restored = MedicalSample.from_dict(json.loads(blob))
    if restored.to_dict() != sample.to_dict():
        raise RuntimeError("MedicalSample canonical round-trip is not lossless")


def _check_core_batch_contract() -> None:
    import torch

    from medfm.core import BucketId, BucketKind, MedicalBatch, Modality, ShapeContractError

    batch = MedicalBatch(
        modality=Modality.PATHOLOGY_WSI,
        pixel_values=torch.randn(2, 4, 3, 8, 8),
        image_mask=torch.ones(2, 4, dtype=torch.bool),
        tile_coordinates=torch.zeros(2, 4, 2, dtype=torch.int64),
        sample_ids=["a", "b"],
        bucket=BucketId(kind=BucketKind.WSI_TILES, shape=(4,)),
    )
    moved = batch.to("cpu")
    if moved.modality is not Modality.PATHOLOGY_WSI or moved.sample_ids != ["a", "b"]:
        raise RuntimeError("device transfer lost non-tensor metadata")
    try:
        MedicalBatch(
            modality=Modality.XRAY_2D,
            pixel_values=torch.randn(2, 1, 8, 8, 8),
            sample_ids=["a", "b"],
        )
    except ShapeContractError:
        pass
    else:
        raise RuntimeError("rank/modality mismatch was not rejected")


PHASE_01_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("doctor_json_schema", _check_doctor_json),
    ("monai_3d_load_crop", _check_monai_3d_pipeline),
    ("tiny_lora_step", _check_tiny_lora_step),
]

PHASE_02_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("core_sample_roundtrip", _check_core_sample_roundtrip),
    ("core_batch_contract", _check_core_batch_contract),
]

SMOKE_CHECKS: dict[str, list[tuple[str, Callable[[], None]]]] = {
    "01": PHASE_01_CHECKS,
    "02": PHASE_02_CHECKS,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the phase smoke checks.")
    parser.add_argument("--phase", required=True, help="phase number, e.g. 01")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    phase = args.phase.zfill(2)

    checks = SMOKE_CHECKS.get(phase)
    if checks is None:
        print(f"no smoke checks registered for phase {phase}", file=sys.stderr)
        return 2

    results: list[dict[str, str]] = []
    failed = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            results.append({"name": name, "status": "failed", "detail": str(exc)})
            failed += 1
        else:
            results.append({"name": name, "status": "passed", "detail": ""})

    if args.json:
        print(json.dumps({"phase": phase, "checks": results}, indent=2, sort_keys=True))
    else:
        for r in results:
            print(f"[{r['status']}] {r['name']}" + (f" — {r['detail']}" if r["detail"] else ""))
    if failed:
        print(f"smoke FAILED ({failed}/{len(results)} checks)", file=sys.stderr)
        return 1
    print(f"smoke passed ({len(results)} checks)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
