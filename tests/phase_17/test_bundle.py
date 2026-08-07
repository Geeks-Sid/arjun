from __future__ import annotations

import json

import pytest
import torch

from medfm.inference import (
    BundleChecksumError,
    BundleCompatibilityError,
    BundleValidationError,
    load_bundle,
)


def test_adapter_only_bundle_is_independent_and_checksum_verified(bundle_builder) -> None:
    bundle_builder.add_adapter("classification", {"weight": torch.ones(2, requires_grad=True)})
    bundle_builder.add_tensor_group("bridge", {"projection": torch.eye(2)})
    bundle_builder.set_calibration({"temperature": 1.2})
    bundle_builder.add_example("request.json", '{"task": "classification"}\n')
    bundle = bundle_builder.build()

    loaded = load_bundle(bundle.root, expected_base={"model_id": "base-model", "revision": "base-rev"})
    assert loaded.manifest.adapter_only is True
    assert loaded.adapter_names == ("classification",)
    assert loaded.load_adapter("classification")["weight"].device.type == "cpu"
    checksums = json.loads((bundle.root / "checksums.json").read_text(encoding="utf-8"))
    assert "files" in checksums and "bundle.json" in checksums["files"]

    with pytest.raises(BundleCompatibilityError):
        load_bundle(bundle.root, base_revision="wrong-revision")

    tensor_path = bundle.root / "adapters" / "classification" / "model.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")
    with pytest.raises(BundleChecksumError):
        load_bundle(bundle.root)


def test_bundle_rejects_resume_artifacts_and_unsafe_metadata(bundle_builder) -> None:
    bundle_builder.add_adapter("a", {"weight": torch.ones(1)})
    bundle = bundle_builder.build()
    (bundle.root / "optimizer").mkdir()
    (bundle.root / "optimizer" / "state.pt").write_bytes(b"not a deployment artifact")
    with pytest.raises((BundleChecksumError, BundleValidationError)):
        load_bundle(bundle.root)


def test_merged_artifact_is_secondary_and_requires_documented_conversion(bundle_builder) -> None:
    with pytest.raises(BundleValidationError):
        bundle_builder.add_tensor_group("merged", {"weight": torch.ones(1)})
    bundle_builder.add_adapter("a", {"weight": torch.ones(1)})
    bundle_builder.add_merged_artifact({"weight": torch.ones(1)}, explicit_conversion="reviewed merge step 17.1")
    bundle = bundle_builder.build()
    assert bundle.manifest.tensor_groups == ("adapters/a",)
    assert bundle.manifest.merged_artifacts == ("merged",)
