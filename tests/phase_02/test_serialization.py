"""Canonical serialization: determinism, dtype names, config hashes, devices."""

import pytest
import torch
from contract_fixtures import make_batch, make_sample, make_spatial

from medfm.core import (
    Modality,
    SerializationError,
    TensorMeta,
    canonical_dtype_name,
    canonical_json,
    canonical_yaml,
    config_hash,
    dtype_from_canonical,
    materialize_cpu,
    register_schema_migration,
    tensor_from_data,
    tensor_to_data,
)
from medfm.core.errors import SchemaVersionError
from medfm.core.sample import MedicalSample, SpatialMetadata


def test_canonical_json_deterministic_regardless_of_key_order():
    left = {"b": 1, "a": [1, 2], "c": {"y": True, "x": None}}
    right = {"c": {"x": None, "y": True}, "a": [1, 2], "b": 1}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_yaml(left) == canonical_yaml(right)


def test_config_hash_stable_across_equivalent_configs():
    config_a = {"model": "medsiglip", "size": [448, 448], "precision": "bf16"}
    config_b = {"precision": "bf16", "size": [448, 448], "model": "medsiglip"}
    assert config_hash(config_a) == config_hash(config_b)
    assert config_hash(config_a) != config_hash({**config_a, "size": [224, 224]})


def test_accelerator_neutral_dtype_names_roundtrip():
    for name in ("float32", "float16", "bfloat16", "int64", "uint8", "bool"):
        assert canonical_dtype_name(dtype_from_canonical(name)) == name
    with pytest.raises(SerializationError, match="unknown canonical dtype"):
        dtype_from_canonical("nvfp4")


def test_tensor_metadata_never_contains_device():
    tensor = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    meta = TensorMeta.of(tensor)
    payload = meta.to_dict()
    assert payload == {"shape": [2, 3, 4], "dtype": "bfloat16"}
    assert "device" not in payload
    assert TensorMeta.from_dict(payload) == meta


def test_inline_tensor_limit_protects_payloads():
    big = torch.zeros(200, 200)  # 40k elements > inline metadata limit
    with pytest.raises(SerializationError, match="payload"):
        tensor_to_data(big)


def test_materialize_cpu_detaches():
    tensor = torch.randn(3, 3, requires_grad=True)
    cpu = materialize_cpu(tensor)
    assert cpu.device.type == "cpu"
    assert not cpu.requires_grad


def test_batch_serialization_has_no_device_requirement():
    batch = make_batch(Modality.PATHOLOGY_WSI)
    blob = canonical_json(batch.to_metadata_dict())
    assert "cuda" not in blob and "xla" not in blob and "device" not in blob


def test_sample_serialization_has_no_device_requirement():
    sample = make_sample(Modality.CT_3D)
    blob = canonical_json(sample.to_dict())
    assert "cuda" not in blob and "xla" not in blob and '"device"' not in blob


def test_schema_version_rejects_newer_payloads():
    data = make_spatial((4, 4, 4)).to_dict()
    data["schema_version"] = 99
    with pytest.raises(SchemaVersionError, match="newer|at most"):
        SpatialMetadata.from_dict(data)


def test_versioned_migration_hooks_upgrade_old_payloads():
    """Schema-migration and enum-migration hooks compose end to end."""
    from medfm.core import register_enum_migration

    register_enum_migration("Modality", 0, "CXR_2D", "XRAY_2D")

    def migrate_v0(payload):
        payload = dict(payload)
        payload["schema_version"] = 1
        return payload

    register_schema_migration("MedicalSample", 0, migrate_v0)

    # Schema-version upgrade.
    data = make_sample(Modality.TEXT_ONLY).to_dict()
    data["schema_version"] = 0
    restored = MedicalSample.from_dict(data)
    assert restored.schema_version == 1

    # Retired enum value inside an old payload is migrated via the
    # pre-migration version context (from_value sees version 0).
    raw = make_sample(Modality.XRAY_2D).to_dict()
    raw["schema_version"] = 0
    raw["modality"] = "CXR_2D"
    migrated = MedicalSample.from_dict(raw)
    assert migrated.modality is Modality.XRAY_2D


def test_tensor_from_data_roundtrip_dtype():
    tensor = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    restored = tensor_from_data(tensor_to_data(tensor), torch.int32)
    assert torch.equal(restored, tensor)
