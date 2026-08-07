"""Phase 05 tests: dummy plugin smoke and run-artifact model identity."""

import json

import pytest

from medfm.core.enums import LoadingMode, Modality, TaskType
from medfm.data.transforms.specs import NormalizationSpec, PreprocessSpec
from medfm.registry import (
    BACKEND_KEYS,
    BackendSupport,
    DummyPlugin,
    LicenseClass,
    LicenseSpec,
    LicenseStatus,
    MemoryEstimate,
    MemoryProfile,
    ModelCapability,
    ModelRegistry,
    ModelSpec,
    ModelStatus,
    NoAdapterError,
    OutputCapability,
    clear_plugins,
    register_plugin,
    run_smoke,
)
from medfm.registry.catalog import load_v1_catalog
from medfm.registry.schema import BackendStatus

SHA_A = "a" * 40


@pytest.fixture(autouse=True)
def clean_state():
    ModelRegistry.clear()
    clear_plugins()
    yield
    clear_plugins()
    ModelRegistry.clear()


def register_dummy() -> ModelSpec:
    spec = ModelSpec(
        model_id="dummy-tiny-2d",
        repository="local/dummy",
        revision=SHA_A,
        license=LicenseSpec(
            name="Apache-2.0",
            status=LicenseStatus.APPROVED,
            class_type=LicenseClass.DEPLOYMENT,
            approved_use_cases=("registry smoke testing",),
        ),
        capabilities=ModelCapability(
            modalities=(Modality.XRAY_2D,),
            tasks=(TaskType.BINARY_CLASSIFICATION,),
            output_types=(OutputCapability.POOLED_EMBEDDINGS,),
        ),
        memory=MemoryProfile(
            parameters_b=0.0001,
            max_seq_len=None,
            loading_modes={LoadingMode.FULL: MemoryEstimate(host_bytes=1024, device_bytes=1024)},
        ),
        preprocess=PreprocessSpec(
            model_id="dummy-tiny-2d",
            spatial_shape=(8, 8),
            channels=1,
            normalization=NormalizationSpec(mean=(0.5,), std=(0.25,)),
        ),
        backend_support={k: BackendSupport() for k in BACKEND_KEYS},
    )
    ModelRegistry.register(spec)
    register_plugin("dummy-tiny-2d", DummyPlugin())
    return spec


def test_dummy_plugin_completes_local_smoke(tmp_path):
    register_dummy()
    result = run_smoke("dummy-tiny-2d", backend="cpu", artifact_dir=tmp_path)
    assert result.success
    assert result.model_id == "dummy-tiny-2d"
    assert result.revision == SHA_A
    assert result.artifact_path is not None

    # Every run artifact receives the exact model ID and revision.
    artifact = json.loads((tmp_path / "dummy-tiny-2d" / "smoke_cpu.json").read_text())
    assert artifact["model_id"] == "dummy-tiny-2d"
    assert artifact["revision"] == SHA_A
    assert artifact["backend"] == "cpu"

    # The smoke promoted CPU only; accelerators stay UNTESTED.
    spec = ModelRegistry.get("dummy-tiny-2d")
    assert spec.backend_support["cpu"].status == BackendStatus.CPU_CONTRACT_ONLY
    assert spec.backend_support["cuda_single"].status == BackendStatus.UNTESTED
    assert spec.backend_support["tpu_single_host"].status == BackendStatus.UNTESTED


def test_smoke_without_adapter_raises_structured_error():
    spec = register_dummy()
    clear_plugins()  # READY model, but no plugin registered
    with pytest.raises(NoAdapterError, match="no adapter plugin"):
        run_smoke(spec.model_id, backend="cpu", record=False)


def test_smoke_blocked_model_raises():
    load_v1_catalog()
    # medsiglip is BLOCKED (license pending review); blocked models never smoke.
    with pytest.raises(RuntimeError, match="BLOCKED"):
        run_smoke("medsiglip", backend="cpu", record=False)


def test_v1_catalog_registers_ready_or_blocked_with_reasons():
    import yaml

    from medfm.registry.catalog import LICENSES_PATH

    roster_size = len(yaml.safe_load(LICENSES_PATH.read_text()))
    specs = load_v1_catalog()
    assert len(specs) == roster_size
    for spec in specs:
        assert spec.status in (ModelStatus.READY, ModelStatus.BLOCKED)
        if spec.status == ModelStatus.BLOCKED:
            assert spec.blocked_reason, f"{spec.model_id} blocked without reason"
        assert set(spec.backend_support) == set(BACKEND_KEYS)


def test_v1_catalog_research_never_in_deployment():
    load_v1_catalog()
    deploy_ids = {
        m.model_id for m in ModelRegistry.list_models(license_class=LicenseClass.DEPLOYMENT, include_blocked=True)
    }
    research_ids = {
        m.model_id for m in ModelRegistry.list_models(license_class=LicenseClass.RESEARCH, include_blocked=True)
    }
    assert not deploy_ids & research_ids
    # Non-commercial licenses (conch, titan) must be research-class.
    assert "conch" in research_ids
    assert "titan" in research_ids
    assert "conch" not in deploy_ids
