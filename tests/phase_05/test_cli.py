"""Phase 05 tests: CLI behavior, network isolation, and accelerator reports."""

import json

import pytest

from medfm.cli import accelerator as accel_cli
from medfm.cli import models as models_cli
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
    OutputCapability,
    clear_plugins,
    register_plugin,
)
from medfm.registry.schema import WeightFormat

SHA_A = "a" * 40


@pytest.fixture(autouse=True)
def clean_state():
    ModelRegistry.clear()
    clear_plugins()
    yield
    clear_plugins()
    ModelRegistry.clear()


def register_cli_model(model_id: str = "cli-model", **kwargs) -> ModelSpec:
    defaults = dict(
        model_id=model_id,
        repository="org/repo",
        revision=SHA_A,
        license=LicenseSpec(
            name="Apache-2.0",
            status=LicenseStatus.APPROVED,
            class_type=LicenseClass.DEPLOYMENT,
            approved_use_cases=("research", "commercial"),
            prohibited_use_cases=("clinical use without validation",),
        ),
        capabilities=ModelCapability(
            modalities=(Modality.XRAY_2D,),
            tasks=(TaskType.BINARY_CLASSIFICATION,),
            output_types=(OutputCapability.POOLED_EMBEDDINGS,),
        ),
        memory=MemoryProfile(
            parameters_b=7.0,
            max_seq_len=2048,
            loading_modes={LoadingMode.FULL: MemoryEstimate(host_bytes=1000, device_bytes=1000)},
        ),
        backend_support={k: BackendSupport() for k in BACKEND_KEYS},
    )
    spec = ModelSpec(**(defaults | kwargs))
    ModelRegistry.register(spec)
    return spec


def test_cli_list_text_and_json(capsys):
    register_cli_model()
    assert models_cli.main(["list"]) == 0
    out, _ = capsys.readouterr()
    assert "cli-model" in out and "DEPLOYMENT" in out

    assert models_cli.main(["list", "--format", "json"]) == 0
    out, _ = capsys.readouterr()
    data = json.loads(out)
    ids = {m["model_id"] for m in data}
    assert "cli-model" in ids
    assert data[0]["schema_version"] == 1


def test_cli_list_filters(capsys):
    register_cli_model()
    assert models_cli.main(["list", "--modality", "XRAY_2D", "--format", "json"]) == 0
    out, _ = capsys.readouterr()
    assert any(m["model_id"] == "cli-model" for m in json.loads(out))

    assert models_cli.main(["list", "--modality", "MRI_3D", "--format", "json"]) == 0
    out, _ = capsys.readouterr()
    assert all(m["model_id"] != "cli-model" for m in json.loads(out))

    # Backend filter: nothing is SUPPORTED yet.
    assert models_cli.main(["list", "--backend", "cuda_single", "--format", "json"]) == 0
    out, _ = capsys.readouterr()
    assert json.loads(out) == []


def test_cli_show_includes_use_cases(capsys):
    register_cli_model()
    assert models_cli.main(["show", "cli-model"]) == 0
    out, _ = capsys.readouterr()
    assert "Model ID: cli-model" in out
    assert "Approved use cases: research, commercial" in out
    assert "Prohibited use cases: clinical use without validation" in out
    assert "cuda_single: UNTESTED" in out


def test_cli_validate_metadata(capsys):
    register_cli_model()
    assert models_cli.main(["validate", "cli-model"]) == 0
    out, _ = capsys.readouterr()
    assert "metadata is valid" in out


def test_cli_validate_local_weights(capsys, tmp_path):
    register_cli_model()
    (tmp_path / "model.safetensors").write_text("ok")
    assert models_cli.main(["validate", "cli-model", "--local-weights", str(tmp_path)]) == 0
    out, _ = capsys.readouterr()
    assert "Local weight validation passed" in out

    (tmp_path / "blob.incomplete").write_text("partial")
    assert models_cli.main(["validate", "cli-model", "--local-weights", str(tmp_path)]) == 1


def test_cli_estimate_memory(capsys):
    register_cli_model()
    assert models_cli.main(["estimate-memory", "cli-model", "--loading-mode", "FULL"]) == 0
    out, _ = capsys.readouterr()
    assert "Host Bytes: 1000" in out
    assert "Weight Format: BF16" in out


def test_cli_inspect_modules(capsys):
    register_cli_model()
    # PEFT unsupported by default: exit 1 with a clear message.
    assert models_cli.main(["inspect-modules", "cli-model"]) == 1
    out, _ = capsys.readouterr()
    assert "does not support PEFT" in out


def test_cli_accelerator_report(capsys):
    register_cli_model()
    assert models_cli.main(["accelerator-report", "--format", "json"]) == 0
    out, _ = capsys.readouterr()
    report = json.loads(out)
    assert set(report["cli-model"]) == set(BACKEND_KEYS)
    assert report["cli-model"]["cuda_single"] == "UNTESTED"


def test_metadata_commands_use_no_network(monkeypatch, capsys):
    """list/show/validate (metadata) must work with the network hard-disabled."""
    import huggingface_hub

    def no_network(**kwargs):  # pragma: no cover - must never be called
        raise AssertionError("network access attempted from a metadata command")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    register_cli_model()

    assert models_cli.main(["list"]) == 0
    assert models_cli.main(["show", "cli-model"]) == 0
    assert models_cli.main(["validate", "cli-model"]) == 0
    assert models_cli.main(["estimate-memory", "cli-model", "--loading-mode", "FULL"]) == 0
    assert models_cli.main(["accelerator-report"]) == 0
    capsys.readouterr()


def _register_dummy_for_smoke() -> None:
    spec = register_cli_model(
        model_id="dummy-tiny-2d",
        preprocess=PreprocessSpec(
            model_id="dummy-tiny-2d",
            spatial_shape=(8, 8),
            channels=1,
            normalization=NormalizationSpec(mean=(0.5,), std=(0.25,)),
        ),
    )
    register_plugin(spec.model_id, DummyPlugin())


def test_cli_smoke_dummy(capsys, tmp_path):
    _register_dummy_for_smoke()
    rc = models_cli.main(["smoke", "dummy-tiny-2d", "--backend", "cpu", "--artifact-dir", str(tmp_path)])
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "Smoke OK: dummy-tiny-2d" in out
    artifact = json.loads((tmp_path / "dummy-tiny-2d" / "smoke_cpu.json").read_text())
    assert artifact["model_id"] == "dummy-tiny-2d"
    assert artifact["revision"] == SHA_A


def test_cli_smoke_no_adapter(capsys):
    register_cli_model()  # READY but no plugin registered
    rc = models_cli.main(["smoke", "cli-model", "--backend", "cpu"])
    assert rc == 2
    _, err = capsys.readouterr()
    assert "no adapter plugin" in err


def test_accelerator_validate_model_cpu(capsys, tmp_path):
    _register_dummy_for_smoke()
    rc = accel_cli.main(
        [
            "validate-model",
            "dummy-tiny-2d",
            "--backend",
            "cpu",
            "--artifact-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "validate-model OK: dummy-tiny-2d" in out
    assert SHA_A in out


def test_accelerator_validate_model_rejects_nf4_on_tpu_preallocation(capsys):
    register_cli_model(
        memory=MemoryProfile(
            parameters_b=7.0,
            max_seq_len=2048,
            loading_modes={
                LoadingMode.QLORA_NF4: MemoryEstimate(host_bytes=1, device_bytes=1, weight_format=WeightFormat.NF4)
            },
        ),
    )
    rc = accel_cli.main(["validate-model", "cli-model", "--backend", "tpu_single_host", "--loading-mode", "QLORA_NF4"])
    assert rc == 2
    _, err = capsys.readouterr()
    assert "REJECTED (pre-allocation)" in err
