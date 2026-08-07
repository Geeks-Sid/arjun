"""Phase 05 tests: weight acquisition, integrity, and gated access."""

import hashlib

import pytest

from medfm.core.enums import LoadingMode, Modality, TaskType
from medfm.registry import (
    BACKEND_KEYS,
    BackendSupport,
    GatedAccessError,
    LicenseClass,
    LicenseSpec,
    LicenseStatus,
    MemoryEstimate,
    MemoryProfile,
    ModelCapability,
    ModelRegistry,
    ModelSpec,
    OutputCapability,
    WeightIntegrityError,
    download_weights,
    has_accepted,
    inspect_weights,
    record_acceptance,
    resolve_local_path,
    verify_file_hashes,
    verify_weight_integrity,
)
from medfm.registry.weights import find_partial_downloads

SHA_A = "a" * 40


@pytest.fixture(autouse=True)
def clear_registry():
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


def make_spec(**kwargs) -> ModelSpec:
    defaults = dict(
        model_id="test-model",
        repository="org/repo",
        revision=SHA_A,
        license=LicenseSpec(
            name="Apache-2.0",
            status=LicenseStatus.APPROVED,
            class_type=LicenseClass.DEPLOYMENT,
            approved_use_cases=("research",),
        ),
        capabilities=ModelCapability(
            modalities=(Modality.XRAY_2D,),
            tasks=(TaskType.BINARY_CLASSIFICATION,),
            output_types=(OutputCapability.POOLED_EMBEDDINGS,),
        ),
        memory=MemoryProfile(
            parameters_b=1.0,
            max_seq_len=None,
            loading_modes={LoadingMode.FULL: MemoryEstimate(host_bytes=1, device_bytes=1)},
        ),
        backend_support={k: BackendSupport() for k in BACKEND_KEYS},
    )
    return ModelSpec(**(defaults | kwargs))


def gated_spec() -> ModelSpec:
    return make_spec(
        license=LicenseSpec(
            name="ProviderTerms",
            status=LicenseStatus.APPROVED,
            class_type=LicenseClass.RESEARCH,
            approved_use_cases=("research",),
            terms_url="https://provider.example/terms",
            gated=True,
            acceptance_required=True,
        ),
    )


def test_verify_file_hashes_ok_and_mismatch(tmp_path):
    content = b"weights-bytes"
    (tmp_path / "model.safetensors").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    verify_file_hashes(tmp_path, {"model.safetensors": digest})

    with pytest.raises(WeightIntegrityError, match="hash mismatch"):
        verify_file_hashes(tmp_path, {"model.safetensors": "0" * 64})


def test_verify_file_hashes_missing_and_extra(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(WeightIntegrityError, match="missing expected file"):
        verify_file_hashes(tmp_path, {"model.safetensors": "0" * 64})

    (tmp_path / "rogue.bin").write_bytes(b"rogue")
    with pytest.raises(WeightIntegrityError, match="unexpected weight file"):
        verify_file_hashes(tmp_path, {})


def test_partial_download_detection(tmp_path):
    d = tmp_path / "weights"
    d.mkdir()
    (d / "model.safetensors").write_text("ok")
    assert verify_weight_integrity(d)
    assert find_partial_downloads(d) == []

    (d / "chunk" / "abc.incomplete").parent.mkdir()
    (d / "chunk" / "abc.incomplete").write_text("partial")
    assert find_partial_downloads(d)
    assert not verify_weight_integrity(d)


def test_inspect_weights_report(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"1234")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    report = inspect_weights(tmp_path)
    assert report["exists"] and report["integrity_ok"]
    assert report["uses_safetensors"] and report["uses_pickle_formats"]
    assert {f["path"] for f in report["files"]} == {"model.safetensors", "pytorch_model.bin"}


def test_download_policy(monkeypatch, tmp_path):
    import huggingface_hub

    calls = {}

    def mock_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", mock_snapshot_download)

    spec = make_spec(trust_remote_code_allowed=False)
    download_weights(spec, tmp_path)
    assert "*.py" in calls["ignore_patterns"]
    assert "*.safetensors" in calls["allow_patterns"]
    assert "*.bin" not in calls["allow_patterns"]
    assert calls["revision"] == SHA_A
    assert calls["local_files_only"] is False

    download_weights(spec, tmp_path, allow_unsafe_formats=True)
    assert "*.bin" in calls["allow_patterns"]

    spec_trusted = make_spec(trust_remote_code_allowed=True)
    download_weights(spec_trusted, tmp_path)
    assert calls["ignore_patterns"] is None


def test_download_rejects_partial_result(monkeypatch, tmp_path):
    import huggingface_hub

    def mock_snapshot_download(**kwargs):
        (tmp_path / "model.safetensors.incomplete").write_text("partial")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", mock_snapshot_download)
    with pytest.raises(WeightIntegrityError, match="partial download"):
        download_weights(make_spec(), tmp_path)


def test_gated_access_blocks_until_accepted(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    store = tmp_path / "gated_access.json"
    spec = gated_spec()

    with pytest.raises(GatedAccessError, match="explicit acceptance"):
        download_weights(spec, tmp_path, acceptance_store=store)
    assert not has_accepted(spec.model_id, spec.repository, store_path=store)

    # Acceptance is explicit, recorded outside source control, then unblocks.
    path = record_acceptance(spec.model_id, spec.repository, accepted_by="Siddhesh", store_path=store)
    assert path == store
    assert has_accepted(spec.model_id, spec.repository, store_path=store)
    download_weights(spec, tmp_path, acceptance_store=store)

    # A different repository (typosquat) is not covered by the acceptance.
    other = make_spec(
        model_id="other",
        repository="org/repo-evil",
        license=spec.license,
    )
    with pytest.raises(GatedAccessError):
        download_weights(other, tmp_path, acceptance_store=store)


def test_resolve_local_path_uses_no_network(monkeypatch, tmp_path):
    import huggingface_hub

    calls = {}

    def mock_snapshot_download(**kwargs):
        calls.update(kwargs)
        (tmp_path / "model.safetensors").write_text("ok")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", mock_snapshot_download)
    path = resolve_local_path(make_spec(), tmp_path)
    assert path == tmp_path
    assert calls["local_files_only"] is True
    assert "token" not in calls
