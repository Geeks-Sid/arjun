"""Phase 05 tests: registry schema, lifecycle, and capability discovery."""

import pytest

from medfm.core.enums import LoadingMode, Modality, PrecisionMode, TaskType
from medfm.data.transforms.specs import NormalizationSpec, PreprocessSpec
from medfm.registry import (
    BACKEND_KEYS,
    REGISTRY_SCHEMA_VERSION,
    AttentionBackend,
    BackendStatus,
    BackendSupport,
    FeatureStatus,
    LicenseClass,
    LicenseSpec,
    LicenseStatus,
    MemoryEstimate,
    MemoryProfile,
    ModelCapability,
    ModelRegistry,
    ModelSpec,
    ModelStatus,
    OutputCapability,
    PeftCapability,
    WeightFormat,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(autouse=True)
def clear_registry():
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


def untested_backends() -> dict[str, BackendSupport]:
    return {k: BackendSupport() for k in BACKEND_KEYS}


def make_license(**kwargs) -> LicenseSpec:
    defaults = dict(
        name="Apache-2.0",
        status=LicenseStatus.APPROVED,
        class_type=LicenseClass.DEPLOYMENT,
        approved_use_cases=("research", "commercial"),
        prohibited_use_cases=("clinical use without validation",),
    )
    return LicenseSpec(**(defaults | kwargs))


def make_memory(**kwargs) -> MemoryProfile:
    defaults = dict(
        parameters_b=7.0,
        max_seq_len=2048,
        loading_modes={LoadingMode.FULL: MemoryEstimate(host_bytes=1000, device_bytes=1000)},
    )
    return MemoryProfile(**(defaults | kwargs))


def make_capability(**kwargs) -> ModelCapability:
    defaults = dict(
        modalities=(Modality.XRAY_2D,),
        tasks=(TaskType.BINARY_CLASSIFICATION,),
        output_types=(OutputCapability.POOLED_EMBEDDINGS,),
    )
    return ModelCapability(**(defaults | kwargs))


def make_spec(**kwargs) -> ModelSpec:
    defaults = dict(
        model_id="test-model",
        repository="org/repo",
        revision=SHA_A,
        license=make_license(),
        capabilities=make_capability(),
        memory=make_memory(),
        backend_support=untested_backends(),
    )
    return ModelSpec(**(defaults | kwargs))


# --- License and governance gates -------------------------------------------


def test_reject_incomplete_license_record():
    with pytest.raises(ValueError, match="non-APPROVED licenses must have status=BLOCKED"):
        make_spec(license=LicenseSpec(name="?", status=LicenseStatus.UNKNOWN, class_type=LicenseClass.RESEARCH))


def test_reject_approved_license_without_use_cases():
    with pytest.raises(ValueError, match="approved_use_cases"):
        LicenseSpec(
            name="MIT",
            status=LicenseStatus.APPROVED,
            class_type=LicenseClass.DEPLOYMENT,
        )


def test_reject_gated_license_without_acceptance_fields():
    with pytest.raises(ValueError, match="acceptance_required"):
        LicenseSpec(
            name="Gated",
            status=LicenseStatus.UNKNOWN,
            class_type=LicenseClass.RESEARCH,
            gated=True,
        )
    with pytest.raises(ValueError, match="terms_url"):
        LicenseSpec(
            name="Gated",
            status=LicenseStatus.UNKNOWN,
            class_type=LicenseClass.RESEARCH,
            gated=True,
            acceptance_required=True,
        )


def test_blocked_unapproved_license_allowed_with_reason():
    spec = make_spec(
        license=LicenseSpec(name="?", status=LicenseStatus.UNKNOWN, class_type=LicenseClass.RESEARCH),
        status=ModelStatus.BLOCKED,
        blocked_reason="license unresolved",
        revision="unresolved",
    )
    assert spec.status == ModelStatus.BLOCKED


def test_research_models_never_in_deployment_catalog():
    ModelRegistry.register(make_spec(model_id="deploy-model"))
    ModelRegistry.register(
        make_spec(
            model_id="research-model",
            license=make_license(class_type=LicenseClass.RESEARCH),
        )
    )
    deploy = ModelRegistry.list_models(license_class=LicenseClass.DEPLOYMENT)
    assert [m.model_id for m in deploy] == ["deploy-model"]
    research = ModelRegistry.list_models(license_class=LicenseClass.RESEARCH)
    assert [m.model_id for m in research] == ["research-model"]


# --- Schema lifecycle --------------------------------------------------------


def test_reject_unpinned_revision_for_ready():
    with pytest.raises(ValueError, match="commit SHA"):
        make_spec(revision="main")
    # Blocked records may carry a placeholder revision.
    spec = make_spec(
        revision="unresolved",
        license=LicenseSpec(name="?", status=LicenseStatus.UNKNOWN, class_type=LicenseClass.RESEARCH),
        status=ModelStatus.BLOCKED,
        blocked_reason="revision pending acceptance",
    )
    assert spec.revision == "unresolved"


def test_reject_contradictory_capabilities():
    with pytest.raises(ValueError, match="At least one modality"):
        make_capability(modalities=())
    with pytest.raises(ValueError, match="At least one task"):
        make_capability(tasks=())
    with pytest.raises(ValueError, match="contradicts"):
        make_capability(cuda_only_extensions=True, pure_pytorch_fallback=True)


def test_reject_task_output_mismatch():
    # Classification requires pooled embeddings.
    with pytest.raises(ValueError, match="POOLED_EMBEDDINGS"):
        make_capability(output_types=(OutputCapability.FEATURE_MAPS,))
    # Segmentation requires feature maps or spatial tokens.
    with pytest.raises(ValueError, match="FEATURE_MAPS or SPATIAL_TOKENS"):
        make_capability(
            tasks=(TaskType.SEMANTIC_SEGMENTATION,),
            output_types=(OutputCapability.POOLED_EMBEDDINGS,),
        )
    # Report generation requires native text.
    with pytest.raises(ValueError, match="NATIVE_TEXT"):
        make_capability(
            tasks=(TaskType.REPORT_GENERATION,),
            output_types=(OutputCapability.POOLED_EMBEDDINGS,),
        )


def test_reject_missing_normalization():
    with pytest.raises(ValueError, match="missing normalization"):
        make_spec(preprocess=PreprocessSpec(model_id="test-model", spatial_shape=(8, 8), channels=1))
    # Per-channel mismatch is already rejected by PreprocessSpec itself.
    from medfm.data.errors import PreprocessSpecError

    with pytest.raises(PreprocessSpecError, match="channels"):
        PreprocessSpec(
            model_id="test-model",
            spatial_shape=(8, 8),
            channels=3,
            normalization=NormalizationSpec(mean=(0.5,), std=(0.5,)),
        )


def test_reject_duplicate_ids_and_unsafe_aliases():
    ModelRegistry.register(make_spec(aliases=("tm1",)))
    with pytest.raises(ValueError, match="Duplicate model ID"):
        ModelRegistry.register(make_spec())
    with pytest.raises(ValueError, match="Duplicate alias"):
        ModelRegistry.register(make_spec(model_id="m2", aliases=("tm1",)))
    with pytest.raises(ValueError, match="conflicts with an existing model ID"):
        ModelRegistry.register(make_spec(model_id="m3", aliases=("test-model",)))
    with pytest.raises(ValueError, match="unsafe alias"):
        make_spec(model_id="m4", aliases=("bad alias",))


def test_backend_support_must_cover_all_backends():
    with pytest.raises(ValueError, match="every backend"):
        make_spec(backend_support={"cpu": BackendSupport()})
    with pytest.raises(ValueError, match="unknown backend keys"):
        make_spec(backend_support={**untested_backends(), "quantum": BackendSupport()})


def test_supported_status_requires_smoke_evidence():
    with pytest.raises(ValueError, match="smoke evidence"):
        BackendSupport(status=BackendStatus.SUPPORTED_SINGLE_DEVICE)
    ok = BackendSupport(
        status=BackendStatus.SUPPORTED_SINGLE_DEVICE,
        smoke_revision=SHA_A,
        smoke_date="2026-08-05",
    )
    assert ok.status == BackendStatus.SUPPORTED_SINGLE_DEVICE


def test_blocked_backend_requires_reason():
    with pytest.raises(ValueError, match="blocked_reason"):
        BackendSupport(status=BackendStatus.BLOCKED_CUSTOM_OP)


def test_deprecation_requires_replacement():
    with pytest.raises(ValueError, match="replaced_by"):
        make_spec(deprecated=True)
    spec = make_spec(deprecated=True, replaced_by="test-model-v2")
    assert spec.deprecated


def test_registry_schema_version_frozen():
    assert REGISTRY_SCHEMA_VERSION == 1
    with pytest.raises(ValueError, match="schema_version"):
        make_spec(schema_version=999)


# --- Per-backend evidence ----------------------------------------------------


def test_cuda_success_does_not_mutate_tpu_status():
    ModelRegistry.register(make_spec())
    updated = ModelRegistry.record_backend_result("test-model", "cuda_single", SHA_A, success=True, date="2026-08-05")
    assert updated.backend_support["cuda_single"].status == BackendStatus.SUPPORTED_SINGLE_DEVICE
    assert updated.backend_support["cuda_single"].smoke_revision == SHA_A
    # TPU (and every other backend) stay exactly as registered.
    assert updated.backend_support["tpu_single_host"].status == BackendStatus.UNTESTED
    assert updated.backend_support["tpu_single_host"].smoke_revision is None
    assert updated.backend_support["cpu"].status == BackendStatus.UNTESTED
    assert updated.last_smoke_revision == SHA_A


def test_record_backend_result_rejects_wrong_revision():
    ModelRegistry.register(make_spec())
    with pytest.raises(ValueError, match="does not match"):
        ModelRegistry.record_backend_result("test-model", "cuda_single", SHA_B, success=True, date="2026-08-05")


def test_record_backend_result_failure_keeps_untested():
    ModelRegistry.register(make_spec())
    updated = ModelRegistry.record_backend_result("test-model", "cuda_single", SHA_A, success=False, date="2026-08-05")
    assert updated.backend_support["cuda_single"].status == BackendStatus.UNTESTED


def test_backend_filter_and_accelerator_report():
    ModelRegistry.register(make_spec())
    # Nothing supported yet: backend filter finds nothing.
    assert ModelRegistry.list_models(backend="cuda_single") == []
    ModelRegistry.record_backend_result("test-model", "cuda_single", SHA_A, success=True, date="2026-08-05")
    assert [m.model_id for m in ModelRegistry.list_models(backend="cuda_single")] == ["test-model"]
    # TPU still untested, so it does not match.
    assert ModelRegistry.list_models(backend="tpu_single_host") == []

    report = ModelRegistry.accelerator_report()
    assert report["test-model"]["cuda_single"] == "SUPPORTED_SINGLE_DEVICE"
    assert report["test-model"]["tpu_single_host"] == "UNTESTED"
    assert set(report["test-model"]) == set(BACKEND_KEYS)


# --- Pre-allocation backend validation ---------------------------------------


def test_validate_backend_rejects_nf4_on_tpu_before_allocation():
    spec = make_spec(
        memory=make_memory(
            loading_modes={
                LoadingMode.QLORA_NF4: MemoryEstimate(host_bytes=1, device_bytes=1, weight_format=WeightFormat.NF4),
                LoadingMode.FULL: MemoryEstimate(host_bytes=1, device_bytes=1),
            }
        ),
    )
    with pytest.raises(ValueError, match="not supported on tpu"):
        ModelRegistry.validate_backend(spec, "tpu_single_host", LoadingMode.QLORA_NF4)
    # Same combination is fine on CUDA.
    ModelRegistry.validate_backend(spec, "cuda_single", LoadingMode.QLORA_NF4)


def test_validate_backend_rejects_cuda_only_extensions_on_tpu():
    spec = make_spec(
        capabilities=make_capability(
            cuda_only_extensions=True,
            pure_pytorch_fallback=False,
            custom_operators=("flash_attn_custom",),
        ),
    )
    with pytest.raises(ValueError, match="CUDA-only extensions"):
        ModelRegistry.validate_backend(spec, "tpu_single_host", LoadingMode.FULL)
    ModelRegistry.validate_backend(spec, "cuda_single", LoadingMode.FULL)


def test_validate_backend_rejects_unknown_backend_and_mode():
    spec = make_spec()
    with pytest.raises(ValueError, match="unknown backend"):
        ModelRegistry.validate_backend(spec, "xla_tpu", LoadingMode.FULL)
    with pytest.raises(ValueError, match="not supported"):
        ModelRegistry.validate_backend(spec, "cpu", LoadingMode.QLORA_NF4)


def test_capability_metadata_fields():
    spec = make_spec(
        capabilities=make_capability(
            spatial_tokens_status=FeatureStatus.HOOKED,
            peft=PeftCapability(supported=True, known_target_modules=("q_proj", "v_proj")),
        ),
        tested_precisions=(PrecisionMode.BF16,),
        tested_attention_backends=(AttentionBackend.SDPA,),
    )
    assert spec.capabilities.spatial_tokens_status == FeatureStatus.HOOKED
    assert spec.capabilities.peft.known_target_modules == ("q_proj", "v_proj")
    assert spec.tested_precisions == (PrecisionMode.BF16,)


def test_peft_capability_requires_modules_or_confirmation():
    with pytest.raises(ValueError, match="known_target_modules"):
        PeftCapability(supported=True)
    ok = PeftCapability(supported=True, unknown_family_confirmation_required=True)
    assert ok.supported
