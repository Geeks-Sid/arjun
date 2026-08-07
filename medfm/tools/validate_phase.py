"""Phase acceptance gate: python -m medfm.tools.validate_phase --phase <NN>.

Verifies for a completed phase:
  * required files exist,
  * the phase report is populated,
  * acceptance.json validates against agent/acceptance_schema.json,
  * no acceptance criterion is 'unknown',
  * (phase 00) license records and the v1 scope registry are valid and consistent.

Exit code 0 = gate passed, 1 = gate failed, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from medfm.tools import governance as gov

REPORT_FILES = [
    "summary.md",
    "files_changed.txt",
    "commands_executed.txt",
    "test_results.json",
    "acceptance.json",
    "unresolved_issues.md",
    "next_phase_handoff.md",
]

PHASE_00_REQUIRED_FILES = [
    "docs/product_requirements.md",
    "docs/supported_modalities.md",
    "docs/supported_tasks.md",
    "docs/clinical_safety_scope.md",
    "docs/data_governance.md",
    "docs/model_governance.md",
    "docs/licensing_policy.md",
    "docs/reproducibility_policy.md",
    "docs/architecture/adr_0001_single_framework_multiple_backbones.md",
    "docs/architecture/adr_0002_peft_first_training.md",
    "docs/architecture/adr_0003_external_encoder_vlm_bridge.md",
    "docs/architecture/adr_0004_patient_level_splitting.md",
    "docs/architecture/adr_0005_native_3d_and_slice_sequence_vlm.md",
    "docs/architecture/adr_0006_adapter_only_checkpoints.md",
    "docs/architecture/adr_0007_pytorch_cuda_and_xla_backends.md",
    "docs/architecture/adr_0008_tpu_static_shape_buckets.md",
    "docs/architecture/adr_0009_cuda_qlora_vs_tpu_bf16_lora.md",
    "model_registry/license_schema.json",
    "model_registry/licenses.yaml",
    "model_registry/v1_scope.yaml",
    "agent/README.md",
    "agent/phase_template.md",
    "agent/acceptance_schema.json",
    "agent/prompts/implement_phase.md",
    "agent/prompts/review_phase.md",
    "agent/prompts/test_phase.md",
    "agent/prompts/repair_phase.md",
]

PHASE_01_REQUIRED_FILES = [
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "Makefile",
    ".gitignore",
    "docker/Dockerfile",
    "docker/Dockerfile.ci",
    "docker/compose.yaml",
    "docker/README.md",
    "scripts/tpu_vm_bootstrap.sh",
    "medfm/cli/__init__.py",
    "medfm/cli/main.py",
    "medfm/tools/doctor.py",
    "medfm/tools/doctor_schema.json",
    "medfm/tools/smoke.py",
    "medfm/training/run_metadata.py",
    "medfm/training/tracking.py",
    "tests/phase_01/conftest.py",
]


PHASE_02_REQUIRED_FILES = [
    "medfm/core/__init__.py",
    "medfm/core/enums.py",
    "medfm/core/errors.py",
    "medfm/core/versioning.py",
    "medfm/core/sample.py",
    "medfm/core/batch.py",
    "medfm/core/encoder.py",
    "medfm/core/language.py",
    "medfm/core/task.py",
    "medfm/core/serialization.py",
    "docs/core_contracts.md",
    "tests/phase_02/conftest.py",
    "tests/phase_02/contract_fixtures.py",
    "tests/phase_02/test_contract_smoke.py",
]


PHASE_03_REQUIRED_FILES = [
    "medfm/data/__init__.py",
    "medfm/data/errors.py",
    "medfm/data/manifests/__init__.py",
    "medfm/data/manifests/schema.py",
    "medfm/data/manifests/io.py",
    "medfm/data/readers/__init__.py",
    "medfm/data/readers/base.py",
    "medfm/data/readers/radiology.py",
    "medfm/data/readers/dicom.py",
    "medfm/data/readers/pathology.py",
    "medfm/data/caching/__init__.py",
    "medfm/data/caching/keys.py",
    "medfm/data/caching/base.py",
    "medfm/data/caching/disk.py",
    "medfm/data/caching/typed.py",
    "medfm/data/splits.py",
    "medfm/data/fingerprint.py",
    # Converted from medfm/data/samplers.py to a package in Phase 04 (public
    # imports unchanged); the gate tracks the current layout.
    "medfm/data/samplers/__init__.py",
    "medfm/cli/data.py",
    "medfm/tools/data_tools.py",
    "tests/fixtures/manifests/mixed_synthetic.parquet",
    "tests/phase_03/conftest.py",
    "tests/phase_03/synthetic.py",
]


PHASE_04_REQUIRED_FILES = [
    "medfm/data/transforms/__init__.py",
    "medfm/data/transforms/base.py",
    "medfm/data/transforms/specs.py",
    "medfm/data/transforms/pipeline.py",
    "medfm/data/transforms/radiology2d.py",
    "medfm/data/transforms/spatial3d.py",
    "medfm/data/transforms/ct.py",
    "medfm/data/transforms/mri.py",
    "medfm/data/transforms/pathology.py",
    "medfm/data/transforms/timing.py",
    "medfm/data/samplers/__init__.py",
    "medfm/data/samplers/distributed.py",
    "medfm/data/samplers/patches.py",
    "medfm/data/textprep/__init__.py",
    "medfm/data/textprep/unicode.py",
    "medfm/data/textprep/phi.py",
    "medfm/data/textprep/sections.py",
    "medfm/data/textprep/prompts.py",
    "medfm/data/textprep/tokenize.py",
    "medfm/data/collators/__init__.py",
    "medfm/data/collators/base.py",
    "medfm/data/collators/buckets.py",
    "medfm/data/collators/classification.py",
    "medfm/data/collators/segmentation.py",
    "medfm/data/collators/contrastive.py",
    "medfm/data/collators/vl.py",
    "medfm/data/collators/multitask.py",
    "tests/phase_04/conftest.py",
    "tests/phase_04/test_end_to_end_transforms.py",
    "tests/phase_04/test_radiology2d.py",
    "tests/phase_04/test_ct.py",
    "tests/phase_04/test_mri.py",
    "tests/phase_04/test_patch_samplers.py",
    "tests/phase_04/test_pathology.py",
    "tests/phase_04/test_textprep.py",
    "tests/phase_04/test_vlm_masking.py",
    "tests/phase_04/test_collators.py",
]

PHASE_06_REQUIRED_FILES = [
    "medfm/models/visual/__init__.py",
    "medfm/models/visual/base.py",
    "medfm/models/visual/hf_generic.py",
    "medfm/models/visual/medsiglip.py",
    "medfm/models/visual/raddino.py",
    "medfm/models/visual/hoptimus0.py",
    "medfm/models/visual/medgemma_vision.py",
    "medfm/models/visual/specs.py",
    "medfm/models/__init__.py",
    "tests/phase_06/conftest.py",
    "tests/phase_06/test_adapter_contract.py",
    "tests/phase_06/test_lora.py",
    "tests/phase_06/test_checkpoint.py",
    "tests/phase_06/test_raddino_medsiglip.py",
    "tests/phase_06/test_other_adapters.py",
    "tests/phase_06/test_registry_cli.py",
    "tests/phase_06/test_backend_neutrality.py",
]

PHASE_07_REQUIRED_FILES = [
    "medfm/models/visual/native_3d.py",
    "medfm/models/visual/ct_fm.py",
    "medfm/models/visual/triad.py",
    "medfm/models/visual/native_tasks.py",
    "medfm/models/visual/research_3d.py",
    "tests/phase_07/conftest.py",
    "tests/phase_07/test_native_3d.py",
    "tests/phase_07/test_registry.py",
]


PHASE_08_REQUIRED_FILES = [
    "medfm/models/pathology/__init__.py",
    "medfm/models/pathology/adapters.py",
    "medfm/models/pathology/aggregation.py",
    "medfm/models/pathology/pipeline.py",
    "medfm/models/pathology/selection.py",
    "docs/architecture/adr_0011_pathology_embedding_store.md",
    "tests/phase_08/conftest.py",
    "tests/phase_08/test_pathology.py",
]


PHASE_09_REQUIRED_FILES = [
    "medfm/models/language/__init__.py",
    "medfm/models/language/base.py",
    "medfm/models/language/configs.py",
    "medfm/models/language/gemma.py",
    "medfm/models/language/medgemma.py",
    "medfm/models/language/m3d_lamed.py",
    "medfm/models/language/registry.py",
    "medfm/models/bridges/__init__.py",
    "medfm/models/bridges/base.py",
    "medfm/models/bridges/coordinates.py",
    "medfm/models/bridges/placement.py",
    "medfm/models/bridges/resampler.py",
    "medfm/models/bridges/training.py",
    "configs/phase_09_language_bridges.yaml",
    "tests/phase_09/conftest.py",
    "tests/phase_09/test_bridges.py",
    "tests/phase_09/test_language.py",
]

PHASE_10_REQUIRED_FILES = [
    "medfm/peft/__init__.py",
    "medfm/peft/config.py",
    "medfm/peft/errors.py",
    "medfm/peft/resolver.py",
    "medfm/peft/lora.py",
    "medfm/peft/quantization.py",
    "medfm/peft/checkpoint.py",
    "medfm/cli/peft.py",
    "tests/phase_10/conftest.py",
    "tests/phase_10/test_config_and_backend.py",
    "tests/phase_10/test_resolver_and_injection.py",
    "tests/phase_10/test_quantization_safety.py",
    "tests/phase_10/test_checkpoint_and_merge.py",
    "tests/phase_10/test_cli_and_validation.py",
]
PHASE_11_REQUIRED_FILES = [
    "medfm/models/heads/__init__.py",
    "medfm/models/heads/pooling.py",
    "medfm/models/heads/classification.py",
    "medfm/models/heads/retrieval.py",
    "medfm/models/heads/localization.py",
    "medfm/models/heads/losses.py",
    "medfm/models/decoders/__init__.py",
    "medfm/models/decoders/base.py",
    "medfm/models/decoders/unet.py",
    "medfm/models/decoders/fpn.py",
    "medfm/models/decoders/masks.py",
    "medfm/models/decoders/language.py",
    "medfm/tasks/__init__.py",
    "medfm/tasks/base.py",
    "medfm/tasks/classification.py",
    "medfm/tasks/segmentation.py",
    "medfm/tasks/language_segmentation.py",
    "medfm/tasks/retrieval.py",
    "medfm/tasks/localization.py",
    "medfm/tasks/generation.py",
    "medfm/tasks/structured.py",
    "medfm/tasks/structured_findings_v1.json",
    "medfm/tasks/multitask.py",
    "medfm/tasks/reductions.py",
    "tests/phase_11/conftest.py",
    "tests/phase_11/test_heads_and_losses.py",
    "tests/phase_11/test_segmentation.py",
    "tests/phase_11/test_alignment_boxes_generation.py",
]


PHASE_12_REQUIRED_FILES = [
    "medfm/training/__init__.py",
    "medfm/training/config.py",
    "medfm/training/backend.py",
    "medfm/training/steps.py",
    "medfm/training/optimizer.py",
    "medfm/training/memory.py",
    "medfm/training/data.py",
    "medfm/training/distributed.py",
    "medfm/training/checkpoint.py",
    "medfm/training/evaluation.py",
    "medfm/training/pipeline.py",
    "medfm/training/trainer.py",
    "medfm/cli/train.py",
    "configs/smoke/tiny_multitask.yaml",
    "tests/phase_12/conftest.py",
    "tests/phase_12/test_config_backend.py",
    "tests/phase_12/test_trainer_memory_checkpoint.py",
]
PHASE_13_REQUIRED_FILES = [
    "medfm/recipes/__init__.py",
    "medfm/recipes/phase13.py",
    "medfm/evaluation/__init__.py",
    "medfm/evaluation/metrics.py",
    "medfm/evaluation/ablation.py",
    "medfm/evaluation/report.py",
    "configs/recipes/2d/classification_smoke.yaml",
    "configs/recipes/2d/classification_medsiglip.yaml",
    "configs/recipes/2d/classification_raddino.yaml",
    "configs/recipes/2d/segmentation_smoke.yaml",
    "configs/recipes/2d/segmentation_medsam2_promptable.yaml",
    "configs/recipes/2d/native_vlm_smoke.yaml",
    "configs/recipes/2d/native_structured_findings_smoke.yaml",
    "configs/recipes/2d/external_vlm_linear_64.yaml",
    "configs/recipes/2d/external_vlm_smoke.yaml",
    "configs/recipes/2d/external_vlm_perceiver_32.yaml",
    "configs/recipes/2d/external_vlm_perceiver_128.yaml",
    "configs/recipes/2d/external_vlm_cuda_qlora.yaml",
    "configs/recipes/2d/native_vlm_cuda_nf4.yaml",
    "configs/recipes/2d/native_vlm_tpu_bf16_lora.yaml",
    "tests/phase_13/conftest.py",
    "tests/phase_13/test_recipes.py",
    "tests/phase_13/test_evaluation.py",
    "docs/recipes/phase13_2d_model_cards.md",
]


PHASE_14_REQUIRED_FILES = [
    "medfm/recipes/__init__.py",
    "medfm/recipes/phase14.py",
    "medfm/recipes/slice_selectors.py",
    "medfm/models/visual/native_3d.py",
    "medfm/models/bridges/coordinates.py",
    "medfm/models/bridges/resampler.py",
    "medfm/models/decoders/language.py",
    "medfm/tasks/language_segmentation.py",
    "medfm/evaluation/metrics.py",
    "medfm/evaluation/report.py",
    "configs/recipes/3d/classification_ct_fm_cuda.yaml",
    "configs/recipes/3d/classification_smoke.yaml",
    "configs/recipes/3d/classification_flexict3d.yaml",
    "configs/recipes/3d/classification_triad_mri.yaml",
    "configs/recipes/3d/classification_ct_fm_tpu_bf16.yaml",
    "configs/recipes/3d/segmentation_ct_fm_baseline.yaml",
    "configs/recipes/3d/segmentation_triad_lora.yaml",
    "configs/recipes/3d/native_vlm_cached_tpu_bf16.yaml",
    "configs/recipes/3d/native_vlm_ct_fm_cuda_lora.yaml",
    "configs/recipes/3d/native_vlm_structured_findings.yaml",
    "configs/recipes/3d/slice_sequence_vlm_uniform.yaml",
    "tests/phase_14/__init__.py",
    "configs/recipes/3d/language_conditioned_segmentation.yaml",
    "tests/phase_14/conftest.py",
    "tests/phase_14/test_recipes.py",
    "tests/phase_14/test_evaluation.py",
    "docs/recipes/phase14_3d_model_cards.md",
]


PHASE_15_REQUIRED_FILES = [
    "medfm/recipes/__init__.py",
    "medfm/recipes/phase15.py",
    "medfm/recipes/pathology_stitching.py",
    "medfm/cli/train.py",
    "medfm/evaluation/metrics.py",
    "medfm/evaluation/report.py",
    "configs/recipes/pathology/tile_classification_hoptimus_linear.yaml",
    "configs/recipes/pathology/tile_classification_mlp.yaml",
    "configs/recipes/pathology/tile_classification_tpu_bf16.yaml",
    "configs/recipes/pathology/tile_classification_vision_lora.yaml",
    "configs/recipes/pathology/tile_classification_contrastive.yaml",
    "configs/recipes/pathology/wsi_classification_smoke.yaml",
    "configs/recipes/pathology/wsi_classification_attention_mil.yaml",
    "configs/recipes/pathology/wsi_classification_gated_attention_mil.yaml",
    "configs/recipes/pathology/wsi_classification_transformer.yaml",
    "configs/recipes/pathology/wsi_classification_tpu_bf16.yaml",
    "configs/recipes/pathology/wsi_vlm_cached_smoke.yaml",
    "configs/recipes/pathology/wsi_vlm_organ.yaml",
    "configs/recipes/pathology/wsi_vlm_subtype.yaml",
    "configs/recipes/pathology/wsi_vlm_grade.yaml",
    "configs/recipes/pathology/wsi_vlm_biomarker.yaml",
    "configs/recipes/pathology/wsi_vlm_report.yaml",
    "configs/recipes/pathology/wsi_vlm_vqa.yaml",
    "configs/recipes/pathology/wsi_vlm_retrieval.yaml",
    "configs/recipes/pathology/wsi_vlm_evidence.yaml",
    "configs/recipes/pathology/wsi_vlm_evidence_tpu_bf16.yaml",
    "configs/recipes/pathology/segmentation_smoke.yaml",
    "configs/recipes/pathology/segmentation_tile_unet.yaml",
    "tests/phase_15/__init__.py",
    "tests/phase_15/conftest.py",
    "tests/phase_15/test_recipes.py",
    "tests/phase_15/test_stitching.py",
    "tests/phase_15/test_evaluation.py",
    "docs/recipes/phase15_pathology_model_cards.md",
]

PHASE_16_REQUIRED_FILES = [
    "medfm/evaluation/__init__.py",
    "medfm/evaluation/ablation.py",
    "medfm/evaluation/advanced.py",
    "medfm/evaluation/artifacts.py",
    "medfm/evaluation/calibration.py",
    "medfm/evaluation/distributed.py",
    "medfm/evaluation/human_review.py",
    "medfm/evaluation/metrics.py",
    "medfm/evaluation/report.py",
    "medfm/evaluation/schemas.py",
    "medfm/evaluation/specialized.py",
    "medfm/evaluation/uncertainty.py",
    "medfm/cli/evaluate.py",
    "configs/smoke/evaluation.yaml",
    "docs/evaluation/metric_result_schema.json",
    "docs/evaluation/prediction_schema.json",
    "docs/evaluation/phase16_report_template.md",
    "tests/phase_16/__init__.py",
    "tests/phase_16/test_specialized.py",
    "tests/phase_16/conftest.py",
    "tests/phase_16/test_evaluation.py",
    "tests/phase_16/test_distributed.py",
]
PHASE_17_REQUIRED_FILES = [
    "medfm/inference/__init__.py",
    "medfm/inference/audit.py",
    "medfm/inference/bundle.py",
    "medfm/inference/errors.py",
    "medfm/inference/export_dicom.py",
    "medfm/inference/export_nifti.py",
    "medfm/inference/generation.py",
    "medfm/inference/pipeline.py",
    "medfm/inference/schemas.py",
    "medfm/inference/server.py",
    "medfm/inference/sliding_window.py",
    "medfm/cli/export.py",
    "medfm/cli/infer.py",
    "configs/smoke/inference.yaml",
    "docs/inference/bundle_schema.json",
    "docs/inference/deployment_matrix.md",
    "docs/inference/request_schema.json",
    "docs/inference/response_schema.json",
    "docs/inference/license_catalog.md",
    "tests/phase_17/__init__.py",
    "tests/phase_17/conftest.py",
    "tests/phase_17/test_bundle.py",
    "tests/phase_17/test_geometry.py",
    "tests/phase_17/test_inference.py",
    "tests/phase_17/test_serving.py",
]


def _check_phase_08_pathology() -> list[str]:
    errors: list[str] = []
    from medfm.models.pathology import (
        EmbeddingStore,
        MeanPoolingAggregator,
        PathologyVLMAdapter,
        TinyPathologyTileEncoder,
        TokenBudget,
        WSITokenSelector,
    )

    try:
        encoder = TinyPathologyTileEncoder(embedding_dim=8)
        if encoder.embedding_dim != 8:
            errors.append("tiny pathology encoder dimension mismatch")
        if MeanPoolingAggregator(8).embedding_dim != 8:
            errors.append("mean pooling aggregator dimension mismatch")
        if TokenBudget().visual_tokens != 64 or TokenBudget().precompression != 256:
            errors.append("pathology default token budgets changed")
        if WSITokenSelector().visual_tokens != 64:
            errors.append("WSI selector does not expose the default fixed token count")
        if PathologyVLMAdapter(8, max_tokens=32).max_tokens != 32:
            errors.append("pathology VLM bridge does not enforce fixed token count")
        if EmbeddingStore.schema_version != 1:
            errors.append("embedding store schema version is not 1")
    except Exception as exc:
        errors.append(f"pathology contracts failed: {type(exc).__name__}: {exc}")

    try:
        from medfm.registry import ModelRegistry, clear_plugins, get_plugin
        from medfm.registry.catalog import load_v1_catalog

        ModelRegistry.clear()
        clear_plugins()
        specs = {spec.model_id: spec for spec in load_v1_catalog()}
        for model_id in ("h-optimus-0", "gigapath-flash", "titan"):
            spec = specs.get(model_id)
            if spec is None:
                errors.append(f"{model_id} missing from catalog")
                continue
            if spec.preprocess is None:
                errors.append(f"{model_id} missing pathology preprocess")
            if get_plugin(model_id) is None:
                errors.append(f"{model_id} missing pathology smoke plugin")
    except Exception as exc:
        errors.append(f"Phase 08 catalog check failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            ModelRegistry.clear()
            clear_plugins()
        except UnboundLocalError:
            pass
    return errors


def _check_phase_09_language_bridges() -> list[str]:
    """Phase 09: language modes, fixed bridges, masks, and tiny losses."""
    errors: list[str] = []
    try:
        import torch

        from medfm.core.enums import Modality
        from medfm.models.bridges import (
            MLPBridge,
            ThreeDCoordinateEncoder,
            TrainingStage,
            TwoDCoordinateEncoder,
            WSICoordinateEncoder,
            stage_config,
        )
        from medfm.models.language import (
            GemmaCausalLMAdapter,
            GenericHFCausalLMAdapter,
            M3DLaMedAdapter,
            MedGemmaAdapter,
            language_descriptors,
        )

        descriptors = language_descriptors()
        if descriptors["medgemma_native"].mode.value != "native":
            errors.append("MedGemma descriptor is not native")
        if descriptors["generic_hf_causal"].mode.value != "external":
            errors.append("generic descriptor is not external")
        for builder in (
            GenericHFCausalLMAdapter.build_tiny,
            GemmaCausalLMAdapter.build_tiny,
            MedGemmaAdapter.build_tiny,
            M3DLaMedAdapter.build_tiny,
        ):
            adapter = builder(hidden_size=16, vocab_size=48, visual_token_buckets=(4,))
            if not adapter.capabilities.accepts_visual_tokens:
                errors.append(f"{type(adapter).__name__} does not accept its declared visual path")
            if not adapter.verify_tied_weights():
                errors.append(f"{type(adapter).__name__} lost tied input/output weights")

        language = GenericHFCausalLMAdapter.build_tiny(
            hidden_size=16,
            vocab_size=48,
            max_text_tokens=64,
            visual_token_buckets=(4,),
        )
        bridge = MLPBridge(
            source_dim=8,
            target_dim=16,
            output_tokens=4,
            max_input_tokens=4,
            source_modality=Modality.XRAY_2D,
        )
        text = language.tokenize(["prompt output"])
        labels = torch.full_like(text.input_ids, -100)
        labels[:, -1] = 4
        for modality in (Modality.XRAY_2D, Modality.CT_3D, Modality.PATHOLOGY_WSI):
            visual = bridge(torch.randn(1, 4, 8))
            visual = type(visual)(
                visual.tokens,
                source_modality=modality,
                token_mask=visual.token_mask,
                coordinate_system=visual.coordinate_system,
            )
            output = language.forward_with_visual_tokens(text, visual, labels)
            if output.loss is None or not torch.isfinite(output.loss):
                errors.append(f"invalid external loss for {modality.value}")
        native = MedGemmaAdapter.build_tiny(hidden_size=16, vocab_size=48, visual_token_buckets=(4,))
        native_visual = bridge(torch.randn(1, 4, 8))
        native_output = native.forward_with_visual_tokens(text, native_visual, labels)
        if native_output.loss is None or not torch.isfinite(native_output.loss):
            errors.append("invalid native MedGemma loss")

        if ThreeDCoordinateEncoder(output_dim=8)(torch.rand(1, 2, 3)).shape != (1, 2, 8):
            errors.append("3D coordinate encoder shape mismatch")
        if TwoDCoordinateEncoder(output_dim=8)(torch.rand(1, 2, 2)).shape != (1, 2, 8):
            errors.append("2D coordinate encoder shape mismatch")
        if WSICoordinateEncoder(output_dim=8)(torch.rand(1, 2, 2)).shape != (1, 2, 8):
            errors.append("WSI coordinate encoder shape mismatch")
        if stage_config(TrainingStage.BRIDGE_ONLY).trainable_modules != ("bridge", "boundary"):
            errors.append("Stage 1 trainable-module declaration changed")
    except Exception as exc:
        errors.append(f"Phase 09 contract check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_10_peft() -> list[str]:
    """Phase 10: PEFT targeting, trainability, quantization, and checkpoints."""
    errors: list[str] = []
    try:
        import tempfile

        import torch
        from torch import nn

        from medfm.models.language import GenericHFCausalLMAdapter
        from medfm.models.visual.native_3d import GenericMONAI3DAdapter
        from medfm.peft import (
            BackendCapabilityError,
            LoRAConfig,
            QLoRAConfig,
            QuantizationCapabilityError,
            audit_trainable_parameters,
            compare_merged_unmerged,
            inject_language_lora,
            inject_lora,
            inject_visual_lora,
            inspect_modules,
            load_adapter_checkpoint,
            save_adapter_checkpoint,
            validate_backend_combination,
        )

        vision = nn.Sequential(nn.Linear(8, 8))
        vision_config = LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_policy="explicit",
            target_modules=("0",),
            architecture="vision",
        )
        resolution = inspect_modules(vision, architecture="vision", config=vision_config)
        if resolution.selected_names != ("0",):
            errors.append("2D explicit target resolution changed")
        inject_lora(vision, vision_config, architecture="vision")
        vision(torch.randn(2, 8)).sum().backward()
        vision_audit = audit_trainable_parameters(vision)
        if vision_audit.adapter_parameters <= 0 or vision_audit.other_trainable_parameters:
            errors.append("2D LoRA trainability audit failed")
        vision_input = torch.randn(2, 8)
        if not compare_merged_unmerged(vision, lambda: vision(vision_input), atol=2e-5):
            errors.append("2D merged/unmerged equivalence failed")

        native_3d = GenericMONAI3DAdapter.build_tiny()
        result_3d = inject_visual_lora(
            native_3d,
            LoRAConfig(rank=2, alpha=4, dropout=0.0, architecture="3d_transformer"),
        )
        if not all("blocks.layers.1." in name for name in result_3d.selected_modules):
            errors.append("3D LoRA targeted an unexpected transformer stage")

        language = GenericHFCausalLMAdapter.build_tiny(
            hidden_size=16,
            vocab_size=48,
            max_text_tokens=64,
            visual_token_buckets=(4,),
        )
        inject_language_lora(
            language,
            LoRAConfig(rank=2, alpha=4, dropout=0.0, architecture="llm"),
        )
        language_audit = audit_trainable_parameters(language)
        if language_audit.adapter_parameters <= 0 or language_audit.bridge_parameters <= 0:
            errors.append("language LoRA/boundary trainability audit failed")
        with tempfile.TemporaryDirectory() as directory:
            save_adapter_checkpoint(
                directory,
                language,
                base_model_id="tiny-language",
                base_revision="phase-10",
                architecture="llm",
            )
            restored = GenericHFCausalLMAdapter.build_tiny(
                hidden_size=16,
                vocab_size=48,
                max_text_tokens=64,
                visual_token_buckets=(4,),
            )
            load_adapter_checkpoint(
                directory,
                restored,
                base_model_id="tiny-language",
                base_revision="phase-10",
                architecture="llm",
            )

        try:
            validate_backend_combination(
                LoRAConfig(architecture="llm"),
                QLoRAConfig(enabled=True, method="bitsandbytes_nf4", load_in_4bit=True),
                "xla_tpu",
                model_family="language",
            )
        except (BackendCapabilityError, QuantizationCapabilityError):
            pass
        else:
            errors.append("TPU QLoRA policy did not fail closed")
    except Exception as exc:
        errors.append(f"Phase 10 PEFT check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_11_task_modules() -> list[str]:
    """Phase 11: shared task heads, decoders, losses, and schema guards."""

    errors: list[str] = []
    try:
        import torch

        from medfm.core.batch import MedicalBatch
        from medfm.core.encoder import EncoderOutput
        from medfm.core.enums import Modality
        from medfm.core.task import LossOutput
        from medfm.models.decoders import LanguageConditionedMaskDecoder, UNetDecoder2D
        from medfm.models.heads import (
            BoxHead2D,
            ImageTextProjectionHead,
            LinearClassificationHead,
            SymmetricContrastiveLoss,
            normalized_to_physical_boxes,
        )
        from medfm.tasks import (
            BinarySegmentationTask,
            ClassificationTask,
            MultiTaskLossComposer,
            StructuredFindingsValidator,
        )

        encoded = EncoderOutput(
            pooled_embedding=torch.randn(2, 8, requires_grad=True),
            spatial_tokens=torch.randn(2, 4, 8),
            token_mask=torch.ones(2, 4, dtype=torch.bool),
            feature_maps=(torch.randn(2, 4, 4, 4), torch.randn(2, 8, 8, 8)),
        )
        batch = MedicalBatch(
            modality=Modality.XRAY_2D,
            sample_ids=["p0", "p1"],
            pixel_values=torch.randn(2, 1, 8, 8),
            labels=torch.tensor([0, 1]),
            task_targets={"segmentation": torch.zeros(2, 1, 8, 8)},
        )
        classification_loss = ClassificationTask(LinearClassificationHead(8, 2)).compute_loss(encoded, batch)
        segmentation_loss = BinarySegmentationTask(UNetDecoder2D((4, 8), hidden_channels=4)).compute_loss(
            encoded, batch
        )
        projection = ImageTextProjectionHead(8, 6, projection_dim=4)
        retrieval = projection(encoded, torch.randn(2, 6))
        alignment_loss = SymmetricContrastiveLoss()(retrieval)
        combined = MultiTaskLossComposer({"classification": 1.0, "segmentation": 1.0, "alignment": 1.0})(
            {
                "classification": classification_loss,
                "segmentation": segmentation_loss,
                "alignment": LossOutput(alignment_loss, sample_count=2),
            }
        )
        if not torch.isfinite(combined.total):
            errors.append("combined classification/segmentation/alignment loss is non-finite")
        if BoxHead2D(8)(encoded).boxes.shape != (2, 4):
            errors.append("2D box head output shape changed")
        physical = normalized_to_physical_boxes(
            torch.tensor([[0.25, 0.5, 0.75, 1.0]]),
            spatial_shape=(10, 20),
            spacing=(2.0, 3.0),
        )
        if not torch.allclose(physical, torch.tensor([[15.0, 10.0, 45.0, 20.0]])):
            errors.append("physical box conversion changed")
        language = LanguageConditionedMaskDecoder(8, 6, hidden_dim=8)
        conditioned = language(encoded.feature_maps or (), torch.randn(2, 3, 6))
        if conditioned.logits.shape != (2, 1, 8, 8):
            errors.append("language-conditioned mask output shape changed")
        schema_report = StructuredFindingsValidator().validate_batch(
            [{"findings": [], "impression": "clear"}, "invalid json"]
        )
        if schema_report.invalid != 1 or schema_report.parse_errors != 1:
            errors.append("structured schema errors were not counted")
    except Exception as exc:
        errors.append(f"Phase 11 task check failed: {type(exc).__name__}: {exc}")
    return errors


PHASE_05_REQUIRED_FILES = [
    "medfm/registry/__init__.py",
    "medfm/registry/schema.py",
    "medfm/registry/core.py",
    "medfm/registry/weights.py",
    "medfm/registry/acceptance.py",
    "medfm/registry/catalog.py",
    "medfm/registry/plugins.py",
    "medfm/registry/smoke.py",
    "medfm/cli/models.py",
    "medfm/cli/accelerator.py",
    "tests/phase_05/conftest.py",
    "tests/phase_05/test_model_registry.py",
    "tests/phase_05/test_weights.py",
    "tests/phase_05/test_smoke.py",
    "tests/phase_05/test_cli.py",
]


def _check_v1_catalog() -> list[str]:
    """Phase 05: every roster model registers as READY or BLOCKED with reason."""
    from medfm.registry import BACKEND_KEYS, ModelRegistry
    from medfm.registry.catalog import load_v1_catalog
    from medfm.registry.schema import ModelStatus

    errors: list[str] = []
    try:
        ModelRegistry.clear()
        specs = load_v1_catalog()
    except Exception as e:  # catalog must load deterministically
        return [f"v1 catalog failed to load: {type(e).__name__}: {e}"]
    roster = gov.load_yaml(gov.REPO_ROOT / gov.LICENSES_PATH)
    registered = {s.model_id: s for s in specs}
    for model_id in roster:
        spec = registered.get(model_id)
        if spec is None:
            errors.append(f"roster model '{model_id}' not registered by v1 catalog")
            continue
        if spec.status == ModelStatus.BLOCKED and not spec.blocked_reason:
            errors.append(f"model '{model_id}' BLOCKED without a structured reason")
        if set(spec.backend_support) != set(BACKEND_KEYS):
            errors.append(f"model '{model_id}' missing per-backend support keys")
    ModelRegistry.clear()
    return errors


def _check_phase_06_2d_adapters() -> list[str]:
    """Phase 06: 2D visual adapters are importable, their registry
    records carry pinned revisions, real preprocess specs, and declared
    Peft targets; plugins are registered for the 2D adapter model ids."""
    errors: list[str] = []
    from medfm.registry import ModelRegistry, clear_plugins, get_plugin
    from medfm.registry.catalog import load_v1_catalog
    from medfm.registry.schema import ModelStatus

    ModelRegistry.clear()
    clear_plugins()
    try:
        specs = load_v1_catalog()
    except Exception as e:
        return [f"Phase 06 catalog load failed: {type(e).__name__}: {e}"]
    by_id = {s.model_id: s for s in specs}
    for model_id in ("medsiglip", "rad-dino", "h-optimus-0", "medgemma-1.5-4b"):
        spec = by_id.get(model_id)
        if spec is None:
            errors.append(f"2D adapter model {model_id!r} not registered")
            continue
        if spec.preprocess is None:
            errors.append(f"{model_id} is missing a preprocess spec")
        if not spec.revision or len(spec.revision) < 40:
            errors.append(f"{model_id} revision is not a pinned SHA")
        if not spec.capabilities.peft.known_target_modules:
            errors.append(f"{model_id} Peft target modules are empty")
        if model_id == "rad-dino":
            if spec.status != ModelStatus.READY:
                errors.append("rad-dino must be READY (license approved)")
        else:
            if spec.status != ModelStatus.BLOCKED or not spec.blocked_reason:
                errors.append(f"{model_id} must be BLOCKED with a reason")

    if get_plugin("conch") is not None:
        errors.append("conch must have no plugin")
    for model_id in ("medsiglip", "rad-dino", "h-optimus-0", "medgemma-1.5-4b"):
        if get_plugin(model_id) is None:
            errors.append(f"no smoke plugin registered for {model_id}")
    ModelRegistry.clear()
    clear_plugins()
    return errors


def _check_phase_07_native_3d() -> list[str]:
    """Phase 07: native-volume adapters expose honest local contracts."""
    errors: list[str] = []
    from medfm.models.visual import (
        CTFMAdapter,
        GenericMONAI3DAdapter,
        MedSAM2Adapter,
        NVSegmentCTMRAdapter,
        TriadAdapter,
    )
    from medfm.registry import ModelRegistry, clear_plugins, get_plugin
    from medfm.registry.catalog import load_v1_catalog

    for builder, name in (
        (GenericMONAI3DAdapter.build_tiny, "generic"),
        (CTFMAdapter.build_tiny, "ct-fm"),
        (TriadAdapter.build_tiny, "triad"),
        (NVSegmentCTMRAdapter.build_tiny, "nv-segment-ctmr"),
        (MedSAM2Adapter.build_tiny, "medsam2"),
    ):
        try:
            adapter = builder()
            if len(adapter.preprocess.spatial_shape) != 3:
                errors.append(f"{name} does not declare a 3D preprocess shape")
            if not adapter.lora_target_patterns():
                errors.append(f"{name} has no reviewed LoRA targets")
            if not adapter.tpu_smoke_config()["static_batch"]:
                errors.append(f"{name} TPU smoke config is not static")
        except Exception as exc:
            errors.append(f"{name} tiny adapter failed: {type(exc).__name__}: {exc}")

    ModelRegistry.clear()
    clear_plugins()
    try:
        specs = {spec.model_id: spec for spec in load_v1_catalog()}
        for model_id in ("ct-fm", "flexict-3d", "triad", "nv-segment-ctmr", "medsam2", "merlin", "m3d-lamed"):
            spec = specs.get(model_id)
            if spec is None:
                errors.append(f"{model_id} missing from catalog")
                continue
            if spec.preprocess is None or len(spec.preprocess.spatial_shape) != 3:
                errors.append(f"{model_id} registry preprocess is not native 3D")
            if get_plugin(model_id) is None:
                errors.append(f"{model_id} has no offline smoke plugin")
    except Exception as exc:
        errors.append(f"Phase 07 catalog load failed: {type(exc).__name__}: {exc}")
    finally:
        ModelRegistry.clear()
        clear_plugins()
    return errors


def _check_phase_12_training() -> list[str]:
    """Phase 12: CPU contract smoke for config, trainer, checkpoint, memory."""
    errors: list[str] = []
    try:
        from tempfile import TemporaryDirectory

        from medfm.cli.train import tiny_builders
        from medfm.training.config import RunConfig
        from medfm.training.memory import CUDA_OOM_SUGGESTIONS, diagnose_oom
        from medfm.training.pipeline import TrainingPipeline
        from medfm.training.trainer import Trainer

        with TemporaryDirectory(prefix="medfm-phase12-") as directory:
            config = RunConfig.from_dict(
                {
                    "model_id": "tiny_multitask",
                    "task": {"name": "multiclass_classification"},
                    "accelerator": {"backend": "cpu", "precision": "fp32"},
                    "batch": {"microbatch_per_device": 2, "gradient_accumulation_steps": 1},
                    "max_steps": 1,
                    "output_dir": directory,
                }
            )
            built = TrainingPipeline(config, builders=tiny_builders()).build()
            if not isinstance(built.trainer, Trainer):
                errors.append("tiny Phase 12 pipeline did not construct Trainer")
            else:
                result = built.trainer.train()
                if not result.success or result.optimizer_steps != 1:
                    errors.append("tiny Phase 12 trainer did not complete one optimizer step")
                diagnostic = diagnose_oom(RuntimeError("simulated OOM"), backend="cuda", run_config=config)
                if diagnostic.suggestions != CUDA_OOM_SUGGESTIONS:
                    errors.append("CUDA OOM suggestion order changed")
    except Exception as exc:
        errors.append(f"Phase 12 training smoke failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_13_recipes() -> list[str]:
    """Phase 13: pinned offline builders, task families, and metric contracts."""
    errors: list[str] = []
    try:
        import torch

        from medfm.core.enums import TaskType
        from medfm.evaluation import classification_metrics, segmentation_metrics
        from medfm.recipes.phase13 import build_phase13_recipe, restore_mask_to_original
        from medfm.training.config import RunConfig

        recipe_dir = gov.REPO_ROOT / "configs" / "recipes" / "2d"
        families: set[str] = set()
        offline_builds = 0
        for path in sorted(recipe_dir.glob("*.yaml")):
            config = RunConfig.load(path)
            family = str(config.recipe.get("family", config.recipe.get("type", ""))).lower().replace("-", "_")
            families.add(family)
            if not bool(config.recipe.get("offline_tiny", False)):
                continue
            build = build_phase13_recipe(config)
            offline_builds += 1
            if not build.train_data:
                errors.append(f"{path.name} produced no offline training data")
            if build.metadata.dataset_revision == "":
                errors.append(f"{path.name} omitted dataset revision")
        required_families = {"classification", "segmentation", "promptable_segmentation", "native_vlm", "external_vlm"}
        missing_families = required_families - families
        if missing_families:
            errors.append(f"Phase 13 recipe families missing: {sorted(missing_families)}")
        if offline_builds < 6:
            errors.append("Phase 13 has too few offline contract recipes")

        promptable = RunConfig.load(recipe_dir / "segmentation_promptable_smoke.yaml")
        if build_phase13_recipe(promptable).task.task_type != TaskType.PROMPTABLE_SEGMENTATION:
            errors.append("promptable segmentation task type is not preserved")
        restored = restore_mask_to_original(torch.ones(1, 1, 2, 2), original_size=(8, 8), crop_box=(2, 2, 6, 6))
        if tuple(restored.shape) != (1, 1, 8, 8):
            errors.append("segmentation mask restoration shape changed")
        classification = classification_metrics([0, 1], [0.1, 0.9])
        if classification["auroc"].unit != "per_patient":
            errors.append("classification metric unit contract changed")
        segmentation = segmentation_metrics(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
        if "dice/class_0" not in segmentation:
            errors.append("segmentation metric contract is missing per-class Dice")
    except Exception as exc:
        errors.append(f"Phase 13 recipe check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_14_recipes() -> list[str]:
    """Phase 14: native-volume, slice-sequence, and language-query contracts."""
    errors: list[str] = []
    try:
        import torch

        from medfm.recipes.phase14 import (
            build_phase14_recipe,
            native_3d_segmentation_metrics,
            select_volume_input_policy,
        )
        from medfm.training.config import RunConfig

        recipe_dir = gov.REPO_ROOT / "configs" / "recipes" / "3d"
        families: set[str] = set()
        offline_builds = 0
        for path in sorted(recipe_dir.glob("*.yaml")):
            config = RunConfig.load(path)
            family = str(config.recipe.get("family", config.recipe.get("type", ""))).lower().replace("-", "_")
            families.add(family)
            if not bool(config.recipe.get("offline_tiny", False)):
                continue
            build = build_phase14_recipe(config)
            offline_builds += 1
            if not build.train_data:
                errors.append(f"{path.name} produced no offline training data")
            if not build.metadata.dataset_revision or not build.metadata.preprocessing_revision:
                errors.append(f"{path.name} omitted pinned dataset/preprocessing revisions")
            if not build.metadata.shape_buckets:
                errors.append(f"{path.name} omitted static 3D shape buckets")

        required_families = {
            "native_3d_classification",
            "native_3d_segmentation",
            "native_3d_vlm",
            "slice_sequence_vlm",
            "language_conditioned_3d_segmentation",
        }
        missing = required_families - families
        if missing:
            errors.append(f"Phase 14 recipe families missing: {sorted(missing)}")
        if offline_builds < 8:
            errors.append("Phase 14 has too few offline contract recipes")

        policy = select_volume_input_policy(
            {
                "input_strategy": "global_local",
                "recommended_shape_buckets": [{"kind": "3d_patch", "shape": [96, 96, 96]}],
                "global_shape": [128, 128, 128],
                "local_shape": [96, 96, 96],
            }
        )
        if policy.strategy != "global_local" or policy.global_shape != (128, 128, 128):
            errors.append("fingerprint-driven global/local shape policy changed")
        target = torch.zeros(1, 1, 4, 4, 4)
        target[..., 1:3, 1:3, 1:3] = 1
        metrics = native_3d_segmentation_metrics(torch.where(target > 0, 8.0, -8.0), target)
        required_metrics = {
            "dice/class_0",
            "surface_dice/class_0",
            "hd95/class_0",
            "lesion_recall/class_0",
            "false_positives_per_scan/class_0",
            "volume_error_mm3/class_0",
        }
        if not required_metrics.issubset(metrics):
            errors.append("native 3D segmentation metrics are incomplete")
    except Exception as exc:
        errors.append(f"Phase 14 recipe check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_15_recipes() -> list[str]:
    """Phase 15: bounded pathology families, evidence, and metric contracts."""
    errors: list[str] = []
    try:
        import torch

        from medfm.recipes.pathology_stitching import (
            TilePrediction,
            evidence_payload,
            stitch_tile_predictions,
            validate_evidence_json,
        )
        from medfm.recipes.phase15 import (
            build_phase15_recipe,
            pathology_classification_metrics,
            pathology_segmentation_metrics,
            patient_disjoint_split,
        )
        from medfm.training.config import RunConfig

        recipe_dir = gov.REPO_ROOT / "configs" / "recipes" / "pathology"
        families: set[str] = set()
        offline_builds = 0
        tpu_profiles = 0
        for path in sorted(recipe_dir.glob("*.yaml")):
            config = RunConfig.load(path)
            family = str(config.recipe.get("family", config.recipe.get("type", ""))).lower().replace("-", "_")
            families.add(family)
            if not bool(config.recipe.get("offline_tiny", False)):
                continue
            build = build_phase15_recipe(config)
            offline_builds += 1
            if not build.train_data:
                errors.append(f"{path.name} produced no offline training data")
            metadata = build.metadata
            if not metadata.dataset_revision or not metadata.preprocessing_revision or not metadata.model_revision:
                errors.append(f"{path.name} omitted pinned provenance revisions")
            if metadata.max_tiles_per_slide <= 0 or metadata.shard_unit != "slide":
                errors.append(f"{path.name} omitted bounded slide metadata")
            if any(count > metadata.max_tiles_per_slide for count in metadata.sampled_tile_counts):
                errors.append(f"{path.name} exceeded max tiles per slide")
            if str(config.accelerator.backend).lower() in {"xla_tpu", "tpu"}:
                tpu_profiles += 1
            if family == "wsi_vlm":
                if metadata.visual_token_count not in {32, 64, 128}:
                    errors.append(f"{path.name} did not use a fixed visual token bucket")
                if metadata.selector_revision != "phase15-selector-v1":
                    errors.append(f"{path.name} omitted selector revision")

        required_families = {"tile_classification", "wsi_classification", "wsi_vlm", "pathology_segmentation"}
        missing = required_families - families
        if missing:
            errors.append(f"Phase 15 recipe families missing: {sorted(missing)}")
        if offline_builds < 15:
            errors.append("Phase 15 has too few offline contract recipes")
        if tpu_profiles < 2:
            errors.append("Phase 15 is missing separate TPU classification and VLM profiles")

        split = patient_disjoint_split(
            [
                {"slide_id": "s0", "patient_id": "p0"},
                {"slide_id": "s1", "patient_id": "p0"},
                {"slide_id": "s2", "patient_id": "p1"},
                {"slide_id": "s3", "patient_id": "p2"},
            ],
            seed=15,
        )
        groups = {split.patient_by_slide[slide] for slide in split.train}
        if groups & {split.patient_by_slide[slide] for slide in split.validation + split.test}:
            errors.append("patient-disjoint split crossed a patient boundary")

        metrics = pathology_classification_metrics([0, 1], [0.1, 0.9], patient_ids=["p0", "p1"], slide_ids=["s0", "s1"])
        if {"tile/auroc", "slide/auroc", "patient/auroc"} - set(metrics):
            errors.append("pathology classification metrics omitted tile/slide/patient units")
        target = torch.zeros(1, 1, 4, 4)
        segmentation = pathology_segmentation_metrics(target, target, slide_predicted=target, slide_target=target)
        if "tile/dice/class_0" not in segmentation or "slide/dice/class_0" not in segmentation:
            errors.append("pathology segmentation metrics omitted tile/slide units")

        stitched = stitch_tile_predictions(
            [TilePrediction("slide", "tile", torch.ones(2, 2), 0, 0, 2, 2)],
            (2, 2),
        )
        if not bool(stitched.coverage_mask.all()):
            errors.append("host stitching did not cover a valid slide")
        evidence = evidence_payload(
            [{"tile_id": "tile", "x": 0, "y": 0, "width": 2, "height": 2}],
            slide_id="slide",
            slide_shape=(2, 2),
        )
        if validate_evidence_json(evidence):
            errors.append("evidence JSON contract rejected valid level-0 coordinates")
    except Exception as exc:
        errors.append(f"Phase 15 recipe check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_16_evaluation() -> list[str]:
    """Phase 16: schemas, clinical-unit metrics, parity, and report safety."""

    errors: list[str] = []
    try:
        import torch

        from medfm.evaluation import (
            ClinicalIdentity,
            ClinicalUnit,
            EvaluationSchemaError,
            EvaluationSplit,
            PredictionArtifact,
            PredictionRecord,
            RuntimeProvenance,
            build_evaluation_report,
            classification_metrics,
            compare_backend_predictions,
            fit_threshold,
            generation_metrics,
            retrieval_metrics,
            segmentation_metrics,
            visual_grounding_gate,
        )

        metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], unit="per_patient")
        required_classification = {
            "auroc",
            "auprc",
            "brier",
            "ece",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "confusion_matrix",
        }
        if not required_classification.issubset(metrics):
            errors.append("Phase 16 classification metrics are incomplete")
        if metrics["auroc"].unit != "per_patient" or metrics["auroc"].sample_count != 4:
            errors.append("Phase 16 classification metric unit/count contract changed")

        target = torch.zeros(1, 1, 4, 4, 4)
        target[..., 1:3, 1:3, 1:3] = 1
        segmentation = segmentation_metrics(torch.where(target > 0, 8.0, -8.0), target, unit="per_scan")
        required_segmentation = {
            "dice/class_0",
            "iou/class_0",
            "surface_dice/class_0",
            "hd95/class_0",
            "assd/class_0",
            "lesion_sensitivity/class_0",
            "false_positive_lesions/class_0",
            "volume_error_mm3/class_0",
        }
        if not required_segmentation.issubset(segmentation):
            errors.append("Phase 16 segmentation metrics are incomplete")

        retrieval = retrieval_metrics([[1.0, 0.0], [0.0, 1.0]], query_ids=["a", "b"], candidate_ids=["a", "b"])
        if retrieval["image_to_text/recall@1"].value != 1.0 or retrieval["text_to_image/recall@1"].value != 1.0:
            errors.append("bidirectional retrieval contract changed")
        generation = generation_metrics(['{"findings": []}'], ['{"findings": []}'], schema={"type": "object"})
        if generation["schema_validity"].value != 1.0:
            errors.append("structured generation schema validity contract changed")
        if visual_grounding_gate(0.80, 0.79, margin=0.05)["passed"]:
            errors.append("visual grounding gate did not fail for shuffled performance within margin")
        if not compare_backend_predictions([1.0], [1.0], backend="cuda").within_tolerance:
            errors.append("backend parity identity fixture failed")
        try:
            fit_threshold([0, 1], [0.1, 0.9], split="test")
        except EvaluationSchemaError:
            pass
        else:
            errors.append("threshold fitting accepted test data")

        artifact = PredictionArtifact(
            artifact_id="phase16-validator",
            task="classification",
            clinical_unit=ClinicalUnit.PATIENT,
            split=EvaluationSplit.TEST,
            predictions=(
                PredictionRecord(
                    sample_id="p0",
                    prediction=0.1,
                    target=0,
                    clinical_unit=ClinicalUnit.PATIENT,
                    identity=ClinicalIdentity(patient_id="p0"),
                    split=EvaluationSplit.TEST,
                ),
                PredictionRecord(
                    sample_id="p1",
                    prediction=0.9,
                    target=1,
                    clinical_unit=ClinicalUnit.PATIENT,
                    identity=ClinicalIdentity(patient_id="p1"),
                    split=EvaluationSplit.TEST,
                ),
            ),
            provenance=RuntimeProvenance(
                model_hash="validator-model",
                data_hash="validator-data",
                preprocess_hash="validator-preprocess",
                backend="cpu",
                precision="fp32",
                topology="single",
                bucket="tiny",
                checkpoint_format="safetensors",
            ),
        )
        report = build_evaluation_report(artifact, metrics)
        if report.to_dict()["claims"].get("clinically_validated") is not False:
            errors.append("evaluation report contains an unsupported clinical-validation claim")
    except Exception as exc:
        errors.append(f"Phase 16 evaluation check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_phase_17_inference() -> list[str]:
    """Phase 17: portable bundles, bounded pipelines, VLM, and audit safety."""

    errors: list[str] = []
    try:
        import tempfile

        import torch

        from medfm.inference import (
            AuditLogger,
            BaseModelReference,
            BundleBuilder,
            ClassificationPipeline,
            GenerationConfig,
            InferenceLimits,
            RuntimeSupport,
            generate,
            load_bundle,
            sliding_window_inference,
        )

        def model(value: torch.Tensor) -> torch.Tensor:
            return torch.zeros((value.shape[0], 2), device=value.device)

        pipeline = ClassificationPipeline(model, limits=InferenceLimits(max_batch_size=2, max_image_pixels=1024))
        result = pipeline.predict(torch.zeros(1, 1, 8, 8), modality="XRAY_2D")
        if tuple(result["probabilities"].shape) != (1, 2):
            errors.append("classification inference smoke returned an unexpected shape")
        volume = torch.zeros(1, 1, 4, 4, 4)
        if not torch.allclose(sliding_window_inference(volume, lambda crop: crop, window_shape=(2, 2, 2)), volume):
            errors.append("Gaussian sliding-window identity parity failed")
        generated = generate(
            type("Generator", (), {"generate": lambda self, **kwargs: '{"ok": true}'})(),
            config=GenerationConfig(output_schema={"type": "object", "required": ["ok"]}),
        )
        if generated.schema_valid is not True:
            errors.append("structured VLM output validation failed")
        with tempfile.TemporaryDirectory() as directory:
            builder = BundleBuilder(
                directory,
                bundle_id="phase17-validator",
                model_id="validator-model",
                model_revision="validator-revision",
                task="classification",
                base_models=[BaseModelReference("validator-base", "validator-base-revision")],
                model_card="validator model card",
                license_summary="validator license summary",
                preprocessing={"name": "identity"},
                postprocessing={"name": "identity"},
                task_schema={"type": "object"},
                inference_config={"backend": "cpu"},
                runtime=RuntimeSupport(backends={"cpu": "tested"}),
            )
            builder.add_adapter("validator-adapter", {"weight": torch.ones(1)})
            bundle = builder.build()
            loaded = load_bundle(bundle.root, base_model_id="validator-base", base_revision="validator-base-revision")
            if loaded.adapter_names != ("validator-adapter",):
                errors.append("adapter-only bundle load contract changed")
            audit_path = gov.REPO_ROOT / "artifacts" / "smoke" / "phase17-validator-audit.jsonl"
            audit = AuditLogger(audit_path)
            event = audit.create_event(
                model_id="validator-model",
                model_revision="validator-revision",
                adapter_id="validator-adapter",
                adapter_revision="validator-revision",
                preprocess_hash="identity",
                prompt_version=None,
                input_value={"report": "redacted"},
                output_value={"ok": True},
                runtime="cpu",
                peak_vram_bytes=0,
                error_status=None,
            )
            if "report" in json.dumps(event.to_dict()):
                errors.append("operational audit log retained a raw sensitive field")
    except Exception as exc:
        errors.append(f"Phase 17 inference check failed: {type(exc).__name__}: {exc}")
    return errors


def _check_report(phase: str, errors: list[str]) -> None:
    report_dir = gov.REPO_ROOT / "agent" / "reports" / f"phase_{phase}"
    for name in REPORT_FILES:
        path = report_dir / name
        if not path.exists():
            errors.append(f"missing phase report file: {path.relative_to(gov.REPO_ROOT)}")
        elif path.stat().st_size == 0:
            errors.append(f"empty phase report file: {path.relative_to(gov.REPO_ROOT)}")

    acceptance_path = report_dir / "acceptance.json"
    if acceptance_path.exists() and acceptance_path.stat().st_size > 0:
        report = gov.load_json(acceptance_path)
        schema = gov.load_json(gov.REPO_ROOT / gov.ACCEPTANCE_SCHEMA_PATH)
        errors.extend(validate_acceptance(report, schema, phase))

    test_results = report_dir / "test_results.json"
    if test_results.exists() and test_results.stat().st_size > 0:
        data = gov.load_json(test_results)
        if not data.get("tests"):
            errors.append("test_results.json contains no tests")
        for t in data.get("tests", []):
            if t.get("status") not in {"passed", "failed", "skipped"}:
                errors.append(f"test_results.json: test '{t.get('name')}' has invalid status")
            if t.get("status") == "skipped" and not t.get("reason"):
                errors.append(f"test_results.json: skipped test '{t.get('name')}' lacks a reason")


def validate_acceptance(report: dict[str, Any], schema: dict[str, Any], phase: str) -> list[str]:
    errors = gov.validate_acceptance_report(report, schema)
    if errors:
        return errors
    if report["phase"] != phase:
        errors.append(f"acceptance.json phase '{report['phase']}' != requested '{phase}'")
    if report["status"] != "passed":
        errors.append(f"acceptance.json status is '{report['status']}', gate requires 'passed'")
    return errors


def validate_phase(phase: str) -> list[str]:
    errors: list[str] = []
    if phase == "00":
        for rel in PHASE_00_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        for model_id, errs in gov.validate_license_file().items():
            errors.extend(f"license '{model_id}': {e}" for e in errs)
        scope = gov.load_yaml(gov.REPO_ROOT / gov.SCOPE_PATH)
        license_ids = set(gov.load_yaml(gov.REPO_ROOT / gov.LICENSES_PATH))
        errors.extend(gov.check_scope_consistency(scope, license_ids))
        errors.extend(gov.check_accelerator_policy(scope))
    elif phase == "01":
        for rel in PHASE_01_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
    elif phase == "02":
        for rel in PHASE_02_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
    elif phase == "03":
        for rel in PHASE_03_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
    elif phase == "04":
        for rel in PHASE_04_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
    elif phase == "05":
        for rel in PHASE_05_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_v1_catalog())
    elif phase == "06":
        for rel in PHASE_06_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_06_2d_adapters())
    elif phase == "07":
        for rel in PHASE_07_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_07_native_3d())
    elif phase == "08":
        for rel in PHASE_08_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_08_pathology())
    elif phase == "09":
        for rel in PHASE_09_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_09_language_bridges())
    elif phase == "10":
        for rel in PHASE_10_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_10_peft())
    elif phase == "11":
        for rel in PHASE_11_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_11_task_modules())
    elif phase == "12":
        for rel in PHASE_12_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_12_training())
    elif phase == "13":
        for rel in PHASE_13_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_13_recipes())
    elif phase == "14":
        for rel in PHASE_14_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_14_recipes())
    elif phase == "15":
        for rel in PHASE_15_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_15_recipes())
    elif phase == "16":
        for rel in PHASE_16_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_16_evaluation())
    elif phase == "17":
        for rel in PHASE_17_REQUIRED_FILES:
            if not (gov.REPO_ROOT / rel).exists():
                errors.append(f"missing required file: {rel}")
        errors.extend(_check_phase_17_inference())

    else:
        errors.append(f"no validator registered for phase {phase}")
    _check_report(phase, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a phase acceptance gate.")
    parser.add_argument("--phase", required=True, help="phase number, e.g. 00")
    args = parser.parse_args(argv)
    phase = args.phase.zfill(2)

    errors = validate_phase(phase)
    if errors:
        print(f"Phase {phase} gate FAILED ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Phase {phase} gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
