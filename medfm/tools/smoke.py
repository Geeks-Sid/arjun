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


def _check_data_fingerprint_fixture() -> None:
    from medfm.data.fingerprint import fingerprint_manifest
    from medfm.data.manifests.io import read_manifest

    fixture = gov.REPO_ROOT / "tests" / "fixtures" / "manifests" / "mixed_synthetic.parquet"
    if not fixture.is_file():
        raise RuntimeError(f"missing fingerprint fixture: {fixture}")
    report = fingerprint_manifest(read_manifest(fixture))
    if not report["split_leakage"]["ok"]:
        raise RuntimeError("committed fixture manifest has split leakage")
    again = fingerprint_manifest(read_manifest(fixture))
    if again["fingerprint_hash"] != report["fingerprint_hash"]:
        raise RuntimeError("dataset fingerprint is not deterministic")


def _check_dicom_sort_and_cache_invalidation() -> None:
    import importlib

    import numpy as np
    import torch

    from medfm.data.caching import PreprocessingCache
    from medfm.data.readers.dicom import DICOMSeriesReader

    fixture_dir = str(gov.REPO_ROOT / "tests" / "phase_03")
    sys.path.insert(0, fixture_dir)
    try:
        synthetic = importlib.import_module("synthetic")  # tests/phase_03 fixture builder
    finally:
        sys.path.remove(fixture_dir)

    with tempfile.TemporaryDirectory() as tmp:
        series_dir = Path(tmp) / "series"
        _, raw = synthetic.write_dicom_series(series_dir, num_slices=4, shuffle_files=True, value_seed=3)
        read = DICOMSeriesReader().read(series_dir)
        expected = raw.astype(np.float64) * 2.0 - 1000.0
        if not np.allclose(read.image.numpy().transpose(2, 1, 0), expected):
            raise RuntimeError("DICOM physical sort / CT calibration mismatch")

        cache = PreprocessingCache.on_disk(Path(tmp) / "cache")
        key = PreprocessingCache.key(source_file_hash="0" * 64, reader_version="1.0.0", preprocessing_hash="1" * 64)
        cache.put(key, {"image": torch.randn(4)})
        if cache.get(key) is None:
            raise RuntimeError("cache round-trip failed")
        altered = PreprocessingCache.key(source_file_hash="0" * 64, reader_version="1.0.0", preprocessing_hash="2" * 64)
        if cache.get(altered) is not None:
            raise RuntimeError("cache failed to invalidate on preprocessing change")


PHASE_01_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("doctor_json_schema", _check_doctor_json),
    ("monai_3d_load_crop", _check_monai_3d_pipeline),
    ("tiny_lora_step", _check_tiny_lora_step),
]


def _check_phase_07_native_3d() -> None:
    import torch

    from medfm.core.batch import MedicalBatch
    from medfm.core.encoder import OutputSpec
    from medfm.core.enums import Modality
    from medfm.core.sample import SpatialMetadata
    from medfm.models.visual import GenericMONAI3DAdapter
    from medfm.models.visual.native_3d import sliding_window_inference

    adapter = GenericMONAI3DAdapter.build_tiny()
    adapter.eval()
    metadata = [
        SpatialMetadata(
            original_shape=(16, 16, 16),
            current_shape=(16, 16, 16),
            affine=torch.eye(4, dtype=torch.float64),
            spacing_mm=(1.0, 1.0, 1.0),
            orientation="RAS",
        )
    ]
    volume = torch.randn(1, 1, 16, 16, 16)
    output = adapter.encode(
        MedicalBatch(
            modality=Modality.CT_3D,
            sample_ids=["smoke-3d"],
            pixel_values=volume,
            spatial_metadata=metadata,
        ),
        output_spec=OutputSpec(spatial_tokens=True, token_coordinates=True, feature_maps=True),
    )
    if output.spatial_tokens is None or output.token_coordinates is None or output.feature_maps is None:
        raise RuntimeError("generic native 3D smoke omitted a requested output")
    reconstructed = sliding_window_inference(
        volume,
        lambda crop: crop,
        window_shape=(16, 16, 16),
        overlap=0.0,
    )
    if not torch.equal(reconstructed, volume):
        raise RuntimeError("native 3D sliding-window identity reconstruction failed")


def _check_phase_08_pathology() -> None:
    from types import SimpleNamespace

    import torch

    from medfm.models.pathology import (
        AttentionMILAggregator,
        EmbeddingStore,
        MeanPoolingAggregator,
        TinyPathologyTileEncoder,
        TokenBudget,
        WSITokenSelector,
        extract_slide_embeddings,
    )

    class Record:
        def __init__(self, index: int) -> None:
            self.tile_id = f"tile-{index}"
            self.x, self.y = index * 8, 0
            self.width = self.height = 8
            self.level, self.mpp = 0, 0.5
            self.quality = {"blur": float(index + 1)}

    class Reader:
        def read_tiles(self, locations, *, level, size, on_corrupt="skip"):
            tiles = torch.stack([torch.full((3, size[1], size[0]), float(x + y)) for x, y in locations])
            return SimpleNamespace(tiles=tiles, coords=torch.tensor(locations), errors=())

    records = [Record(i) for i in range(5)]
    encoder = TinyPathologyTileEncoder(embedding_dim=16)
    with tempfile.TemporaryDirectory() as directory:
        store = EmbeddingStore(directory)
        stats = extract_slide_embeddings("smoke-slide", records, Reader(), encoder, store, chunk_size=2)
        if not stats.complete or store.validate("smoke-slide"):
            raise RuntimeError("pathology extraction did not commit a valid store")
        cached = store.read_slide("smoke-slide")
        mask = torch.ones(1, cached.embeddings.shape[0], dtype=torch.bool)
        tile_embeddings = cached.embeddings.unsqueeze(0)
        mean = MeanPoolingAggregator(16)(tile_embeddings, mask)
        mil = AttentionMILAggregator(16)(tile_embeddings, mask)
        if mean.shape != (1, 16) or mil.shape != (1, 16):
            raise RuntimeError("slide aggregators returned an unexpected shape")
        selected = WSITokenSelector(budget=TokenBudget(precompression=128, visual_tokens=32)).select(
            cached.embeddings, records
        )
        if selected.tokens.shape != (32, 16) or int(selected.mask.sum()) != 5:
            raise RuntimeError("WSI selector violated its fixed visual-token contract")
        head = torch.nn.Linear(16, 2)
        head(mean).sum().backward()
        if not any(parameter.grad is not None for parameter in head.parameters()):
            raise RuntimeError("classifier backward from cached pathology embeddings failed")


def _check_phase_09_language_bridges() -> None:
    import torch

    from medfm.core.enums import Modality
    from medfm.core.language import GenerationConfig, ProjectedVisualTokens
    from medfm.models.bridges import LinearBridge, MLPBridge, TrainingStage, apply_stage_freeze, stage_config
    from medfm.models.language import GenericHFCausalLMAdapter, MedGemmaAdapter

    language = GenericHFCausalLMAdapter.build_tiny(
        model_id="phase09-external-tiny",
        hidden_size=16,
        vocab_size=64,
        max_text_tokens=64,
        visual_token_buckets=(32,),
    )
    bridge = MLPBridge(
        source_dim=8,
        target_dim=16,
        output_tokens=32,
        max_input_tokens=32,
        source_modality=Modality.XRAY_2D,
    )
    text = language.tokenize(["assistant output"])
    labels = torch.full_like(text.input_ids, -100)
    labels[:, -1] = 5
    for modality in (Modality.XRAY_2D, Modality.CT_3D, Modality.PATHOLOGY_WSI):
        visual = bridge(
            torch.randn(1, 32, 8),
            torch.ones(1, 32, dtype=torch.bool),
        )
        visual = ProjectedVisualTokens(
            visual.tokens,
            source_modality=modality,
            token_mask=visual.token_mask,
            coordinate_system=visual.coordinate_system,
        )
        result = language.forward_with_visual_tokens(text, visual, labels)
        if result.loss is None or not torch.isfinite(result.loss):
            raise RuntimeError(f"external {modality.value} language loss is invalid")
        language.zero_grad(set_to_none=True)
        bridge.zero_grad(set_to_none=True)
        result.loss.backward()
        if not any(parameter.grad is not None for parameter in bridge.parameters()):
            raise RuntimeError("visual bridge did not receive gradients")

    native = MedGemmaAdapter.build_tiny(
        model_id="phase09-native-tiny",
        hidden_size=16,
        vocab_size=64,
        visual_token_buckets=(32,),
    )
    native_visual = LinearBridge(
        source_dim=8,
        target_dim=16,
        output_tokens=32,
        max_input_tokens=32,
        source_modality=Modality.CT_3D,
    )(torch.randn(1, 32, 8))
    native_result = native.forward_with_visual_tokens(text, native_visual, labels)
    if native_result.loss is None or not torch.isfinite(native_result.loss):
        raise RuntimeError("native MedGemma language loss is invalid")
    generated = native.generate(text, native_visual, GenerationConfig(max_new_tokens=2))
    if generated.token_ids is None or generated.token_ids.shape[1] > 2:
        raise RuntimeError("native generation exceeded its output limit")
    if not native.verify_tied_weights():
        raise RuntimeError("tiny native language model lost tied embeddings")

    frozen_vision = torch.nn.Linear(8, 8)
    stage = stage_config(TrainingStage.BRIDGE_ONLY)
    apply_stage_freeze(
        {"vision": frozen_vision, "language": language, "bridge": bridge, "boundary": language.boundary_embeddings},
        stage,
    )
    if any(parameter.requires_grad for parameter in frozen_vision.parameters()) or any(
        parameter.requires_grad for parameter in language.model.parameters()
    ):
        raise RuntimeError("Stage 1 did not freeze vision and language modules")


def _check_phase_11_task_modules() -> None:
    """Exercise one synthetic path through each Phase 11 task family."""

    import torch

    from medfm.core.batch import MedicalBatch
    from medfm.core.encoder import EncoderOutput
    from medfm.core.enums import Modality
    from medfm.core.task import LossOutput
    from medfm.models.decoders import LanguageConditionedMaskDecoder, UNetDecoder2D
    from medfm.models.heads import (
        ImageTextProjectionHead,
        LinearClassificationHead,
        SymmetricContrastiveLoss,
    )
    from medfm.tasks import (
        BinarySegmentationTask,
        ClassificationTask,
        MultiTaskLossComposer,
        StructuredFindingsValidator,
    )

    visual = EncoderOutput(
        pooled_embedding=torch.randn(2, 8, requires_grad=True),
        spatial_tokens=torch.randn(2, 4, 8, requires_grad=True),
        token_mask=torch.ones(2, 4, dtype=torch.bool),
        feature_maps=(torch.randn(2, 4, 4, 4), torch.randn(2, 8, 8, 8)),
    )
    classification_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["cls-0", "cls-1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        labels=torch.tensor([0, 1]),
    )
    classification = ClassificationTask(LinearClassificationHead(8, 2))
    cls_loss = classification.compute_loss(visual, classification_batch)
    segmentation_batch = MedicalBatch(
        modality=Modality.XRAY_2D,
        sample_ids=["seg-0", "seg-1"],
        pixel_values=torch.randn(2, 1, 8, 8),
        task_targets={"segmentation": torch.zeros(2, 1, 8, 8)},
    )
    segmentation = BinarySegmentationTask(UNetDecoder2D((4, 8), hidden_channels=4))
    seg_loss = segmentation.compute_loss(visual, segmentation_batch)
    text = torch.randn(2, 3, 6, requires_grad=True)
    language_decoder = LanguageConditionedMaskDecoder(8, 6, hidden_dim=8)
    language_output = language_decoder(visual.feature_maps or (), text)
    language_loss = language_output.logits.square().mean()
    language_loss.backward()
    retrieval_head = ImageTextProjectionHead(8, 6, projection_dim=4)
    retrieval_output = retrieval_head(visual, text.mean(dim=1))
    align_loss = SymmetricContrastiveLoss()(retrieval_output)
    combined = MultiTaskLossComposer({"classification": 1.0, "segmentation": 1.0, "alignment": 1.0})(
        {
            "classification": cls_loss,
            "segmentation": seg_loss,
            "alignment": LossOutput(align_loss, sample_count=2),
        }
    )
    if not torch.isfinite(combined.total):
        raise RuntimeError("multitask Phase 11 smoke loss is non-finite")
    combined.total.backward()
    valid = {"findings": [], "impression": "clear"}
    report = StructuredFindingsValidator().validate_batch([valid, "not json"])
    if report.invalid != 1 or report.parse_errors != 1:
        raise RuntimeError("structured findings smoke did not count invalid output")

def _check_phase_12_training_engine() -> None:
    """Build the tiny recipe, run one step, export, and inspect diagnostics."""
    from dataclasses import replace

    from medfm.cli.train import tiny_builders
    from medfm.training.checkpoint import CheckpointManager
    from medfm.training.config import RunConfig
    from medfm.training.memory import CUDA_OOM_SUGGESTIONS, diagnose_oom
    from medfm.training.pipeline import TrainingPipeline
    from medfm.training.trainer import Trainer

    with tempfile.TemporaryDirectory(prefix="medfm-phase12-") as directory:
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
            raise RuntimeError("Phase 12 tiny pipeline did not construct Trainer")
        result = built.trainer.train()
        if not result.success or result.optimizer_steps != 1:
            raise RuntimeError(f"tiny trainer did not complete one optimizer step: {result.to_dict()}")
        adapter = built.trainer.export_adapter("adapter")
        if not (adapter / "adapter.safetensors").exists():
            raise RuntimeError("portable adapter export was not written")
        diagnostic = diagnose_oom(RuntimeError("simulated out of memory"), backend="cuda", run_config=config)
        if diagnostic.scientific_configuration_mutated or diagnostic.suggestions != CUDA_OOM_SUGGESTIONS:
            raise RuntimeError("OOM diagnostic mutated configuration or changed suggestion order")


def _check_phase_15_pathology() -> None:
    """Build one bounded recipe per pathology family and validate evidence."""
    from dataclasses import replace

    import torch

    from medfm.recipes.pathology_stitching import (
        TilePrediction,
        stitch_tile_predictions,
        validate_evidence_json,
    )
    from medfm.recipes.phase15 import build_phase15_recipe, phase15_builders
    from medfm.training.config import RunConfig
    from medfm.training.pipeline import TrainingPipeline

    recipe_root = gov.REPO_ROOT / "configs" / "recipes" / "pathology"
    names = (
        "tile_classification_hoptimus_linear.yaml",
        "wsi_classification_smoke.yaml",
        "wsi_vlm_cached_smoke.yaml",
        "segmentation_smoke.yaml",
    )
    with tempfile.TemporaryDirectory(prefix="medfm-phase15-") as directory:
        for name in names:
            config = RunConfig.load(recipe_root / name)
            config = replace(
                config,
                accelerator=replace(config.accelerator, backend="cpu", precision="fp32", distribution="single"),
                max_steps=1,
                output_dir=str(Path(directory) / Path(name).stem),
            )
            built = build_phase15_recipe(config)
            if not built.train_data or built.metadata.shard_unit != "slide":
                raise RuntimeError(f"{name} did not expose bounded pathology metadata")
            run = TrainingPipeline(config, builders=phase15_builders()).build()
            result = run.trainer.train()
            if not result.success or result.optimizer_steps != 1:
                raise RuntimeError(f"{name} did not complete one optimizer step")
            if name.startswith("wsi_vlm"):
                output = built.model.forward_mode(built.train_data[0], mode="image")
                payload = built.model.evidence_json(slide_id="smoke-slide")
                if not output.evidence_tiles or validate_evidence_json(payload, slide_shape=(64, 64)):
                    raise RuntimeError("WSI VLM evidence JSON failed validation")

        stitched = stitch_tile_predictions(
            [TilePrediction("smoke-slide", "tile-0", torch.ones(4, 4), 0, 0, 4, 4)],
            (4, 4),
        )
        if not bool(stitched.coverage_mask.all()):
            raise RuntimeError("pathology host stitching did not cover the slide")


PHASE_15_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("pathology_recipes_stitching_evidence", _check_phase_15_pathology),
]

PHASE_12_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("training_engine_checkpoint_memory_smoke", _check_phase_12_training_engine),
]



PHASE_08_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("pathology_extraction_store_aggregation", _check_phase_08_pathology),
]


PHASE_09_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("language_bridges_and_native_vlm", _check_phase_09_language_bridges),
]


PHASE_07_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("native_3d_contract_and_reconstruction", _check_phase_07_native_3d),
]


PHASE_02_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("core_sample_roundtrip", _check_core_sample_roundtrip),
    ("core_batch_contract", _check_core_batch_contract),
]

PHASE_03_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("data_fingerprint_fixture", _check_data_fingerprint_fixture),
    ("dicom_sort_and_cache_invalidation", _check_dicom_sort_and_cache_invalidation),
]
PHASE_11_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("task_heads_decoders_losses", _check_phase_11_task_modules),
]


SMOKE_CHECKS: dict[str, list[tuple[str, Callable[[], None]]]] = {
    "01": PHASE_01_CHECKS,
    "02": PHASE_02_CHECKS,
    "03": PHASE_03_CHECKS,
    "07": PHASE_07_CHECKS,
    "08": PHASE_08_CHECKS,
    "09": PHASE_09_CHECKS,
    "11": PHASE_11_CHECKS,
    "12": PHASE_12_CHECKS,
    "15": PHASE_15_CHECKS,
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
