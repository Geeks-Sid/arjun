# Phase 13 handoff

## Recipe entry points

Use `RunConfig.load` and `medfm.recipes.phase13.phase13_builders()` for all 2D training. Do not add backbone or dataset branches to `Trainer`.

Offline contract smokes:

```bash
python -m medfm.cli.train --config configs/recipes/2d/classification_smoke.yaml --format json
python -m medfm.cli.train --config configs/recipes/2d/segmentation_smoke.yaml --format json
python -m medfm.cli.train --config configs/recipes/2d/native_vlm_smoke.yaml --format json
python -m medfm.cli.train --config configs/recipes/2d/external_vlm_smoke.yaml --format json
```

Use the matching production/checkpoint-gated configs only after approved local checkpoints and manifest hashes are available:

- Classification: `classification_medsiglip.yaml`, `classification_raddino.yaml`, `classification_hoptimus_tile.yaml`, `classification_flexict_2d.yaml`, `classification_cuda_bf16.yaml`, `classification_tpu_bf16.yaml`.
- Segmentation: `segmentation_smoke.yaml`, `segmentation_development_lora.yaml`, `segmentation_medsam2_promptable.yaml`.
- Native VLM: `native_vlm_cuda_nf4.yaml`, `native_vlm_tpu_bf16_lora.yaml`, `native_structured_findings_smoke.yaml`.
- External VLM: `external_vlm_linear_64.yaml`, `external_vlm_perceiver_32.yaml`, `external_vlm_perceiver_128.yaml`, `external_vlm_cuda_qlora.yaml`, `external_vlm_tpu_cached.yaml`, `external_vlm_stage3_vision_lora.yaml`.

## Contract details

- `recipe.family` selects the recipe-owned builder; accepted values are `classification`, `segmentation`, `promptable_segmentation`, `native_vlm`, and `external_vlm`.
- `offline_tiny: true` is only a deterministic test fixture and must never be described as a clinical model.
- Classification task-head and decoder parameters are optimizer components outside the backbone model; run metadata counts both model and task parameters and reports configured accumulation.
- `mixed_task_weights` is validated as finite, non-negative, and positive-sum; values are retained in recipe metadata and VLM batch targets.
- VLM visual-token counts are fixed within a recipe bucket. External bridge comparisons use 32/64/128 token configs; TPU cached mode remains a separate baseline.
- Structured findings use `StructuredGenerationTask` and `medfm.tasks.structured` schema validation. Invalid generated structures must remain counted and must not silently enter scoring.
- `restore_mask_to_original` accepts `[B,H,W]` or `[B,1,H,W]` masks and requires an explicit original size; crop placement is validated.
- `run_visual_dependence_ablation` must be reported with image, no-image, and shuffled deltas. Offline random results do not establish grounding.

## Protected accelerator commands

```bash
PJRT_DEVICE=TPU python -m medfm.cli.train --config configs/recipes/2d/classification_tpu_bf16.yaml --backend xla_tpu --dry-run --format json
PJRT_DEVICE=TPU python -m medfm.cli.train --config configs/recipes/2d/native_vlm_tpu_bf16_lora.yaml --backend xla_tpu --format json
python -m medfm.cli.train --config configs/recipes/2d/native_vlm_cuda_nf4.yaml --backend cuda --dry-run --format json
python -m medfm.cli.train --config configs/recipes/2d/external_vlm_cuda_qlora.yaml --backend cuda --dry-run --format json
```

Never auto-reduce scientific shapes, token budgets, precision, or effective batch after an OOM. Edit the recipe explicitly and preserve the resulting config hash.

## Next evidence gates

1. Register approved de-identified manifests, preprocessing hashes, and local checkpoint directories.
2. Install and capability-check `torch_xla` for TPU BF16/no-recompile/export acceptance and `bitsandbytes` for CUDA NF4 acceptance.
3. Run one-step and short-run CPU/CUDA/TPU parity with matching global batch and fixed buckets.
4. Run clinical-unit classification/segmentation metrics, calibrated operating points, subgroup analysis, conventional/majority baselines, and external-site/human-review evidence.
5. Require shuffled-visual degradation after training before calling VLM grounding accepted.
6. Export accepted adapter/checkpoint candidates for Phase 17 with full provenance and limitations.
