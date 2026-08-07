# Phase 13 2D recipe model cards

These cards describe the recipe contracts delivered in Phase 13. The offline recipes use deterministic random-weight adapters and synthetic data. They are test fixtures, not clinical models.

## Classification

- **Recipes:** `classification_medsiglip.yaml`, `classification_raddino.yaml`, `classification_hoptimus_tile.yaml`, `classification_flexict_2d.yaml`, `classification_development_lora.yaml`, and smoke variants.
- **Input/task:** fixed 2D radiology image buckets with binary or multilabel classification targets.
- **Training:** Stage A freezes the visual adapter and trains a task head; Stage B adds declared late visual LoRA targets. The optimizer records `task_head` and `vision_lora` groups separately.
- **Reported metrics:** AUROC, AUPRC, Brier/ECE calibration, threshold operating points, subgroup results, sample counts, and `per_patient` units.
- **Known limitations:** production checkpoint loading, calibrated thresholds, prevalence estimates, patient-level external validation, majority/conventional comparator results, and human review require approved data and release-specific evidence.

## Segmentation

- **Recipes:** `segmentation_smoke.yaml`, `segmentation_development_lora.yaml`, and `segmentation_medsam2_promptable.yaml`.
- **Input/task:** fixed 2D image/crop buckets with binary masks; promptable recipes carry prompts in `task_targets["prompt_map"]` rather than mutating pixels.
- **Training:** decoder-first freeze schedule followed by declared late vision LoRA. Promptable tasks preserve `PROMPTABLE_SEGMENTATION` task identity.
- **Reported metrics:** per-class Dice, surface Dice, lesion sensitivity, false positives per image, sample counts, and `per_image` units. `restore_mask_to_original` maps crop predictions to original image coordinates.
- **Known limitations:** full-resolution/tiled clinical validation, lesion sampling quality, surface-distance calibration, conventional U-Net comparisons, and approved MedSAM2 checkpoint execution remain unavailable in this environment.

## Native VLM

- **Recipes:** `native_vlm_smoke.yaml`, `native_structured_findings_smoke.yaml`, `native_vlm_cuda_nf4.yaml`, and `native_vlm_tpu_bf16_lora.yaml`.
- **Input/task:** fixed image and text buckets with masked prompt tokens, explicit visual-token counts, and no KV cache during training. Structured findings use versioned JSON schema validation before scoring.
- **Training:** the primary production profiles distinguish CUDA NF4 QLoRA from TPU BF16 LoRA; neither is mislabeled or silently substituted. Mixed VQA/findings/impression/report weights are preserved in metadata.
- **Known limitations:** MedGemma production checkpoints, `bitsandbytes` NF4 execution, TPU PJRT/no-recompile/export evidence, deterministic generation quality, shuffled-image grounding, clinical report quality, and human review are not available here.

## External-encoder VLM

- **Recipes:** `external_vlm_linear_64.yaml`, `external_vlm_perceiver_32.yaml`, `external_vlm_perceiver_128.yaml`, `external_vlm_cuda_qlora.yaml`, `external_vlm_tpu_cached.yaml`, and `external_vlm_stage3_vision_lora.yaml`.
- **Input/task:** external 2D visual encoders feed an LLM through linear or fixed-query Perceiver bridges. Visual metadata retains view, image index, timepoint, coordinates, masks, and token bucket identity.
- **Training:** Stage 1 trains the bridge; Stage 2 adds language LoRA; Stage 3 adds late vision LoRA. Cached TPU mode is a separate baseline.
- **Known limitations:** bridge grounding is not accepted from offline random weights. Image/no-image/shuffled-image deltas must be reported after approved-data training; external-site performance, clinical calibration, and production checkpoint parity remain open.

## Shared safety and reproducibility limits

Every accepted artifact must retain configuration, dataset, preprocessing, prompt/model revisions, seed, effective batch geometry, trainable/total counts, memory information, clinical units, and explicit limitations. No offline result may be presented as clinical validation, and no runtime failure may silently change scientific shapes, token budgets, precision, or batch geometry.
