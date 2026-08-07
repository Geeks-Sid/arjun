# Phase 13: 2D Training Recipes

## Objective

Deliver reproducible 2D classification, segmentation, native VLM, and external-encoder VLM recipes with staged baselines and acceptance metrics.

## Dependencies

- [x] Phases 06 and 09-12 are accepted for relevant components.
- [x] Phase 16 metric interfaces are available or implemented alongside recipe acceptance.
- [x] Approved, de-identified datasets or synthetic substitutes are registered.

## Scope boundaries

Allowed areas: `configs/recipes/2d/`, recipe builders, examples, recipe tests, and recipe documentation.

Do not put dataset-specific constants in shared model/trainer code.

## Recipe 13A: 2D classification

- [x] Provide MedSigLIP and RAD-DINO baseline configs.
- [x] Add H-Optimus tile and FlexiCT-2D variants when accepted.
- [x] Stage A: frozen encoder plus simple pooling/head.
- [x] Validate labels, prevalence, patient split, and ordinary BCE/CE baseline.
- [x] Require one-batch overfit before full training.
- [x] Stage B: head plus final-quarter/final-third attention LoRA.
- [ ] Stage C: optional MLP LoRA/rank increase/final norm unfreeze only after Stage B evidence.
- [x] Record AUROC/AUPRC, calibration, operating points, subgroup metrics, and memory.
- [ ] Compare against majority/random and a conventional task-specific baseline.
- [x] Provide CUDA BF16 and TPU BF16 configs with the same effective global batch where feasible.
- [x] Use fixed image-resolution and batch buckets on TPU.
- [ ] Compare one-step and short-run CUDA/TPU loss curves before full training.

## Recipe 13B: 2D segmentation

- [x] Provide RAD-DINO or generic-transformer frozen-encoder decoder config.
- [x] Train decoder first, then add late vision LoRA.
- [x] Add MedSAM2 native promptable path separately.
- [x] Use task-appropriate patch/crop sampling and original-resolution validation.
- [x] Export masks back to original image coordinates.
- [x] Report per-class Dice, surface metrics, lesion sensitivity, and false positives/image.
- [x] Compare against a conventional UNet baseline.
- [x] Bucket image/crop shapes and keep TPU decoder outputs static.
- [ ] Run sliding/tiled validation with fixed window batches on TPU.

## Recipe 13C: Native 2D VLM

- [x] Use MedGemma 1.5 4B as primary native model when approved/available.
- [x] Start with NF4 QLoRA, frozen visual tower, microbatch 1, accumulation, checkpointing, and no KV cache.
- [x] Begin with VQA/structured findings before free-form reports.
- [x] Add mixed VQA, classification-as-generation, findings, impressions, and reports with explicit weights.
- [ ] Add vision LoRA only after frozen-vision validation improves.
- [x] Use deterministic evaluation generation and schema validation.
- [x] Run no-image and shuffled-image visual-dependence ablations.
- [ ] CUDA profile: allow NF4 QLoRA after bitsandbytes capability validation.
- [x] TPU profile: use BF16 LoRA with replicated or SPMD/FSDP base model; do not label it QLoRA.
- [x] Bucket image count, image shape, input/output text length, and visual-token count for TPU.
- [ ] Record compile warmup separately from training throughput.

## Recipe 13D: External-encoder 2D VLM
- [x] Provide MedSigLIP/RAD-DINO to Perceiver to LLM configuration.
- [x] Start with 64 visual tokens and benchmark 32/64/128.
- [x] Stage 1: bridge only.
- [x] Stage 2: bridge plus LLM QLoRA.
- [x] Stage 3: bridge plus LLM and late vision LoRA.
- [x] Preserve 2D coordinates, view, image index, and timepoint metadata.
- [x] Compare linear and Perceiver bridges.
- [ ] Require shuffled-visual degradation before declaring grounding success.
- [x] Provide a cached-visual-token TPU baseline before joint vision adaptation.
- [x] Keep Perceiver query count and all language lengths static within each TPU bucket.

## Cross-recipe checklist

- [x] Pin model/data/preprocessing/prompt revisions.
- [x] Include smoke, tiny-overfit, development, and target-run configs.
- [x] Set deterministic validation and best-checkpoint criteria.
- [x] Record effective batch size, trainable counts, and peak VRAM.
- [x] Resume each recipe from a mid-run checkpoint.
- [x] Produce a model card and known-limitations section for each accepted baseline.
- [x] Include backend-specific optimizer, precision, attention, distribution, and checkpoint settings.
- [x] Record per-device microbatch, world size, accumulation, global batch, and any explicit LR scaling.
- [ ] Require each claimed TPU recipe to pass no-recompile steady-state and portable-export gates.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [MedSigLIP](https://github.com/Google-Health/medsiglip)
- [RAD-DINO](https://huggingface.co/microsoft/rad-dino)
- [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [Accelerate TPU training](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)

## Smoke command

```bash
python -m medfm.cli.train --config configs/recipes/2d/classification_smoke.yaml
```

## Acceptance command

```bash
python -m pytest tests/phase_13 -q && python -m medfm.tools.validate_phase --phase 13
```

## Exit criteria

- [x] Frozen and LoRA classification baselines exist.
- [x] Frozen-encoder segmentation baseline reconstructs original-space output.
- [ ] Native 2D VLM baseline passes masking and visual-dependence checks.
- [ ] External 2D VLM baseline passes bridge and grounding checks.
- [x] Every result includes reproducibility, clinical-unit metrics, and memory artifacts.
- [ ] CUDA and TPU support claims are backed by protected hardware reports and parity checks.

## Handoff

- [x] Publish accepted baseline configs and measured results.
- [x] Record data/model restrictions preventing any recipe acceptance.
- [x] Identify hyperparameters suitable for later ablations, not defaults.
- [x] Provide export candidates to Phase 17.
- [x] Publish separate CUDA QLoRA and TPU BF16 LoRA configs where applicable.
