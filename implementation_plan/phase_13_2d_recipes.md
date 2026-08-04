# Phase 13: 2D Training Recipes

## Objective

Deliver reproducible 2D classification, segmentation, native VLM, and external-encoder VLM recipes with staged baselines and acceptance metrics.

## Dependencies

- [ ] Phases 06 and 09-12 are accepted for relevant components.
- [ ] Phase 16 metric interfaces are available or implemented alongside recipe acceptance.
- [ ] Approved, de-identified datasets or synthetic substitutes are registered.

## Scope boundaries

Allowed areas: `configs/recipes/2d/`, recipe builders, examples, recipe tests, and recipe documentation.

Do not put dataset-specific constants in shared model/trainer code.

## Recipe 13A: 2D classification

- [ ] Provide MedSigLIP and RAD-DINO baseline configs.
- [ ] Add H-Optimus tile and FlexiCT-2D variants when accepted.
- [ ] Stage A: frozen encoder plus simple pooling/head.
- [ ] Validate labels, prevalence, patient split, and ordinary BCE/CE baseline.
- [ ] Require one-batch overfit before full training.
- [ ] Stage B: head plus final-quarter/final-third attention LoRA.
- [ ] Stage C: optional MLP LoRA/rank increase/final norm unfreeze only after Stage B evidence.
- [ ] Record AUROC/AUPRC, calibration, operating points, subgroup metrics, and memory.
- [ ] Compare against majority/random and a conventional task-specific baseline.
- [ ] Provide CUDA BF16 and TPU BF16 configs with the same effective global batch where feasible.
- [ ] Use fixed image-resolution and batch buckets on TPU.
- [ ] Compare one-step and short-run CUDA/TPU loss curves before full training.

## Recipe 13B: 2D segmentation

- [ ] Provide RAD-DINO or generic-transformer frozen-encoder decoder config.
- [ ] Train decoder first, then add late vision LoRA.
- [ ] Add MedSAM2 native promptable path separately.
- [ ] Use task-appropriate patch/crop sampling and original-resolution validation.
- [ ] Export masks back to original image coordinates.
- [ ] Report per-class Dice, surface metrics, lesion sensitivity, and false positives/image.
- [ ] Compare against a conventional UNet baseline.
- [ ] Bucket image/crop shapes and keep TPU decoder outputs static.
- [ ] Run sliding/tiled validation with fixed window batches on TPU.

## Recipe 13C: Native 2D VLM

- [ ] Use MedGemma 1.5 4B as primary native model when approved/available.
- [ ] Start with NF4 QLoRA, frozen visual tower, microbatch 1, accumulation, checkpointing, and no KV cache.
- [ ] Begin with VQA/structured findings before free-form reports.
- [ ] Add mixed VQA, classification-as-generation, findings, impressions, and reports with explicit weights.
- [ ] Add vision LoRA only after frozen-vision validation improves.
- [ ] Use deterministic evaluation generation and schema validation.
- [ ] Run no-image and shuffled-image visual-dependence ablations.
- [ ] CUDA profile: allow NF4 QLoRA after bitsandbytes capability validation.
- [ ] TPU profile: use BF16 LoRA with replicated or SPMD/FSDP base model; do not label it QLoRA.
- [ ] Bucket image count, image shape, input/output text length, and visual-token count for TPU.
- [ ] Record compile warmup separately from training throughput.

## Recipe 13D: External-encoder 2D VLM

- [ ] Provide MedSigLIP/RAD-DINO to Perceiver to LLM configuration.
- [ ] Start with 64 visual tokens and benchmark 32/64/128.
- [ ] Stage 1: bridge only.
- [ ] Stage 2: bridge plus LLM QLoRA.
- [ ] Stage 3: bridge plus LLM and late vision LoRA.
- [ ] Preserve 2D coordinates, view, image index, and timepoint metadata.
- [ ] Compare linear and Perceiver bridges.
- [ ] Require shuffled-visual degradation before declaring grounding success.
- [ ] Provide a cached-visual-token TPU baseline before joint vision adaptation.
- [ ] Keep Perceiver query count and all language lengths static within each TPU bucket.

## Cross-recipe checklist

- [ ] Pin model/data/preprocessing/prompt revisions.
- [ ] Include smoke, tiny-overfit, development, and target-run configs.
- [ ] Set deterministic validation and best-checkpoint criteria.
- [ ] Record effective batch size, trainable counts, and peak VRAM.
- [ ] Resume each recipe from a mid-run checkpoint.
- [ ] Produce a model card and known-limitations section for each accepted baseline.
- [ ] Include backend-specific optimizer, precision, attention, distribution, and checkpoint settings.
- [ ] Record per-device microbatch, world size, accumulation, global batch, and any explicit LR scaling.
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
pytest tests/phase_13 -q && python -m medfm.tools.validate_phase --phase 13
```

## Exit criteria

- [ ] Frozen and LoRA classification baselines exist.
- [ ] Frozen-encoder segmentation baseline reconstructs original-space output.
- [ ] Native 2D VLM baseline passes masking and visual-dependence checks.
- [ ] External 2D VLM baseline passes bridge and grounding checks.
- [ ] Every result includes reproducibility, clinical-unit metrics, and memory artifacts.
- [ ] CUDA and TPU support claims are backed by protected hardware reports and parity checks.

## Handoff

- [ ] Publish accepted baseline configs and measured results.
- [ ] Record data/model restrictions preventing any recipe acceptance.
- [ ] Identify hyperparameters suitable for later ablations, not defaults.
- [ ] Provide export candidates to Phase 17.
- [ ] Publish separate CUDA QLoRA and TPU BF16 LoRA configs where applicable.
