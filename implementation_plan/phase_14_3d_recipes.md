# Phase 14: 3D CT/MRI Training Recipes

## Objective

Deliver distinct, reproducible recipes for native 3D classification, native 3D segmentation, native 3D VLMs, slice-sequence VLMs, and language-conditioned 3D segmentation.

## Dependencies

- [x] Phases 07 and 09-12 are accepted.
- [x] Phase 03 fingerprints and Phase 04 3D transforms/samplers are accepted.
- [x] Approved synthetic CT/MRI smoke datasets are registered in the Phase 14 recipe matrix.

## Scope boundaries

Allowed areas: `configs/recipes/3d/`, recipe builders, slice selectors, examples, and Phase 14 tests.

Do not treat slice-sequence processing as native 3D encoding.

## Recipe 14A: Native 3D classification

- [x] Provide CT-FM/FlexiCT-3D config and one MRI config where available.
- [x] Select full volume, fixed crop, multicrop, low-resolution global, or global+local input from the dataset fingerprint.
- [x] Start with a low-resolution volume or 96-128 cubed task crop, subject to model requirements (offline contract fixtures use a bounded 16^3 shape).
- [x] Stage 1: frozen encoder plus pooled head.
- [x] Stage 2: frozen encoder plus attention pooling.
- [x] Stage 3: final-stage vision LoRA.
- [x] Stage 4: multicrop attention aggregation only if needed.
- [x] Report patient/study-level metrics and memory by crop strategy.
- [x] Choose a small bounded set of fixed 3D shape buckets for TPU.
- [x] Provide CUDA and TPU BF16 configs with explicit global-batch semantics.

## Recipe 14B: Native 3D segmentation

- [x] Establish MONAI/nnU-Net-style conventional baseline.
- [x] Train a decoder over frozen CT-FM/Triad/generic features.
- [x] Add late encoder LoRA only after decoder baseline.
- [x] Use positive-lesion sampling and log actual positive-patch rate.
- [x] Add deep supervision only after baseline acceptance.
- [x] Validate with sliding-window inference and configurable blending.
- [x] Invert outputs to original physical space.
- [x] Report Dice, surface/HD95, lesion recall, false positives/scan, and volume error.
- [x] Keep TPU train and sliding-window inference patch shapes static.
- [x] Separate host transform/inversion time from accelerator execution time.
- [x] Verify positive sampling and padding masks are identical across distributed backends.

## Recipe 14C: Native 3D VLM

- [x] Connect native 3D spatial tokens to physical coordinate embeddings and a Perceiver.
- [x] Start with 32-64 visual tokens and 512 text tokens (offline contract fixtures use an 8-token text length).
- [x] Stage 1: bridge/contrastive alignment with frozen 3D encoder and LLM.
- [x] Stage 2: bridge plus LLM QLoRA for VQA and structured findings.
- [x] Stage 3: add final-stage 3D vision LoRA after validation evidence.
- [x] Stage 4: add boxes, region tokens, language-conditioned masks, and coordinate output.
- [x] Support cached spatial-token training before expensive joint tuning.
- [x] Compare against no-image/shuffled-image and slice-sequence baselines.
- [x] Use cached fixed-shape 3D tokens as the first TPU bridge/LLM baseline.
- [x] Use TPU BF16 LoRA/SPMD rather than bitsandbytes QLoRA.
- [x] Add joint 3D vision adaptation on TPU only after the cached-token baseline and operator audit pass.

## Recipe 14D: Slice-sequence VLM

- [x] Implement uniform selector first with a fixed slice count.
- [x] Add anatomy-aware, report-conditioned, entropy, lesion-aware, and multi-window selectors behind separate experiments.
- [x] Preserve index, normalized/physical z, series order, window, and MRI sequence.
- [x] Use a frozen 2D/native multi-image visual tower for projector warm-up.
- [x] Add LLM QLoRA after projector acceptance.
- [x] Keep config names, metrics, and artifacts distinct from native 3D VLM.
- [x] Benchmark number of slices and token budget under the 48 GB cap.
- [x] Bucket slice count, per-slice shape, position tensors, and text lengths for TPU.
- [x] Perform any data-dependent slice selection on the host before fixed-shape collation.
- [x] Benchmark the same selector/bucket on CUDA and TPU.

## Recipe 14E: Language-conditioned 3D segmentation

- [x] Produce text embeddings from explicit anatomy/lesion queries.
- [x] Fuse text with the 3D feature pyramid through cross-attention.
- [x] Produce masks through the 3D decoder.
- [x] Test absent-target, multiple-target, laterality, and ambiguous-query behavior.
- [x] Evaluate mask accuracy and query grounding separately.
- [x] Use fixed query count and static decoder output shapes per TPU bucket.

## Cross-recipe checklist

- [x] Include one-batch overfit and checkpoint-resume tests.
- [x] Log spacing, crop origin, positive sampling, visual/text tokens, and peak VRAM.
- [x] Compare frozen and LoRA stages before broadening trainable parameters.
- [x] Pin all model/data/preprocess revisions.
- [x] Separate CT and MRI sequence/channel assumptions.
- [ ] Produce representative original-space visualizations in protected artifacts.
- [x] Record compiler count, input wait, throughput, HBM/VRAM, and fallback operators by backend.
- [x] Require a generic MONAI 3D TPU baseline even if an upstream foundation model is TPU-blocked.
- [x] Keep unsupported custom CUDA models marked blocked rather than replacing their internals silently.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [CT-FM](https://github.com/project-lighter/CT-FM)
- [FlexiCT](https://github.com/ricklisz/FlexiCT)
- [Triad](https://github.com/wangshansong1/Triad)
- [MONAI sliding-window inferers](https://docs.monai.io/en/stable/inferers.html)
- [PyTorch/XLA troubleshooting](https://docs.pytorch.org/xla/master/debug.html)

## Smoke command

```bash
python -m medfm.cli.train --config configs/recipes/3d/classification_smoke.yaml
```

## Acceptance command

```bash
pytest tests/phase_14 -q && python -m medfm.tools.validate_phase --phase 14
```

## Exit criteria

- [x] Native 3D classification and sliding-window segmentation pass.
- [x] Native 3D tokens produce grounded LM loss and generation metrics.
- [x] Slice-sequence VLM passes as a separate experiment family.
- [x] Language-conditioned 3D segmentation produces spatial masks.
- [x] Every accepted recipe remains within the configured memory cap.
- [ ] At least one 3D task recipe passes TPU hardware acceptance; each remaining model has an explicit TPU status.

## Handoff

- [x] Publish accepted CT/MRI configs, crop policies, and memory results.
- [x] Publish native-3D versus slice-sequence comparison protocol.
- [x] Record models/datasets blocked by license or access.
- [x] Provide export candidates and original-space examples to Phase 17.
- [ ] Publish CUDA/TPU shape buckets, parity results, and blocked custom operators.
