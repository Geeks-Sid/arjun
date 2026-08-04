# Phase 14: 3D CT/MRI Training Recipes

## Objective

Deliver distinct, reproducible recipes for native 3D classification, native 3D segmentation, native 3D VLMs, slice-sequence VLMs, and language-conditioned 3D segmentation.

## Dependencies

- [ ] Phases 07 and 09-12 are accepted.
- [ ] Phase 03 fingerprints and Phase 04 3D transforms/samplers are accepted.
- [ ] Approved CT/MRI datasets or synthetic smoke datasets are registered.

## Scope boundaries

Allowed areas: `configs/recipes/3d/`, recipe builders, slice selectors, examples, and Phase 14 tests.

Do not treat slice-sequence processing as native 3D encoding.

## Recipe 14A: Native 3D classification

- [ ] Provide CT-FM/FlexiCT-3D config and one MRI config where available.
- [ ] Select full volume, fixed crop, multicrop, low-resolution global, or global+local input from the dataset fingerprint.
- [ ] Start with a low-resolution volume or 96-128 cubed task crop, subject to model requirements.
- [ ] Stage 1: frozen encoder plus pooled head.
- [ ] Stage 2: frozen encoder plus attention pooling.
- [ ] Stage 3: final-stage vision LoRA.
- [ ] Stage 4: multicrop attention aggregation only if needed.
- [ ] Report patient/study-level metrics and memory by crop strategy.
- [ ] Choose a small bounded set of fixed 3D shape buckets for TPU.
- [ ] Provide CUDA and TPU BF16 configs with explicit global-batch semantics.

## Recipe 14B: Native 3D segmentation

- [ ] Establish MONAI/nnU-Net-style conventional baseline.
- [ ] Train a decoder over frozen CT-FM/Triad/generic features.
- [ ] Add late encoder LoRA only after decoder baseline.
- [ ] Use positive-lesion sampling and log actual positive-patch rate.
- [ ] Add deep supervision only after baseline acceptance.
- [ ] Validate with sliding-window inference and configurable blending.
- [ ] Invert outputs to original physical space.
- [ ] Report Dice, surface/HD95, lesion recall, false positives/scan, and volume error.
- [ ] Keep TPU train and sliding-window inference patch shapes static.
- [ ] Separate host transform/inversion time from accelerator execution time.
- [ ] Verify positive sampling and padding masks are identical across distributed backends.

## Recipe 14C: Native 3D VLM

- [ ] Connect native 3D spatial tokens to physical coordinate embeddings and a Perceiver.
- [ ] Start with 32-64 visual tokens and 512 text tokens.
- [ ] Stage 1: bridge/contrastive alignment with frozen 3D encoder and LLM.
- [ ] Stage 2: bridge plus LLM QLoRA for VQA and structured findings.
- [ ] Stage 3: add final-stage 3D vision LoRA after validation evidence.
- [ ] Stage 4: add boxes, region tokens, language-conditioned masks, and coordinate output.
- [ ] Support cached spatial-token training before expensive joint tuning.
- [ ] Compare against no-image/shuffled-image and slice-sequence baselines.
- [ ] Use cached fixed-shape 3D tokens as the first TPU bridge/LLM baseline.
- [ ] Use TPU BF16 LoRA/SPMD rather than bitsandbytes QLoRA.
- [ ] Add joint 3D vision adaptation on TPU only after the cached-token baseline and operator audit pass.

## Recipe 14D: Slice-sequence VLM

- [ ] Implement uniform selector first with a fixed slice count.
- [ ] Add anatomy-aware, report-conditioned, entropy, lesion-aware, and multi-window selectors behind separate experiments.
- [ ] Preserve index, normalized/physical z, series order, window, and MRI sequence.
- [ ] Use a frozen 2D/native multi-image visual tower for projector warm-up.
- [ ] Add LLM QLoRA after projector acceptance.
- [ ] Keep config names, metrics, and artifacts distinct from native 3D VLM.
- [ ] Benchmark number of slices and token budget under the 48 GB cap.
- [ ] Bucket slice count, per-slice shape, position tensors, and text lengths for TPU.
- [ ] Perform any data-dependent slice selection on the host before fixed-shape collation.
- [ ] Benchmark the same selector/bucket on CUDA and TPU.

## Recipe 14E: Language-conditioned 3D segmentation

- [ ] Produce text embeddings from explicit anatomy/lesion queries.
- [ ] Fuse text with the 3D feature pyramid through cross-attention.
- [ ] Produce masks through the 3D decoder.
- [ ] Test absent-target, multiple-target, laterality, and ambiguous-query behavior.
- [ ] Evaluate mask accuracy and query grounding separately.
- [ ] Use fixed query count and static decoder output shapes per TPU bucket.

## Cross-recipe checklist

- [ ] Include one-batch overfit and checkpoint-resume tests.
- [ ] Log spacing, crop origin, positive sampling, visual/text tokens, and peak VRAM.
- [ ] Compare frozen and LoRA stages before broadening trainable parameters.
- [ ] Pin all model/data/preprocess revisions.
- [ ] Separate CT and MRI sequence/channel assumptions.
- [ ] Produce representative original-space visualizations in protected artifacts.
- [ ] Record compiler count, input wait, throughput, HBM/VRAM, and fallback operators by backend.
- [ ] Require a generic MONAI 3D TPU baseline even if an upstream foundation model is TPU-blocked.
- [ ] Keep unsupported custom CUDA models marked blocked rather than replacing their internals silently.

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

- [ ] Native 3D classification and sliding-window segmentation pass.
- [ ] Native 3D tokens produce grounded LM loss and generation metrics.
- [ ] Slice-sequence VLM passes as a separate experiment family.
- [ ] Language-conditioned 3D segmentation produces spatial masks.
- [ ] Every accepted recipe remains within the configured memory cap.
- [ ] At least one 3D task recipe passes TPU hardware acceptance; each remaining model has an explicit TPU status.

## Handoff

- [ ] Publish accepted CT/MRI configs, crop policies, and memory results.
- [ ] Publish native-3D versus slice-sequence comparison protocol.
- [ ] Record models/datasets blocked by license or access.
- [ ] Provide export candidates and original-space examples to Phase 17.
- [ ] Publish CUDA/TPU shape buckets, parity results, and blocked custom operators.
