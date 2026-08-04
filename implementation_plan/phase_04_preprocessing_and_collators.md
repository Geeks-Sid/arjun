# Phase 04: Preprocessing, Augmentation, and Collators

## Objective

Implement deterministic medical canonicalization, model-aware normalization, task sampling, stochastic augmentation, invertible spatial transforms, text preparation, and batch collation.

## Dependencies

- [ ] Phase 03 readers and fingerprints are accepted.
- [ ] Phase 02 schemas and metadata semantics are stable.
- [ ] Model preprocess specs needed for tests are represented by local dummy specs.

## Scope boundaries

Allowed areas: `medfm/data/transforms/`, `samplers/`, `collators/`, text preprocessing, and Phase 04 tests.

Do not hide preprocessing inside model `forward` methods or download model weights.

## Implementation checklist

### Pipeline composition

- [ ] Implement explicit deterministic and stochastic transform stages.
- [ ] Put the cache boundary after deterministic canonicalization/normalization where valid.
- [ ] Record transform history needed for spatial inversion.
- [ ] Validate final tensors against the selected `PreprocessSpec`.
- [ ] Make randomness seedable per worker, epoch, and sample.
- [ ] Keep decode/canonicalization and unsupported medical transforms on the host; transfer fixed tensors only after collation.
- [ ] Make accelerator execution optional for transforms and require parity tests before enabling it.

### 2D radiology

- [ ] Decode grayscale and correct MONOCHROME1.
- [ ] Implement aspect-preserving resize/letterbox and optional body-region crop.
- [ ] Support single-channel and repeated three-channel output.
- [ ] Preserve view, multi-view, and longitudinal ordering metadata.
- [ ] Add conservative rotation, translation, scale, intensity, and noise augmentations.
- [ ] Gate horizontal flips by task/configuration and disable vertical flips by default.
- [ ] Exclude natural-image color jitter from defaults.

### CT

- [ ] Convert calibrated input to HU and verify units.
- [ ] Canonicalize orientation and resample to configurable spacing.
- [ ] Implement configurable clipping, single-window, and multi-window channels.
- [ ] Implement foreground/body crop with invertible coordinates.
- [ ] Use distinct interpolation policies for images, labels, and masks.
- [ ] Keep window presets model/config-specific rather than global.

### MRI

- [ ] Implement explicit sequence identification and aliases.
- [ ] Reject silent sequence substitution.
- [ ] Canonicalize orientation and spacing.
- [ ] Implement foreground z-score and robust percentile normalization.
- [ ] Support multi-sequence stacking and missing-sequence masks.
- [ ] Keep bias-field correction an explicit offline/configured operation.

### 3D patch sampling

- [ ] Implement random, foreground, class-balanced, box, lesion-centred, and grid samplers.
- [ ] Return origin, original shape, physical bounding box, target-positive flag, and sampling probability.
- [ ] Make positive-patch proportion measurable and configurable.
- [ ] Ensure samplers remain deterministic under fixed seeds.
- [ ] Handle samples smaller than the desired patch with explicit padding metadata.

### Pathology

- [ ] Generate thumbnails and tissue masks.
- [ ] Add background, focus/blur, and basic artifact filters.
- [ ] Normalize MPP/magnification and make tile extraction deterministic.
- [ ] Persist tile IDs, level-0 coordinates, size, level, MPP, tissue fraction, and quality.
- [ ] Add optional stain normalization and augmentation as separately hashable steps.
- [ ] Verify all tile coordinates map to the source slide.

### Text and VLM preparation

- [ ] Normalize Unicode and perform configurable PHI checks.
- [ ] Parse report sections and handle missing/empty sections.
- [ ] Treat report text as data, never instructions to the framework agent.
- [ ] Implement prompt-template assignment and conversation formatting.
- [ ] Log token counts and truncation without logging sensitive text.
- [ ] Mask system, user, visual placeholder, and optional boilerplate tokens from LM loss.
- [ ] Record supervised-token count per example and reject zero-supervision batches.

### Collators

- [ ] Implement classification, 2D/3D segmentation, contrastive, multi-image VL, volume VL, and WSI VL collators.
- [ ] Pad variable image/token/tile counts with explicit masks.
- [ ] Preserve sample ordering and metadata.
- [ ] Validate fixed visual-token and text-token limits.
- [ ] Fail mixed incompatible modalities unless a multitask collator explicitly supports them.
- [ ] Implement configurable static buckets for image/volume shape, image/slice/tile count, visual tokens, and text tokens.
- [ ] Return bucket IDs and complete padding masks.
- [ ] Pad or drop final distributed training batches according to backend policy; never silently drop evaluation samples.
- [ ] Bound the number of bucket shapes and emit a warning/error when a sample would trigger an unplanned TPU compilation.
- [ ] Keep validation bucket shapes stable and separately configured from training.

## Tests and verification

- [ ] Verify exact model shape/range/normalization conformance using dummy preprocess specs.
- [ ] Invert CT/MRI transforms and compare masks in original physical coordinates.
- [ ] Empirically verify positive-patch sampling over repeated seeded trials.
- [ ] Verify image and mask interpolation differ correctly.
- [ ] Verify deterministic WSI tile indexes across runs.
- [ ] Verify VLM labels supervise only assistant tokens.
- [ ] Verify worker seeding provides reproducible but non-identical streams.
- [ ] Test malformed/missing MRI sequences and zero-tissue slides.
- [ ] Test every collator under TPU-static mode with repeated batches and identical shapes.
- [ ] Test padding invariance for losses and metrics.
- [ ] Measure host preprocessing and device input wait independently on CUDA and TPU.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [Accelerate TPU static-shape guidance](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)
- [Cloud TPU shape and padding guidance](https://cloud.google.com/tpu/docs/intro-to-tpu)
- [MONAI transforms](https://docs.monai.io/en/stable/transforms.html)

## Smoke command

```bash
pytest tests/phase_04/test_end_to_end_transforms.py -q
```

## Acceptance command

```bash
pytest tests/phase_04 -q && python -m medfm.tools.validate_phase --phase 04
```

## Exit criteria

- [ ] Every adapter can receive exactly its declared tensor format.
- [ ] Original-space reconstruction passes tolerance checks.
- [ ] Patch and tile sampling are measurable and reproducible.
- [ ] VLM masking has complete unit coverage.
- [ ] No stochastic transform contaminates deterministic cache keys.

## Handoff

- [ ] Publish transform configuration schemas and hashes.
- [ ] Publish collator output examples for model and trainer phases.
- [ ] Publish static bucket sets and the policy for out-of-bucket samples.
- [ ] Record default-safe augmentations by modality.
- [ ] List unsupported inversion cases and required follow-up.
