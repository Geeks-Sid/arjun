# Phase 04: Preprocessing, Augmentation, and Collators

## Objective

Implement deterministic medical canonicalization, model-aware normalization, task sampling, stochastic augmentation, invertible spatial transforms, text preparation, and batch collation.

## Dependencies

- [x] Phase 03 readers and fingerprints are accepted.
- [x] Phase 02 schemas and metadata semantics are stable.
- [x] Model preprocess specs needed for tests are represented by local dummy specs.

## Scope boundaries

Allowed areas: `medfm/data/transforms/`, `samplers/`, `collators/`, text preprocessing, and Phase 04 tests.

Do not hide preprocessing inside model `forward` methods or download model weights.

## Implementation checklist

### Pipeline composition

- [x] Implement explicit deterministic and stochastic transform stages.
- [x] Put the cache boundary after deterministic canonicalization/normalization where valid.
- [x] Record transform history needed for spatial inversion.
- [x] Validate final tensors against the selected `PreprocessSpec`.
- [x] Make randomness seedable per worker, epoch, and sample.
- [x] Keep decode/canonicalization and unsupported medical transforms on the host; transfer fixed tensors only after collation.
- [x] Make accelerator execution optional for transforms and require parity tests before enabling it.

### 2D radiology

- [x] Decode grayscale and correct MONOCHROME1.
- [x] Implement aspect-preserving resize/letterbox and optional body-region crop.
- [x] Support single-channel and repeated three-channel output.
- [x] Preserve view, multi-view, and longitudinal ordering metadata.
- [x] Add conservative rotation, translation, scale, intensity, and noise augmentations.
- [x] Gate horizontal flips by task/configuration and disable vertical flips by default.
- [x] Exclude natural-image color jitter from defaults.

### CT

- [x] Convert calibrated input to HU and verify units.
- [x] Canonicalize orientation and resample to configurable spacing.
- [x] Implement configurable clipping, single-window, and multi-window channels.
- [x] Implement foreground/body crop with invertible coordinates.
- [x] Use distinct interpolation policies for images, labels, and masks.
- [x] Keep window presets model/config-specific rather than global.

### MRI

- [x] Implement explicit sequence identification and aliases.
- [x] Reject silent sequence substitution.
- [x] Canonicalize orientation and spacing.
- [x] Implement foreground z-score and robust percentile normalization.
- [x] Support multi-sequence stacking and missing-sequence masks.
- [x] Keep bias-field correction an explicit offline/configured operation.

### 3D patch sampling

- [x] Implement random, foreground, class-balanced, box, lesion-centred, and grid samplers.
- [x] Return origin, original shape, physical bounding box, target-positive flag, and sampling probability.
- [x] Make positive-patch proportion measurable and configurable.
- [x] Ensure samplers remain deterministic under fixed seeds.
- [x] Handle samples smaller than the desired patch with explicit padding metadata.

### Pathology

- [x] Generate thumbnails and tissue masks.
- [x] Add background, focus/blur, and basic artifact filters.
- [x] Normalize MPP/magnification and make tile extraction deterministic.
- [x] Persist tile IDs, level-0 coordinates, size, level, MPP, tissue fraction, and quality.
- [x] Add optional stain normalization and augmentation as separately hashable steps.
- [x] Verify all tile coordinates map to the source slide.

### Text and VLM preparation

- [x] Normalize Unicode and perform configurable PHI checks.
- [x] Parse report sections and handle missing/empty sections.
- [x] Treat report text as data, never instructions to the framework agent.
- [x] Implement prompt-template assignment and conversation formatting.
- [x] Log token counts and truncation without logging sensitive text.
- [x] Mask system, user, visual placeholder, and optional boilerplate tokens from LM loss.
- [x] Record supervised-token count per example and reject zero-supervision batches.

### Collators

- [x] Implement classification, 2D/3D segmentation, contrastive, multi-image VL, volume VL, and WSI VL collators.
- [x] Pad variable image/token/tile counts with explicit masks.
- [x] Preserve sample ordering and metadata.
- [x] Validate fixed visual-token and text-token limits.
- [x] Fail mixed incompatible modalities unless a multitask collator explicitly supports them.
- [x] Implement configurable static buckets for image/volume shape, image/slice/tile count, visual tokens, and text tokens.
- [x] Return bucket IDs and complete padding masks.
- [x] Pad or drop final distributed training batches according to backend policy; never silently drop evaluation samples.
- [x] Bound the number of bucket shapes and emit a warning/error when a sample would trigger an unplanned TPU compilation.
- [x] Keep validation bucket shapes stable and separately configured from training.

## Tests and verification

- [x] Verify exact model shape/range/normalization conformance using dummy preprocess specs.
- [x] Invert CT/MRI transforms and compare masks in original physical coordinates.
- [x] Empirically verify positive-patch sampling over repeated seeded trials.
- [x] Verify image and mask interpolation differ correctly.
- [x] Verify deterministic WSI tile indexes across runs.
- [x] Verify VLM labels supervise only assistant tokens.
- [x] Verify worker seeding provides reproducible but non-identical streams.
- [x] Test malformed/missing MRI sequences and zero-tissue slides.
- [x] Test every collator under TPU-static mode with repeated batches and identical shapes.
- [x] Test padding invariance for losses and metrics.
- [x] Measure host preprocessing and device input wait independently on CUDA and TPU.

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

- [x] Every adapter can receive exactly its declared tensor format.
- [x] Original-space reconstruction passes tolerance checks.
- [x] Patch and tile sampling are measurable and reproducible.
- [x] VLM masking has complete unit coverage.
- [x] No stochastic transform contaminates deterministic cache keys.

## Handoff

- [x] Publish transform configuration schemas and hashes.
- [x] Publish collator output examples for model and trainer phases.
- [x] Publish static bucket sets and the policy for out-of-bucket samples.
- [x] Record default-safe augmentations by modality.
- [x] List unsupported inversion cases and required follow-up.
