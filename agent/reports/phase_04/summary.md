# Phase 04 Summary: Preprocessing, Augmentation, and Collators

## Outcome

Deterministic medical canonicalization, model-aware normalization, stochastic
augmentation, invertible spatial transforms, 3D patch sampling, WSI tile
planning, text/VLM preparation with loss masking, and bucketed batch collation
are implemented and gated. 198 phase-local tests pass on CPU (2 protected
hardware skips); the full repository suite passes (538 + phase-04 tests, 8
protected-hardware skips). `ruff check`, `ruff format --check`, and strict
`mypy` are clean.

## What was built

- `medfm/data/transforms/` — two-stage pipeline machinery (ADR 0010):
  `TransformPipeline` enforces an explicit deterministic (cacheable) stage and
  a stochastic (augmentation) stage; `deterministic_config_hash()` excludes
  stochastic transforms by construction so augmentation can never contaminate
  cache keys. Every transform records a JSON-able `TransformRecord`; spatial
  transforms register mode-aware inverters (image = trilinear/bilinear, label
  = nearest) and `invert_history` reconstructs original physical coordinates.
  `PreprocessSpec` declares each adapter's exact tensor contract (shape,
  channels, dtype, value range, normalization statistics) and pipelines
  validate final tensors against it. Randomness flows only through
  `TransformContext` seeds derived per (base_seed, epoch, worker, sample).
  - `radiology2d.py` — MONOCHROME1 correction, aspect-preserving letterbox
    (invertible), body-region crop (invertible), 1/3-channel output, and
    conservative rotation/translation/scale/intensity/noise augmentation;
    horizontal flips gated by config, vertical flips off by default, no color
    jitter. View/multi-view/longitudinal metadata passes through untouched.
  - `spatial3d.py` / `ct.py` — HU conversion with unit verification,
    orientation canonicalization, resample-to-spacing, configurable HU
    clipping, single/multi-window channels (presets are constructor config,
    no global registry), foreground/body crop with invertible coordinates,
    distinct image/label interpolation orders.
  - `mri.py` — `SequenceResolver` with explicit aliases (unknown names
    rejected, never silently substituted), foreground z-score and robust
    percentile normalization, multi-sequence stacking with missing-sequence
    masks (allowed-missing is explicit config), N4-style bias-field
    correction as an explicit offline-only function.
  - `pathology.py` — thumbnails, tissue masks, tissue-fraction/blur/artifact
    quality filters, MPP normalization, deterministic tile planning with
    persisted `TileRecord`s (tile id, level-0 coords, size, level, MPP,
    tissue fraction, quality), coordinate-in-bounds validation, zero-tissue
    slide handling, optional separately-hashable Reinhard stain normalization
    and stochastic stain augmentation.
  - `timing.py` — host preprocessing and device input wait measured
    independently (`PreprocessTimer`), backend-neutral with lazy CUDA/XLA
    synchronization.
- `medfm/data/samplers/` — the Phase 03 distributed sampler moved unchanged
  into `distributed.py` (public imports preserved); `patches.py` adds random,
  foreground (configurable/measurable positive ratio), class-balanced, box,
  lesion-centred, and deterministic grid patch samplers returning `PatchInfo`
  with origin, original shape, physical bounding box, target-positive flag,
  sampling probability, and explicit padding metadata for small volumes.
- `medfm/data/textprep/` — Unicode normalization, configurable pattern-based
  PHI screening that logs counts only (never text), radiology section parsing
  with missing/empty-section handling (report text is data, never
  instructions), prompt-template registry with conversation formatting, and
  supervised-example construction that supervises only assistant tokens
  (system/user/visual-placeholder/boilerplate masked at -100), records
  supervised-token counts, logs token counts and truncation without text, and
  rejects zero-supervision examples and batches.
- `medfm/data/collators/` — `BucketPlan` (bounded static bucket sets, smallest
  covering assignment, error/pad_to_max out-of-bucket policy with unplanned
  TPU compilation messaging, first-exercise and low-utilization warnings,
  hashed config; validation plans configured separately) plus classification,
  2D/3D segmentation, contrastive, multi-image VL, volume/multi-series VL,
  and WSI VL collators producing validated `MedicalBatch`es with complete
  padding masks, preserved ordering/metadata, fixed visual/text token limit
  validation, train-only pad/drop final-batch policy (evaluation samples
  never dropped), and a `MultitaskCollator` as the only declared mixed-modality
  path.

## Decisions recorded

- ADR 0010 (`docs/architecture/adr_0010_transform_pipeline_stages_inversion_host_only.md`):
  two-stage pipelines with inversion history, per-sample seeding, host-only
  execution, spec validation, opt-in bias-field/stain steps, declared
  multitask mixing.
- `medfm/data/samplers.py` converted to a package (behavior-preserving;
  Phase 03 gate entry updated to the package path).
- mypy overrides extended with `scipy.*`/`skimage.*` (same policy as other
  untyped third-party packages).

## Verification

- `pytest tests/phase_04 -q` → 198 passed, 2 skipped (protected GPU/TPU timing tests).
- `pytest tests/ -q` → all pass (8 protected-hardware skips).
- `ruff check medfm tests`, `ruff format --check medfm tests` → clean.
- `mypy` (strict) → clean.
- Smoke: `pytest tests/phase_04/test_end_to_end_transforms.py -q` → 5 passed, 2 skipped.
- `python -m medfm.tools.validate_phase --phase 04` → passed.

## Notes

- Implementation was split across parallel sub-agents; the CT/MRI test files,
  VL/multitask collators, collator tests, end-to-end smoke test, timing
  module, validator registration, and ADR were completed inline by the
  coordinator after quota interruptions. All code was reviewed and verified
  by the commands above.
