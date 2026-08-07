# Phase 04 → Phase 05 Handoff

## Adapter input contract (build model adapters against this)

- `medfm.data.transforms.specs.PreprocessSpec` is the declared tensor contract
  of an adapter: `spatial_shape` ((H, W) or (D, H, W)), `channels`, canonical
  `dtype`, `value_range`, and `NormalizationSpec` (per-channel mean/std).
  Pipelines validate final tensors with `spec.validate(tensor)`; adapters
  must receive exactly `spec.expected_tensor_shape()` per sample. Register one
  spec per adapter; window presets/crop policies stay in transform configs
  (never global). `spec.spec_hash()` is the cache-invalidation identity.
- Pipeline output per sample is `TransformData` (image `[C, *spatial]` CPU,
  `targets` incl. masks and MRI `sequence_mask`, `spatial`/`pathology`
  metadata preserved, `history` of `TransformRecord`s). Cache boundary: store
  the output of `pipeline.run_deterministic(...)`; key it with
  `pipeline.deterministic_config_hash()` (+ spec hash) folded into
  `CacheKey.preprocessing_hash`. Stochastic transforms never affect the key
  (ADR 0010).
- Inversion: `invert_history(history, tensor, mode="image"|"label")` maps
  outputs back to original physical coordinates; label mode uses nearest
  interpolation. Unsupported inversion cases (stochastic spatial
  augmentation, intensity/noise/stain steps) are listed in
  `unresolved_issues.md`; `strict=True` raises on them.

## Collator outputs (trainer/model-phase contract)

- All collators return validated `medfm.core.batch.MedicalBatch` (never
  dicts): `ClassificationCollator`, `Segmentation2D/3DCollator`,
  `ContrastiveCollator`, `MultiImageVLCollator`, `VolumeVLCollator`
  (incl. MULTI_SERIES_3D), `WSIVLCollator` (tile pixels or precomputed
  `visual_tokens`), and `MultitaskCollator` (the only mixed-modality path;
  returns `MultitaskBatch` with per-modality batches + input-order
  `modality_index`).
- Example dict keys consumed: `sample_id`, `modality` (required),
  `image`, `images`, `volumes`, `tiles`, `tile_coordinates`, `label`,
  `mask`, `input_ids`/`attention_mask`, `lm_labels` (-100 masked),
  `visual_tokens`/`visual_token_mask`, `spatial`.
- Static mode: pass a `BucketPlan` (mode="static"); batches land on declared
  `BucketId`s with complete masks (`image_mask`, `attention_mask`,
  `visual_token_mask`). Out-of-bucket samples raise with an unplanned-TPU-
  compilation message (policy "error") or crop+pad to the max bucket
  ("pad_to_max"). `BucketPlan.from_config/to_config/config_hash` — hash the
  plan into every run. Validation plans are separately configured.
- Final batches: `FinalBatchPolicy.PAD` injects fully-masked replicas
  (`::padN` sample ids, `is_padded_example`); `DROP` returns `None` — both
  training-only. Evaluation never drops or pads.
- Text supervision: `medfm.data.textprep.tokenize.build_supervised_example`
  yields `SupervisedExample` (input_ids, attention_mask, labels with -100
  outside assistant spans, `supervised_token_count`, truncation flags);
  zero-supervision examples/batches are rejected. Feed
  `input_ids`/`attention_mask`/`lm_labels` to collators.

## Patch and tile sampling

- `medfm.data.samplers.patches`: six samplers returning `PatchInfo` (origin,
  original shape, physical bbox from spacing, `target_positive`,
  `sampling_probability`, padding metadata) + `Patch` (image/mask tensors).
  Deterministic under fixed seeds; positive-patch proportion is configurable
  and empirically verified. Grid sampler is RNG-free and covers the volume.
- WSI: `medfm.data.transforms.pathology.plan_tiles` returns deterministic
  `TileRecord`s (tile_id hash, level-0 coords, size, level, MPP,
  tissue_fraction, quality) validated to map inside the source slide;
  zero-tissue slides yield no tiles. Stain steps are separately hashable.

## Default-safe augmentations by modality

- 2D radiology: small rotation/translation/scale, intensity shift/scale,
  Gaussian noise; horizontal flip only when configured per task; vertical
  flip never by default; no color jitter.
- CT: HU clip/window are deterministic config; augmentation via the same
  conservative 2D/3D-safe set on slices/patches.
- MRI: normalization is deterministic (foreground z-score / robust
  percentile); bias-field correction is explicit offline only.
- Pathology: stain augmentation is opt-in (`StainAugment`), never default.

## Timing instrumentation

- `medfm.data.transforms.timing.PreprocessTimer` measures host preprocessing
  and device input wait independently (CUDA/TPU tests are protected-hardware
  marked). Use it in trainer phases to attribute input starvation.

## Gate status

- `pytest tests/phase_04 -q` → 198 passed, 2 protected-hardware skips.
- `pytest tests/ -q` → all pass (8 protected-hardware skips).
- `ruff check` / `ruff format --check` / `mypy` (strict) → clean.
- Smoke: `pytest tests/phase_04/test_end_to_end_transforms.py -q` → 5 passed, 2 skipped.
- `python -m medfm.tools.validate_phase --phase 04` → passed.
- CACHE_KEY_VERSION unchanged (1); preprocessing config hashes feed
  `CacheKey.preprocessing_hash` as documented above.
