# Phase 03 → Phase 04 Handoff

## Reader contract (build preprocessing against this)

- `medfm.data.readers.base.Reader` → `PayloadRead(tensors, spatial, pathology,
  source_metadata)`. `tensors["image"]` is always a CPU tensor with a canonical
  dtype; `source_metadata` carries only hashed identifiers + acquisition facts.
- **Volumetric axis contract:** every volume reader returns `(i, j, k)` such
  that `affine @ [i, j, k, 1]` yields patient mm. NIfTI is native; MHA is
  transposed from SimpleITK `(k, j, i)`; DICOM is transposed from the stacked
  `(slice, row, col)`. Phase 04 resampling must consume `SpatialMetadata.affine`
  + `spacing_mm` and never assume axis identity from shape alone.
- `sample_from_manifest_row(row, read)` builds a `MedicalSample` from a manifest
  row plus a `PayloadRead` (identity from the row, geometry from the reader).
  `tests/phase_03/test_sample_bridge.py` shows the pattern.
- Fixture manifests + synthetic payload builders live in
  `tests/phase_03/synthetic.py` (DICOM/NIfTI/MHA/NumPy/PNG/pyramid-TIFF/tiles/
  embeddings). Committed manifest fixture:
  `tests/fixtures/manifests/mixed_synthetic.parquet`.

## Dataset fingerprint schema (for recipe planning)

`medfm.data.fingerprint.fingerprint_manifest(df)` returns a deterministic dict:
`counts`, `modality_counts`, `split_counts`, `shape_stats`, `spacing_stats`,
`intensity_stats`, `label_prevalence`, `missing_values`, `site_distribution`,
`vendor_distribution`, `duplicate_stats`, `split_leakage`, `report_chars_stats`,
`wsi_microns_per_pixel_stats`, `wsi_magnification_stats`,
`segmentation_volume_stats`, `recommended_shape_buckets`, and `fingerprint_hash`.
The hash is stable across row order and read format; record it in run metadata.

## Static shape buckets (recommended, bounded)

`recommend_shape_buckets(df)` emits bucket kinds aligned with the Phase 02
`BucketKind` vocabulary: `2d_resolution` (H, W), `3d_patch` (D, H, W),
`slice_count` (I), `tile_count` (T), `text_length` (L). Each carries a concrete
`shape` and `samples_considered`. These are *recommendations*; Phase 04 owns the
final bucket table and must re-derive geometry from payloads (manifest
`shape_bucket_*` columns are hints, never authoritative).

## Distributed sample ownership rules

- Use `medfm.data.samplers.GroupAwareDistributedSampler(df, num_ranks, rank,
  seed, split=..., group_column=...)`. All samples of a group (patient/study/
  slide via `group_id_hash`) land on ONE rank; per-rank shards are disjoint and
  cover the split exactly.
- Call `set_epoch(epoch)` per epoch; worker seeds come from
  `worker_seed(seed, epoch, rank, worker_id)` / `worker_init_fn`.
- Padded final batches use `PADDING_INDEX` sentinels; drop them before metrics
  (`SamplerShard.real_indices` or `combine_shards_for_metrics`).
- Resolve corrupt samples rank-locally before any collective with
  `resolve_samples_before_collective(indices, check)` to avoid cross-rank hangs.

## Cache-key version and storage expectations

- `medfm.data.caching.keys.CACHE_KEY_VERSION = 1` is mixed into every key
  string. Bump it on any storage-format or key-semantics change and record it
  here.
- On-disk layout: `<root>/<kind>/<partition>/<key_string>/{payload.safetensors,
  meta.json}`; `<partition>` is empty for the shared (coordinator) layout, else
  `rank-<n>`. Writes are atomic (tmp-dir + rename); corrupt entries are
  quarantined and counted. Tensors are CPU/canonical-dtype (safetensors), so
  CPU/CUDA/XLA consumers load identically.

## Unsupported variants (explicit)

- **DICOM:** multiframe objects, scout/localizer acquisitions, mixed-series
  directories (unless a series is explicitly selected), non-grayscale
  photometric interpretations (RGB/palette), and missing
  `ImagePositionPatient`/`ImageOrientationPatient` are all rejected with
  actionable errors. Compressed transfer syntaxes are not covered by synthetic
  fixtures (see unresolved_issues.md).
- **WSI:** cuCIM decode requires the optional CUDA-only extra; vendor-specific
  formats were not validated against real scanner files.

## Gate status

- `pytest tests/phase_03 -q` → 151 passed.
- `pytest tests/ -q` → 345 passed, 8 protected-hardware skips.
- `ruff check` / `ruff format --check` / `mypy` (strict) → clean.
- Smoke: `python -m medfm.cli.data fingerprint --manifest
  tests/fixtures/manifests/mixed_synthetic.parquet` → exit 0;
  `python -m medfm.tools.smoke --phase 03` → 2/2 checks.
- `python -m medfm.tools.validate_phase --phase 03` → passed.
