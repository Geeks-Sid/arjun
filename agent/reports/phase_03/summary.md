# Phase 03 Summary: Dataset Manifests, Ingestion, and Provenance

## Outcome

The modality-aware dataset layer is implemented and gated: canonical Parquet
manifests with fail-closed validation, physical-coordinate-aware radiology and
pathology readers, patient/site/temporal split generation with leakage
protection, deterministic dataset fingerprinting, accelerator-neutral caching
with exact invalidation, group-aware distributed sampling, and a `medfm data`
CLI. 151 phase-local tests pass on CPU; the smoke fingerprint command runs
against a committed synthetic fixture.

## What was built

- `medfm/data/manifests/` — versioned Parquet manifest schema (all columns from
  `idea.md`), fail-closed validation that collects every problem, URI scheme +
  path-traversal guards, identifier-hygiene checks that reject raw MRNs/DICOM
  UIDs, JSONL as a debugging-only interchange, content hashing, schema
  introspection, and a migration registry. Restricted report text is referenced
  (`report_uri`), never embedded.
- `medfm/data/readers/` — a shared `Reader`/`PayloadRead` contract. Radiology:
  NIfTI, MHA, NumPy volume, and PNG/JPEG readers that preserve affine, spacing,
  on-disk dtype, and orientation; a DICOM series reader that sorts by physical
  position (never filename), validates consistent orientation/spacing, applies
  CT rescale slope/intercept and MONOCHROME1 inversion, rejects multiframe /
  scout / mixed / unsupported-pixel series, and hashes UIDs at the boundary.
  Pathology: an OpenSlide/TiffSlide/cuCIM slide contract (dimensions, levels,
  MPP, thumbnails, regions, tiles) with validated pyramid-level coordinate
  conversion, plus pre-extracted tile and embedding-store readers; corrupt
  regions are recoverable at tile scope.
- `medfm/data/splits.py` — patient-level-first split generation (then site, then
  time), deterministic in `(seed, policy, grouping key)` and row-order-free;
  `group_id_hash` keeps WSI tiles/slides/cases together. Leakage checks across
  patient/study/series/group/content-hash keys, an auditable `SplitReport`, and
  a training gate that refuses leaking manifests unless a `ResearchOverride` is
  recorded (ADR 0004, governance §6).
- `medfm/data/fingerprint.py` — deterministic dataset fingerprint: counts,
  modality/shape/spacing/intensity/label/missing-value/site/vendor/duplicate
  statistics, report-length and WSI MPP/magnification statistics, segmentation
  class volumes, embedded split-leakage results, bounded shape-bucket
  recommendations, and a stable `fingerprint_hash` for Phase 04 and run metadata.
- `medfm/data/caching/` — versioned cache keys covering source hash, reader
  version, preprocessing hash, model id/revision, output layer, dtype, and
  adapter `extra`; an atomic on-disk store (tmp-dir + rename) with corruption
  detection/quarantine, LRU eviction, and rank-safe coordinator-only or
  per-rank-partition writes. Tensors are stored CPU/canonical-dtype so CPU,
  CUDA, and XLA consumers load them identically. Typed `PreprocessingCache`,
  `VisualEmbeddingCache`, and `TokenizationCache` wrappers enforce per-kind key
  fields.
- `medfm/data/samplers.py` — group-aware distributed sampler with disjoint,
  exactly-covering per-rank shards, group integrity, deterministic per-epoch
  rank/worker seeds, padded final batches via an explicit sentinel, and
  pre-collective corrupt-sample resolution so one rank's bad sample cannot hang
  the others.
- `medfm/cli/data.py` + `medfm/tools/data_tools.py` — `medfm data
  fingerprint|inspect|migrate|split`, wired into the top-level CLI and runnable
  as `python -m medfm.cli.data`.
- `tests/phase_03/synthetic.py` + `tests/fixtures/manifests/mixed_synthetic.parquet`
  — legally-generated synthetic fixtures (DICOM series, NIfTI, MHA, pyramid
  TIFF, tile/embedding stores) reused across the tests and the committed
  fingerprint fixture.
- Gate tooling: `medfm/tools/validate_phase.py` phase-03 file list, phase-03
  smoke checks in `medfm/tools/smoke.py`.

## Verification

- `pytest tests/phase_03 -q` → 151 passed.
- `pytest tests/ -q` → 345 passed, 8 protected-hardware skips.
- `ruff check` / `ruff format --check` / `mypy` (strict) → clean.
- Smoke: `python -m medfm.cli.data fingerprint --manifest
  tests/fixtures/manifests/mixed_synthetic.parquet` and
  `python -m medfm.tools.smoke --phase 03` → pass.
