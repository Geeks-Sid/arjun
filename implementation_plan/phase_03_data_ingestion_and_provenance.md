# Phase 03: Dataset Manifests, Ingestion, and Provenance

## Objective

Build a modality-aware dataset layer with canonical manifests, physical-coordinate-safe readers, patient-level split protection, provenance, fingerprinting, and cache keys.

## Dependencies

- [x] Phase 02 contracts are accepted.
- [x] Data governance and restricted-text policies are available.
- [x] Synthetic radiology and pathology fixtures can be generated legally.

## Scope boundaries

Allowed areas: `medfm/data/manifests/`, `readers/`, `caching/`, `fingerprint.py`, data CLI, and Phase 03 tests.

Do not add stochastic augmentation, model-specific normalization, or real patient data.

## Implementation checklist

### Canonical manifests

- [x] Define a versioned Parquet schema with all columns from `idea.md`.
- [x] Support JSONL only as a debugging interchange.
- [x] Validate URI schemes and prevent unsafe path traversal.
- [x] Allow restricted reports to be referenced rather than embedded.
- [x] Store dataset name, version, license, and provenance for every row.
- [x] Add manifest migration and schema-inspection commands.
- [x] Store optional shape-bucket hints derived from fingerprints without making them authoritative raw metadata.

### Split generation and leakage checks

- [x] Implement patient-disjoint splitting as the default.
- [x] Support site-disjoint and temporal holdouts.
- [x] Keep studies, derived images, adjacent slices, WSI tiles, slides, and cases together.
- [x] Detect duplicate content hashes across splits.
- [x] Produce an auditable split report with seed and grouping keys.
- [x] Refuse to train on a manifest with known split leakage unless an explicit research override is recorded.

### Radiology readers

- [x] Implement NIfTI, MHA, NumPy volume, and PNG/JPEG readers.
- [x] Implement DICOM series discovery and physical-position sorting.
- [x] Validate orientation, spacing, frame of reference, and mixed-series conditions.
- [x] Apply CT rescale slope/intercept and MONOCHROME1 correction.
- [x] Detect multiframe objects, scouts/localizers, and unsupported pixel data.
- [x] Hash UIDs and preserve enough metadata for original-space output mapping.
- [x] Return actionable errors for ambiguous or corrupt series.

### Pathology readers

- [x] Define a common slide-reader contract for dimensions, levels, MPP, thumbnails, regions, and tiles.
- [x] Implement OpenSlide and TiffSlide backends.
- [x] Add cuCIM behind an optional dependency/capability check.
- [x] Implement pre-extracted tile and embedding-store readers.
- [x] Validate coordinate conversion between pyramid levels.
- [x] Make corrupt regions recoverable at sample/tile scope.

### Dataset fingerprinting

- [x] Implement `medfm data fingerprint --manifest <path>`.
- [x] Report patient/study/sample counts and modality distributions.
- [x] Report shape, spacing, intensity, label, missing-value, site, vendor, and duplicate statistics.
- [x] Report text length, WSI MPP/magnification, and segmentation-volume statistics where present.
- [x] Include split leakage results and a deterministic fingerprint hash.
- [x] Serialize a machine-readable report for preprocessing configuration.
- [x] Recommend bounded shape buckets for 2D resolution, 3D patches, slice counts, tile counts, and text lengths.

### Caching

- [x] Define preprocessing, visual-embedding, and tokenization cache interfaces.
- [x] Include source hash, reader version, preprocessing hash, model ID/revision, layer, and dtype in keys.
- [x] Use atomic writes and detect partial/corrupt cache entries.
- [x] Define cache metadata and eviction policy.
- [x] Test invalidation for normalization, resolution, model, adapter, layer, and dtype changes.
- [x] Keep cache tensors accelerator-neutral and loadable by CPU, CUDA, and XLA backends.
- [x] Support rank-safe reads and coordinator-only writes or atomic per-rank partitions.

### Distributed input ownership

- [x] Define patient/study/slide-aware distributed sampling with no leakage across ranks.
- [x] Ensure each epoch has deterministic rank and worker seeds.
- [x] Handle padded final batches without duplicating samples in evaluation metrics.
- [x] Resolve corrupt samples before distributed collectives to prevent rank hangs.

## Tests and verification

- [x] Generate a synthetic DICOM series and prove physical sorting.
- [x] Reject a deliberately inconsistent DICOM orientation/spacing series.
- [x] Preserve NIfTI affine, spacing, dtype, and orientation metadata.
- [x] Read a synthetic pyramid slide and verify coordinate conversion.
- [x] Detect deliberately corrupted patient and slide split leakage.
- [x] Verify manifest fingerprints are deterministic.
- [x] Verify cache invalidation and corrupt-entry recovery.
- [x] Verify exceptions and logs do not expose raw identifiers or report text.
- [x] Verify distributed samplers cover the intended dataset exactly after removing explicit padding duplicates.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [PyTorch data loading and distributed sampler](https://docs.pytorch.org/docs/stable/data.html)
- [pydicom](https://pydicom.github.io/pydicom/stable/)
- [OpenSlide](https://openslide.org/)

## Smoke command

```bash
python -m medfm.cli.data fingerprint --manifest tests/fixtures/manifests/mixed_synthetic.parquet
```

## Acceptance command

```bash
pytest tests/phase_03 -q && python -m medfm.tools.validate_phase --phase 03
```

## Exit criteria

- [x] Synthetic DICOM, NIfTI, ordinary image, and WSI data load correctly.
- [x] Physical and slide coordinate metadata is preserved.
- [x] Leakage checks fail a corrupted split.
- [x] Fingerprints are deterministic and useful to Phase 04.
- [x] Cache changes invalidate exactly when required.

## Handoff

- [x] Provide reader contracts and fixture manifests to Phase 04.
- [x] Provide dataset fingerprint schema to recipe planning.
- [x] Provide recommended static-shape buckets and distributed sample ownership rules.
- [x] List unsupported DICOM/WSI variants explicitly.
- [x] Record cache-key version and storage expectations.
