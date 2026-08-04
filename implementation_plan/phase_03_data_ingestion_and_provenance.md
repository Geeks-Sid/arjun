# Phase 03: Dataset Manifests, Ingestion, and Provenance

## Objective

Build a modality-aware dataset layer with canonical manifests, physical-coordinate-safe readers, patient-level split protection, provenance, fingerprinting, and cache keys.

## Dependencies

- [ ] Phase 02 contracts are accepted.
- [ ] Data governance and restricted-text policies are available.
- [ ] Synthetic radiology and pathology fixtures can be generated legally.

## Scope boundaries

Allowed areas: `medfm/data/manifests/`, `readers/`, `caching/`, `fingerprint.py`, data CLI, and Phase 03 tests.

Do not add stochastic augmentation, model-specific normalization, or real patient data.

## Implementation checklist

### Canonical manifests

- [ ] Define a versioned Parquet schema with all columns from `idea.md`.
- [ ] Support JSONL only as a debugging interchange.
- [ ] Validate URI schemes and prevent unsafe path traversal.
- [ ] Allow restricted reports to be referenced rather than embedded.
- [ ] Store dataset name, version, license, and provenance for every row.
- [ ] Add manifest migration and schema-inspection commands.
- [ ] Store optional shape-bucket hints derived from fingerprints without making them authoritative raw metadata.

### Split generation and leakage checks

- [ ] Implement patient-disjoint splitting as the default.
- [ ] Support site-disjoint and temporal holdouts.
- [ ] Keep studies, derived images, adjacent slices, WSI tiles, slides, and cases together.
- [ ] Detect duplicate content hashes across splits.
- [ ] Produce an auditable split report with seed and grouping keys.
- [ ] Refuse to train on a manifest with known split leakage unless an explicit research override is recorded.

### Radiology readers

- [ ] Implement NIfTI, MHA, NumPy volume, and PNG/JPEG readers.
- [ ] Implement DICOM series discovery and physical-position sorting.
- [ ] Validate orientation, spacing, frame of reference, and mixed-series conditions.
- [ ] Apply CT rescale slope/intercept and MONOCHROME1 correction.
- [ ] Detect multiframe objects, scouts/localizers, and unsupported pixel data.
- [ ] Hash UIDs and preserve enough metadata for original-space output mapping.
- [ ] Return actionable errors for ambiguous or corrupt series.

### Pathology readers

- [ ] Define a common slide-reader contract for dimensions, levels, MPP, thumbnails, regions, and tiles.
- [ ] Implement OpenSlide and TiffSlide backends.
- [ ] Add cuCIM behind an optional dependency/capability check.
- [ ] Implement pre-extracted tile and embedding-store readers.
- [ ] Validate coordinate conversion between pyramid levels.
- [ ] Make corrupt regions recoverable at sample/tile scope.

### Dataset fingerprinting

- [ ] Implement `medfm data fingerprint --manifest <path>`.
- [ ] Report patient/study/sample counts and modality distributions.
- [ ] Report shape, spacing, intensity, label, missing-value, site, vendor, and duplicate statistics.
- [ ] Report text length, WSI MPP/magnification, and segmentation-volume statistics where present.
- [ ] Include split leakage results and a deterministic fingerprint hash.
- [ ] Serialize a machine-readable report for preprocessing configuration.
- [ ] Recommend bounded shape buckets for 2D resolution, 3D patches, slice counts, tile counts, and text lengths.

### Caching

- [ ] Define preprocessing, visual-embedding, and tokenization cache interfaces.
- [ ] Include source hash, reader version, preprocessing hash, model ID/revision, layer, and dtype in keys.
- [ ] Use atomic writes and detect partial/corrupt cache entries.
- [ ] Define cache metadata and eviction policy.
- [ ] Test invalidation for normalization, resolution, model, adapter, layer, and dtype changes.
- [ ] Keep cache tensors accelerator-neutral and loadable by CPU, CUDA, and XLA backends.
- [ ] Support rank-safe reads and coordinator-only writes or atomic per-rank partitions.

### Distributed input ownership

- [ ] Define patient/study/slide-aware distributed sampling with no leakage across ranks.
- [ ] Ensure each epoch has deterministic rank and worker seeds.
- [ ] Handle padded final batches without duplicating samples in evaluation metrics.
- [ ] Resolve corrupt samples before distributed collectives to prevent rank hangs.

## Tests and verification

- [ ] Generate a synthetic DICOM series and prove physical sorting.
- [ ] Reject a deliberately inconsistent DICOM orientation/spacing series.
- [ ] Preserve NIfTI affine, spacing, dtype, and orientation metadata.
- [ ] Read a synthetic pyramid slide and verify coordinate conversion.
- [ ] Detect deliberately corrupted patient and slide split leakage.
- [ ] Verify manifest fingerprints are deterministic.
- [ ] Verify cache invalidation and corrupt-entry recovery.
- [ ] Verify exceptions and logs do not expose raw identifiers or report text.
- [ ] Verify distributed samplers cover the intended dataset exactly after removing explicit padding duplicates.

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

- [ ] Synthetic DICOM, NIfTI, ordinary image, and WSI data load correctly.
- [ ] Physical and slide coordinate metadata is preserved.
- [ ] Leakage checks fail a corrupted split.
- [ ] Fingerprints are deterministic and useful to Phase 04.
- [ ] Cache changes invalidate exactly when required.

## Handoff

- [ ] Provide reader contracts and fixture manifests to Phase 04.
- [ ] Provide dataset fingerprint schema to recipe planning.
- [ ] Provide recommended static-shape buckets and distributed sample ownership rules.
- [ ] List unsupported DICOM/WSI variants explicitly.
- [ ] Record cache-key version and storage expectations.
