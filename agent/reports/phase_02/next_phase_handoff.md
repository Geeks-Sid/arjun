# Phase 02 → Phase 03 Handoff

## Contract version

- `medfm.core.versioning.SCHEMA_VERSION = 1`. Every serializable contract
  object carries `schema_version`; old payloads upgrade only through
  registered `register_schema_migration` steps, retired enum values only
  through `register_enum_migration`. Bump policy: `docs/core_contracts.md`.

## What Phase 03 (readers/manifests) should build against

- Produce `MedicalSample` objects (never raw dicts) at reader output.
  `sample_id` + `patient_id_hash` (+ optional study/series hashes) are
  required; hash factories in `medfm.core.sample` reject raw MRNs/DICOM UIDs —
  hash UIDs at the reader boundary (`_hash_id` accepts lowercase hex digests,
  32–128 chars).
- `ProvenanceMetadata(dataset_name, dataset_version, ...)` is mandatory on
  every sample; manifest columns map 1:1 (split → `SplitName`, license,
  site_id, source_uri/sha256, acquisition_date_bucket).
- 3D readers must populate `SpatialMetadata` (affine or spacing required;
  never discard). WSI readers must populate `PathologyMetadata` (MPP,
  pyramid `level_dimensions`, `tile_coordinates [T, 2|4]` int64).
- `MedicalSample.validate_for_task(task)` enforces task-specific targets.

## Static bucket and device contracts (Phase 04 collators)

- `BucketId(kind, shape)`: `IMAGE_2D (H,W)`, `VOLUME_3D (D,H,W)`,
  `MULTI_IMAGE (I,)`, `WSI_TILES (T,)`, `VISUAL_TOKENS (N,)`,
  `TEXT_TOKENS (L,)`. A bucketed `MedicalBatch` must carry the mask for every
  padded dimension or `BucketError` is raised at construction.
- `MedicalBatch.to(device)` / `pin_memory()` are backend-neutral and preserve
  non-tensor metadata; collators must not call `.cuda()`.
- `MULTI_SERIES_3D` batches are rank-6 `[B, S, C, D, H, W]` (not lists) so
  static-shape paths stay uniform.

## Protocol conformance fixtures (model authors, Phases 06–11)

`tests/phase_02/contract_fixtures.py` contains the reference dummies:
`DummyVisualEncoder`, `PoolingOnlyEncoder` (must refuse spatial requests),
`DummyLanguageModelAdapter`, `TextOnlyLanguageModelAdapter` (must refuse
visual tokens), `DummyTaskModule`. Real adapters must pass the same
`isinstance` checks and output-semantics assertions.

## Serialization rules for artifacts

- Configs/manifests: `canonical_json` / `canonical_yaml` + `config_hash`.
- Tensor artifacts: `TensorMeta` (shape + canonical dtype name) in JSON,
  payloads via a tensor store; `materialize_cpu` before export. Devices are
  never serialized.

## Deferred schema questions (ADR candidates)

1. **Instance-segmentation sample representation**: `SegmentationTarget`
   currently references one class-indexed mask; instance masks (RLE vs
   per-instance channels) need a decision before INSTANCE_SEGMENTATION
   recipes land.
2. **Prompt representation for PROMPTABLE_SEGMENTATION**: points/boxes/prior-
   slice prompts are not yet typed in the sample schema (boxes exist; point
   prompts do not).
3. **Structured-finding schema**: STRUCTURED_FINDING_GENERATION samples reuse
   `report` text today; a typed findings structure may be warranted.
4. **Multi-sequence MRI labeling**: `MULTI_SERIES_3D` samples carry multiple
   `ImageReference`s but no per-series sequence label (T1/T2/FLAIR...) — the
   role field is free text; a `SequenceType` enum may be needed in Phase 03.

## Gate status

- `pytest tests/phase_02 -q`: 102 passed, 2 protected skips (gpu/tpu fixtures).
- `pytest tests/phase_02/test_contract_smoke.py -q`: 4 passed.
- `python -m medfm.tools.smoke --phase 02`: 2/2 checks.
- `python -m medfm.tools.validate_phase --phase 02`: passed.
- `MEDFM_RUN_GPU_TESTS=1 pytest -m gpu`: 3 passed (incl. CUDA transfer fixture).
- `make lint` / `make typecheck`: clean.
