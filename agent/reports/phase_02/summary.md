# Phase 02 Summary: Core Type System and Contracts

## Outcome

The contract layer in `medfm/core/` is implemented and gated: canonical
enums, sample/batch schemas, encoder/language/task protocols, deterministic
serialization, and typed errors. 103 phase-local tests (194 total repo tests)
pass on CPU; the CUDA device-transfer fixture passes on the host GPU.

## What was built

- `medfm/core/enums.py` — `Modality` (10 canonical values), `TaskType` (16),
  `LoadingMode`, `CoordinateSystem`, `PrecisionMode`, `SplitName` on a strict
  `StrEnum` base; unknown values raise `UnknownEnumValueError` unless a
  versioned migration registered via `register_enum_migration` applies.
- `medfm/core/errors.py` — `ContractError` hierarchy (schema, identifier,
  shape, bucket, serialization, unsupported modality/task/capability).
- `medfm/core/versioning.py` — `SCHEMA_VERSION = 1` plus append-only payload
  migration hooks (`register_schema_migration` / `migrate_payload`); newer
  payloads are rejected, never downgraded.
- `medfm/core/sample.py` — `MedicalSample` with nested `ImageReference`,
  `LabelTarget`, `SegmentationTarget`, `BoxTarget`, `ConversationTurn`,
  `ProvenanceMetadata`, `SpatialMetadata`, `PathologyMetadata`; typed ID hash
  factories that reject raw MRNs/DICOM UIDs; modality- and task-specific
  requirement validation; lossless canonical `to_dict`/`from_dict`.
- `medfm/core/batch.py` — `MedicalBatch` with authoritative `modality` (rank
  validated against it, never inferred), per-modality shape/mask/segmentation
  validation, `BucketId`/`BucketKind` static buckets that require masks for
  every padded dimension, and backend-neutral `to(device)` / `pin_memory()`
  that preserve all non-tensor metadata.
- `medfm/core/encoder.py` — `VisualEncoder` protocol, `EncoderCapabilities`,
  `PreprocessSpec`, `InputSpec`, `OutputSpec`, `EncoderOutput` with documented
  pooled/spatial-token/feature-map/token-mask/coordinate semantics; missing
  requested output raises `UnsupportedCapabilityError` (no silent pooling).
- `medfm/core/language.py` — `LanguageModelAdapter` protocol with mandatory
  capability declaration (`accepts_inputs_embeds` and/or
  `native_visual_connector`); visual tokens against a text-only adapter fail
  loudly. `TokenizedText`, `ProjectedVisualTokens`, `LanguageOutput`,
  `GenerationConfig`, `GeneratedText`.
- `medfm/core/task.py` — `TaskModule` protocol with metric lifecycle
  (`reset_metrics`/`update_metrics`/`compute_metrics`) and distributed
  reduction expectations (true sample/token counts, sufficient statistics);
  `LossOutput` with scalar total, named components, counts, diagnostics.
- `medfm/core/serialization.py` — canonical JSON/YAML, `TensorMeta`
  (shape + accelerator-neutral dtype name, never a device), `config_hash`
  (SHA-256 over canonical JSON), inline metadata tensor limit, and
  `materialize_cpu` for export.
- `docs/core_contracts.md` — shape/coordinate/encoder/bucket semantics and
  the public-contract stability and deprecation policy.
- `tests/phase_02/` — 103 tests including protocol conformance fixtures
  (`contract_fixtures.py`: dummy encoder, pooling-only encoder, LM adapters,
  task module), AST-level backend-neutrality enforcement, protected
  CUDA/XLA transfer fixtures, and the smoke module.
- `medfm/tools/validate_phase.py` + `medfm/tools/smoke.py` — phase 02 gate
  registration and smoke checks.

## Deviations from the plan sketch

- `MedicalBatch` adds a required `modality` field (idea.md sketch omitted it,
  but the plan requires modality to be authoritative).
- `MULTI_SERIES_3D` batches use rank-6 `[B, S, C, D, H, W]` tensors instead of
  Python lists of rank-5 tensors so static-shape/TPU paths stay uniform.
- Hash-typed IDs are `NewType`s over `str` with validating factories rather
  than wrapper classes, keeping serialization trivial.
