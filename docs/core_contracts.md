# Core Contracts (Phase 02)

Owner: Project Maintainer (Siddhesh) — framework
Contract version: **schema 1** (`medfm.core.versioning.SCHEMA_VERSION`)

`medfm/core/` is the stable contract layer. Downstream phases (readers,
collators, adapters, task heads, training) depend only on these types — never
on concrete adapter implementations.

## Module map

| Module | Contents |
|---|---|
| `medfm/core/enums.py` | `Modality`, `TaskType`, `LoadingMode`, `CoordinateSystem`, `PrecisionMode`, `SplitName`; strict parsing with versioned enum migrations |
| `medfm/core/errors.py` | Typed errors: `ContractError` and subclasses (schema, shape, bucket, unsupported modality/task/capability) |
| `medfm/core/versioning.py` | `SCHEMA_VERSION`, payload migration hooks (`register_schema_migration`, `migrate_payload`) |
| `medfm/core/sample.py` | `MedicalSample`, `ImageReference`, `LabelTarget`, `SegmentationTarget`, `BoxTarget`, `ConversationTurn`, `ProvenanceMetadata`, `SpatialMetadata`, `PathologyMetadata`, typed ID hashes |
| `medfm/core/batch.py` | `MedicalBatch`, `BucketId`/`BucketKind`, device transfer |
| `medfm/core/encoder.py` | `VisualEncoder` protocol, `EncoderCapabilities`, `PreprocessSpec`, `InputSpec`, `OutputSpec`, `EncoderOutput` |
| `medfm/core/language.py` | `LanguageModelAdapter` protocol, `TokenizedText`, `ProjectedVisualTokens`, `LanguageOutput`, `GenerationConfig`, `GeneratedText` |
| `medfm/core/task.py` | `TaskModule` protocol, `LossOutput` |
| `medfm/core/serialization.py` | Canonical JSON/YAML, `TensorMeta`, accelerator-neutral dtype names, `config_hash`, `materialize_cpu` |

## Tensor shape semantics

`MedicalBatch.modality` is **authoritative**. Rank is validated against the
declared modality and is never used to infer it.

| Content | Shape |
|---|---|
| 2D image (`XRAY_2D`, `CT_2D_SLICE`, `MRI_2D_SLICE`, `PATHOLOGY_TILE`) | `[B, C, H, W]` |
| 3D volume (`CT_3D`, `MRI_3D`) | `[B, C, D, H, W]` |
| Multi-image (`MULTI_IMAGE_2D`) | `[B, I, C, H, W]` |
| WSI tiles (`PATHOLOGY_WSI`) | `[B, T, C, H, W]` |
| Multi-series 3D (`MULTI_SERIES_3D`) | `[B, S, C, D, H, W]` |
| Visual tokens (precomputed embeddings) | `task_targets["visual_tokens"] [B, N, Dv]` + `["visual_token_mask"] [B, N]` |
| Text tokens | `input_ids [B, L]`, `attention_mask [B, L]` |
| Segmentation target | `[B, K, H, W]` (2D) or `[B, K, D, H, W]` (3D), spatial dims matching `pixel_values` |

Masks are boolean or 0/1; True/1 marks real content, False/0 marks padding.

## Coordinate semantics

`BoxTarget.coordinate_system` and `EncoderOutput.token_coordinate_system`
are explicit and required whenever coordinates are present:

- `NORMALIZED_IMAGE`: x/y in [0, 1] relative to image width/height.
- `MILLIMETERS`: patient-space mm (radiology; affine-defined).
- `MICRONS`: slide microns (pathology; MPP-defined).
- `SLIDE_PIXELS`: level-0 slide pixels (pathology tile coordinates).

## Encoder output semantics

See `medfm/core/encoder.py` module docstring: pooled embeddings `[B, Dp]`,
spatial tokens `[B, N, Dv]` (row-major spatial order), feature-map pyramids
for segmentation decoders, `token_mask [B, N]` (real vs padded), and
`token_coordinates [B, N, 2|3]` with a mandatory coordinate system. An
adapter that cannot produce a requested output raises
`UnsupportedCapabilityError`; it never fabricates output or silently pools.

## Language adapter contract

Adapters declare `LanguageModelCapabilities.accepts_inputs_embeds` and/or
`native_visual_connector`. Passing `ProjectedVisualTokens` to an adapter
declaring neither must raise `UnsupportedCapabilityError` — visual input is
never silently dropped.

## Static-shape buckets (TPU path, ADR 0008)

`BucketId(kind, shape)` identifies a fixed-shape bucket: `IMAGE_2D (H, W)`,
`VOLUME_3D (D, H, W)`, `MULTI_IMAGE (I,)`, `WSI_TILES (T,)`,
`VISUAL_TOKENS (N,)`, `TEXT_TOKENS (L,)`. A bucketed batch must carry the
mask for every padded dimension (`attention_mask`, `image_mask`, or
`visual_token_mask`); a bucket missing its mask raises `BucketError`.

## Device transfer and accelerator neutrality

- `MedicalBatch.to(device)` / `pin_memory()` move tensors only; all
  non-tensor metadata (modality, sample IDs, bucket, spatial scalars) is
  preserved. Pinned state is a host concept and resets off-CPU.
- Core modules contain no `.cuda()` calls and no `torch_xla`/`bitsandbytes`
  imports (enforced by `tests/phase_02/test_backend_neutrality.py`).
- Serialization never records a device: `TensorMeta` carries shape plus an
  accelerator-neutral dtype name only; canonical artifacts are materialized
  on CPU (`materialize_cpu`) before export.

## Serialization and hashing

- Non-tensor fields serialize to canonical JSON (`sort_keys`, fixed
  separators) and YAML (`sort_keys`); `config_hash` is SHA-256 over canonical
  JSON and is stable across dict orderings.
- Small metadata tensors (affines, tile coordinates, spacing) serialize
  inline as nested lists with their dtype recorded; round-trips are exact.
  Tensors above `MAX_INLINE_TENSOR_ELEMENTS` are payloads and must go through
  a tensor store, never JSON.

## Identifier hygiene

`patient_id_hash`, `study_id_hash`, `series_id_hash`, and
`frame_of_reference_hash` accept only lowercase hex digests (32–128 chars).
Values resembling raw MRNs or DICOM UIDs raise `IdentifierError`.

## Stability and deprecation policy

- The public surface is `medfm.core.__all__`; everything else is private.
- **Additive changes** (new enum value, new optional field with a default)
  are minor changes and keep `SCHEMA_VERSION`. Enum additions still require
  an ADR per the Phase 00 governance docs.
- **Breaking changes** (renamed/removed fields or enum values, changed shape
  semantics) require: an ADR, a `SCHEMA_VERSION` bump, a registered
  `register_schema_migration` step from the previous version, and — for
  retired enum values — a `register_enum_migration` entry so old payloads
  fail loudly or upgrade explicitly. Silent acceptance of unknown values is
  never allowed.
- Payloads newer than the running code are rejected; downgrades are never
  attempted.
- Deprecations are announced in the phase report of the phase that
  introduces them and take effect no earlier than the following phase.
