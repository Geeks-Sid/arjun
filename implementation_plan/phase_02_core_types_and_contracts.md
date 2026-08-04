# Phase 02: Core Type System and Contracts

## Objective

Define stable, validated contracts for 2D, 3D, multi-image, WSI, language, model output, and task execution.

## Dependencies

- [x] Phases 00 and 01 are accepted.
- [x] Canonical modalities and tasks are frozen.
- [x] Serialization and validation libraries are available.

## Scope boundaries

Allowed areas: `medfm/core/`, contract fixtures, and Phase 02 tests/docs.

Do not implement readers, model-specific adapters, or training loops.

## Implementation checklist

### Enums and identifiers

- [x] Implement `Modality` with all canonical values from `idea.md`.
- [x] Implement `TaskType` with all canonical task values.
- [x] Define loading modes, coordinate systems, precision modes, and split names.
- [x] Reject unknown values by default while supporting explicit versioned migrations.
- [x] Define typed IDs without exposing raw patient/study/series identifiers.

### Sample schemas

- [x] Implement `MedicalSample` and nested image, label, segmentation, box, conversation, and provenance types.
- [x] Implement `SpatialMetadata` preserving current/original shape, affine, spacing, orientation, and frame reference.
- [x] Implement `PathologyMetadata` preserving MPP, magnification, pyramid dimensions, stain, scanner, and coordinates.
- [x] Validate required fields by modality and task.
- [x] Validate that de-identified/hash fields do not accept obvious raw identifiers where enforceable.
- [x] Add explicit schema version fields and migration hooks.

### Batch schema

- [x] Implement `MedicalBatch` with image, token, target, metadata, and sample ID fields.
- [x] Support 2D, 3D, multi-image, WSI tile, visual-token, text-token, and segmentation shapes.
- [x] Make `modality` authoritative; never derive it only from tensor rank.
- [x] Validate masks and target shapes against batch dimensions.
- [x] Define device-transfer and pin-memory behavior without losing metadata.
- [x] Keep device movement backend-neutral; prohibit `.cuda()` and backend imports in core schemas.
- [x] Represent fixed-shape bucket IDs for image, volume, tile, visual-token, and text batches.
- [x] Require padding masks whenever a static bucket contains padded samples/tokens/tiles.

### Encoder and language contracts

- [x] Implement `EncoderCapabilities`, `PreprocessSpec`, `InputSpec`, and `OutputSpec`.
- [x] Implement the `VisualEncoder` protocol and `EncoderOutput`.
- [x] Document pooled, spatial-token, feature-map, token-mask, and coordinate semantics.
- [x] Implement the `LanguageModelAdapter` protocol and tokenized/generated output types.
- [x] Require adapters to declare support for `inputs_embeds` or a native connector.
- [x] Prevent silent token pooling when spatial output was requested.

### Task and loss contracts

- [x] Implement the `TaskModule` protocol.
- [x] Define `LossOutput` with total loss, named components, sample counts, and diagnostics.
- [x] Define metric lifecycle methods and distributed-reduction expectations.
- [x] Define typed errors for unsupported modality/task/capability combinations.

### Serialization and compatibility

- [x] Provide deterministic JSON/YAML serialization for non-tensor schema fields.
- [x] Define tensor metadata serialization separately from tensor payloads.
- [x] Add configuration hashes based on canonical serialization.
- [x] Document public contract stability and deprecation policy.
- [x] Define accelerator-neutral dtype names and avoid serializing device-specific tensor locations.
- [x] Ensure canonical tensor artifacts can be materialized on CPU before export.

## Tests and verification

- [x] Construct and validate synthetic 2D, 3D, multi-image, and WSI samples.
- [x] Construct and validate each supported batch shape.
- [x] Fail incorrect tensor-rank/modality combinations with actionable errors.
- [x] Round-trip spatial metadata through serialization without loss.
- [x] Round-trip pathology coordinates and MPP without loss.
- [x] Test device transfer retains non-tensor metadata.
- [x] Test unsupported capability requests fail rather than fabricate output.
- [x] Add static protocol conformance fixtures for dummy encoders, LMs, and tasks.
- [x] Run schema/device-transfer fixtures on CPU, CUDA, and XLA tensors where hardware is available.
- [x] Verify static bucket padding and masks preserve unpadded results.
- [x] Verify serialization never stores a hard-coded CUDA/XLA device requirement.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [PyTorch/XLA tensor/device semantics](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html)

## Smoke command

```bash
pytest tests/phase_02/test_contract_smoke.py -q
```

## Acceptance command

```bash
pytest tests/phase_02 -q && python -m medfm.tools.validate_phase --phase 02
```

## Exit criteria

- [x] Synthetic samples and batches validate for every modality family.
- [x] Invalid shape/modality combinations fail clearly.
- [x] Spatial and pathology metadata survive round-trips.
- [x] Every tensor field has documented shape and coordinate semantics.
- [x] Downstream phases can depend on contracts without importing concrete adapters.

## Handoff

- [x] Publish schema examples for reader and collator authors.
- [x] Publish static bucket and accelerator-neutral device movement contracts.
- [x] Publish protocol conformance fixtures for model authors.
- [x] Record contract version and compatibility rules.
- [x] List deferred schema questions as ADR candidates.
