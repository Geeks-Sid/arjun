# Phase 02: Core Type System and Contracts

## Objective

Define stable, validated contracts for 2D, 3D, multi-image, WSI, language, model output, and task execution.

## Dependencies

- [ ] Phases 00 and 01 are accepted.
- [ ] Canonical modalities and tasks are frozen.
- [ ] Serialization and validation libraries are available.

## Scope boundaries

Allowed areas: `medfm/core/`, contract fixtures, and Phase 02 tests/docs.

Do not implement readers, model-specific adapters, or training loops.

## Implementation checklist

### Enums and identifiers

- [ ] Implement `Modality` with all canonical values from `idea.md`.
- [ ] Implement `TaskType` with all canonical task values.
- [ ] Define loading modes, coordinate systems, precision modes, and split names.
- [ ] Reject unknown values by default while supporting explicit versioned migrations.
- [ ] Define typed IDs without exposing raw patient/study/series identifiers.

### Sample schemas

- [ ] Implement `MedicalSample` and nested image, label, segmentation, box, conversation, and provenance types.
- [ ] Implement `SpatialMetadata` preserving current/original shape, affine, spacing, orientation, and frame reference.
- [ ] Implement `PathologyMetadata` preserving MPP, magnification, pyramid dimensions, stain, scanner, and coordinates.
- [ ] Validate required fields by modality and task.
- [ ] Validate that de-identified/hash fields do not accept obvious raw identifiers where enforceable.
- [ ] Add explicit schema version fields and migration hooks.

### Batch schema

- [ ] Implement `MedicalBatch` with image, token, target, metadata, and sample ID fields.
- [ ] Support 2D, 3D, multi-image, WSI tile, visual-token, text-token, and segmentation shapes.
- [ ] Make `modality` authoritative; never derive it only from tensor rank.
- [ ] Validate masks and target shapes against batch dimensions.
- [ ] Define device-transfer and pin-memory behavior without losing metadata.
- [ ] Keep device movement backend-neutral; prohibit `.cuda()` and backend imports in core schemas.
- [ ] Represent fixed-shape bucket IDs for image, volume, tile, visual-token, and text batches.
- [ ] Require padding masks whenever a static bucket contains padded samples/tokens/tiles.

### Encoder and language contracts

- [ ] Implement `EncoderCapabilities`, `PreprocessSpec`, `InputSpec`, and `OutputSpec`.
- [ ] Implement the `VisualEncoder` protocol and `EncoderOutput`.
- [ ] Document pooled, spatial-token, feature-map, token-mask, and coordinate semantics.
- [ ] Implement the `LanguageModelAdapter` protocol and tokenized/generated output types.
- [ ] Require adapters to declare support for `inputs_embeds` or a native connector.
- [ ] Prevent silent token pooling when spatial output was requested.

### Task and loss contracts

- [ ] Implement the `TaskModule` protocol.
- [ ] Define `LossOutput` with total loss, named components, sample counts, and diagnostics.
- [ ] Define metric lifecycle methods and distributed-reduction expectations.
- [ ] Define typed errors for unsupported modality/task/capability combinations.

### Serialization and compatibility

- [ ] Provide deterministic JSON/YAML serialization for non-tensor schema fields.
- [ ] Define tensor metadata serialization separately from tensor payloads.
- [ ] Add configuration hashes based on canonical serialization.
- [ ] Document public contract stability and deprecation policy.
- [ ] Define accelerator-neutral dtype names and avoid serializing device-specific tensor locations.
- [ ] Ensure canonical tensor artifacts can be materialized on CPU before export.

## Tests and verification

- [ ] Construct and validate synthetic 2D, 3D, multi-image, and WSI samples.
- [ ] Construct and validate each supported batch shape.
- [ ] Fail incorrect tensor-rank/modality combinations with actionable errors.
- [ ] Round-trip spatial metadata through serialization without loss.
- [ ] Round-trip pathology coordinates and MPP without loss.
- [ ] Test device transfer retains non-tensor metadata.
- [ ] Test unsupported capability requests fail rather than fabricate output.
- [ ] Add static protocol conformance fixtures for dummy encoders, LMs, and tasks.
- [ ] Run schema/device-transfer fixtures on CPU, CUDA, and XLA tensors where hardware is available.
- [ ] Verify static bucket padding and masks preserve unpadded results.
- [ ] Verify serialization never stores a hard-coded CUDA/XLA device requirement.

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

- [ ] Synthetic samples and batches validate for every modality family.
- [ ] Invalid shape/modality combinations fail clearly.
- [ ] Spatial and pathology metadata survive round-trips.
- [ ] Every tensor field has documented shape and coordinate semantics.
- [ ] Downstream phases can depend on contracts without importing concrete adapters.

## Handoff

- [ ] Publish schema examples for reader and collator authors.
- [ ] Publish static bucket and accelerator-neutral device movement contracts.
- [ ] Publish protocol conformance fixtures for model authors.
- [ ] Record contract version and compatibility rules.
- [ ] List deferred schema questions as ADR candidates.
