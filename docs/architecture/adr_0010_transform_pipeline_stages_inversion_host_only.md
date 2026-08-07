# ADR 0010: Two-stage transform pipelines with inversion history and host-only execution

Status: Accepted (2026-08-05)
Deciders: Project Maintainer

## Context

Phase 04 preprocessing must satisfy four forces at once: deterministic
cacheability (Phase 03 cache keys), per-epoch augmentation variability,
original-space reconstruction for segmentation evaluation (masks compared in
original physical coordinates), and the TPU static-shape policy (ADR 0008).
MONAI-style free-form transform composition makes the cache boundary and
inversion support implicit and therefore unauditable.

## Decision

1. **Two explicit stages.** Every pipeline is a `TransformPipeline` of a
   *deterministic* stage (decode/canonicalization/normalization) followed by
   a *stochastic* stage (augmentation). Placement is enforced at construction
   by each transform's declared `stage`. The cache boundary sits after the
   deterministic stage: `deterministic_config_hash()` excludes stochastic
   transforms by construction, so augmentation can never contaminate a cache
   key.
2. **Recorded, invertible history.** Every transform appends a JSON-able
   `TransformRecord` to `TransformData.history`. Spatial transforms register
   mode-aware inverters (`image` = trilinear/bilinear, `label` = nearest) and
   `invert_history` replays records in reverse for original-space
   reconstruction. Non-invertible steps (intensity/noise augmentation, stain
   augmentation) register no inverter and are skipped — unsupported inversion
   cases are explicit, never silent (`strict=True` raises).
3. **Seeded randomness per sample.** Stochastic transforms draw only from a
   `TransformContext` generator seeded by
   `derive_seed(base_seed, epoch, worker_id, sample_key)` — reproducible under
   fixed seeds, non-identical across workers/epochs/samples.
4. **Host-only execution.** Decode, canonicalization, and all medical
   transforms run on the CPU; fixed tensors transfer to the accelerator only
   after collation (`MedicalBatch.to(device)`). Accelerator execution of
   transforms is opt-in future work and requires host/device parity tests
   before being enabled.
5. **Model-aware validation.** A `PreprocessSpec` (shape, channels, dtype,
   value range, normalization statistics) is attached to pipelines and the
   final tensor is validated against it, so adapters always receive exactly
   their declared tensor format. Window presets, crop policies, and
   augmentation settings live in transform configs (model/config-specific),
   never in global registries.
6. **Bias-field correction and stain steps are opt-in.** N4-style bias-field
   correction is a standalone explicitly-invoked function; stain
   normalization/augmentation are separately hashable transforms. Neither
   appears in any default pipeline.
7. **Multitask mixing is declared, not implicit.** Single-modality collators
   reject foreign modalities; only `MultitaskCollator` mixes, over an
   explicitly declared modality set, dispatching to per-modality delegates
   and returning per-modality `MedicalBatch`es plus an input-order modality
   index (a single `MedicalBatch` cannot hold mixed modalities by contract).

## Alternatives considered

- **MONAI `Compose` everywhere:** cache boundary, inversion support, and
  stage separation are conventions, not enforced structure; MONAI remains an
  optional dependency but the contract layer is ours. Rejected.
- **Invert by re-reading payloads:** doubles I/O and breaks patch-level
  evaluation; recorded-history inversion is exact and cheap. Rejected.
- **Single shared bucket/collator for all modalities:** contradicts the
  per-modality `MedicalBatch` contract and hides incompatible mixes. Rejected.

## Consequences

- Phase 05+ model/adapters consume `PreprocessSpec`-validated tensors only.
- Cache-key components for preprocessing come from
  `TransformPipeline.deterministic_config_hash()` plus `PreprocessSpec.spec_hash()`.
- Segmentation evaluation inverts masks via `invert_history`; transforms that
  cannot be inverted must be listed as unsupported cases in the phase handoff.
- Any future accelerator-resident transform path must add parity tests
  (host vs device output equality) before being enabled.

## Reversal conditions

Reverse or amend if profiling shows host preprocessing cannot feed TPU/CUDA
at target utilization even with worker parallelism and caching, or if a
mature upstream (e.g. MONAI) adds equivalent enforced staging/inversion;
adopt via a new ADR with measurements.
