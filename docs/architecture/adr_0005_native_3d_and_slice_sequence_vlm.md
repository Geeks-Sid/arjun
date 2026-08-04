# ADR 0005: Native 3D and slice-sequence VLMs are distinct

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

Two different approaches claim "3D VLM" capability: native volumetric encoders bridged into an LM (Merlin, M3D-LaMed) and slice/multi-image VLMs that consume collections of 2D slices through a 2D vision tower (MedGemma 1.5 4B). Treating them as interchangeable hides real differences in spatial reasoning, memory profile, and data requirements.

## Decision

`CT_3D`/`MRI_3D` (native volumetric) and `MULTI_IMAGE_2D` (slice-sequence) are **distinct modalities** with distinct adapter families and distinct acceptance criteria. A slice-sequence VLM result is never reported as a native-3D result. MedGemma 1.5 4B is classified as a slice/multi-image architecture, not a substitute for a native volumetric encoder.

## Alternatives considered

- **Unified "3D-ish" modality:** simpler bookkeeping, but conflates architectures and invites invalid metric comparisons. Rejected.
- **Native 3D only:** excludes MedGemma 1.5, the primary generative model, which cannot consume native volumes. Rejected.
- **Slice-sequence only:** loses volumetric context for segmentation/3D classification. Rejected.

## Consequences

- Phase 07 (native 3D adapters) and Phase 09 (bridges) have separate vertical slices.
- Benchmark reports must name which architecture family produced the result.
- Shape bucketing differs: volume-shape buckets vs. images-per-sample buckets (ADR 0008).

## Reversal conditions

Reverse only if a future backbone genuinely fuses both paths (native 3D tokens and slice tokens in one tower); then introduce a new modality value via ADR rather than merging these two.
