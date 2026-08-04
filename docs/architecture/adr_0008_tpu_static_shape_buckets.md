# ADR 0008: Static shape buckets and bounded token/tile/slice counts on TPU

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

XLA compiles per shape; dynamic shapes cause recompilation storms and host fallbacks. Medical inputs are highly variable: WSI tile counts, slice counts, images per sample, visual-token counts, text lengths, 3D patch shapes.

## Decision

On TPU, all variable dimensions are **bucketed and padded to static shapes with masks**: 2D image sizes, 3D patch shapes, images/slices per sample, WSI tile counts, visual-token counts, and text lengths each use a bounded, documented set of buckets (canonical defaults in `implementation_plan/accelerator_training_strategy.md`). Sample content must never change control flow inside a compiled training step; data-dependent Python loops in forward are prohibited; the final distributed batch is padded or dropped so per-replica shapes are stable. Steady-state recompilation beyond the configured threshold (default 0) fails the TPU acceptance gate.

## Alternatives considered

- **Dynamic shapes with eager fallback:** steady-state CPU fallback destroys TPU economics. Rejected.
- **Per-shape compilation without bounds:** unbounded compile time and memory. Rejected.
- **Padding everything to the global maximum:** simple but wastes compute; buckets are the bounded compromise. Rejected.

## Consequences

- Collators (Phase 04) must implement bucket assignment and masks; bucket sets are config, hashed into every run.
- Modality design must bound T (WSI tiles), I (images), and L (tokens) at ingestion.
- CUDA runs may use dynamic shapes; the bucketed path is shared so parity tests compare like with like.

## Reversal conditions

Reverse if PyTorch/XLA dynamic-shape support (e.g. bounded dynamism) matures and parity/performance tests show no regression; adopt via a new ADR with measured compile counts.
