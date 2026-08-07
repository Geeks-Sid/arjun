# ADR 0011: HDF5 per-slide pathology embedding stores

Status: Accepted (2026-08-06)
Deciders: Project Maintainer

## Context

Phase 08 pathology recipes need to persist slide-level tile embeddings so that
later stages (retrieval, MIL aggregation, WSI-VLM training) can re-read them
without re-encoding whole gigapixel pyramids. The store has four forces at
once: bounded reads (a consumer asks for a subset of rows, not a whole cohort
in memory), backend-neutral tensor output, deterministic invalidation tied to
schema/model/preprocess identity, and crash-safe, resumable writes.

## Decision

1. **One HDF5 file per slide**, with chunked/gzip-compressed `embeddings` and
   aligned `tile_ids`, `coords`, `level`, `mpp`, and quality columns. A JSON
   sidecar carries the schema and encoder identity; a digest marker is written
   last.
2. **Atomicity/resume.** Writers create a sibling temporary file and atomically
   replace the slide file. Completion markers are written after the sidecar.
   Extraction chunks are independently atomically persisted under
   `<slide>.chunks/` and finalized only after all healthy tiles are present.
3. **Invalidation.** The store identity includes schema version, model
   revision, preprocessing hash, layer, and dtype. Any mismatch invalidates the
   existing complete store.
4. **Corrupt-tile policy.** Reader failures are counted per requested tile.
   Extraction continues for `on_corrupt="skip"` until the configured failure
   threshold is exceeded; a store is never marked complete when the threshold
   is exceeded.

## Alternatives considered

- **Whole-cohort embedding store:** one file for the entire dataset avoids
  per-slide bookkeeping but forces opening a gigapixel pyramid or loading the
  full cohort for a single slide's rows. Rejected.
- **Per-slide safetensors arrays:** already used for checkpoints/embedding
  reads elsewhere, but lacks the aligned columnar metadata, chunked/compressed
  tile reads, and bounded row subsetting HDF5 provides here. Rejected.
- **zarr (pathology extra):** available, but HDF5 already satisfies the
  concurrent read-only subset access and compression requirements without an
  additional store schema. Rejected for this phase.

## Consequences

- Consumers read embeddings plus aligned diagnostic columns from one slide file
  with bounded row reads and no whole-cohort materialization.
- Store identity is a hard-build key: model revision, preprocessing hash, layer,
  dtype, and schema version changes invalidate complete stores by construction.
- Crash-safe writes come at the cost of sibling-temp-file replacement and
  chunk-finalization bookkeeping.
- Corrupted tiles degrade to skipped extraction rather than failing the store,
  bounded by the configured failure threshold.

## Reversal conditions

Reverse or amend if HDF5 concurrency or platform support becomes a deployment
problem, if a mature upstream store abstraction covers per-slide columnar
access with equivalent identity/invalidation semantics, or if storage profiling
shows per-slide files are wasteful for small slides — adopt via a new ADR with
measurements.
