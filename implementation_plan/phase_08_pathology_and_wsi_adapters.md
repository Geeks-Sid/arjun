# Phase 08: Pathology Tile and WSI Adapters

## Objective

Implement bounded-memory tile and slide processing with deterministic coordinates, persistent embeddings, slide aggregation, token selection, and pathology multimodal outputs.

## Dependencies

- [ ] Phase 05 registry is accepted.
- [ ] Phase 03 slide readers and Phase 04 tile indexing are accepted.
- [ ] License status is resolved for every selected pathology checkpoint.

## Scope boundaries

Allowed areas: pathology model adapters, tile/slide abstractions, embedding stores, selectors/aggregators, registry records, and Phase 08 tests.

Do not attempt end-to-end gigapixel training or implement general LLM behavior.

## Architecture checklist

- [x] Implement `PathologyTileEncoder`, `SlideAggregator`, `TileSampler`, `EmbeddingStore`, `WSITokenSelector`, and `PathologyVLMAdapter` contracts.
- [x] Enforce the two-stage WSI flow: index tiles, encode, persist, aggregate/select.
- [x] Never load an entire WSI pixel pyramid into memory.
- [x] Preserve slide ID, tile ID, coordinates, level, MPP, quality, model revision, preprocess hash, and dtype.
- [x] Use atomic embedding-store writes and detect incomplete slides.
- [x] Support resumable extraction at tile/chunk granularity.
- [x] Keep OpenSlide/TiffSlide decoding and tile quality work on CPU hosts.
- [x] Treat cuCIM as an optional CUDA acceleration, never a TPU dependency.
- [x] Emit fixed tile batches and fixed selected-token counts for TPU execution.
- [x] Shard slides/embedding chunks deterministically across GPU/TPU ranks.

## Adapter checklist

### Tile encoders

- [x] Integrate H-Optimus-0 once and reuse it with Phase 06.
- [x] Integrate the GigaPath tile encoder with its native preprocessing boundary and offline fallback.
- [x] Keep optional CONCH disabled pending license approval.
- [x] Support frozen extraction, batched inference, cache metadata, and bounded queues.

### Slide encoders

- [x] Preserve GigaPath tile/slide separation.
- [x] Implement GigaPath-Flash as the first preferred slide-level integration boundary where available.
- [x] Integrate TITAN for slide representation and image-text alignment boundary.
- [x] Implement a generic mean-pooling baseline and attention MIL baseline.
- [x] Return slide embeddings and selected/evidence tile metadata.

### Selection and token budgets

- [x] Implement random tissue, quality-weighted, diversity, top-k attention, grid, multiresolution, and text-conditioned selector interfaces.
- [x] Start acceptance with deterministic grid and seeded random selectors.
- [x] Bound raw tile samples per batch.
- [x] Bound post-resampler LLM visual tokens to a configurable 32-128 default range.
- [x] Make 128-1,024 pre-compression embeddings configurable; real-checkpoint benchmarking is hardware/license gated.
- [x] Keep evaluation selection deterministic.

## Embedding store checklist

- [x] Select HDF5 based on concurrent access and deployment constraints; record the choice in ADR 0011.
- [x] Store array shapes, chunks, compression, dtype, and schema version.
- [x] Validate coordinate and embedding row alignment.
- [x] Support read subsets without loading all embeddings.
- [x] Invalidate stores on encoder, revision, preprocess, layer, or dtype changes.
- [x] Detect missing/corrupt tiles and continue according to an explicit failure threshold.

## Tests and verification

- [x] Tile a synthetic pyramid slide twice and compare stable IDs/coordinates.
- [x] Extract/cache embeddings with a tiny local tile encoder.
- [x] Train one step of a slide classifier from cached embeddings.
- [x] Verify mean-pooling and attention-MIL output contracts.
- [x] Verify selectors respect tile and visual-token budgets.
- [x] Verify a missing/corrupt tile does not terminate an otherwise valid epoch.
- [x] Verify evidence coordinates map to the source slide.
- [x] Record extraction throughput/store size for the local smoke path; real peak CPU/GPU memory is hardware gated.
- [x] Run fixed-shape CPU smoke; CUDA/TPU execution remains an explicit blocked status with the CPU-host alternative.
- [x] Verify padded tile/token entries have no effect on aggregation or loss.
- [x] Verify deterministic rank sharding assigns each slide/chunk to one rank.
- [x] Record host input stalls and XLA compilation count as hardware-gated follow-up metrics.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [TRIDENT](https://github.com/mahmoodlab/Trident)
- [OpenSlide](https://openslide.org/)
- [cuCIM](https://docs.rapids.ai/api/cucim/stable/)
- [H-Optimus](https://www.bioptimus.com/h-optimus)
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
- [TITAN](https://github.com/mahmoodlab/TITAN)

## Smoke command

```bash
python -m medfm.tools.smoke --phase 08
```

## Acceptance command

```bash
pytest tests/phase_08 -q && python -m medfm.tools.validate_phase --phase 08
```

## Exit criteria

- [x] Synthetic WSI extraction and resumable caching pass.
- [x] One local tile encoder and one slide aggregation path pass smoke tests; real upstream weights remain registry-gated.
- [x] Cached embeddings support classifier backward.
- [x] Token budgets are fixed and enforced.
- [x] The accepted local WSI representation path exposes tokens for Phase 09.
- [x] WSI aggregation has a static-shape CPU path and an explicit CUDA/TPU blocked status with CPU-host alternative.

## Handoff

- [x] Identify the pathology path that unblocks Phase 09.
- [x] Publish embedding-store schema and invalidation version.
- [x] Publish tile-selector determinism and budget behavior.
- [x] Record throughput/memory limits and corrupt-tile policy.
- [x] Record backend-specific tile throughput, fixed buckets, and optional CUDA accelerations.
