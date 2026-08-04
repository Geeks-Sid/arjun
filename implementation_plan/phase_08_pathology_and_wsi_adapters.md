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

- [ ] Implement `PathologyTileEncoder`, `SlideAggregator`, `TileSampler`, `EmbeddingStore`, `WSITokenSelector`, and `PathologyVLMAdapter` contracts.
- [ ] Enforce the two-stage WSI flow: index tiles, encode, persist, aggregate/select.
- [ ] Never load an entire WSI pixel pyramid into memory.
- [ ] Preserve slide ID, tile ID, coordinates, level, MPP, quality, model revision, preprocess hash, and dtype.
- [ ] Use atomic embedding-store writes and detect incomplete slides.
- [ ] Support resumable extraction at tile/chunk granularity.
- [ ] Keep OpenSlide/TiffSlide decoding and tile quality work on CPU hosts.
- [ ] Treat cuCIM as an optional CUDA acceleration, never a TPU dependency.
- [ ] Emit fixed tile batches and fixed selected-token counts for TPU execution.
- [ ] Shard slides/embedding chunks deterministically across GPU/TPU ranks.

## Adapter checklist

### Tile encoders

- [ ] Integrate H-Optimus-0 once and reuse it with Phase 06.
- [ ] Integrate the GigaPath tile encoder with its native preprocessing.
- [ ] Add optional CONCH only after license approval.
- [ ] Support frozen extraction, batched inference, cache metadata, and bounded queues.

### Slide encoders

- [ ] Preserve GigaPath tile/slide separation.
- [ ] Implement GigaPath-Flash as the first preferred slide-level integration where available.
- [ ] Integrate TITAN for slide representation and image-text alignment.
- [ ] Implement a generic mean-pooling baseline and attention MIL baseline.
- [ ] Return slide embeddings and selected/evidence tile metadata.

### Selection and token budgets

- [ ] Implement random tissue, quality-weighted, diversity, top-k attention, grid, multiresolution, and text-conditioned selector interfaces.
- [ ] Start acceptance with deterministic grid and seeded random selectors.
- [ ] Bound raw tile samples per batch.
- [ ] Bound post-resampler LLM visual tokens to a configurable 32-128 default range.
- [ ] Make 128-1,024 pre-compression embeddings configurable and benchmarked.
- [ ] Keep evaluation selection deterministic.

## Embedding store checklist

- [ ] Select Zarr/HDF5/Arrow based on concurrent access and deployment constraints; record the choice in an ADR.
- [ ] Store array shapes, chunks, compression, dtype, and schema version.
- [ ] Validate coordinate and embedding row alignment.
- [ ] Support read subsets without loading all embeddings.
- [ ] Invalidate stores on encoder, revision, preprocess, layer, or dtype changes.
- [ ] Detect missing/corrupt tiles and continue according to an explicit failure threshold.

## Tests and verification

- [ ] Tile a synthetic pyramid slide twice and compare stable IDs/coordinates.
- [ ] Extract/cache embeddings with a tiny local tile encoder.
- [ ] Train one step of a slide classifier from cached embeddings.
- [ ] Verify mean-pooling and attention-MIL output contracts.
- [ ] Verify selectors respect tile and visual-token budgets.
- [ ] Verify a missing/corrupt tile does not terminate an otherwise valid epoch.
- [ ] Verify evidence coordinates map to the source slide.
- [ ] Record extraction throughput, store size, and peak CPU/GPU memory.
- [ ] Run tile encoder and slide aggregator fixed-shape steps on CUDA and TPU when declared supported.
- [ ] Verify padded tile/token entries have no effect on aggregation or loss.
- [ ] Verify distributed ranks do not process the same slide except explicit padding and that metrics deduplicate it.
- [ ] Record host input stalls and XLA compilation count for WSI TPU runs.

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

- [ ] Synthetic WSI extraction and resumable caching pass.
- [ ] One real tile encoder and one slide aggregation path pass smoke tests.
- [ ] Cached embeddings support classifier backward.
- [ ] Token budgets are fixed and enforced.
- [ ] At least one accepted WSI representation path exposes tokens for Phase 09.
- [ ] WSI aggregation has a static-shape TPU path or an explicit blocked status with a CPU-host/CUDA alternative.

## Handoff

- [ ] Identify the pathology path that unblocks Phase 09.
- [ ] Publish embedding-store schema and invalidation version.
- [ ] Publish tile-selector determinism and budget behavior.
- [ ] Record throughput/memory limits and corrupt-tile policy.
- [ ] Record backend-specific tile throughput, fixed buckets, and optional CUDA accelerations.
