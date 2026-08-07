# Phase 15 handoff

## Delivered contracts

- `build_phase15_recipe` and `phase15_builders()` expose tile classification, WSI classification, WSI VLM, and tiled segmentation through the model-agnostic Phase 12 pipeline.
- `PathologyRecipeMetadata` is the source for bounded tile counts, visual-token buckets, selector revisions, MPP/magnification, embedding-store revision, split policy, shard unit, batch formula, memory cap, and backend observability.
- `pad_slide_embeddings`/`select_wsi_visual_tokens` return fixed-width tensors, masks, level-0 tile geometry, retained records, selected indices, and actual counts.
- `stitch_tile_predictions` and evidence JSON helpers are host-side and coordinate-system explicit. Evidence payloads contain no pixels, patient names, free-form clinical text, or paths.
- Phase 17 can consume the published configs, protected evidence artifacts, and `make_phase15_artifact` provenance wrapper.

## Recommended next work

1. Register approved production pathology datasets/manifests and checkpoint revisions/licenses before enabling non-offline recipe modes.
2. Run the cached-embedding static aggregator and fixed-selector Perceiver profiles on TPU BF16/SPMD hardware; capture compile count, input stalls, HBM, and throughput.
3. Run CUDA tile LoRA and real slide aggregation profiles with protected de-identified data; verify slide-level sharding and multi-rank resume.
4. Compare tile-count/magnification condition rows with patient/slide clustered metrics and scanner/site/organ subgroup slices.
5. Perform protected artifact review for stitched masks/evidence heatmaps and hand export candidates to Phase 17.
