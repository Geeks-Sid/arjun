# Phase 15: Pathology Task Recipes

## Objective

Deliver bounded-memory recipes for pathology tile classification, WSI classification, WSI VLM training, evidence localization, and tiled pathology segmentation.

## Dependencies

- [ ] Phases 08-12 are accepted for required components.
- [ ] Phase 03 patient/case/slide split controls are accepted.
- [ ] Phase 04 deterministic tile indexes and quality filters are accepted.
- [ ] Approved pathology datasets and checkpoint licenses are registered.

## Scope boundaries

Allowed areas: `configs/recipes/pathology/`, pathology recipe builders, stitching/evidence utilities, examples, and Phase 15 tests.

Do not train gigapixel images end-to-end or discard tile coordinates after aggregation.

## Recipe 15A: Tile classification

- [ ] Provide H-Optimus-0 frozen linear-head baseline.
- [ ] Add GigaPath tile, CONCH, or MedSigLIP variants only when approved.
- [ ] Stage 1: cached/frozen embeddings plus linear head.
- [ ] Stage 2: cached/frozen embeddings plus MLP head.
- [ ] Stage 3: late-block vision LoRA with direct image batches.
- [ ] Stage 4: optional text contrastive alignment.
- [ ] Report tile metrics and patient/slide-clustered metrics where relevant.
- [ ] Provide fixed tile-shape CUDA and TPU BF16 configs for supported encoders.

## Recipe 15B: WSI classification

- [ ] Precompute versioned tile embeddings.
- [ ] Establish deterministic mean-pooling baseline.
- [ ] Add attention MIL and gated attention MIL.
- [ ] Add transformer/GigaPath-Flash/TITAN slide paths separately.
- [ ] Use patient-disjoint splits and deterministic evaluation tile selection.
- [ ] Bound sampled tiles per training batch and log actual tile counts.
- [ ] Retain coordinates through pooling for interpretability/evidence analysis.
- [ ] Evaluate performance versus tile count and magnification.
- [ ] Bucket sampled tile counts and pad with masks for TPU.
- [ ] Shard slides, not arbitrary dependent tiles, across distributed ranks.
- [ ] Start TPU acceptance from cached embeddings plus a static aggregator.

## Recipe 15C: WSI VLM

- [ ] Use cached tile embeddings as the default starting point.
- [ ] Aggregate/select tiles with coordinates before visual resampling.
- [ ] Compress to a configurable 32-128 LLM visual-token budget.
- [ ] Stage 1: frozen tile/slide encoders plus WSI bridge.
- [ ] Stage 2: bridge plus LLM QLoRA.
- [ ] Stage 3: fine-tune slide aggregator.
- [ ] Stage 4: add tile-encoder LoRA only for narrowly scoped experiments.
- [ ] Support organ/site, subtype, grade, biomarker, report, VQA, retrieval, and evidence tasks as separate configs.
- [ ] Validate evidence-tile JSON and map normalized coordinates back to the WSI.
- [ ] Run no-slide, shuffled-tile, and shuffled-coordinate ablations.
- [ ] Use fixed selector output and Perceiver query counts on TPU.
- [ ] Use TPU BF16 LoRA/SPMD rather than bitsandbytes QLoRA.
- [ ] Measure host embedding-store input stalls separately from TPU compile/step time.

## Recipe 15D: Pathology segmentation

- [ ] Use ROI-annotated tile/mask pairs.
- [ ] Train a 2D decoder with frozen encoder baseline first.
- [ ] Stitch predictions using overlap and blending.
- [ ] Map outputs to level-0 slide coordinates.
- [ ] Handle slide boundaries, missing tiles, and multiple pyramid levels.
- [ ] Evaluate tile and slide levels separately.
- [ ] Compare against a conventional tile UNet baseline.
- [ ] Keep tile/crop/output shapes fixed per TPU bucket and stitch on the host.

## Cross-recipe checklist

- [ ] Pin slide-reader, tile-index, encoder, embedding-store, and selection versions.
- [ ] Test resume at embedding extraction and training stages.
- [ ] Record tiles/slide, selected tiles, magnification, MPP, throughput, storage, and memory.
- [ ] Keep train/evaluation selection behavior explicit.
- [ ] Include scanner/site/organ subgroup metrics.
- [ ] Produce evidence heatmaps and stitched masks as protected artifacts without PHI.
- [ ] Record failure rates for corrupt/low-quality slides and tiles.
- [ ] Record world size, global batch, tile bucket, input throughput, VRAM/HBM, and compile count.
- [ ] Verify padded tiles and slides do not enter loss or patient-level metrics.
- [ ] Require evidence-backed accelerator status for tile encoders and aggregators separately.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [TRIDENT](https://github.com/mahmoodlab/Trident)
- [H-Optimus](https://www.bioptimus.com/h-optimus)
- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
- [TITAN](https://github.com/mahmoodlab/TITAN)
- [OpenSlide](https://openslide.org/)

## Smoke command

```bash
python -m medfm.cli.train --config configs/recipes/pathology/wsi_classification_smoke.yaml
```

## Acceptance command

```bash
pytest tests/phase_15 -q && python -m medfm.tools.validate_phase --phase 15
```

## Exit criteria

- [ ] Cached embeddings train a slide classifier.
- [ ] Mean pooling and one stronger slide aggregator produce accepted baselines.
- [ ] WSI tokens condition an LLM within the token/memory cap.
- [ ] Evidence tiles map back to original slide coordinates.
- [ ] Tiled segmentation reconstructs a valid slide-level mask.
- [ ] At least one cached-embedding pathology recipe passes TPU acceptance.

## Handoff

- [ ] Publish accepted tile/slide/VLM/segmentation configs.
- [ ] Publish embedding storage and throughput requirements.
- [ ] Record selection-bias and quality-filter limitations.
- [ ] Provide export candidates and evidence examples to Phase 17.
- [ ] Publish separate CUDA/TPU input, bucket, precision, and checkpoint settings.
