# Phase 15: Pathology Task Recipes

## Objective

Deliver bounded-memory recipes for pathology tile classification, WSI classification, WSI VLM training, evidence localization, and tiled pathology segmentation.

## Dependencies

- [x] Phases 08-12 are accepted for required components.
- [x] Phase 03 patient/case/slide split controls are accepted.
- [x] Phase 04 deterministic tile indexes and quality filters are accepted.
- [ ] Approved pathology datasets and checkpoint licenses are registered.

## Scope boundaries

Allowed areas: `configs/recipes/pathology/`, pathology recipe builders, stitching/evidence utilities, examples, and Phase 15 tests.

Do not train gigapixel images end-to-end or discard tile coordinates after aggregation.

## Recipe 15A: Tile classification

- [x] Provide H-Optimus-0 frozen linear-head baseline.
- [x] Add GigaPath tile, CONCH, or MedSigLIP variants only when approved.
- [x] Stage 1: cached/frozen embeddings plus linear head.
- [x] Stage 2: cached/frozen embeddings plus MLP head.
- [x] Stage 3: late-block vision LoRA with direct image batches.
- [x] Stage 4: optional text contrastive alignment.
- [x] Report tile metrics and patient/slide-clustered metrics where relevant.
- [x] Provide fixed tile-shape CUDA and TPU BF16 configs for supported encoders.

## Recipe 15B: WSI classification

- [x] Precompute versioned tile embeddings.
- [x] Establish deterministic mean-pooling baseline.
- [x] Add attention MIL and gated attention MIL.
- [x] Add transformer/GigaPath-Flash/TITAN slide paths separately.
- [x] Use patient-disjoint splits and deterministic evaluation tile selection.
- [x] Bound sampled tiles per training batch and log actual tile counts.
- [x] Retain coordinates through pooling for interpretability/evidence analysis.
- [x] Evaluate performance versus tile count and magnification.
- [x] Bucket sampled tile counts and pad with masks for TPU.
- [x] Shard slides, not arbitrary dependent tiles, across distributed ranks.
- [x] Start TPU acceptance from cached embeddings plus a static aggregator.

## Recipe 15C: WSI VLM

- [x] Use cached tile embeddings as the default starting point.
- [x] Aggregate/select tiles with coordinates before visual resampling.
- [x] Compress to a configurable 32-128 LLM visual-token budget.
- [x] Stage 1: frozen tile/slide encoders plus WSI bridge.
- [x] Stage 2: bridge plus LLM QLoRA.
- [x] Stage 3: fine-tune slide aggregator.
- [x] Stage 4: add tile-encoder LoRA only for narrowly scoped experiments.
- [x] Support organ/site, subtype, grade, biomarker, report, VQA, retrieval, and evidence tasks as separate configs.
- [x] Validate evidence-tile JSON and map normalized coordinates back to the WSI.
- [x] Run no-slide, shuffled-tile, and shuffled-coordinate ablations.
- [x] Use fixed selector output and Perceiver query counts on TPU.
- [x] Use TPU BF16 LoRA/SPMD rather than bitsandbytes QLoRA.
- [x] Measure host embedding-store input stalls separately from TPU compile/step time.

## Recipe 15D: Pathology segmentation

- [x] Use ROI-annotated tile/mask pairs.
- [x] Train a 2D decoder with frozen encoder baseline first.
- [x] Stitch predictions using overlap and blending.
- [x] Map outputs to level-0 slide coordinates.
- [x] Handle slide boundaries, missing tiles, and multiple pyramid levels.
- [x] Evaluate tile and slide levels separately.
- [x] Compare against a conventional tile UNet baseline.
- [x] Keep tile/crop/output shapes fixed per TPU bucket and stitch on the host.

## Cross-recipe checklist

- [x] Pin slide-reader, tile-index, encoder, embedding-store, and selection versions.
- [x] Test resume at embedding extraction and training stages.
- [x] Record tiles/slide, selected tiles, magnification, MPP, throughput, storage, and memory.
- [x] Keep train/evaluation selection behavior explicit.
- [x] Include scanner/site/organ subgroup metrics.
- [x] Produce evidence heatmaps and stitched masks as protected artifacts without PHI.
- [x] Record failure rates for corrupt/low-quality slides and tiles.
- [x] Record world size, global batch, tile bucket, input throughput, VRAM/HBM, and compile count.
- [x] Verify padded tiles and slides do not enter loss or patient-level metrics.
- [x] Require evidence-backed accelerator status for tile encoders and aggregators separately.

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

- [x] Cached embeddings train a slide classifier.
- [x] Mean pooling and one stronger slide aggregator produce accepted baselines.
- [x] WSI tokens condition an LLM within the token/memory cap.
- [x] Evidence tiles map back to original slide coordinates.
- [x] Tiled segmentation reconstructs a valid slide-level mask.
- [ ] At least one cached-embedding pathology recipe passes TPU acceptance.

## Handoff

- [x] Publish accepted tile/slide/VLM/segmentation configs.
- [x] Publish embedding storage and throughput requirements.
- [x] Record selection-bias and quality-filter limitations.
- [x] Provide export candidates and evidence examples to Phase 17.
- [x] Publish separate CUDA/TPU input, bucket, precision, and checkpoint settings.
