# Phase 15 summary

Phase 15 delivers bounded-memory pathology tile classification, WSI classification, WSI VLM, evidence localization, and tiled segmentation recipe contracts. WSI pixels remain outside accelerator memory in the recipe layer: bounded tile embeddings are selected/padded with masks, coordinates and provenance records are retained, and segmentation/evidence reconstruction is host-side.

## Delivered

- Added `medfm/recipes/phase15.py` with typed pinned metadata, deterministic offline contract builders, tile stages 1–4, WSI mean/attention/gated-attention/transformer/TITAN/GigaPath aggregator selection, WSI VLM bridges, fixed visual-token budgets, ablations, metrics, observability, and pipeline builders.
- Added `medfm/recipes/pathology_stitching.py` with bounded host stitching, overlap blending, level-to-level-0 geometry mapping, missing-tile reporting, PHI-safe evidence JSON, normalized-coordinate round trips, and defensive validation.
- Added tile profiles for frozen linear, MLP, fixed-shape TPU BF16, late-block vision LoRA, and optional text contrastive alignment; WSI profiles for deterministic mean, attention MIL, gated attention MIL, transformer, and cached TPU static aggregation.
- Added separate WSI VLM configs for organ/site, subtype, grade, biomarker, report, VQA, retrieval, evidence, cached smoke, and TPU BF16/LoRA contracts.
- Added ROI tile/mask segmentation smoke and conventional tile UNet baseline profiles with fixed buckets, overlap blending, level-0 output policy, missing-tile policy, and host stitching.
- Added Phase 15 CLI family dispatch, focused behavioral tests, smoke registration, validator registration, model-card documentation, and acceptance artifacts.

## Verification

- `python -m pytest tests/phase_15 -q`: 41 passed.
- `python -m medfm.cli.train --config configs/recipes/pathology/wsi_classification_smoke.yaml --format json`: completed one CPU optimizer step with effective batch size 2 and wrote `checkpoints/last`.
- `python -m medfm.tools.smoke --phase 15 --json`: one pathology recipe/stitching/evidence smoke check passed.
- `python -m pytest tests/phase_12 tests/phase_13 tests/phase_14 -q`: 50 passed with no adjacent-phase regressions.
- `python -m medfm.tools.validate_phase --phase 15`: passed after the acceptance artifacts were populated.
- All published offline pathology YAML profiles parse and build through `build_phase15_recipe`; focused tests exercise every recipe family, WSI ablations, coordinate evidence, metrics, padding masks, and host reconstruction.

## Runtime limits

This workstation has no `torch_xla`/PJRT TPU runtime, protected pathology cohorts, or approved production checkpoint access. TPU BF16/SPMD and CUDA/QLoRA profiles are published as explicit static-shape/configuration contracts; no hardware, clinical, external-site, human-reader, or protected evidence-artifact result is claimed. Production checkpoints remain registry/license gated and offline tiny/synthetic results are contract evidence only.
