# Phase 15 Pathology Recipe Cards

Phase 15 publishes bounded-memory pathology recipes. The offline profiles use deterministic local PyTorch modules and synthetic batches as contract fixtures; they are not clinical checkpoints or validation results. Production profiles remain fail-closed until a de-identified dataset manifest, pinned preprocessing revision, checkpoint revision, and license review are recorded.

## Published recipe matrix

| Profile | Family | Stage | Main contract | Accelerator declaration |
| --- | --- | ---: | --- | --- |
| `tile_classification_hoptimus_linear.yaml` | tile classification | 1 | frozen/cached tile embeddings plus linear head | fixed tile CUDA BF16 contract |
| `tile_classification_mlp.yaml` | tile classification | 2 | frozen/cached embeddings plus MLP head | fixed tile CUDA BF16 contract |
| `tile_classification_vision_lora.yaml` | tile classification | 3 | direct image batches plus late-block vision LoRA | CUDA BF16; target modules explicit |
| `tile_classification_tpu_bf16.yaml` | tile classification | 1 | fixed-shape cached tile encoder and linear head | TPU BF16 static-shape contract, not hardware acceptance |
| `tile_classification_contrastive.yaml` | tile classification | 4 | optional image/text alignment loss | fixed tile shape; text pairs explicit |
| `wsi_classification_smoke.yaml` | WSI classification | 1 | versioned cached embeddings and deterministic mean pooling | CPU contract smoke |
| `wsi_classification_attention_mil.yaml` | WSI classification | 2 | attention MIL over masked tile buckets | CPU contract; host aggregation |
| `wsi_classification_gated_attention_mil.yaml` | WSI classification | 2 | gated attention MIL over masked tile buckets | CPU contract; host aggregation |
| `wsi_classification_transformer.yaml` | WSI classification | 3 | transformer/GigaPath-Flash slide path | CPU contract; host aggregation |
| `wsi_classification_tpu_bf16.yaml` | WSI classification | 1 | cached embeddings plus static mean aggregator | TPU BF16 static-shape contract, not hardware acceptance |
| `wsi_vlm_cached_smoke.yaml` | WSI VLM | 1 | cached embeddings, coordinate-aware Perceiver, 32 visual tokens | CPU contract smoke |
| `wsi_vlm_report.yaml` | WSI VLM | 2 | bridge plus language LoRA report generation | CPU contract; TPU profile separate |
| `wsi_vlm_evidence_tpu_bf16.yaml` | WSI VLM | 2 | fixed selector/query budget and BF16 LoRA policy | TPU BF16 SPMD contract, not hardware acceptance |
| `wsi_vlm_evidence.yaml` | evidence localization | 1 | level-0 coordinate evidence JSON | CPU contract smoke |
| `wsi_vlm_organ.yaml`, `wsi_vlm_subtype.yaml` | WSI VLM | 1 | organ/site and subtype task declarations | CPU contract |
| `wsi_vlm_grade.yaml`, `wsi_vlm_biomarker.yaml` | WSI VLM | 2 | grade and biomarker task declarations with language LoRA | CPU contract |
| `wsi_vlm_vqa.yaml`, `wsi_vlm_retrieval.yaml` | WSI VLM | 1 | VQA and image/text retrieval declarations | CPU contract |
| `segmentation_smoke.yaml` | tiled segmentation | 1 | frozen encoder plus 2D decoder, host stitch | CPU contract smoke |
| `segmentation_tile_unet.yaml` | tiled segmentation | 1 | conventional tile UNet baseline | CPU contract |

All WSI paths retain `tile_records`, level-0 geometry, MPP, magnification, selector revision, actual tile counts, and padding masks. Slide IDs—not dependent tiles—are the declared distributed shard unit. Training selection and deterministic evaluation selection are separate fields.

## Bounded-memory and evidence policy

1. A reader/encoder processes bounded tile chunks; no recipe loads a complete WSI pyramid into accelerator memory.
2. Embeddings are versioned against the Phase 08 HDF5 schema/revision. Padded tile entries have false masks and are excluded from losses and metrics.
3. WSI VLM selection occurs before fixed Perceiver resampling. The visual-token buckets are 32, 64, or 128; precompression is bounded to 128–1024 candidates.
4. `image`, `none`, `shuffle_tiles`, and `shuffle_coordinates` modes are available for visual-dependence and coordinate-dependence checks.
5. Evidence JSON contains only de-identified slide IDs, ranked tile IDs, scores, and level-0 slide-pixel geometry. It contains no pixel payload, patient name, free-form report text, or filesystem path.
6. Segmentation blends overlap on a host-side level-0 canvas. Boundaries, missing tiles, multiple pyramid levels, coverage masks, and coordinate-system declarations remain explicit.

## Metrics and observability

Tile/slide/patient clustered classification metrics are emitted separately, with scanner/site/organ subgroup metrics when identifiers are present. WSI benchmark rows keep tile count and magnification as independent conditions. Segmentation emits separate tile and reconstructed-slide metrics. Recipe metadata records microbatch, world size, global batch formula, memory cap, actual tile counts, throughput placeholders, host embedding-store input stalls, compiler count, VRAM/HBM fields, and fallback-operator status.

## Acceptance and limitations

The focused acceptance command is:

```bash
python -m pytest tests/phase_15 -q && python -m medfm.tools.validate_phase --phase 15
```

The CPU smoke command is:

```bash
python -m medfm.cli.train --config configs/recipes/pathology/wsi_classification_smoke.yaml
```

This workstation does not provide a TPU PJRT runtime, protected pathology cohorts, or approved production checkpoint access. TPU BF16/SPMD and CUDA/QLoRA claims therefore remain explicit configuration contracts, not executed hardware results. Clinical performance, external-site generalization, reader agreement, and protected evidence-artifact handling require later acceptance work.
