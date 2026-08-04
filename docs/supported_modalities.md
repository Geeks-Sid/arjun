# Supported Modalities (v1)

Owner: Project Maintainer (Siddhesh) — product
Review date: 2026-11-02

Canonical modality names. These are the **only** legal values of the `modality` field in `MedicalSample` and dataset manifests (canonical enum handed to Phase 02). The `modality` field is authoritative; modality must never be inferred from tensor rank.

| Modality | Definition | Example tensor shape | v1 backbone candidates |
|---|---|---|---|
| `XRAY_2D` | Single 2D projectional radiograph | `[B, C, H, W]` | MedSigLIP (preferred), RAD-DINO (fallback), MedGemma 1.5 visual pathway |
| `CT_2D_SLICE` | Single axial/coronal/sagittal CT slice | `[B, C, H, W]` | RAD-DINO, FlexiCT 2D pathway |
| `CT_3D` | Volumetric CT study or patch | `[B, C, D, H, W]` | CT-FM (preferred), FlexiCT-3D (fallback), Merlin |
| `MRI_2D_SLICE` | Single MRI slice, one sequence | `[B, C, H, W]` | RAD-DINO, MedSigLIP |
| `MRI_3D` | Volumetric MRI, single or stacked sequences | `[B, C, D, H, W]` | Triad (preferred), NV-Segment-CTMR (segmentation) |
| `PATHOLOGY_TILE` | Single WSI tile at a recorded MPP | `[B, C, H, W]` | H-Optimus-0 (preferred), CONCH (optional) |
| `PATHOLOGY_WSI` | Whole-slide image as bounded tile sets | `[B, T, C, H, W]` | H-Optimus-0 + GigaPath-Flash aggregator, TITAN |
| `MULTI_IMAGE_2D` | Bounded set of 2D images per sample (multi-view, slice sequences for MedGemma-style VLMs) | `[B, I, C, H, W]` | MedGemma 1.5 4B (slice-sequence), MedSigLIP per-image |
| `MULTI_SERIES_3D` | Multiple co-registered 3D series per sample (e.g. multi-sequence MRI) | list of `[B, C, D, H, W]` | Triad multi-sequence, NV-Segment-CTMR |
| `TEXT_ONLY` | Text with no image input | `[B, L]` | MedGemma 1.5 4B language pathway, generic Gemma/Qwen |

Rules:

- Adding, renaming, or removing a modality requires an ADR and a Phase 02 contract update.
- `MULTI_IMAGE_2D` covers MedGemma-style slice/multi-image input; it is **not** a substitute for a native volumetric encoder (see `docs/architecture/adr_0005_native_3d_and_slice_sequence_vlm.md`).
- WSI tile counts, slice counts, and image counts are bounded per the TPU static-shape policy (`docs/architecture/adr_0008_tpu_static_shape_buckets.md`).
