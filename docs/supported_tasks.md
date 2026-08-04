# Supported Tasks (v1)

Owner: Project Maintainer (Siddhesh) — product
Review date: 2026-11-02

Canonical task names. These are the **only** legal task identifiers in configs, manifests, and the task registry (canonical enum handed to Phase 02). Implementation paths are realized in Phases 11–15.

| Task | Definition | v1 implementation path |
|---|---|---|
| `BINARY_CLASSIFICATION` | Single binary label per sample | Task head on pooled encoder embedding |
| `MULTICLASS_CLASSIFICATION` | Single categorical label | Task head on pooled encoder embedding |
| `MULTILABEL_CLASSIFICATION` | Multiple independent binary labels | Sigmoid task head on pooled embedding |
| `ORDINAL_CLASSIFICATION` | Ordered categorical label | Ordinal (cumulative-link) task head |
| `IMAGE_TEXT_RETRIEVAL` | Rank texts for an image | Dual-tower contrastive model (MedSigLIP-style) |
| `TEXT_IMAGE_RETRIEVAL` | Rank images for a text | Dual-tower contrastive model (MedSigLIP-style) |
| `SEMANTIC_SEGMENTATION` | Dense per-pixel/voxel class map | Encoder feature maps + segmentation decoder head |
| `INSTANCE_SEGMENTATION` | Per-object masks and labels | Deferred — decoder + instance head, post-v1 |
| `PROMPTABLE_SEGMENTATION` | Mask from points/boxes/prior slices | MedSAM2-style prompt encoder + mask decoder |
| `LANGUAGE_CONDITIONED_SEGMENTATION` | Mask from a text phrase | Text-conditioned decoder on visual tokens (deferred to recipe phases) |
| `BOUNDING_BOX_LOCALIZATION` | Boxes for findings | Box regression head on spatial tokens (deferred) |
| `VISUAL_QUESTION_ANSWERING` | Answer a question about image(s) | VLM: external encoder + bridge + MedGemma 1.5 4B |
| `REPORT_GENERATION` | Free-text report from image(s) | VLM: external encoder + bridge + MedGemma 1.5 4B |
| `STRUCTURED_FINDING_GENERATION` | Structured (JSON-like) findings | VLM with constrained/structured decoding |
| `CONTRASTIVE_ALIGNMENT` | Joint image-text embedding training | Dual-tower contrastive loss (Phase 11) |
| `MULTITASK` | Multiple heads sharing one encoder | Shared encoder + task-router heads |

## Modality × task dispositions

Every modality/task pair is mapped to **supported**, **deferred**, or **unsupported**. The machine-readable source of truth is `modality_task_matrix` in `model_registry/v1_scope.yaml`; the consistency tests in `tests/phase_00/test_scope_consistency.py` prove the matrix is a complete partition of all 10 × 16 pairs and that every supported pair has an implementation path. Summary of the deferred/unsupported edges:

- `INSTANCE_SEGMENTATION` is deferred for most image modalities and **unsupported** for `PATHOLOGY_WSI`, `MULTI_IMAGE_2D`, and `TEXT_ONLY` in v1.
- `PROMPTABLE_SEGMENTATION` is supported for `CT_3D` (MedSAM2), deferred for other image modalities, **unsupported** for `PATHOLOGY_WSI` and `TEXT_ONLY`.
- Retrieval and contrastive tasks are **unsupported** for `TEXT_ONLY`.
- Segmentation tasks are **unsupported** for `TEXT_ONLY` and `MULTI_IMAGE_2D` (an aggregation modality, not a dense-prediction modality).
- `LANGUAGE_CONDITIONED_SEGMENTATION` is deferred for all 3D and pathology modalities pending recipe-phase validation.

Rules:

- Adding, renaming, or removing a task requires an ADR and a Phase 02 contract update.
- Moving a pair from deferred/unsupported to supported requires a tracked ADR plus a passing vertical-slice or recipe acceptance test.
