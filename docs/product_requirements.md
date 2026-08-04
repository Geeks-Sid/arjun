# Product Requirements — Unified Medical Foundation-Model Framework (v1)

Owner: Project Maintainer (Siddhesh) — product
Review date: 2026-11-02
Status: Frozen for v1 (changes require an ADR and a phase gate)

## 1. Confirmed platform assumptions

| Assumption in `idea.md` | Verification result (2026-08-04) | Consequence |
|---|---|---|
| Linux workstation | Confirmed (Linux) | None |
| One NVIDIA GPU, 48 GB VRAM | **Confirmed as the v1 build target** (project owner direction, 2026-08-04): a single 48 GB CUDA GPU is the reference training device. Note: the local dev shell currently exposes an 8 GB RTX 4060 Laptop GPU (nvidia-smi, driver 595.84); it is used for Tier 0/1 development only and does not change the 48 GB planning baseline. | Tier 1 recipes are sized for 48 GB. PEFT-first remains mandatory: full fine-tuning of multi-billion-parameter models does not fit in 48 GB either. See `docs/architecture/adr_0002_peft_first_training.md`. |
| PEFT-first training (LoRA/QLoRA, frozen encoders, projector/head training) | Confirmed as design intent | Locked by ADR 0002 |
| PyTorch + MONAI + Hugging Face runtime | Confirmed as design intent | Locked by ADR 0001 and ADR 0007 |
| Full pretraining of billion-parameter models out of scope | Confirmed | Locked as a non-goal (§4) |
| Clinical data de-identified before entering the training environment | Confirmed as policy | Enforced by `docs/data_governance.md` |

## 2. Users

- **Research engineers** adapting medical foundation models to institutional datasets.
- **ML platform engineers** operating training on CPU (tests), CUDA workstations, and TPU slices.
- **Clinical research collaborators** evaluating task-specific outputs under human review (never end users of raw model output).
- **Coding agents** executing phase-gated implementation under the protocol in `agent/`.

## 3. v1 use cases

1. Adapt 2D radiology encoders (X-ray, CT/MRI slices) to classification and retrieval tasks.
2. Adapt native 3D CT/MRI encoders to classification and segmentation tasks.
3. Adapt pathology tile/WSI encoders to tile classification and slide-level (MIL) classification and retrieval.
4. Promptable and language-conditioned segmentation on medical images/volumes.
5. 2D VLM instruction tuning (VQA, report and structured-finding generation) via an external-encoder bridge.
6. Slice-sequence VLM training (MedGemma-style multi-image input) for volumetric studies.
7. Research integration of native 3D VLMs (M3D-LaMed, Merlin) behind the same contracts.
8. Contrastive alignment between medical images and text.

## 4. Non-goals (explicitly outside v1)

- **Full pretraining of billion-parameter foundation models is outside v1.** Only PEFT-scale adaptation of existing checkpoints.
- A single monolithic multimodal model across modalities. **Models remain modality-specific internally; they are unified only behind shared framework contracts** (sample/batch schemas, encoder/decoder/task protocols, checkpoint format).
- Clinical deployment, autonomous diagnosis, or any use without a qualified human reviewer (see `docs/clinical_safety_scope.md`).
- Multi-host TPU SPMD (Tier 5) acceptance — supported only after single-host TPU acceptance.
- bitsandbytes QLoRA on TPU (see `docs/architecture/adr_0009_cuda_qlora_vs_tpu_bf16_lora.md`).
- DICOMweb ingestion, serving infrastructure beyond local export (Phase 17 scope).

## 5. v1 model roster

Preferred and fallback backbones per modality family (full details and license state in `model_registry/v1_scope.yaml` and `model_registry/licenses.yaml`):

| Family | Preferred | Fallback | Optional |
|---|---|---|---|
| 2D radiology | MedSigLIP | RAD-DINO | MedGemma 1.5 native visual pathway, H-Optimus-0 (patch), CONCH |
| 3D CT | CT-FM | FlexiCT-3D | Merlin (3D VLM), M3D-LaMed (research) |
| 3D MRI | Triad | NV-Segment-CTMR (seg) | BrainIAC (deferred) |
| Pathology tile/WSI | H-Optimus-0 | GigaPath-Flash | TITAN, CONCH |
| Promptable segmentation | MedSAM2 | NV-Segment-CTMR interactive | — |
| Generative VLM / language | MedGemma 1.5 4B | M3D-LaMed (research) | Generic Gemma/Qwen behind the same interface |

## 6. Minimum end-to-end vertical slices (v1)

Each slice must run data → adapt → train (PEFT) → evaluate → export adapter-only checkpoint:

- **VS-2D:** XRAY_2D multilabel classification — RAD-DINO/MedSigLIP frozen encoder + trained head.
- **VS-3D:** CT_3D classification — CT-FM frozen encoder + trained head.
- **VS-WSI:** PATHOLOGY_WSI slide classification — H-Optimus-0 tile embeddings + slide aggregator (GigaPath-Flash-style).
- **VS-SEG:** CT_3D semantic segmentation — Triad/NV-Segment-CTMR decoder path.
- **VS-RETRIEVAL:** XRAY_2D image↔text retrieval — MedSigLIP dual towers with contrastive objective.
- **VS-VLM:** 2D VLM report generation / VQA — external visual encoder + bridge + MedGemma 1.5 4B (QLoRA on CUDA, BF16 LoRA on TPU).

## 7. Accelerator support tiers

| Tier | Definition | v1 status |
|---|---|---|
| T0 | CPU contract tests, tiny local models, synthetic data | Required |
| T1 | One CUDA GPU, 48 GB VRAM (v1 reference device; local dev shell may be smaller) | Required |
| T2 | Multi-GPU DDP (replicated) | Required where hardware available |
| T3 | Multi-GPU FSDP (sharded) | Required where hardware available |
| T4 | Single-host TPU via PyTorch/XLA PJRT | Required (minimum target below) |
| T5 | Multi-host TPU SPMD/FSDP | Deferred — only after T4 acceptance |

**Backend support is certified per model/task/topology, never framework-wide by assumption.** Every registry entry carries a per-backend status from the capability matrix in `implementation_plan/accelerator_training_strategy.md` (`UNTESTED`, `CPU_CONTRACT_ONLY`, `SUPPORTED_*`, `BLOCKED_*`, `NOT_APPLICABLE`), backed by recorded smoke evidence before any `SUPPORTED_*` claim.

**Minimum v1 TPU target (T4):** tiny-model coverage for every task family (classification, segmentation, retrieval/contrastive, VLM generation) **plus at least one accepted real Hugging Face vision or language backbone** trained end-to-end on a single-host TPU slice.

## 8. Measurable v1 outcomes

1. All phase acceptance gates 00–18 pass (`python -m medfm.tools.validate_phase --phase <N>`).
2. All six vertical slices run end-to-end on T0 (tiny models) and T1 (real backbones on the 48 GB reference device via PEFT).
3. T4 minimum target met with recorded XLA compilation metrics.
4. 100% of registry models have a populated license record and per-backend accelerator status; no `SUPPORTED_*` status without recorded evidence.
5. Patient-level split leakage test fails a deliberately corrupted split (proving the check works).
6. Every training run records the mandatory metadata in `docs/reproducibility_policy.md`.
7. No patient data or model weights committed to Git (checked in CI, Phase 18).

## 9. Ownership

| Area | Owner | Notes |
|---|---|---|
| Product | Project Maintainer (Siddhesh) | Scope changes require ADR + phase gate |
| Clinical safety | Project Maintainer (acting); a designated clinical safety officer is **required before any clinical-data use** | See `docs/clinical_safety_scope.md` |
| Data governance | Project Maintainer (acting) | See `docs/data_governance.md` |
| Model licensing | Project Maintainer (acting) — named per-record review owners in `model_registry/licenses.yaml` | See `docs/licensing_policy.md` |
