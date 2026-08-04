# Phase 00 → Phase 01 Handoff

## Canonical enum values for Phase 02

Machine-readable source: `model_registry/v1_scope.yaml`.

- **Modalities:** `XRAY_2D`, `CT_2D_SLICE`, `CT_3D`, `MRI_2D_SLICE`, `MRI_3D`, `PATHOLOGY_TILE`, `PATHOLOGY_WSI`, `MULTI_IMAGE_2D`, `MULTI_SERIES_3D`, `TEXT_ONLY`.
- **Tasks:** `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `MULTILABEL_CLASSIFICATION`, `ORDINAL_CLASSIFICATION`, `IMAGE_TEXT_RETRIEVAL`, `TEXT_IMAGE_RETRIEVAL`, `SEMANTIC_SEGMENTATION`, `INSTANCE_SEGMENTATION`, `PROMPTABLE_SEGMENTATION`, `LANGUAGE_CONDITIONED_SEGMENTATION`, `BOUNDING_BOX_LOCALIZATION`, `VISUAL_QUESTION_ANSWERING`, `REPORT_GENERATION`, `STRUCTURED_FINDING_GENERATION`, `CONTRASTIVE_ALIGNMENT`, `MULTITASK`.
- **Backend keys:** `cpu`, `cuda_single`, `cuda_distributed`, `tpu_single_host`, `tpu_multi_host`.
- **Backend statuses:** `UNTESTED`, `CPU_CONTRACT_ONLY`, `SUPPORTED_SINGLE_DEVICE`, `SUPPORTED_REPLICATED`, `SUPPORTED_SHARDED`, `BLOCKED_CUSTOM_OP`, `BLOCKED_MEMORY`, `BLOCKED_UPSTREAM`, `NOT_APPLICABLE`.
- **License statuses:** `approved_research`, `approved_commercial`, `pending_review`, `blocked_unresolved`, `rejected`.
- **Acceptance criterion statuses:** `passed`, `failed`, `blocked`, `not_applicable` (`unknown` is illegal).

## Models that must remain disabled (unresolved licenses)

`medsiglip`, `ct-fm`, `flexict-3d`, `merlin`, `m3d-lamed`, `triad`, `nv-segment-ctmr`, `brainiac`, `medsam2`, `gigapath-flash`, `medgemma-1.5-4b` are `blocked_unresolved`; `h-optimus-0`, `conch`, `titan`, `gemma-generic`, `rad-dino`, `qwen-generic` are `pending_review` (not yet approved). The Phase 05 registry must enforce: no approval ⇒ not loadable. Phase 05 must prioritize resolving `ct-fm` and `triad` (preferred backbones with unconfirmed sources).

## Mandatory run metadata for Phase 01

Phase 01's environment/reproducibility work must capture, per run (full list in `docs/reproducibility_policy.md`): git commit, dirty-tree state, lockfile hash, Python/PyTorch/CUDA-or-XLA versions, accelerator generation/device count/memory/topology, random seed, dataset-manifest hash, preprocessing-config hash, base-model revision, adapter config + hash, trainable-parameter count, precision mode, effective batch size (`microbatch × world_size × accumulation`), max allocated memory, resolved backend config hash, XLA compilation metrics (TPU).

## Accelerator tiers and v1 per-backend acceptance targets

- T0 CPU contract tests — required.
- T1 single CUDA GPU, 48 GB VRAM reference device — required.
- T2 DDP / T3 FSDP — required where hardware available.
- T4 single-host TPU — required; minimum v1 target: tiny-model coverage of all task families **plus at least one accepted real HF vision or language backbone**.
- T5 multi-host TPU — deferred until T4 accepted.
- Policy: backend support is certified per model/task/topology; any `SUPPORTED_*` claim needs recorded `smoke_config` + `last_success_date` (enforced by `tests/phase_00/test_accelerator_policy.py`).

## Policy owners and review dates

All Phase 00 documents: owner Project Maintainer (Siddhesh, acting; clinical safety officer role unfilled — see unresolved issues), review date **2026-11-02**. Per-record license review owners and dates are in `model_registry/licenses.yaml`.

## Tooling available to Phase 01

- `python -m medfm.tools.validate_phase --phase <NN>` — extend `PHASE_00_REQUIRED_FILES`-style checks per phase.
- `medfm.tools.governance` — license/acceptance/scope validators reused by tests.
- `agent/prompts/*.md` — implement/review/test/repair workflow.
- `agent/phase_template.md` — template for future phase specs.
