# Medical Foundation-Model Framework Implementation Plan

This directory turns `idea.md` into executable, phase-gated work. Each phase has one primary objective, explicit dependencies, implementation and verification checklists, primary implementation sources, and a required handoff.

Start with:

- [Accelerator training strategy](accelerator_training_strategy.md) for the mandatory CPU/CUDA/TPU execution contract.
- [Primary implementation references](references.md) for upstream documentation, model cards, and repositories.

## Operating assumptions

- [ ] Optimize the first CUDA path for one Linux workstation with one NVIDIA GPU and 48 GB VRAM.
- [ ] Support single/multi-GPU training and Google Cloud TPU training through the accelerator contract.
- [ ] Use PyTorch, MONAI, Hugging Face Transformers, PEFT, TRL, Accelerate, and bitsandbytes.
- [ ] Prefer frozen encoders, projector/head training, and LoRA over full-model training.
- [ ] Use QLoRA only on a verified quantization backend; use BF16 LoRA/frozen-model strategies on TPU until TPU quantization is separately accepted.
- [ ] Keep clinical data de-identified before it reaches this framework.
- [ ] Keep every model modality-specific internally while enforcing shared framework contracts.
- [ ] Treat all generated clinical output as research output until task-specific validation is complete.
- [ ] Keep model weights, patient data, secrets, and gated-license tokens out of Git.

## Phase order

- [ ] [Phase 00: Requirements and governance](phase_00_requirements_and_governance.md).
- [ ] [Phase 01: Repository and environment](phase_01_repository_and_environment.md).
- [ ] [Phase 02: Core type system and contracts](phase_02_core_types_and_contracts.md).
- [ ] [Phase 03: Dataset manifests and ingestion](phase_03_data_ingestion_and_provenance.md).
- [ ] [Phase 04: Preprocessing and collators](phase_04_preprocessing_and_collators.md).
- [ ] [Phase 05: Model registry and weight management](phase_05_model_registry_and_weights.md).
- [ ] [Phase 06: 2D visual adapters](phase_06_2d_visual_adapters.md).
- [ ] [Phase 07: Native 3D visual adapters](phase_07_native_3d_adapters.md).
- [ ] [Phase 08: Pathology and WSI adapters](phase_08_pathology_and_wsi_adapters.md).
- [ ] [Phase 09: Language models and visual bridges](phase_09_language_models_and_bridges.md).
- [ ] [Phase 10: LoRA, QLoRA, and PEFT](phase_10_peft_and_quantization.md).
- [ ] [Phase 11: Task heads, decoders, and losses](phase_11_task_modules_and_losses.md).
- [ ] [Phase 12: Unified trainer and memory planner](phase_12_training_engine_and_memory.md).
- [ ] [Phase 13: 2D task recipes](phase_13_2d_recipes.md).
- [ ] [Phase 14: 3D task recipes](phase_14_3d_recipes.md).
- [ ] [Phase 15: Pathology task recipes](phase_15_pathology_recipes.md).
- [ ] [Phase 16: Evaluation and clinical validation](phase_16_evaluation_and_validation.md).
- [ ] [Phase 17: Inference, export, and serving](phase_17_inference_export_and_serving.md).
- [ ] [Phase 18: CI, hardening, and release](phase_18_ci_hardening_and_release.md).

Phases 06, 07, and 08 may proceed in parallel after Phase 05. Phase 09 may begin only after each adapter family has at least one accepted implementation. Later recipe phases may be developed incrementally, but their final acceptance depends on the shared trainer and task modules.

## Accelerator support summary

| Training capability | CUDA GPU baseline | TPU baseline |
| --- | --- | --- |
| Precision | BF16 preferred; FP16 with scaling when needed | XLA BF16; FP32-sensitive reductions |
| LoRA | Supported after module audit | Supported in BF16 after XLA operator audit |
| QLoRA | bitsandbytes NF4 on a supported CUDA build | Not part of baseline; do not use bitsandbytes |
| Replicated training | DDP/Accelerate | PJRT/XLA replicated execution |
| Sharded training | PyTorch FSDP | PyTorch/XLA SPMD/FSDP after replicated acceptance |
| Attention | SDPA default; optional tested FlashAttention | XLA-lowered PyTorch attention/SDPA path |
| Shapes | Dynamic where safe, bounded by recipe | Fixed documented buckets with masks |
| Checkpoints | Portable adapter export plus DCP for sharded resume | Portable adapter export plus XLA DCP planners for sharded resume |

This matrix describes framework policy. A model is supported only after its registry record links to a successful hardware report.

## Standard phase gate

Every phase is complete only when all of the following are true:

- [ ] The phase objective and non-goals remain unchanged or an ADR records the change.
- [ ] Only the phase's allowed areas were modified.
- [ ] Required code, configuration, tests, and documentation exist.
- [ ] Unit tests pass with no unexplained skips.
- [ ] Integration tests pass where required.
- [ ] The smoke command passes.
- [ ] The acceptance command passes.
- [ ] License and data-provenance checks pass where applicable.
- [ ] `agent/reports/phase_<NN>/` contains the required report files.
- [ ] `acceptance.json` has no `unknown` result.
- [ ] Remaining issues are recorded and do not violate exit criteria.
- [ ] `next_phase_handoff.md` describes outputs, interfaces, and known constraints.
- [ ] The model/task accelerator matrix is updated with CPU, CUDA, and TPU evidence.
- [ ] No phase claims TPU support based only on a CUDA test or source-level portability.

## Required report artifacts

Each phase must produce:

```text
agent/reports/phase_<NN>/
├── summary.md
├── files_changed.txt
├── commands_executed.txt
├── test_results.json
├── acceptance.json
├── unresolved_issues.md
└── next_phase_handoff.md
```

## Standard commands

Once the phase validator exists, every phase should expose commands equivalent to:

```bash
python -m medfm.tools.validate_phase --phase <NN>
pytest tests/phase_<NN> -q
python -m medfm.tools.smoke --phase <NN>
```

If a phase needs model weights or a GPU, keep CPU contract tests separate and mark real-checkpoint tests with an explicit, documented marker. A skipped hardware test is acceptable only when the acceptance manifest names the missing capability and the protected GPU acceptance job is still required before release.

TPU acceptance is also a protected hardware test. It must record static-shape buckets, compile count, unsupported-op/fallback counters, steady-state throughput, topology, and a portable checkpoint/export result.

## Delivery milestones

- [ ] Milestone 0, skeleton: Phases 00-02.
- [ ] Milestone 1, first 2D classification: key parts of Phases 03-06, 10-13, and 16.
- [ ] Milestone 2, first 3D classification: key parts of Phases 03-07, 10-12, 14, and 16.
- [ ] Milestone 3, first 3D segmentation: Phases 04, 07, 11, 12, 14, and 16.
- [ ] Milestone 4, native 2D VLM: Phases 09, 10, 12, 13, and 16.
- [ ] Milestone 5, external 2D VLM: Phases 06, 09, 10, 12, 13, and 16.
- [ ] Milestone 6, native 3D VLM: Phases 07, 09, 10, 12, 14, and 16.
- [ ] Milestone 7, slice-sequence VLM: Phases 04, 09, 12, 14, and 16.
- [ ] Milestone 8, pathology: Phases 03, 04, 08, 12, 15, and 16.
- [ ] Milestone 9, multitask release: all phases through 18.
