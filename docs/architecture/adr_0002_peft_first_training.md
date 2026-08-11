# ADR 0002: PEFT-first training

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

`idea.md` assumes a single 48 GB GPU, confirmed as the v1 build target by the project owner (2026-08-04). (The local dev shell currently exposes an 8 GB RTX 4060 Laptop GPU — a development convenience, not the planning baseline.) v1 backbones range from 400M to multi-billion parameters, and 3D volumes and WSI tile sets are memory-hungry. Full fine-tuning of a 4B VLM does not fit in 48 GB, and full pretraining is a stated non-goal.

## Decision

Training in v1 is **PEFT-first**: frozen-encoder feature extraction, linear/head training, LoRA/QLoRA, projector/bridge training, and decoder training. Full fine-tuning of any backbone layer requires an explicit per-model exception recorded in the run config and a memory justification. Billion-parameter full pretraining is outside v1.

- Production native 2D VLM instruction tuning uses Unsloth's
  `FastVisionModel`/`UnslothVisionDataCollator` with TRL `SFTTrainer`; this
  replaces bespoke CUDA optimizer loops while retaining the PEFT-first
  freeze policy.  Custom external-encoder/bridge, 3D, and WSI routes remain
  on modality-specific trainers until an equivalent multimodal backend is
  validated.

## Alternatives considered

- **Full fine-tuning as default:** infeasible on target hardware; catastrophic-forgetting risk on small medical datasets. Rejected.
- **Prompt tuning only:** weaker for segmentation/retrieval heads; kept as an allowed PEFT method, not the default. Rejected as the sole strategy.
- **Offload-based full training (DeepSpeed ZeRO-Offload):** complexity and throughput cost not justified for v1 scope. Deferred.

## Consequences

- Canonical checkpoints are adapter-only (see ADR 0006).
- CUDA path uses NF4 QLoRA for large LMs; TPU path uses BF16 LoRA (see ADR 0009).
- Every recipe starts from frozen-encoder baselines before LoRA, and LoRA before broader unfreezing (per `implementation_plan/accelerator_training_strategy.md`).

## Reversal conditions

Reverse for a specific model if PEFT measurably fails the task acceptance threshold while a frozen-feature baseline succeeds, and hardware exists to full-tune it — record the exception in a new ADR with benchmark evidence.
