# Phase 12 summary

Phase 12 delivers a single resumable training engine with explicit registry preflight, typed configuration, CPU/CUDA/XLA backend boundaries, task-specific training steps, optimizer-group and freeze audits, memory planning, atomic checkpoints, evaluation, provenance, and local-first failure tracking.

## Delivered

- Typed `RunConfig` loading from YAML/JSON with canonical serialization, stable hashing, global-batch validation, precision/backend/distribution gates, static-shape policy, PEFT/quantization policy checks, freeze schedules, and explicit memory controls.
- A fixed registry → dataset → model → PEFT → optimizer → task → evaluator → trainer → checkpoint pipeline. Capability preflight and dry-run/model-summary happen before recipe model construction.
- Backend-neutral `AcceleratorBackend` implementations for CPU and CUDA plus lazy PyTorch/XLA TPU integration. Placement, autocast, accumulation, clipping/unscaling, reductions, topology, memory snapshots, XLA loaders, runtime metrics, KV-cache/gradient-checkpointing toggles, weight retie, and nonblocking pinned transfers remain behind the backend boundary.
- Shared `TrainingStep` protocol with classification, segmentation, 2D VLM, and 3D VLM family adapters; task modules retain model-output and loss semantics while the trainer owns accumulation, reductions, clipping, and optimizer boundaries.
- Explicit bridge/task-head/decoder/vision-LoRA/language-LoRA/other optimizer grouping, duplicate-parameter rejection, quantized-base rejection, gradient audits, staged freezing, scheduler preservation, and safe optimizer rebuilds.
- Deterministic distributed samplers, resumable sampler state, epoch `set_epoch` calls, bounded shape buckets, bucket warmup, backend data loading, padded-sample masking helpers, CUDA pinned/nonblocking transfer support, CUDA process-group launch, and lazy TPU PJRT/Gloo process-group launch.
- CPU/CUDA memory planners with reserved headroom, model/optimizer/gradient/activation/input estimates, ordered CUDA and TPU OOM suggestions, XLA recompilation gates, and no resource-driven scientific configuration mutation.
- Atomic resumable checkpoints containing model/components, optimizer, scheduler/scaler, RNG, sampler state, topology/runtime metadata, configuration hashes, metrics, best criterion, and static bucket schema. CUDA FSDP uses `torch.distributed.checkpoint`; TPU SPMD uses PyTorch/XLA SPMD planners; replicated TPU writes are coordinator-only. Adapter-only CPU safetensors exports are separate artifacts.
- Evaluator reductions by true sample count, NaN/Inf guards before loss collectives, failure reports with redaction and command/config context, interruption checkpoints, and run provenance including source, runtime, topology, memory, and XLA fields.
- `medfm train` dry-run/training entry points plus accelerator `validate-model`, `profile`, and one-step `parity` commands.

## Verification

- `python -m pytest tests/phase_12 -q && python -m medfm.tools.validate_phase --phase 12`: 12 passed; Phase 12 gate passed.
- CPU dry-run/model-summary: passed with `allocated: false` and no model weights allocated.
- CPU tiny classification smoke: passed with one optimizer step and resumable checkpoint.
- CUDA tiny classification smoke: passed with one optimizer step on an NVIDIA GeForce RTX 4060 Laptop GPU; peak allocated memory was recorded.
- CUDA tiny 2D and 3D segmentation smokes: each passed with one optimizer step.
- Tiny 2D and 3D VLM task-step smokes: CPU and CUDA each passed with one optimizer step through `VLMTrainingStep` and `ThreeDVLMTrainingStep`.
- CPU/CUDA tiny one-step parity: passed with absolute loss delta `0.0` at tolerance `1e-3`.
- Fresh-interpreter single-rank DCP save/restore smoke: passed; manifest storage format was `torch_distributed_checkpoint`, DCP metadata/shard files were written, model parameters matched after restore, and optimizer state restored.
- Completed-checkpoint resume, accumulation, freeze boundary, checkpoint corruption rejection, adapter export, ordered OOM diagnostics, and recompilation-gate tests pass in the isolated Phase 12 suite.

## Runtime limits

This workstation has CUDA (`torch.cuda.is_available() == True`) but does not have `torch_xla` or `accelerate` installed. TPU PJRT/SPMD execution, TPU compilation stability, TPU multi-rank reduction, multi-rank checkpoint planners, CUDA multi-device DDP/FSDP, and Accelerate-backed runtime execution are therefore recorded as not exercised rather than fabricated. The TPU and distributed paths remain explicit capability-gated code paths with CPU-contract coverage and a fresh single-rank DCP contract smoke.
