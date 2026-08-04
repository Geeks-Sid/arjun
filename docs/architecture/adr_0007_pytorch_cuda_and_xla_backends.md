# ADR 0007: PyTorch with CUDA and PyTorch/XLA backends — no separate GPU/TPU codebases

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

v1 must train on CUDA GPUs and Google Cloud TPUs. Two options: maintain separate GPU and TPU codepaths, or one PyTorch codebase with an accelerator-backend abstraction.

## Decision

One PyTorch codebase. An `AcceleratorBackend` interface (owned by the training layer) has `CpuBackend`, `CudaBackend`, and `XlaTpuBackend` implementations. Hugging Face Accelerate handles common placement/accumulation/mixed-precision where verified; direct PyTorch/XLA APIs live behind `XlaTpuBackend` for compilation metrics, SPMD, and XLA checkpointing. Application code must not call `.cuda()`, `torch.cuda.current_device()`, or construct tensors on hard-coded devices; CUDA-only and XLA-only imports are isolated in backend/capability modules. Model, task, loss, and metric APIs stay ordinary backend-neutral PyTorch.

## Alternatives considered

- **Separate TPU codebase (e.g. JAX/Flax port):** doubles the surface area, splits the model roster (most medical checkpoints are PyTorch), and breaks the shared-contract goal. Rejected.
- **Accelerate-only abstraction:** hides XLA compilation metrics and SPMD controls we explicitly need for acceptance. Rejected as the sole mechanism.
- **Keras/multi-framework:** no; ecosystem is PyTorch-native for the chosen backbones. Rejected.

## Consequences

- Backend parity tests (CPU FP32 vs CUDA BF16 vs TPU BF16) are part of acceptance.
- Backend support is certified per model/task/topology — never assumed framework-wide.
- TPU-specific constraints (static shapes, ADR 0008) are expressed in config, not code forks.

## Reversal conditions

Reverse if a critical v1 model proves unportable to PyTorch/XLA and is TPU-mandatory; then mark it `BLOCKED_UPSTREAM` for TPU (per the capability matrix) rather than forking the codebase. Full reversal requires evidence that PyTorch/XLA cannot meet the v1 TPU target at all.
