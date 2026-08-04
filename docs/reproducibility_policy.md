# Reproducibility Policy

Owner: Project Maintainer (acting data-governance owner)
Review date: 2026-11-02
Status: Binding for all phases

## 1. Mandatory run metadata

Every training or evaluation run must record:

- Git commit SHA.
- Dirty working-tree state (boolean + diff hash).
- Python lockfile hash.
- Python version; PyTorch, CUDA/driver or PyTorch/XLA runtime versions.
- Accelerator generation, device count, per-device memory, topology (per `implementation_plan/accelerator_training_strategy.md`).
- Random seed (and per-rank/worker seed derivation rule).
- Dataset-manifest hash, dataset name and version.
- Preprocessing-configuration hash.
- Base-model id and pinned revision/commit SHA.
- Adapter configuration (method, rank, targets) and configuration hash.
- Trainable-parameter count.
- Precision mode (BF16/FP16/FP32, quantization method if any).
- Effective batch size: `global_batch = microbatch_per_device × world_size × accumulation_steps`.
- Maximum allocated accelerator memory.
- Resolved backend configuration hash (accelerator, distribution, precision, shape buckets).
- XLA compilation metrics for TPU runs (compile count, graph count, fallbacks).

## 2. Artifact retention

| Artifact | Retention |
|---|---|
| Run metadata (above) | Permanent (small, JSON) |
| Canonical adapter checkpoint (CPU safetensors, ADR 0006) | Permanent for accepted runs |
| Resumable distributed checkpoints | Latest 2 per active run; deleted on run acceptance |
| XLA metrics/profiler traces | 180 days for accepted TPU runs |
| Evaluation outputs | Permanent for accepted runs |
| TensorBoard/local tracker logs | 90 days for rejected/debug runs, permanent for accepted runs |

## 3. Rules

- Tracking is local-first (`LocalJSONTracker`, `TensorBoardTracker`); external hosted trackers are opt-in only (medical metadata may be sensitive).
- A run missing any mandatory metadata field fails its phase acceptance — the validation utility (`medfm.tools.validate_phase`) and Phase 16 evaluation gates check this.
- No fabricated results: reported metrics must come from recorded run artifacts.
- Checkpoints must carry a base-model reference and configuration hash; a checkpoint without them is not exportable.
