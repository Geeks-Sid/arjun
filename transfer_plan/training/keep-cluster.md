# transfer_plan/training/keep-cluster.md

Covers: `training/{memory,checkpoint,distributed,data,steps,trainer,pipeline,evaluation,tracking,run_metadata,config}.py`.
Wave: 2 — **keep by design** (orchestration/contract; no library drop-in without a rewrite).

## Transfer checklist

- [ ] `memory.py` (`MemoryPlanner`, CUDA/TPU planners, CompilationMonitor) → **keep** —
      accelerator-aware budgets are policy; CUDA planner wraps `torch.cuda.memory` APIs
      (library-backed) behind a backend-neutral face.
- [ ] `checkpoint.py` (`CheckpointManager`, DCP paths) → **keep** — already delegates to
      `torch.distributed.checkpoint` + `safetensors` (import-verified); the atomic-rename,
      RNG-state capture/restore, and schema-version manifest are contract glue.
- [ ] `distributed.py` / `data.py` (`DeterministicDistributedSampler`, `BackendDataLoader`,
      `ShapeBucketPlan`) → **keep** — group-aware, resumable, static-shape semantics exceed
      `monai.data.DistributedSampler`/torch defaults.
- [ ] `steps.py` / `trainer.py` / `pipeline.py` / `evaluation.py` → **keep** — the training
      loop is intentionally backend-neutral (CPU/CUDA/XLA) and TPU-bucket-aware; swapping for
      `accelerate.Trainer`/HF trainer violates ground rule §1.3 (and would pull new dtype
      handling). The loop already calls torch-native ops; nothing to adopt at leaf level.
- [ ] `tracking.py` / `run_metadata.py` → **keep** — TensorBoard wrapper already library; JSON
      tracker / metadata capture are reproducibility-contract glue (pinned by
      `tests/phase_01/test_tracking.py`, `test_run_metadata.py`).

## Tests
`tests/phase_10/test_config_backend.py`, `tests/phase_12/test_trainer_memory_checkpoint.py`,
`tests/phase_12/test_config_backend.py`, `tests/phase_01/*`.
