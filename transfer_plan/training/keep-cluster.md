# transfer_plan/training/keep-cluster.md

Covers: `training/{memory,checkpoint,distributed,data,steps,trainer,pipeline,evaluation,tracking,run_metadata,config}.py`.
Wave: 2 — **keep by design** (orchestration/contract; no library drop-in without a rewrite).

## Transfer checklist

- [x] `memory.py` (`MemoryPlanner`, CUDA/TPU planners, CompilationMonitor) → **keep** —
      accelerator-aware budgets are policy; CUDA planner wraps `torch.cuda.memory` APIs
      (library-backed) behind a backend-neutral face.
- [x] `checkpoint.py` (`CheckpointManager`, DCP paths) → **keep** — already delegates to
      `torch.distributed.checkpoint` + `safetensors` (import-verified); the atomic-rename,
      RNG-state capture/restore, and schema-version manifest are contract glue.
- [x] `distributed.py` / `data.py` (`DeterministicDistributedSampler`, `BackendDataLoader`,
      `ShapeBucketPlan`) → **keep** — group-aware, resumable, static-shape semantics exceed
      `monai.data.DistributedSampler`/torch defaults.
- [x] `steps.py` / `trainer.py` / `pipeline.py` / `evaluation.py` → **keep** — the training
      loop is intentionally backend-neutral (CPU/CUDA/XLA) and TPU-bucket-aware; swapping for
      `accelerate.Trainer`/HF trainer violates ground rule §1.3 (and would pull new dtype
      handling). The loop already calls torch-native ops; nothing to adopt at leaf level.
- [x] `tracking.py` / `run_metadata.py` → **keep** — TensorBoard wrapper already library; JSON
      tracker / metadata capture are reproducibility-contract glue (pinned by
      `tests/phase_01/test_tracking.py`, `test_run_metadata.py`).

## Result
- **Keep** memory planning and compilation monitoring: backend-aware memory budgets and CUDA/TPU policy are contract decisions, with CUDA memory APIs already library-backed.
- **Keep** checkpoint orchestration: `CheckpointManager` owns atomic rename, RNG state, schema manifest, and adapter/component contracts around torch/safetensors persistence.
- **Keep** distributed/data helpers: `DeterministicDistributedSampler`, `BackendDataLoader`, and `ShapeBucketPlan` preserve resumable and TPU static-shape semantics not covered by generic samplers.
- **Keep** steps/trainer/pipeline/evaluation: the orchestration remains backend-neutral across CPU/CUDA/XLA and cannot be replaced by `accelerate.Trainer` without changing contracts and dtype handling.
- **Keep** tracking/run metadata: TensorBoard is already wrapped and JSON/reproducibility metadata are glue contracts.
- **Parity drift:** none applicable; no library replacement was attempted.
- **Source reads:** all 12 assigned `medfm/training` modules confirmed backend-neutral orchestration, accelerator policy, persistence contracts, and library-backed leaf calls.
- **Verification:** `uv run --frozen pytest tests/phase_12/test_trainer_memory_checkpoint.py tests/phase_12/test_config_backend.py` (12 passed); `uv run --frozen pytest tests/phase_01` (27 passed, 6 skipped); `uv run --frozen ruff check` on all 12 assigned training sources (pass); `uv run --frozen ruff format --check` on all 12 assigned training sources (12 already formatted); `uv run --frozen mypy` on all 12 assigned training sources (pass). The checklist's `tests/phase_10/test_config_backend.py` path does not exist.

## Tests
`tests/phase_10/test_config_backend.py`, `tests/phase_12/test_trainer_memory_checkpoint.py`,
`tests/phase_12/test_config_backend.py`, `tests/phase_01/*`.
