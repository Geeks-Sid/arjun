# transfer_plan/inference/pipeline.md

Source: `medfm/inference/pipeline.py` (runtime orchestration over sliding window + generation).
Wave: 1 (depends on `inference/sliding_window.py`, `inference/generation.py`, `training/backend.py`).

## Transfer checklist

- [ ] `InferencePipeline` / `InferenceRuntime` / `BucketPolicy` → **keep** — orchestration +
      TPU bucket policy; delegates to already-transferred sliding window/generation. Only
      verify integration after Wave-0 landing.
