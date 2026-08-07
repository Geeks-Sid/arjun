# transfer_plan/inference/pipeline.md

Source: `medfm/inference/pipeline.py` (runtime orchestration over sliding window + generation).
Wave: 1 (depends on `inference/sliding_window.py`, `inference/generation.py`, `training/backend.py`).

## Transfer checklist

- [x] `InferencePipeline` / `InferenceRuntime` / `BucketPolicy` → **keep** — orchestration +
      TPU bucket policy; delegates to already-transferred sliding window/generation. Only
      verify integration after Wave-0 landing.


## Result
- `InferencePipeline` / `InferenceRuntime` / `BucketPolicy`: **keep**; orchestration and TPU
  bucket policy remain bespoke, with integration verified against the transferred/kept
  sliding-window and generation paths.
- Parity drift: none observed in the focused inference tests.
- Files changed: this checklist only.
- Verification: `uv run --frozen pytest tests/phase_17/test_inference.py tests/phase_05/test_model_registry.py`
  (30 passed); `uv run --frozen ruff check medfm/inference/generation.py medfm/inference/pipeline.py
  medfm/inference/bundle.py` (passed); `uv run --frozen ruff format --check
  medfm/inference/generation.py medfm/inference/pipeline.py medfm/inference/bundle.py`
  (3 files already formatted); `uv run --frozen mypy medfm/inference/generation.py
  medfm/inference/pipeline.py medfm/inference/bundle.py` (passed).