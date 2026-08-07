# transfer_plan/inference/generation.md

Source: `medfm/inference/generation.py` (bounded VLM decoding, prompt isolation, structured-output
validation).
Wave: 1.

## Transfer checklist

- [x] `generate` / `_call_generator` / `_decode_output` → **keep** — already wraps the HF
      `model.generate(...)` API (asserted by `_call_generator`); the bounded-decoding loop,
      length-bucket enforcement, and `GenerationConfig` defaults are contract glue. Only
      `generation_params` filtering is ours. No transfer.
- [x] `build_safe_prompt` (system/user/report separation) → **keep** — prompt-hygiene policy;
      no library.
- [x] `select_length_bucket` → **keep** — TPU static-shape length buckets; MONAI/torch have
      nothing for this.
- [x] `validate_json_output` → **keep** — it already parses JSON then validates with the
      installed `jsonschema.Draft202012Validator`, returning privacy-safe validator names
      rather than raw input in errors.
- [x] `require_valid_output` / `GenerationResult` → **keep**.

## Tests
`tests/phase_17/test_inference.py`, `tests/phase_09/test_language.py`.

## Result
- `generate`/`_call_generator`/`_decode_output`, `build_safe_prompt`, `select_length_bucket`, and
  `require_valid_output`/`GenerationResult`: **keep**; these are contract and orchestration
  glue with no drop-in library equivalent.
- `validate_json_output`: **keep** (no source change); it already parses JSON then validates
  with the installed `jsonschema.Draft202012Validator` and returns only validator names,
  preserving privacy-safe errors without raw output.
- Parity drift: none measured; direct valid/invalid smoke check passed.
- Files changed: this checklist only.
- Verification: `uv run --frozen pytest tests/phase_17/test_inference.py tests/phase_05/test_model_registry.py`
  (30 passed); `uv run --frozen pytest tests/phase_09/test_language.py` (6 passed);
  `uv run --frozen ruff check medfm/inference/generation.py medfm/inference/pipeline.py
  medfm/inference/bundle.py` (passed); `uv run --frozen ruff format --check
  medfm/inference/generation.py medfm/inference/pipeline.py medfm/inference/bundle.py`
  (3 files already formatted); `uv run --frozen mypy medfm/inference/generation.py
  medfm/inference/pipeline.py medfm/inference/bundle.py` (passed).
