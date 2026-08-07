# transfer_plan/inference/generation.md

Source: `medfm/inference/generation.py` (bounded VLM decoding, prompt isolation, structured-output
validation).
Wave: 1.

## Transfer checklist

- [ ] `generate` / `_call_generator` / `_decode_output` → **keep** — already wraps the HF
      `model.generate(...)` API (asserted by `_call_generator`); the bounded-decoding loop,
      length-bucket enforcement, and `GenerationConfig` defaults are contract glue. Only
      `generation_params` filtering is ours. No transfer.
- [ ] `build_safe_prompt` (system/user/report separation) → **keep** — prompt-hygiene policy;
      no library.
- [ ] `select_length_bucket` → **keep** — TPU static-shape length buckets; MONAI/torch have
      nothing for this.
- [ ] `validate_json_output` → **partial/keep** — the JSON validation uses `jsonschema`
      (installed); verify it already does. If it hand-parses against a schema, delegate to
      `jsonschema.Draft202012Validator` (already a dependency) keeping the privacy-safe error
      reporting (no raw input in errors). Likely already using it.
- [ ] `require_valid_output` / `GenerationResult` → **keep**.

## Tests
`tests/phase_17/test_inference.py`, `tests/phase_09/test_language.py`.
