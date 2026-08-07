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
- [x] `_rouge_l` (in `medfm/evaluation/advanced.py`) → **keep** — the kernel is owned by the
      advanced-metrics transfer and its exact token-table semantics cannot be delegated from
      `generation.py`; the parity fixture is recorded in
      `tests/phase_17/test_parity_generation_rouge.py`.

## Tests
`tests/phase_17/test_inference.py`, `tests/phase_09/test_language.py`,
`tests/phase_17/test_parity_generation_rouge.py`.

## Result
- `generate`/`_call_generator`/`_decode_output`, `build_safe_prompt`, `select_length_bucket`, and
  `require_valid_output`/`GenerationResult`: **keep**; these are contract and orchestration
  glue with no drop-in library equivalent.
- `validate_json_output`: **keep** (no source change); it already parses JSON then validates
  with the installed `jsonschema.Draft202012Validator` and returns only validator names,
  preserving privacy-safe errors without raw output.
- `_rouge_l`: **keep**; the kernel lives in `medfm/evaluation/advanced.py` (owned by the
  advanced-metrics agent), so `generation.py` was not edited. For ordinary token sequences
  (`["left", "lung", "opacity"]` vs `["left", "lung"]`), RougeScorer parity was exact:
  precision/recall/F1 drift `(0, 0, 0)`. For the clinical punctuation fixture
  (`["left-lower", "lobe"]` vs `["left", "lower", "lobe"]`), repo DP
  precision/recall/F1 was `(0.5, 0.3333333333, 0.4)` while the existing
  `advanced._rouge_l` returned recall `1.0` and RougeScorer's default tokenization returned
  `(1.0, 1.0, 1.0)`, with DP-to-library drift `(0.5, 0.6666666667, 0.6)`; this exceeds
  the `1e-6` gate and prevents transfer.
- Files changed: this checklist and the parity fixture
  `tests/phase_17/test_parity_generation_rouge.py`; `medfm/inference/generation.py` was
  intentionally unchanged.
- Verification: `uv run --frozen pytest tests/phase_17/test_inference.py
  tests/phase_16/test_evaluation.py tests/phase_16/test_specialized.py` (20 passed);
  `uv run --frozen pytest tests/phase_17/test_parity_generation_rouge.py` (2 passed).

## ADR-0013 update (contract bump) — adopted

`advanced._rouge_l` now delegates the LCS to `rouge_score.RougeScorer(["rougeL"])`
with the repository `_tokens` tokenizer (a custom `tokenize()`), keeping ROUGE-L
recall number-for-number identical to the previous DP on all string fixtures
(`tests/phase_17/test_parity_generation_rouge.py` now asserts parity). The
default NLTK WordPunct tokenizer is not used. The kernel lives in
`medfm/evaluation/advanced.py`; `generation.py` remains unchanged.
