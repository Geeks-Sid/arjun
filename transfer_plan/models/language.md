# transfer_plan/models/language.md

Covers: `models/language/{base,gemma,medgemma,m3d_lamed,configs,registry}.py`.
Wave: 2 — **keep by design**.

## Transfer checklist

- [x] `GenericHFCausalLMAdapter` / `_TinyCausalLM` / `_FallbackTokenizer` → **keep** — the
      adapter already wraps `transformers` models; `_TinyCausalLM` is a deliberately small
      offline/cpu test net (torch-native). `_causal_cross_entropy` already reduces to
      `F.cross_entropy` with label masking — keep wrapper (it implements `IGNORE_INDEX`).
- [x] `GemmaCausalLMAdapter` / `MedGemmaAdapter` / `M3DLaMedAdapter` / registry → **keep** —
      pinned-weight + capability wrappers; upstream `transformers` is already the engine.


## Tests
`tests/phase_09/test_language.py`, `tests/phase_09/test_bridges.py`.

## Result

Both language adapter items verified as **keep**: Hugging Face remains the model engine while
medfm owns architecture checks, offline tiny fixtures, tokenizer fallback, masking, capability
gates, and pinned-weight registry contracts. No transfer or parity drift measured. Source and test
files were unchanged; only this checklist was updated.

Validation (shared phase run): `uv run --frozen pytest tests/phase_02 tests/phase_03 tests/phase_04
tests/phase_05 tests/phase_06 tests/phase_07 tests/phase_09` — **PASS** (622 passed, 4 skipped,
1 warning). Scoped `ruff check` — **PASS**; `ruff format --check` — **PASS** (43 files);
scoped `mypy` — **PASS**.
