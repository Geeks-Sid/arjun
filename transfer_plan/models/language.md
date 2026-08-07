# transfer_plan/models/language.md

Covers: `models/language/{base,gemma,medgemma,m3d_lamed,configs,registry}.py`.
Wave: 2 — **keep by design**.

## Transfer checklist

- [ ] `GenericHFCausalLMAdapter` / `_TinyCausalLM` / `_FallbackTokenizer` → **keep** — the
      adapter already wraps `transformers` models; `_TinyCausalLM` is a deliberately small
      offline/cpu test net (torch-native). `_causal_cross_entropy` already reduces to
      `F.cross_entropy` with label masking — keep wrapper (it implements `IGNORE_INDEX`).
- [ ] `GemmaCausalLMAdapter` / `MedGemmaAdapter` / `M3DLaMedAdapter` / registry → **keep** —
      pinned-weight + capability wrappers; upstream `transformers` is already the engine.

## Tests
`tests/phase_09/test_language.py`, `tests/phase_09/test_bridges.py`.
