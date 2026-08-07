# transfer_plan/models/visual/native_3d.md

Source: `medfm/models/visual/native_3d.py` (native 3D adapter, MONAI-3D backbone, its own
sliding-window copy).
Wave: 0.

## Transfer checklist

- [ ] `sliding_window_inference` (duplicate copy in this module) → **partial** — this file
      carries a **second copy** of the sliding-window loop already planned in
      `inference/sliding_window.md`. After that Wave-0 file lands, consolidate this duplicate to
      reuse `medfm.inference.sliding_window.sliding_window_inference` (or the shared MONAI
      delegation) instead of maintaining two loops. Gate on the same parity tests.
- [ ] `_LocalMONAI3DBackbone` (small transformer over volume patches) → **partial/keep** —
      hand-rolled patch-embed + positional tokens + transformer blocks. MONAI
      `monai.networks.blocks.TransformerBlock` / `MLPBlock` are verified and could replace the
      internal attention/MLP blocks **if** a parity test (deterministic init, same seed) passes
      on CPU/CUDA. Given it deliberately avoids MONAI-heavy operators for backend-neutrality
      ("no MONAI/CUDA-only operators"), likely **keep** — do not over-engineer.
- [ ] `GenericMONAI3DAdapter` (preprocess contract, hidden-state extraction, output spec) →
      **keep** — adapter contract glue.
- [ ] `Native3DPreprocess` / checkpoint provenance → **keep**.

## Tests
`tests/phase_07/test_native_3d.py`, `tests/phase_14/test_recipes.py`.
