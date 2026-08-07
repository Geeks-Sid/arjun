# transfer_plan/models/visual/native_3d.md

Source: `medfm/models/visual/native_3d.py` (native 3D adapter, MONAI-3D backbone, its own
sliding-window copy).
Wave: 0.

## Transfer checklist

- [x] `sliding_window_inference` (duplicate copy in this module) → **keep** — the
      shared `medfm.inference.sliding_window.sliding_window_inference` is not
      numerically or behaviorally equivalent: it applies Gaussian blending,
      zero-pads undersized windows, and flattens windows into predictor batches.
      The native adapter's unpadded overlap-average loop is required for the
      existing exact `torch.equal` identity reconstruction contract, so
      consolidating it would change observable behavior.
- [x] `_LocalMONAI3DBackbone` (small transformer over volume patches) → **keep** —
      hand-rolled patch-embed + positional tokens + transformer blocks. MONAI
      `monai.networks.blocks.TransformerBlock` / `MLPBlock` are verified and could
      replace the internal attention/MLP blocks **if** a parity test (deterministic
      init, same seed) passes on CPU/CUDA. Given it deliberately avoids MONAI-heavy
      operators for backend-neutrality ("no MONAI/CUDA-only operators"), likely
      keep — do not over-engineer.
- [x] `GenericMONAI3DAdapter` (preprocess contract, hidden-state extraction, output
      spec) → **keep** — adapter contract glue.
- [x] `Native3DPreprocess` / checkpoint provenance → **keep**.
 
## Result

- `sliding_window_inference`: **keep**. The shared helper's Gaussian weighting,
  padding, and flattened predictor batching differ from the native adapter's
  exact unpadded overlap-average contract; the local implementation remains
  intentionally documented rather than duplicated accidentally. Measured shared
  helper drift for the required identity case (`20×18×16`, window `8³`,
  overlap `0.5`) was `max_abs=0.001220703125` and `2368` unequal voxels.
- `_LocalMONAI3DBackbone`: **keep** for backend neutrality; no MONAI-heavy or
  CUDA-only operators introduced.
- `GenericMONAI3DAdapter`: **keep** adapter contract glue.
- `Native3DPreprocess` / checkpoint provenance: **keep**.
- Files changed: `medfm/models/visual/native_3d.py`,
  `transfer_plan/models/visual/native_3d.md`.

## Tests
`tests/phase_07/test_native_3d.py`, `tests/phase_14/test_recipes.py`.
