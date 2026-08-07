# transfer_plan/inference/sliding_window.md

Source: `medfm/inference/sliding_window.py` (bounded 3D sliding-window inference).

Wave: 0 (leaf — depends only on `medfm.core.errors` + `medfm.core.sample`).

## Transfer checklist

- [x] `sliding_window_inference(...)` → **keep** (MONAI candidate evaluated) — MONAI's
      `monai.inferers.sliding_window_inference(inputs, roi_size, sw_batch_size, predictor,
      overlap, mode="gaussian"|"constant", padding_mode, sigma_scale=0.125, sw_device, device)`
      is the mature, GPU/TPU-ported reference. **Verified** present in monai 1.6.0.
      The public wrapper and bounded-memory `_window_starts`/flush-window behavior remain
      unchanged because the repository contract uses Gaussian importance weighting, while
      the prescribed `mode="constant"` candidate diverges on overlapping non-identity
      predictions (maximum measured drift 97.60758972167969 on the parity fixture).
      MONAI constant identity reconstruction is exact, and the existing phase-17
      reconstruction tolerance remains green, but identity alone does not establish
      weighted-output parity.
      **Parity tests required (already exist — must stay green):**
      `tests/phase_17/test_inference.py::test_sliding_window_gaussian_blending_reconstructs_volume`
      (allclose atol=1e-4 rtol=1e-5) and
      `tests/phase_07/test_native_3d.py` identity reconstruction (`torch.equal(output, volume)`
      for window 8³, overlap 0.5). If MONAI's numeric blending drifts beyond tolerance, keep
      the hand-rolled loop and tick `keep` with measured drift in `## Result`.
- [x] `gaussian_importance_map(...)` → **keep** (MONAI candidate evaluated) — MONAI's
      documented `monai.utils.gaussian_1d` is not exported by the installed monai 1.6.0;
      its available `monai.inferers.utils.compute_importance_map(..., mode="gaussian")`
      uses a `1e-3` minimum-weight clamp rather than this repository's `1e-6` clamp.
      The measured maximum map drift is 0.0009909352520480752 for `(3, 4, 3)`, so the
      hand-rolled map is retained.
- [x] `_window_starts` / `_predict_window` → **keep** helpers regardless; they are the
      bounded-memory loop glue.
 
## Result
 
 - `sliding_window_inference`: **keep**. MONAI's constant blend reconstructs identity
   exactly but fails the repository's Gaussian weighted-output contract; measured maximum
   drift was `97.60758972167969` on the non-identity parity fixture.
 - `gaussian_importance_map`: **keep**. `monai.utils.gaussian_1d` is unavailable in
   installed monai 1.6.0; the available Gaussian map helper differs at the edge clamp,
   with maximum drift `0.0009909352520480752` for `(3, 4, 3)`.
 - `_window_starts` / `_predict_window`: **keep**. These helpers preserve the wrapper's
   bounded-memory batching, metadata fallback, mapping extraction, validation, and padding.
 - Existing implementation was intentionally left unchanged; no candidate met exact parity.
 - Files changed: `tests/phase_17/test_parity_sliding_window.py` and this checklist.
 - Verification: `uv run --frozen pytest tests/phase_17/test_parity_sliding_window.py tests/phase_17/test_inference.py tests/phase_07/test_native_3d.py` (19 passed).

## Sequencing
Wave 0. Consumers (`inference/pipeline.py`, `models/visual/native_3d.py` has its own duplicate
copy) are Wave 1 — after this file is done, consider consolidating the duplicate native_3d
copy to reuse this one (see models/visual/native_3d.md).
