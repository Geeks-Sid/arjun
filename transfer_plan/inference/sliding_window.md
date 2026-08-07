# transfer_plan/inference/sliding_window.md

Source: `medfm/inference/sliding_window.py` (bounded 3D sliding-window inference).

Wave: 0 (leaf — depends only on `medfm.core.errors` + `medfm.core.sample`).

## Transfer checklist

- [ ] `sliding_window_inference(...)` → **partial** — MONAI's
      `monai.inferers.sliding_window_inference(inputs, roi_size, sw_batch_size, predictor,
      overlap, mode="gaussian"|"constant", padding_mode, sigma_scale=0.125, sw_device, device)`
      is the mature, GPU/TPU-ported reference. **Verified** present in monai 1.6.0.
      How to adopt without breaking the contract:
      1. Keep the public wrapper signature (`(volume, predictor, *, window_shape, overlap,
         sw_batch_size, device)` returning `[B,C,D,H,W]`).
      2. Internally call MONAI `sliding_window_inference(..., mode="constant")` (the repo
         currently averages with constant-weight blending, not Gaussian — see test below which
         requires exact reconstruction `torch.allclose(restored, volume)`).
      3. Keep our `ShapeContractError` validation for non-positive/oversized windows and the
         `_window_starts` + flush-window logic ONLY if MONAI's roi placement differs; verify
         with the existing tests. MONAI uses `range(0, H, step)` + tail flush too, so measure
         first.
      **Parity tests required (already exist — must stay green):**
      `tests/phase_17/test_inference.py::test_sliding_window_gaussian_blending_reconstructs_volume`
      (allclose atol=1e-4 rtol=1e-5) and
      `tests/phase_07/test_native_3d.py` identity reconstruction (`torch.equal(output, volume)`
      for window 8³, overlap 0.5). If MONAI's numeric blending drifts beyond tolerance, keep the
      hand-rolled loop and tick `keep` with measured drift in `## Result`.
- [ ] `gaussian_importance_map(...)` → **partial** — MONAI's `monai.utils.gaussian_1d` builds
      the separable Gaussian (sigma_scale default 0.125, same as ours). Verify byte-identical
      weights (both use the same separable construction + `s / (2*sigma^2)` exponential and
      `clamp_min(1e-6)` lower bound). If parity holds, delegate; else keep.
- [ ] `_window_starts` / `_predict_window` → **keep** helpers regardless; they are the
      bounded-memory loop glue.

## Sequencing
Wave 0. Consumers (`inference/pipeline.py`, `models/visual/native_3d.py` has its own duplicate
copy) are Wave 1 — after this file is done, consider consolidating the duplicate native_3d
copy to reuse this one (see models/visual/native_3d.md).
