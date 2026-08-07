# transfer_plan/data/transforms/radiology2d.md

Source: `medfm/data/transforms/radiology2d.py` (2D grayscale canonicalization, geometry,
augmentation).

Wave: 0.

## Transfer checklist

Blocking context: every transform here runs on the `TransformData`/`TransformContext`
contract — stochastic ops draw only from `ctx.rng` (`torch.Generator`), and spatial ops
record an **invertible `TransformRecord`** (via module-level `register_inverter`). MONAI's
dict-transforms use their own RNG and do **not** carry our inversion history. So the transfer
policy is: **delegate the numeric kernel, keep the record/inversion + seeded-RNG glue**.

- [x] `_affine_resample(image, theta)` (grid_sample warp) → **keep** — already a thin
  wrapper over `torch.nn.functional.grid_sample`; torchvision affine has measured
  `max_abs_drift=11.670423508` on the pinned float32 candidate, so the contract kernel stays.
- [x] `RandomRotate2D` / `RandomTranslate2D` / `RandomScale2D` → **keep** — the transforms
  continue generating their 2×3 affine from `ctx.rng` and using `_affine_resample`; the
  torchvision affine candidate did not meet float32 parity (`max_abs_drift=11.670423508`).
- [x] `RandomGaussianNoise` → **keep** — no torchvision equivalent accepts the required
  `ctx.rng`; retain `torch.randn(..., generator=ctx.rng)`.
- [x] `RandomFlip2D` → **partial** — axis decisions remain drawn from `ctx.rng`, while
  torchvision `hflip`/`vflip` provide exact float32 dtype/value parity for the flip kernel.
- [x] `LetterboxResize` / `BodyRegionCrop` + their `_invert_*` → **keep** — inversion-aware
  crop/pad geometry and history bookkeeping remain custom.
- [x] `DecodeGrayscale`, `ToChannels`, `NormalizeImage`, `RescaleIntensity` → **keep** —
  trivial per-channel operations need no library delegation.
- [x] `_draw_uniform` / `_require_ctx` → **keep** — preserve the seeded `ctx.rng` contract.

## Tests
`tests/phase_04/test_radiology2d.py`, `tests/phase_04/test_end_to_end_transforms.py`.

## Result

- Transfers: `RandomFlip2D` delegates only the flip kernels to torchvision; RNG draws,
  history records, and public contracts remain unchanged.
- Keeps: affine resampling and rotate/translate/scale (candidate drift above), Gaussian
  noise, inversion-aware geometry, canonicalization/intensity operations, and RNG helpers.
- Parity: torchvision `hflip`/`vflip` exactly matched `torch.flip` for float32 tensors;
  affine candidate max absolute drift was `11.670423508` on the pinned test case.
- Files changed: `medfm/data/transforms/radiology2d.py`,
  `tests/phase_04/test_parity_radiology2d.py`, and this checklist.
- Verification: focused tests `36 passed, 2 skipped`; ruff check passed for source and
  parity test; source mypy passed; both files are formatted.
