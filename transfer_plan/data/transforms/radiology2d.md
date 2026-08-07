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

- [ ] `_affine_resample(image, theta)` (grid_sample warp) → **partial/keep** — already a thin
      wrapper over `torch.nn.functional.grid_sample`; library-backed, no change.
- [ ] `RandomRotate2D` / `RandomTranslate2D` / `RandomScale2D` → **partial** — the kernels are
      exactly affine warps; `torchvision.transforms.functional.affine`/`rotate`/`resize` are
      candidates **but** they consume built-in RNG, not `ctx.rng`, and don't produce our
      inversion records. The clean split: generate the 2×3 affine from `ctx.rng` in the
      transform (as today), then call `torchvision.transforms.functional.affine` for the
      resample step. Requires verifying dtype parity (float32 tensor path is native). If the
      dtypes mismatch or drift appears, keep current grid_sample path.
- [ ] `RandomGaussianNoise` → **partial** — no torchvision equivalent for tensor Gaussian
      noise with a passed Generator; MONAI `RandGaussianNoised` uses its own RNG/dicts. Keep the
      `torch.randn(..., generator=ctx.rng)` kernel; nothing to adopt.
- [ ] `RandomFlip2D` → **partial** — `torchvision.transforms.functional.hflip`/`vflip` accept
      tensor + are dtype-preserving (float32). Gate on the same RNG caveat (axis choice still
      drawn from `ctx.rng`); the flip itself can delegate.
- [ ] `LetterboxResize` / `BodyRegionCrop` + their `_invert_*` → **keep** — inversion-aware
      crop/pad geometry is contract glue; `torchvision.transforms.functional.resize`/`crop`
      could cover the raw kernel but the asymmetric letterbox bookkeeping stays custom.
- [ ] `DecodeGrayscale`, `ToChannels`, `NormalizeImage`, `RescaleIntensity` → **keep** —
      trivial per-channel ops (no library needed / would add dependency).
- [ ] `_draw_uniform` / `_require_ctx` → **keep** (RNG contract).

## Tests
`tests/phase_04/test_radiology2d.py`, `tests/phase_04/test_end_to_end_transforms.py`.
