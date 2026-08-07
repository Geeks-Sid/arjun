# transfer_plan/data/transforms/spatial3d.md

Source: `medfm/data/transforms/spatial3d.py` (orientation, spacing resampling, foreground crop).

Wave: 0.

## Transfer checklist

Blocking context: these transforms carry `SpatialMetadata` (affines, spacing), record
invertible `TransformRecord`s, and preserve label-vs-image interpolation modes (nearest vs
trilinear). MONAI `Orientationd`/`Spacing`/`CropForegroundd` do the *numeric* job but do not
carry our affine-tracking / inversion history. Policy = delegate kernels, keep bookkeeping.

- [x] `CanonicalizeOrientation` → **partial** — permutation/flip computation behind
      `_orientation_from_affine` / `_permute_spatial` / `_spatial_flip_dims` is exactly what
      `monai.transforms.Orientation` does (RAS default; `axcodes_to_orientation` /
      `orientation_ras_lps`). **Do not** adopt wholesale (it would shortcut our
      `SpatialMetadata` affine updates); instead verify our permutation matches MONAI's on
      fixture volumes (add a parity unit test in `tests/phase_04/`). If parity holds, a private
      helper may call `monai.transforms.spatial.array.Orientation` and re-wrap; otherwise keep.
      dtype note: MONAI works on numpy/torch and preserves dtype; our path already preserves
      `torch.float32` — no forced cast needed.
- [x] `ResampleToSpacing` → **partial** — `_zoom_tensor` already uses
      `scipy.ndimage.zoom` (library-backed, order 3/0). `monai.transforms.Spacing` is the
      canonical alternative but recomputes affine/spacing its own way; ours must stay wired to
      `_updated_spatial`. Keep the scipy kernel; add a parity test vs MONAI `Spacing` on a
      tiny volume for equal output dtype/shape. Expected: keep, with parity test as documentation.
- [x] `ForegroundCrop3D` → **partial** — MONAI `CropForegroundd(..., select_fn=lambda x: x >
      threshold)` is the mature equivalent. Our version adds a fixed margin + invertibility.
      Adopt MONAI's foreground-mask computation (percentile threshold is configurable) and keep
      the margin/crop recording. Verify the empty-foreground behavior matches ours
      (empty → keep whole volume, never error).
- [x] `_zoom_tensor`, `_crop_or_pad_to`, `_permute_spatial`, `_invert_*` → **keep** — the
      inversion helpers are contract glue (mode-dependent label order).

## Result

- `CanonicalizeOrientation`: **partial transfer**. Delegates the exact MONAI
  `Orientation` array kernel when the affine orientation agrees with explicit metadata;
  retains custom permutation/flip and affine/history bookkeeping otherwise. Parity is exact
  for fixture image and `int64` mask tensors, including output affine (`max_abs=0`).
- `ResampleToSpacing`: **partial / keep kernel**. Retains the scipy `zoom` kernel and
  `_updated_spatial` geometry glue; MONAI parity fixture matched output shape and dtype.
  Numeric values are intentionally not substituted: measured max absolute value drift was
  `3.42657470703125` on the fixture.
- `ForegroundCrop3D`: **partial transfer**. Uses MONAI `CropForeground` to compute non-empty
  foreground bounds, while retaining configurable percentile thresholding, fixed margins,
  affine/history recording, inversion, and empty → whole-volume behavior. Bounds and crop
  output matched exactly on the parity fixture.
- Helpers remain **keep** because they implement inversion and label/image padding contracts.

Files changed: `medfm/data/transforms/spatial3d.py`,
`tests/phase_04/test_parity_spatial3d.py`, this checklist.

Validation:

- `uv run --frozen pytest tests/phase_04/test_parity_spatial3d.py tests/phase_04/test_ct.py tests/phase_04/test_mri.py tests/phase_04/test_end_to_end_transforms.py tests/phase_03/test_radiology_readers.py` — **50 passed, 2 skipped**.
- `uv run --frozen ruff check medfm/data/transforms/spatial3d.py tests/phase_04/test_parity_spatial3d.py` — **passed**.
- `uv run --frozen ruff format --check medfm/data/transforms/spatial3d.py tests/phase_04/test_parity_spatial3d.py` — **passed**.
- `uv run --frozen mypy medfm/data/transforms/spatial3d.py` — **passed**.

## Tests
`tests/phase_04/test_ct.py`, `tests/phase_04/test_mri.py`, `tests/phase_04/test_end_to_end_transforms.py`,
plus `tests/phase_03/test_radiology_readers.py` (spacing preservation).
