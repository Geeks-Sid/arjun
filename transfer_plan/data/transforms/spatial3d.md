# transfer_plan/data/transforms/spatial3d.md

Source: `medfm/data/transforms/spatial3d.py` (orientation, spacing resampling, foreground crop).

Wave: 0.

## Transfer checklist

Blocking context: these transforms carry `SpatialMetadata` (affines, spacing), record
invertible `TransformRecord`s, and preserve label-vs-image interpolation modes (nearest vs
trilinear). MONAI `Orientationd`/`Spacing`/`CropForegroundd` do the *numeric* job but do not
carry our affine-tracking / inversion history. Policy = delegate kernels, keep bookkeeping.

- [ ] `CanonicalizeOrientation` → **partial** — permutation/flip computation behind
      `_orientation_from_affine` / `_permute_spatial` / `_spatial_flip_dims` is exactly what
      `monai.transforms.Orientation` does (RAS default; `axcodes_to_orientation` /
      `orientation_ras_lps`). **Do not** adopt wholesale (it would shortcut our
      `SpatialMetadata` affine updates); instead verify our permutation matches MONAI's on
      fixture volumes (add a parity unit test in `tests/phase_04/`). If parity holds, a private
      helper may call `monai.transforms.spatial.array.Orientation` and re-wrap; otherwise keep.
      dtype note: MONAI works on numpy/torch and preserves dtype; our path already preserves
      `torch.float32` — no forced cast needed.
- [ ] `ResampleToSpacing` → **partial** — `_zoom_tensor` already uses
      `scipy.ndimage.zoom` (library-backed, order 3/0). `monai.transforms.Spacing` is the
      canonical alternative but recomputes affine/spacing its own way; ours must stay wired to
      `_updated_spatial`. Keep the scipy kernel; add a parity test vs MONAI `Spacing` on a
      tiny volume for equal output dtype/shape. Expected: keep, with parity test as documentation.
- [ ] `ForegroundCrop3D` → **partial** — MONAI `CropForegroundd(..., select_fn=lambda x: x >
      threshold)` is the mature equivalent. Our version adds a fixed margin + invertibility.
      Adopt MONAI's foreground-mask computation (percentile threshold is configurable) and keep
      the margin/crop recording. Verify the empty-foreground behavior matches ours
      (empty → keep whole volume, never error).
- [ ] `_zoom_tensor`, `_crop_or_pad_to`, `_permute_spatial`, `_invert_*` → **keep** — the
      inversion helpers are contract glue (mode-dependent label order).

## Tests
`tests/phase_04/test_ct.py`, `tests/phase_04/test_mri.py`, `tests/phase_04/test_end_to_end_transforms.py`,
plus `tests/phase_03/test_radiology_readers.py` (spacing preservation).
