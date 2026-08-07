# transfer_plan/data/transforms/mri.md

Source: `medfm/data/transforms/mri.py` (sequence resolution, normalization, stacking, N4).

Wave: 0.

## Transfer checklist

- [ ] `SequenceResolver` / `select_sequences` / `stack_sequences` → **keep** — sequence
      canonicalization + channel-layout stacking is pure contract logic; no library equivalent.
- [ ] `ForegroundZScoreNormalize` → **partial** — z-score over foreground (nonzero) voxels
      matches `monai.transforms.NormalizeIntensityd(..., nonzero=True, channel_wise=True)`.
      Verify parity (mean/std over nonzero only, epsilon floor `_EPS=1e-8`; MONAI uses
      `nonzero=True` semantics). If matched, delegate the numeric kernel; keep the
      `TransformRecord` glue. Dtype: float32 tensor native both sides.
- [ ] `RobustPercentileNormalize` → **partial** — clip to percentiles then scale to [0,1]
      maps to `monai.transforms.ScaleIntensityRangePercentiles(..., lower, upper, b_min=0,
      b_max=1, clip=True, relative=False)`. Add a parity test before delegating.
- [ ] `apply_n4_bias_field_correction` → **partial** — N4 is inherently
      `SimpleITK.N4BiasFieldCorrectionImageFilter`-based (a library already); verify the
      wrapper is already delegating and only the configured-only gating is ours. If it invokes
      SimpleITK already, tick `keep` (correct as-is).
- [ ] `_sequence_of` helpers → **keep**.

## Tests
`tests/phase_04/test_mri.py`.
