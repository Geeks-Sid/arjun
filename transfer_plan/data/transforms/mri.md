# transfer_plan/data/transforms/mri.md

Source: `medfm/data/transforms/mri.py` (sequence resolution, normalization, stacking, N4).

Wave: 0.

## Transfer checklist

- [x] `SequenceResolver` / `select_sequences` / `stack_sequences` → **keep** — sequence
      canonicalization + channel-layout stacking is pure contract logic; no library equivalent.
- [x] `ForegroundZScoreNormalize` → **partial** — delegates qualifying foreground channel
      kernels to `monai.transforms.NormalizeIntensity(nonzero=True)` while retaining the
      `TransformRecord` glue, per-channel history, optional-dependency fallback, and the
      repository's `_EPS=1e-8` no-op floor. Parity drift was 0.0 on float32 qualifying channels.
- [x] `RobustPercentileNormalize` → **partial** — delegates percentile scaling of each
      foreground-value vector to `ScaleIntensityRangePercentiles(..., b_min=0, b_max=1,
      clip=True, relative=False)` while retaining foreground masking and the `_EPS` no-op branch.
      Parity drift was 0.0 on the covered float32 foreground case.
- [x] `apply_n4_bias_field_correction` → **keep** — the current implementation is an explicit,
      deterministic SciPy Gaussian low-pass approximation, not a SimpleITK N4 wrapper; no
      semantics-preserving library transfer was made. Its configured-only policy is retained.
- [x] `_sequence_of` helpers → **keep**.

## Result

- Transfers: foreground z-score and robust percentile numeric kernels partially transfer to
  MONAI array transforms; custom masking, epsilon handling, history, and optional-dependency
  fallback preserve the repository contract.
- Keeps: sequence resolution/selection/stacking, `_sequence_of`, and N4 remain custom. In
  particular, N4 is deliberately not changed to SimpleITK because this source currently
  documents and tests a deterministic SciPy approximation rather than exact N4 behavior.
- Parity: `tests/phase_04/test_parity_mri.py` measured max absolute drift `0.0` (exact tensor
  equality) for z-score and robust-percentile float32 covered cases; the epsilon-floor no-op
  case also remained unchanged.
- Files changed: `medfm/data/transforms/mri.py`, `tests/phase_04/test_parity_mri.py`,
  `transfer_plan/data/transforms/mri.md`.
- Verification: focused CT + MRI + parity tests passed (38 total); Ruff and strict mypy passed
  for both transform sources.

## Tests
`tests/phase_04/test_mri.py`; `tests/phase_04/test_parity_mri.py`.
