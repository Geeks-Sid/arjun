# transfer_plan/data/transforms/ct.md

Source: `medfm/data/transforms/ct.py` (HU calibration, clipping, window channels).

Wave: 0.

## Transfer checklist

- [x] `ToHounsfieldUnits` → **keep** — the DICOM rescale `image*slope + intercept` is a
      one-line `torch` op with units-verification policy; MONAI's `ScaleIntensityRanged` does
      not model the slope/intercept/rescale units contract. Keep.
- [x] `ClipHU` → **keep** — a `torch.clamp`; adding a library call adds no value.
- [x] `WindowChannels` → **partial** — delegates each float32 window to MONAI's array
      `ScaleIntensityRange` (the `ScaleIntensityRanged` symbol is dictionary-only), while
      preserving the single-channel shape check, stacking, history, optional-dependency fallback,
      and native input dtype. Parity drift was 0.0 (exact equality) on the covered float32 cases.
- [x] Intensity-transform no-inverter policy → **keep** (documented behavior; registry note).

## Result

- Transfers: `WindowChannels` partially transfers its numeric window kernel to
  `monai.transforms.ScaleIntensityRange` with `clip=True`, `[0, 1]` bounds, and dtype preserved.
- Keeps: HU slope/intercept units verification, `torch.clamp` HU clipping, and the no-inverter
  intensity policy remain custom because they encode repository-specific contracts.
- Parity: `tests/phase_04/test_parity_ct.py` measured max absolute drift `0.0` (exact tensor
  equality) against the hand-rolled window formula for float32 inputs.
- Files changed: `medfm/data/transforms/ct.py`, `tests/phase_04/test_parity_ct.py`,
  `transfer_plan/data/transforms/ct.md`.
- Verification: focused CT + MRI + parity tests passed (38 total); Ruff and strict mypy passed
  for both transform sources.

## Tests
`tests/phase_04/test_ct.py`; `tests/phase_04/test_parity_ct.py`.
