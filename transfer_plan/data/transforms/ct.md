# transfer_plan/data/transforms/ct.md

Source: `medfm/data/transforms/ct.py` (HU calibration, clipping, window channels).

Wave: 0.

## Transfer checklist

- [ ] `ToHounsfieldUnits` → **keep** — the DICOM rescale `image*slope + intercept` is a
      one-line `torch` op with units-verification policy; MONAI's `ScaleIntensityRanged` does
      not model the slope/intercept/rescale units contract. Keep.
- [ ] `ClipHU` → **keep** — a `torch.clamp`; adding a library call adds no value.
- [ ] `WindowChannels` → **partial** — this is a pure per-sample torch map
      `(clipped - center + width/2) / width`, clamped to [0,1], one channel per window. MONAI's
      `ScaleIntensityRanged(a_min=center-width/2, a_max=center+width/2, b_min=0, b_max=1,
      clip=True)` is a number-for-number equivalent and mature; a thin delegation is reasonable.
      Verify dtype (float32 native both sides) and add a parity unit test in `tests/phase_04/`.
- [ ] Intensity-transform no-inverter policy → **keep** (documented behavior; registry note).

## Tests
`tests/phase_04/test_ct.py`.
