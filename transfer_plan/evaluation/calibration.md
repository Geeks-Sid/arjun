# transfer_plan/evaluation/calibration.md

Source: `medfm/evaluation/calibration.py` (threshold selection, histogram calibration, ECE, Brier).

Wave: 1.

## Transfer checklist

- [ ] `fit_threshold` / `apply_threshold` / `_candidate_thresholds` → **keep** — operating-point
      selection on validation rows only is contract logic; no installed library matches the
      "fit on val / apply to test" split policy.
- [ ] `fit_calibration` (piecewise-constant histogram calibrator) → **keep** — MONAI has no
      calibrator; sklearn's `IsotonicRegression` is **not installed** (new dep, out of scope);
      torchmetrics not installed. Keep the deterministic histogram map.
- [ ] `expected_calibration_error` / `brier_score` → **keep** — same dependency situation
      (torchmetrics `CalibrationError`/`BrierScore` would be the ideal target but is not
      installed; adding it is an ADR-level choice, not part of this transfer wave). The current
      torch implementations are already correct and tested.
- [ ] `ThresholdSelection` / `CalibrationModel` dataclasses → **keep** (contract types).

## Tests
`tests/phase_16/test_evaluation.py` (ECE, fit_calibration, thresholds).
