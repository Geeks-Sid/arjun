# transfer_plan/evaluation/calibration.md

Source: `medfm/evaluation/calibration.py` (threshold selection, histogram calibration, ECE, Brier).

Wave: 1.

## Transfer checklist

- [x] `fit_threshold` / `apply_threshold` / `_candidate_thresholds` → **keep** — operating-point
      selection on validation rows only is contract logic; no installed library matches the
      "fit on val / apply to test" split policy.
- [x] `fit_calibration` (piecewise-constant histogram calibrator) → **keep** — the
      deterministic histogram map is test-pinned; sklearn's `IsotonicRegression` has
      different fitting semantics and is explicitly not adopted.
- [x] `expected_calibration_error` → **transfer** — `torchmetrics.classification.BinaryCalibrationError`
      with equal-width bins and `norm="l1"` matches the repository's weighting and boundary
      semantics across parity fixtures (including empty bins and degenerate predictions).
- [x] `brier_score` → **keep** — torchmetrics 1.9 removed/deprecated `BrierScore`; retain
      the hand-rolled torch operation rather than inventing a library substitute.
- [x] `ThresholdSelection` / `CalibrationModel` dataclasses → **keep** (contract types).

## Tests
`tests/phase_16/test_evaluation.py` and
`tests/phase_16/test_parity_calibration_torchmetrics.py` (ECE, fit_calibration, thresholds).

## Result
Transferred `expected_calibration_error` to `BinaryCalibrationError(n_bins=bins, norm="l1")`;
parity fixtures covered equal-width bin boundaries, weighting, empty bins, all-0 predictions,
all-1 predictions, and empty input contract (`None` vs torchmetrics `nan`). Maximum measured
drift was 0.0 in the executed fixtures. Kept `brier_score` because torchmetrics 1.9 has no
`BrierScore`; kept histogram calibration, threshold helpers, and contract dataclasses because
their semantics are custom/test-pinned. Files changed: `medfm/evaluation/calibration.py`,
`tests/phase_16/test_parity_calibration_torchmetrics.py`, and this checklist.
