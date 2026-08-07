# transfer_plan/evaluation/calibration.md

Source: `medfm/evaluation/calibration.py` (threshold selection, histogram calibration, ECE, Brier).

Wave: 1.

## Transfer checklist

- [x] `fit_threshold` / `apply_threshold` / `_candidate_thresholds` → **keep** — operating-point
      selection on validation rows only is contract logic; no installed library matches the
      "fit on val / apply to test" split policy.
- [x] `fit_calibration` (piecewise-constant histogram calibrator) → **keep** — MONAI has no
      calibrator; sklearn's `IsotonicRegression` is **not installed** (new dep, out of scope);
      torchmetrics not installed. Keep the deterministic histogram map.
- [x] `expected_calibration_error` / `brier_score` → **keep** — same dependency situation
      (torchmetrics `CalibrationError`/`BrierScore` would be the ideal target but is not
      installed; adding it is an ADR-level choice, not part of this transfer wave). The current
      torch implementations are already correct and tested.
- [x] `ThresholdSelection` / `CalibrationModel` dataclasses → **keep** (contract types).

## Tests
`tests/phase_16/test_evaluation.py` (ECE, fit_calibration, thresholds).
## Result
Verified keep: validation-split thresholding, deterministic histogram calibration, ECE/Brier kernels, and contract dataclasses remain custom; no installed library provides the required semantics (`sklearn` and `torchmetrics` are unavailable). Source read confirmed the contracts. No parity drift measured. `uv run --frozen pytest tests/phase_16/test_evaluation.py tests/phase_16/test_specialized.py tests/phase_16/test_distributed.py` — 17 passed.
