# transfer_plan/evaluation/advanced.md

Source: `medfm/evaluation/advanced.py` (Phase 16 clinical metrics — the highest-value
transfer target in the repo).

Wave: 0 (leaf — depends only on `medfm.evaluation.metrics.MetricValue`).

## Transfer checklist

- [x] `_surface(mask)` → **keep** — thin wrapper over `scipy.ndimage.binary_erosion` (already
      library-backed). No change. **Verdict: keep — the existing SciPy kernel is already the
      mature implementation and has the required connectivity/border behavior.**
- [x] `_surface_distances(source, target, spacing)` → **partial** — the EDT itself is
      `scipy.ndimage.distance_transform_edt(..., sampling=spacing)`, already a library. A
      consolidation pass *may* call `monai.metrics.utils.get_surface_distance` for parity
      checks, but the physical-spacing variant here is scipy-native; leave the kernel.
      Add a parity test against `monai.metrics.SurfaceDistanceMetric(distance_metric="euclidean")`
      on a small synthetic volume with spacing=(1.5, 1.0, 0.5) — expect match within 1e-6.
      **Verdict: partial — retained the SciPy physical-spacing array kernel and added
      SurfaceDistanceMetric mean parity at absolute drift below 1e-6.**
- [x] `_spatial_summary(...)` (HD95, ASSD, surface-Dice, Dice, IoU) → **partial**:
      the HD95/ASSD/surface-Dice **denominators and percentile math** map to
      `monai.metrics.HausdorffDistanceMetric(percentile=95, directed=False)`,
      `SurfaceDistanceMetric` (ASSD), and `SurfaceDiceMetric(class_thresholds=[1])`.
      `DiceMetric(include_background=False, ignore_empty=False)` now supplies non-empty
      Dice; IoU is derived from the equivalent non-empty Dice identity. **Blocking caveat**:
      MONAI returns `nan` for empty prediction/target, while this function encodes the repo's
      empty-case contract (`both_empty→dice=1.0, hd95=0.0`, `one-side empty→0.0/None`).
      Therefore the empty-case branch remains hand-rolled and only non-empty kernels delegate.
      **Verdict: partial — MONAI supplies non-empty HD95/ASSD/surface-Dice/Dice/IoU while
      custom empty branches preserve the repository contract.**
- [x] `segmentation_metrics(...)` → **partial** — orchestration (empty-case rules, per-class
      unit mapping, confidence-bootstrap wiring) is contract glue → keep; internal kernels
      use the transfers above. **Verdict: partial — orchestration remains custom and invokes
      the MONAI-backed non-empty spatial helper.**
- [x] `_auroc` / `_rank_auc` / `_average_precision` → **partial**:
      `BinaryAUROC` replaces the non-degenerate rank-AUC kernel; its balanced,
      imbalanced, and tied fixtures stayed within 1e-6. Degenerate inputs retain the
      repo's `None` branch. `BinaryAveragePrecision` drifts on tied scores (>1e-6),
      so average precision remains hand-rolled; its balanced/imbalanced cases match.
      **Verdict: partial — rank-AUC transferred with parity gating; AP kept for tie parity.**
- [x] `_ece` / classification calibration → **transfer**:
      `BinaryCalibrationError(n_bins=bins, norm="l1")` matches the repository's
      fixed-width [0, 1] bins and sample-weighted L1 error within 1e-6 across
      balanced, imbalanced, tied, and degenerate fixtures; empty input remains `None`.
      **Verdict: transfer — torchmetrics preserves bin edges, weighting, and contract.**
- [x] `cluster_bootstrap_ci` / `_statistic` / BootstrapCI → **keep** — cluster bootstrap with
      deterministic seeding + `(coverage, resample)` semantics has no library equivalent
      without adding sklearn `bootstrap` or `statsmodels` (not installed). Tests:
      `tests/phase_16/test_evaluation.py::test_patient_cluster_bootstrap_is_deterministic_and_not_slice_level`.
      **Verdict: keep — deterministic cluster semantics require the existing implementation.**
- [x] `box_iou(predicted, target)` (plain python, 2D/3D half-open) → **keep** — returns a
      scalar and supports 3D; `torchvision.ops.box_iou` is batched 2D `(N,4)` only and would
      require tensor plumb-through + dtype/format normalization → violates "no forced casts".
      **Verdict: keep — torchvision cannot preserve scalar 2D/3D half-open behavior.**
- [x] `generation_metrics` `_token_f1` / `_rouge_l` → **keep** — ROUGE-L via token-table DP is
      bespoke and clinical-unit-wrapped; adding `rouge-score` (not installed) is out of scope.
      **Verdict: keep — no installed dependency matches the clinical wrapper and token DP.**

## Existing tests (must keep passing)
`tests/phase_13/test_evaluation.py`, `tests/phase_14/test_evaluation.py`,
`tests/phase_16/test_evaluation.py` (empty-mask, bootstrap determinism, localization
physical-error).

## Dependencies / sequencing
Wave 0. `metrics.py` (Wave 1) is a facade over `advanced`. No other module imports these
internals; parallel-safe.

## Result

- `_surface`: **keep**; the existing SciPy erosion kernel already matches the contract.
- `_surface_distances`: **partial**; retained SciPy's physical-spacing EDT and MONAI
  `SurfaceDistanceMetric` parity; measured mean-distance drift was `0.0` at spacing
  `(1.5, 1.0, 0.5)`.
- `_spatial_summary`: **partial**; empty branches remain custom. Non-empty HD95/ASSD/
  surface-Dice continue to delegate to MONAI, while non-empty Dice now delegates to
  `DiceMetric(include_background=False, ignore_empty=False)` and IoU is derived from
  Dice. New parity drift was `0.0` for Dice and IoU on the parity fixture.
- `segmentation_metrics`: **partial**; orchestration and clinical aggregation remain custom.
- `_rank_auc`: **transfer**; `BinaryAUROC` matched balanced, imbalanced, and tied fixtures
  within `5.30e-8` maximum observed drift (production float64 inputs were exact); degenerate
  input remains `None` (torchmetrics returns `0.0`).
- `_average_precision`: **keep**; torchmetrics matched balanced and imbalanced fixtures,
  but tied-score drift was `0.0277778`, so the hand-rolled implementation remains.
- `_ece`: **transfer**; `BinaryCalibrationError(norm="l1")` matched the repository's
  `n_bins` binning and weighting within `1.49e-8` maximum observed drift; empty input remains
  `None`.
- Cluster bootstrap/BootstrapCI, box IoU, and generation token F1/ROUGE-L: **keep**;
  deterministic cluster semantics, scalar 2D/3D half-open boxes, and bespoke clinical
  token scoring remain contract-specific.
- Files changed: `medfm/evaluation/advanced.py`,
  `tests/phase_16/test_parity_advanced_torchmetrics.py`,
  `transfer_plan/evaluation/advanced.md`.
- Verification: focused parity and existing evaluation tests, Ruff, and mypy commands
  are recorded in the worker report.
