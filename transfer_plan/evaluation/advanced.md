# transfer_plan/evaluation/advanced.md

Source: `medfm/evaluation/advanced.py` (Phase 16 clinical metrics — the highest-value
transfer target in the repo).

Wave: 0 (leaf — depends only on `medfm.evaluation.metrics.MetricValue`).

## Transfer checklist

- [ ] `_surface(mask)` → **keep** — thin wrapper over `scipy.ndimage.binary_erosion` (already
      library-backed). No change.
- [ ] `_surface_distances(source, target, spacing)` → **partial** — the EDT itself is
      `scipy.ndimage.distance_transform_edt(..., sampling=spacing)`, already a library. A
      consolidation pass *may* call `monai.metrics.utils.get_surface_distance` for parity
      checks, but the physical-spacing variant here is scipy-native; leave the kernel.
      Add a parity test against `monai.metrics.SurfaceDistanceMetric(distance_metric="euclidean")`
      on a small synthetic volume with spacing=(1.5, 1.0, 0.5) — expect match within 1e-6.
- [ ] `_spatial_summary(...)` (HD95, ASSD, surface-Dice, Dice, IoU) → **partial**:
      the HD95/ASSD/surface-Dice **denominators and percentile math** map to
      `monai.metrics.HausdorffDistanceMetric(percentile=95, directed=False)`,
      `SurfaceDistanceMetric` (ASSD), and `SurfaceDiceMetric(class_thresholds=[1])`.
      **Blocking caveat**: MONAI returns `nan` for empty prediction/target, while this
      function encodes the repo's empty-case contract (`both_empty→dice=1.0, hd95=0.0`,
      `one-side empty→0.0/None`). Therefore: keep the empty-case branch hand-rolled; delegate
      ONLY the non-empty distance computation to MONAI in a private helper, then map MONAI's
      `nan`/`inf` to the repo contract. Add a **parity test**:
      `tests/phase_16/test_evaluation.py::test_empty_segmentation_and_physical_localization_metrics`
      already pins `hd95==0.0` for both-empty and `assd==0.0` for perfect — run it after the
      swap; add a new test comparing our HD95/ASSD against MONAI on a non-empty two-object case.
- [ ] `segmentation_metrics(...)` → **partial** — orchestration (empty-case rules, per-class
      unit mapping, confidence-bootstrap wiring) is contract glue → keep; internal kernels
      use the transfers above.
- [ ] `_auroc` / `_rank_auc` / `_average_precision` → **keep** — `sklearn`/`torchmetrics` not
      installed. MONAI has `monai.metrics.ROCAUCMetric`/`monai.metrics.auc` only for ROC-AUC,
      no PR-AUC; not worth a new dep for one metric. Keep hand-rolled (it is already a closed-form
      rank computation, verified consistent with trapezoid AUC).
- [ ] `_ece` / classification calibration → **keep** (see calibration.md; sklearn not installed).
- [ ] `cluster_bootstrap_ci` / `_statistic` / BootstrapCI → **keep** — cluster bootstrap with
      deterministic seeding + `(coverage, resample)` semantics has no library equivalent
      without adding sklearn `bootstrap` or `statsmodels` (not installed). Tests:
      `tests/phase_16/test_evaluation.py::test_patient_cluster_bootstrap_is_deterministic_and_not_slice_level`.
- [ ] `box_iou(predicted, target)` (plain python, 2D/3D half-open) → **keep** — returns a
      scalar and supports 3D; `torchvision.ops.box_iou` is batched 2D `(N,4)` only and would
      require tensor plumb-through + dtype/format normalization → violates "no forced casts".
- [ ] `generation_metrics` `_token_f1` / `_rouge_l` → **keep** — ROUGE-L via token-table DP is
      bespoke and clinical-unit-wrapped; adding `rouge-score` (not installed) is out of scope.

## Existing tests (must keep passing)
`tests/phase_13/test_evaluation.py`, `tests/phase_14/test_evaluation.py`,
`tests/phase_16/test_evaluation.py` (empty-mask, bootstrap determinism, localization
physical-error).

## Dependencies / sequencing
Wave 0. `metrics.py` (Wave 1) is a facade over `advanced`. No other module imports these
internals; parallel-safe.
