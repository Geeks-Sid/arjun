# transfer_plan/evaluation/metrics.md

Source: `medfm/evaluation/metrics.py` (Phase-13 legacy metrics + facade to `advanced.py`).

Wave: 1 (facade depends on `advanced.py`, Wave 0).

## Transfer checklist

- [x] `_auroc` / `_auprc` → **keep** — torchmetrics `BinaryAUROC` and
  `BinaryAveragePrecision` match the legacy kernels within `1e-6` for unique
  scores, but tied-score fixtures drift by as much as `0.5` (and sample order
  changes the legacy result). Degenerate all-positive/all-negative fixtures
  also differ in the required `None` contract.
- [x] `_sorted_binary` / `_operating_point` / `_legacy_classification_metrics`
  → **keep** — these deterministic kernels preserve the existing ordering,
  threshold, and metric contract; no transfer is parity-safe.
- [x] `_boundary(mask)` / `_surface_dice(...)` / `_legacy_segmentation_metrics` → **keep** —
  MONAI's `SurfaceDiceMetric` was parity-tested for the non-empty kernel, but its
  connectivity/edge semantics drift from the repository's max-pool implementation by
  `0.3182051182` on the fixed fixture (greater than 1e-6). The custom kernel remains,
  including the empty-mask convention (both_empty→1.0, one-side→0.0).
- [x] `classification_metrics` / `segmentation_metrics` / `baseline_metrics` / `retrieval_metrics`
      / `generation_metrics` / `localization_metrics` etc. (**lazy facade functions**) → **keep** —
      they are `__getattr__`-style forwarding shims into `advanced`. Nothing to transfer; just
      ensure they keep resolving after `advanced.md` edits.
- [x] `serialize_metrics` / `MetricValue` → **keep** — contract type; `MetricValue.to_dict` is
      stable JSON surface.
 
## Result

- `_auroc`, `_auprc`, `_sorted_binary`, `_operating_point`, `_legacy_classification_metrics`,
  `_boundary`, `_surface_dice`, `_legacy_segmentation_metrics`, `serialize_metrics`, and
  `MetricValue` remain unchanged and kept; no new mandatory dependency was added.
- MONAI `SurfaceDiceMetric` non-empty parity drift was `0.3182051182` on the fixed fixture,
  so delegating would violate the 1e-6 gate. Empty-mask behavior remains both-empty→1.0 and
  one-side→0.0.
- Lazy facade forwarding was verified with a sentinel test against `advanced.classification_metrics`.
- Files changed: `tests/phase_16/test_parity_metrics.py`, this checklist. Source
  `medfm/evaluation/metrics.py` intentionally unchanged.

## Existing tests
`tests/phase_13/test_evaluation.py`, `tests/phase_14/test_evaluation.py`,
`tests/phase_16/test_evaluation.py`.

## Sequencing
Run strictly after `evaluation/advanced.md`. No other parallel work blocks on `metrics.py`.

## Result

- `_auroc` and `_auprc` were parity-tested against torchmetrics 1.9.0 using
  callable `forward()` and `update()`/`compute()` paths. Unique-score drift was
  at most `7.417466907355674e-08`; tied-score drift reached `0.5` for both
  metrics, so both transfers were kept.
- Degenerate behavior remains custom: legacy AUROC returns `None` for both
  all-positive and all-negative targets, while torchmetrics returns `0.0`;
  legacy AUPRC returns `1.0` for all-positive and `None` for all-negative,
  while the candidate returns `1.0` and `-0.0`, respectively.
- `_sorted_binary`, `_operating_point`, `_legacy_classification_metrics`,
  segmentation kernels, lazy facade functions, `serialize_metrics`, and
  `MetricValue` remain kept and unchanged.
- Files changed: `tests/phase_16/test_parity_metrics_torchmetrics.py` and this
  checklist. `medfm/evaluation/metrics.py` intentionally unchanged.
