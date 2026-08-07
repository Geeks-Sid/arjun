# transfer_plan/evaluation/metrics.md

Source: `medfm/evaluation/metrics.py` (Phase-13 legacy metrics + facade to `advanced.py`).

Wave: 1 (facade depends on `advanced.py`, Wave 0).

## Transfer checklist

- [ ] `_auroc` / `_auprc` / `_sorted_binary` / `_operating_point` / `_legacy_classification_metrics`
      → **keep** — these are the deterministic, closed-form torch implementations already used
      by recipes; `_auroc` uses `torch.trapezoid` over the ROC curve. sklearn/torchmetrics not
      installed; no new dep. Verify unchanged against `tests/phase_13/test_evaluation.py`.
- [ ] `_boundary(mask)` / `_surface_dice(...)` / `_legacy_segmentation_metrics` → **partial** —
      `_boundary` (morphological erosion boundary via `nn.functional.max_pool`) and
      `_surface_dice` (tolerance-neighborhood surface Dice) are torch-native and compact.
      MONAI's `SurfaceDiceMetric` is a direct candidate IF and only if the publish-step keeps the
      repo's empty-mask convention (both_empty→1.0, one-side→0.0) — MONAI returns NaN. Mirror the
      approach in `advanced.md`: keep the empty-case branch, delegate the non-empty kernel, and
      gate with a parity test. If that parity test shows drift > 1e-6, keep the current code and
      tick `keep` with the drift recorded.
- [ ] `classification_metrics` / `segmentation_metrics` / `baseline_metrics` / `retrieval_metrics`
      / `generation_metrics` / `localization_metrics` etc. (**lazy facade functions**) → **keep** —
      they are `__getattr__`-style forwarding shims into `advanced`. Nothing to transfer; just
      ensure they keep resolving after `advanced.md` edits.
- [ ] `serialize_metrics` / `MetricValue` → **keep** — contract type; `MetricValue.to_dict` is
      stable JSON surface.

## Existing tests
`tests/phase_13/test_evaluation.py`, `tests/phase_14/test_evaluation.py`,
`tests/phase_16/test_evaluation.py`.

## Sequencing
Run strictly after `evaluation/advanced.md`. No other parallel work blocks on `metrics.py`.
