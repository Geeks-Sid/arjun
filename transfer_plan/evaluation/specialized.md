# transfer_plan/evaluation/specialized.md

Source: `medfm/evaluation/specialized.py` (3D/pathology-specific Phase-16 tables).
Wave: 1 (imports `advanced.classification_metrics`).

## Transfer checklist

- [x] `adjacent_slice_consistency` (mean IoU of adjacent slice predictions) → **partial** —
      IoU over adjacent slices is just `monai.metrics.DiceMetric`-style math; the per-volume
      grouping is custom. Keep unless a helper from `evaluation/advanced.md` already exposes
      the IoU kernel. Likely **keep**.
- [x] `small_lesion_sensitivity` → **keep** — connected-lesion subsetting via
      `scipy.ndimage.label` (already library) + size bound; bespoke semantics.
- [x] `pathology_evaluation_metrics` / `sweep_pathology_sampling` / `compare_native_3d_and_slice_sequence`
      → **keep** — orchestration over `advanced` metrics.

## Tests
`tests/phase_16/test_evaluation.py`.
## Result
Verified partial/keep by design: adjacent-slice IoU retains custom per-volume grouping, lesion sensitivity retains size-bounded connected-component semantics, and pathology comparisons remain orchestration over advanced metrics. Source read confirmed the contracts. No parity drift measured. Focused Phase-16 test run — 17 passed.
