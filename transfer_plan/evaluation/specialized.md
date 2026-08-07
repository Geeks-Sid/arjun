# transfer_plan/evaluation/specialized.md

Source: `medfm/evaluation/specialized.py` (3D/pathology-specific Phase-16 tables).
Wave: 1 (imports `advanced.classification_metrics`).

## Transfer checklist

- [x] `adjacent_slice_consistency` (mean IoU of adjacent slice predictions) → **partial** —
      per-pair IoU now delegates to `advanced._monai_dice_iou` (MONAI `DiceMetric`), while
      per-volume grouping, thresholding, and empty-pair semantics remain custom.
- [x] `small_lesion_sensitivity` → **keep** — connected-lesion subsetting via
      `scipy.ndimage.label` (already library) + size bound; bespoke semantics.
- [x] `pathology_evaluation_metrics` / `sweep_pathology_sampling` / `compare_native_3d_and_slice_sequence`
      → **keep** — orchestration over `advanced` metrics.

## Tests
`tests/phase_16/test_evaluation.py`, `tests/phase_16/test_specialized.py`,
`tests/phase_16/test_parity_specialized_monai.py`.

## Result
- `adjacent_slice_consistency`: **partial transfer**; the MONAI DiceMetric-backed advanced IoU
  helper matched the original non-empty per-pair IoU exactly (maximum drift `0.0`, below `1e-6`)
  on overlapping, identical, and disjoint masks. Custom per-volume grouping and the
  both-empty score of `1.0` remain in `specialized.py`.
- `small_lesion_sensitivity`: **keep** — connected-lesion subsetting via
  `scipy.ndimage.label` plus the size bound has bespoke semantics.
- `pathology_evaluation_metrics` / `sweep_pathology_sampling` / `compare_native_3d_and_slice_sequence`:
  **keep** — orchestration over advanced metrics.
- Files changed: `medfm/evaluation/specialized.py` and this checklist; parity test added at
  `tests/phase_16/test_parity_specialized_monai.py`.
- Verification: `uv run --frozen pytest tests/phase_16/test_specialized.py
  tests/phase_16/test_parity_specialized_monai.py` (4 passed).
