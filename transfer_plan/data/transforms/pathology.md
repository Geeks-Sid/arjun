# transfer_plan/data/transforms/pathology.md

Source: `medfm/data/transforms/pathology.py` (thumbnail, tissue mask, tile planning, stain ops,
quality scores).

Wave: 0.

## Transfer checklist

- [x] `_otsu_threshold(values)` (hand-rolled 256-bin between-class variance) → **keep** — parity tests against `skimage.filters.threshold_otsu` show histogram/constant-channel drift (for example, ours `0.001953125` vs skimage `0.5` for a constant `0.5` fixture, and ours `0.501953125` vs skimage `0.5057010168773981` for a deterministic random fixture). The fixed-bin behavior is part of the tissue-mask contract, so the hand-rolled kernel remains.
- [x] `compute_tissue_mask` → **partial** — delegates to `_otsu_threshold` plus HSV brightness gating; the Otsu kernel remains custom because parity drift makes a full skimage transfer unsafe, while the rest is intentionally numpy contract logic.
- [x] `make_thumbnail` → **keep** — already delegates to Pillow (`Image.resize`, BILINEAR).
- [x] `blur_score` (hand-rolled 4-neighbour Laplacian variance) → **keep** — parity confirms the required 3x3 validation and exact constant-tile score; `scipy.ndimage.laplace` changes edge handling and yields a different score on an edge-sensitive fixture.
- [x] `artifact_score` → **keep** — thresholded HSV fraction; simple numpy.
- [x] `_grayscale` → **keep** (BT.601 luma via matmul) — trivial.
- [x] `plan_tiles` / `TileRecord` / `make_tile_id` → **keep** — deterministic grid + MPP
      normalization + SHA-256 tile IDs are contract glue (TotalSegmentator-style tiling only in
      external repos under non-identical conventions).
- [x] `ReinhardStainNormalize` / `StainAugment` (HED/Lab ops via `_color_ops`) → **keep** —
      uses `skimage.color` already (rgb2hed/hed2rgb/lab conversions are library-backed);
      stain-normalization statistics matching is bespoke.
- [x] `_color_ops` → **keep** (already wraps skimage; lazy-imported).

## Tests
`tests/phase_04/test_pathology.py`, `tests/phase_03/test_pathology_readers.py`.

## Result
- `_otsu_threshold`: **keep** — skimage parity drift measured as `0.001953125` vs `0.5` on constant `0.5` and `0.0037478918773981` vs `0.5057010168773981` on deterministic random values; fixed 256-bin behavior is retained.
- `compute_tissue_mask`: **partial** — retains custom Otsu plus numpy HSV gating.
- `make_thumbnail`, `artifact_score`, `_grayscale`, `plan_tiles`/`TileRecord`/`make_tile_id`, `ReinhardStainNormalize`/`StainAugment`, `_color_ops`: **keep** — existing library-backed or contract-specific implementations match the checklist.
- `blur_score`: **keep** — scipy edge handling does not match the bespoke interior-only stencil; 3x3 rejection and constant `0.0` behavior are covered.
- Files changed: `tests/phase_04/test_parity_pathology.py`, this checklist. `medfm/data/transforms/pathology.py` was intentionally unchanged because no eligible transfer passed parity.
- Validation: `uv run --frozen pytest tests/phase_04/test_parity_pathology.py tests/phase_04/test_pathology.py tests/phase_03/test_pathology_readers.py` (45 passed); `uv run --frozen ruff check medfm/data/transforms/pathology.py tests/phase_04/test_parity_pathology.py` (OK); `uv run --frozen ruff format medfm/data/transforms/pathology.py tests/phase_04/test_parity_pathology.py` (2 files left unchanged); `uv run --frozen mypy medfm/data/transforms/pathology.py` (OK).
