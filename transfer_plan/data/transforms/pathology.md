# transfer_plan/data/transforms/pathology.md

Source: `medfm/data/transforms/pathology.py` (thumbnail, tissue mask, tile planning, stain ops,
quality scores).

Wave: 0.

## Transfer checklist

- [ ] `_otsu_threshold(values)` (hand-rolled 256-bin between-class variance) → **partial /
      transfer** — `skimage.filters.threshold_otsu` (verified installed, scikit-image 0.25) is
      the mature equivalent and returns a float threshold on [0,1] inputs. Add a parity test
      (deterministic: same 256-bin histogram → equal threshold for every fixture). If exact
      match: replace the body. Note: skimage's Otsu histogram bin-width differs slightly
      (255 bins in [0,1] domain) — measure; if drift appears, keep ours and record.
- [ ] `compute_tissue_mask` → **partial** — delegates to `_otsu_threshold` + HSV brightness
      gating; once `_otsu_threshold` transfers, the rest is numpy (keep). 
- [ ] `make_thumbnail` → **keep** — already delegates to Pillow (`Image.resize`, BILINEAR).
- [ ] `blur_score` (hand-rolled 4-neighbour Laplacian variance) → **partial** —
      `scipy.ndimage.laplace` (verified) or `skimage.filters.laplace` computes the Laplacian;
      the interior-only stencil here excludes borders. Adopting the library would change edge
      handling → parity test first (`min(gray.shape)<3` case must still raise; constant tile
      must score exactly 0.0). Likely verdict: **keep** (border-exclusion is bespoke).
- [ ] `artifact_score` → **keep** — thresholded HSV fraction; simple numpy.
- [ ] `_grayscale` → **keep** (BT.601 luma via matmul) — trivial.
- [ ] `plan_tiles` / `TileRecord` / `make_tile_id` → **keep** — deterministic grid + MPP
      normalization + SHA-256 tile IDs are contract glue (TotalSegmentator-style tiling only in
      external repos under non-identical conventions).
- [ ] `ReinhardStainNormalize` / `StainAugment` (HED/Lab ops via `_color_ops`) → **keep** —
      uses `skimage.color` already (rgb2hed/hed2rgb/lab conversions are library-backed);
      stain-normalization statistics matching is bespoke.
- [ ] `_color_ops` → **keep** (already wraps skimage; lazy-imported).

## Tests
`tests/phase_04/test_pathology.py`, `tests/phase_03/test_pathology_readers.py`.
