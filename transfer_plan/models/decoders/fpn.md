# transfer_plan/models/decoders/fpn.md

Source: `medfm/models/decoders/fpn.py` (2D/3D feature-pyramid segmentation decoders).

Wave: 0.

## Transfer checklist

- [x] `_FPNDecoder` (lateral 1×1 convs, top-down upsampling, smooth 3×3, per-level auxiliary
      heads) → **keep** — the installed torchvision FPN uses nearest-neighbor top-down
      interpolation, while this decoder's contract uses bilinear interpolation with
      `align_corners=False`; the parity test measures a 0.75 maximum drift on a tiny
      non-constant feature map. The custom `SegmentationOutput` / `as_feature_maps` glue,
      optionality, deep-supervision heads, and smoothing semantics are therefore retained.
- [x] `FPNDecoder3D` → **keep** — no library equivalent; torchvision FPN is 2D-only.
- [x] `auxiliary_heads` / deep-supervision wiring → **keep** — custom output contract and
      per-level head wiring are retained.
- [x] `_interpolate` / `_group_count` (from base) → **keep** — the decoder relies on the
      repository's bilinear/trilinear interpolation and GroupNorm grouping semantics.

## Result

- `_FPNDecoder`: keep (2D and 3D); torchvision top-down interpolation parity failed with
  measured max absolute drift `0.75` (`atol=rtol=1e-6`), so no source transfer was made.
- `FPNDecoder3D`: keep; torchvision has no 3D equivalent.
- `auxiliary_heads` / deep supervision: keep; custom `SegmentationOutput` contract.
- `_interpolate` / `_group_count`: keep; repository semantics are required.
- Files changed: `tests/phase_11/test_parity_fpn.py`, this checklist.
- Verification: `uv run --frozen pytest tests/phase_11/test_parity_fpn.py` (pass).

## Tests
`tests/phase_11/test_segmentation.py`, `tests/phase_11/test_heads_and_losses.py`.
