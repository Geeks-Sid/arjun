# transfer_plan/tasks/losses.md

Source: `medfm/tasks/losses.py` (classification + segmentation loss zoo, ~712 lines).

Wave: 0 (depends only on `medfm.core.errors`).

## Transfer checklist

Blocking context: every loss must (a) honor mask-aware reduction through
`_masked_reduce(..., valid_mask, reduction)` over padded TPU buckets, (b) return
`LossOutput` with sample-count metadata (via task wrappers, not here), (c) treat
empty-target/prediction cases per the repo's semantics, and (d) stay dtype-transparent
(return logits/input dtype, float16/bfloat16/float32 all valid). MONAI losses are torch-native
and dtype-agnostic on the *kernel*, but they do **not** take a per-sample `valid_mask` and
their empty-case defaults differ.

- [x] `_masked_reduce` / `_broadcast_class_weight` / `_validate_reduction` → **keep** — the
      mask-aware reduction IS the value-add; leaving it custom is correct.
- [x] `BinaryCrossEntropyWithLogitsLoss` → **keep** — it already calls
      `nn.functional.binary_cross_entropy_with_logits`; the wrapper only adds masked
      reduction. No transfer needed.
- [x] `CrossEntropyClassificationLoss` → **keep** — same: delegates to
      `nn.functional.cross_entropy`.
- [x] `FocalLoss` → **keep** — MONAI's softmax focal applies channel-wise focal
      weighting and class-balanced alpha rather than this wrapper's CE focal contract;
      measured drift is 0.801864 (float32) and 0.779766 (float16), and MONAI returns
      float32 for float16 input.
- [x] `AsymmetricMultilabelLoss` → **keep** — asymmetric focal variant not in MONAI's public
      API; bespoke, keep.
- [x] `OrdinalCumulativeLinkLoss` → **keep** — no library equivalent.
- [x] `DiceLoss` / `dice_loss` (soft Dice, empty→perfect) → **keep** — MONAI matches the
      float32 kernel exactly with `smooth_nr=smooth_dr=1`, but float16 drifts by
      0.00048828125 (above the parity tolerance); retaining FP32 accumulation,
      empty handling, and masked semantics is required.
- [x] `DiceCELoss` / `DiceBCELoss` → **keep** — MONAI's compositions use a fixed
      `1e-5` smoothing contract and differ from ours by 0.052701/0.019297 in float32
      (0.050781/0.019531 in float16), respectively.
- [x] `TverskyLoss` → **keep** — MONAI matches the float32 kernel, but float16 drifts
      by 0.00048828125; preserving dtype-transparent output and masked/empty behavior
      takes precedence.
- [x] `BoundaryLoss` → **keep** — local finite-difference boundary surrogate; no MONAI
      equivalent in the installed API surface.
- [x] `DeepSupervisionLoss` → **keep** — MONAI's implementation does not preserve our
      explicit per-level weighting, mask forwarding, resizing contract, or float16 output.
- [x] `FocalSegmentationLoss` → **keep** — inherits `FocalLoss`'s non-parity and dtype
      mismatch with MONAI.
- [x] Functional aliases (`binary_cross_entropy_with_logits`, `cross_entropy`, `focal_loss`,
      `dice_ce_loss`, `dice_bce_loss`, ...) → **keep** — stable config-facing surface; bodies
      forward to the classes above.

## Result

- Verdicts: all checklist items are `keep`; no MONAI transfer was safe under the
  float32 + float16 parity, empty-case, mask-aware reduction, and dtype-transparency
  requirements.
- Parity measurements (MONAI configured with equivalent available options): Focal
  drift 0.801864/0.779766 (float32/float16); Dice 0/0.00048828125; DiceCE
  0.052701/0.050781; DiceBCE 0.019297/0.019531; Tversky 0/0.00048828125.
- Files changed: `tests/phase_11/test_parity_losses.py` and this checklist.
- Validation: `uv run --frozen pytest tests/phase_11/test_parity_losses.py` (7 passed).

## Tests
`tests/phase_11/test_heads_and_losses.py`, `tests/phase_11/test_segmentation.py`,
`tests/phase_11/test_task_wrappers.py`.
