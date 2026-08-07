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

- [ ] `_masked_reduce` / `_broadcast_class_weight` / `_validate_reduction` → **keep** — the
      mask-aware reduction IS the value-add; leaving it custom is correct.
- [ ] `BinaryCrossEntropyWithLogitsLoss` → **keep** — it already calls `nn.functional.binary_cross_entropy_with_logits`;
      the wrapper only adds masked reduction. No transfer needed.
- [ ] `CrossEntropyClassificationLoss` → **keep** — same: delegates to `nn.functional.cross_entropy`.
- [ ] `FocalLoss` → **partial** — MONAI `monai.losses.FocalLoss(gamma, alpha, use_softmax)`
      exists (verified). Adopt only if a parity test confirms identical alpha/gamma/reduction
      math on float32 & float16 logits and identical empty-class behavior. Caveat: MONAI's
      reduction is a plain mean (no valid_mask); keep our `_masked_reduce` wrapper around the
      delegated kernel. Otherwise **keep**.
- [ ] `AsymmetricMultilabelLoss` → **keep** — asymmetric focal variant not in MONAI's public
      API; bespoke, keep.
- [ ] `OrdinalCumulativeLinkLoss` → **keep** — no library equivalent.
- [ ] `DiceLoss` / `dice_loss` (soft Dice, empty→perfect) → **partial** — MONAI
      `monai.losses.DiceLoss(include_background, smooth_nr, smooth_dr, reduction)` is the
      mature reference. **Semantic gap**: this repo treats `empty target/prediction as a
      perfect class`; MONAI's empty-case behavior must be checked. Keep the empty-class branch
      custom, delegate the non-empty Dice kernel, and gate with a parity test
      (float32 + float16). If drift: keep and record.
- [ ] `DiceCELoss` / `DiceBCELoss` → **partial** — MONAI `DiceCELoss(include_background,
      sigmoid/softmax, lambda_dice, lambda_ce, label_smoothing)` matches the composition;
      `DiceBCELoss` maps to `DiceCELoss(sigmoid=True)` or a manual dice+BCE pairing. Same
      empty-case + masked-reduction caveats as DiceLoss.
- [ ] `TverskyLoss` → **partial** — MONAI `monai.losses.TverskyLoss(include_background,
      sigmoid/softmax, alpha, beta, reduction)` is verified; adopt with parity test (ours
      defaults alpha=beta=0.5). Empty-case + masked-reduce caveats apply.
- [ ] `BoundaryLoss` → **keep** — local finite-difference boundary surrogate; no MONAI
      equivalent in the installed API surface.
- [ ] `DeepSupervisionLoss` → **partial** — MONAI `monai.losses.DeepSupervisionLoss` exists
      (verified) but assumes its own spatial weights contract; ours takes explicit
      per-level weights. Keep unless parity on `tests/phase_11/test_heads_and_losses.py` holds.
- [ ] `FocalSegmentationLoss` → **partial** — thin wrapper over `FocalLoss`; inherits its verdict.
- [ ] Functional aliases (`binary_cross_entropy_with_logits`, `cross_entropy`, `focal_loss`,
      `dice_ce_loss`, `dice_bce_loss`, ...) → **keep** — stable config-facing surface; bodies
      forward to the classes above.

## Tests
`tests/phase_11/test_heads_and_losses.py`, `tests/phase_11/test_segmentation.py`,
`tests/phase_11/test_task_wrappers.py`.
