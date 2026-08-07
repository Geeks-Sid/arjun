# transfer_plan/models/heads/localization.md

Source: `medfm/models/heads/localization.py` (2D/3D box heads, coordinate conversion, IoU losses).

Wave: 0.

## Transfer checklist

- [ ] `box_iou(boxes_a, boxes_b)` (aligned 2D/3D, half-open min/max, eps=1e-7) → **partial** —
      for the 2D case `torchvision.ops.box_iou` is the mature, CUDA-optimized reference
      (verified). **Format mismatch**: torchvision expects `(N,4)` xyxy float batches and
      returns a pairwise matrix; this codebase does aligned elementwise 2D **and 3D** half-open
      coords. Transfer only the 2D batched matrix via a small adapter that converts min/max →
      xyxy (pure arithmetic, **not a dtype cast** — both are float) and add a parity test
      comparing against our formula on random 2D boxes. The 3D path stays hand-rolled (no
      library). Given format churn + 3D duality, **keep is acceptable** if the parity test shows
      any drift; document.
- [ ] `generalized_box_iou(...)` → **partial** — same as above vs `torchvision.ops.generalized_box_iou`
      (2D). 3D stays custom. Also note `torchvision.ops` has `complete_box_iou`/`distance_box_iou`
      already if we ever want CIoU/DIoU variants instead of hand-rolling.
- [ ] `IoUBoxLoss`/`GIoULoss`/`BoxL1Loss` → **partial** — `torchvision.ops.giou_loss`/`ciou_loss`
      exist (verified) but are 2D-batched; our loss wraps `(1 - score).mean()`. Verify dtype
      (float32) and reduction parity; 3D path stays custom. Likely **keep** for symmetry with
      the box math above.
- [ ] `normalized_to_physical_boxes` / `physical_to_normalized_boxes` / `boxes_from_spatial_metadata`
      / `_affine_corners` → **keep** — coordinate-system contract (mm/slide-pixels/normalized);
      no library equivalent.
- [ ] `BoxHead2D/3D`, `SpatialBoxHead` → **keep** — network glue (`nn.Conv2d/3d` building
      blocks); nothing library-specific to gain.

## Tests
`tests/phase_11/test_alignment_boxes_generation.py`, `tests/phase_16/test_evaluation.py`
(box_iou scalar form).
