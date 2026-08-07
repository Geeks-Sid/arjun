# transfer_plan/models/decoders/fpn.md

Source: `medfm/models/decoders/fpn.py` (2D/3D feature-pyramid segmentation decoders).

Wave: 0.

## Transfer checklist

- [ ] `_FPNDecoder` (lateral 1×1 convs, top-down upsampling, smooth 3×3, per-level auxiliary
      heads) → **partial** — for the **2D** variant, `torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork`
      (verified in this torchvision pin) is the mature equivalent of the top-down fusion
      (lateral+smooth). Adoption path: keep the `SegmentationOutput` / `as_feature_maps` glue
      and optionality (backbone_fmap_blocks vs precomputed maps; deep-supervision heads);
      internal 2D fusion may delegate to `FeaturePyramidNetwork` **if** parity holds on
      `tests/phase_11/test_segmentation.py` fixtures. The 3D variant has **no** library
      equivalent (torchvision FPN is 2D-only) → keep hand-rolled. Dtype: torch float32 native.
- [ ] `FPNDecoder3D` → **keep** — no library equivalent; keep and only guarantee the 2D path
      shares nothing broken.
- [ ] `auxiliary_heads` / deep-supervision wiring → **keep** — custom.
- [ ] `_interpolate` / `_group_count` (from base) → **keep** (library-backed via
      `torch.nn.functional.interpolate`).

## Tests
`tests/phase_11/test_segmentation.py`, `tests/phase_11/test_heads_and_losses.py`.
