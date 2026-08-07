# transfer_plan/models/decoders/unet.md

Source: `medfm/models/decoders/unet.py` (2D/3D UNet-style decoders).

Wave: 0.

## Transfer checklist

Blocking context: `_UNetDecoder` consumes the encoder's **feature pyramid** (`EncoderOutput`
maps, finest-last) via `as_feature_maps(features)`, returns `SegmentationOutput` (logits +
optional deep supervision), and uses static-shape-friendly lazy convs
(`_lazy_projection`) for fixed-shape TPU buckets. MONAI's `UNet`/`BasicUNet`/`FlexibleUNet`
take a raw input volume through their own encoder/decoder, not a precomputed pyramid.

- [ ] `_boundary`/`_projection`/`_head`/forward core → **keep** — the decoder is thin
      sequential glue over `nn.Conv{2,3}d` + interpolate (`_interpolate` from `base.py`).
      There is **no library drop-in that consumes an EncoderOutput pyramid**; adopting MONAI
      UNet would restructure the whole adapter contract. Do not transfer.
- [ ] Deep-supervision over auxiliary outputs → **keep** — contract glue.
- [ ] `ConvBlock` (from `decoders/base.py`, reused) → **partial** — `monai.networks.blocks.Convolution`
      (normalization+activation+dropout options) is a candidate **if and only if** a parity test
      confirms identical numerics for the same kernel/stride/padding/group config and dtype.
      Low priority: the block is already a few lines over `nn.Conv2d/3d`. Likely **keep**.
- [ ] `_interpolate` / `_group_count` / `as_feature_maps` → **keep** — `_interpolate` wraps
      `torch.nn.functional.interpolate` (library-backed already); group-count/feature-splitting
      are contract logic.

## Tests
`tests/phase_11/test_segmentation.py`, `tests/phase_11/test_heads_and_losses.py`,
`tests/phase_13/test_recipes.py` (2D/3D recipes drive this decoder).
