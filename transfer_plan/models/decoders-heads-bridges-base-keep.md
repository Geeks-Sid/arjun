# transfer_plan/models/decoders-heads-bridges-base-keep.md

Covers (keep-by-design): `models/decoders/{base,masks,language,segmentation}.py`,
`models/heads/{classification,losses}.py`, `models/bridges/{base,coordinates,placement,training}.py`,
`models/pathology/{aggregators,selectors,stores,distributed,encoders,adapters,pipeline}.py`.
Wave: 1/2.

## Transfer checklist

- [x] `decoders/base.py` (`ConvBlock`, `SegmentationOutput`, `as_feature_maps`,
      `_interpolate`) → **keep** — `_interpolate` already wraps `torch.nn.functional.interpolate`;
      the block/feature-pyramid contract is custom. `ConvBlock` could optionally adopt
      `monai.networks.blocks.Convolution` if parity holds (see `decoders/unet.md`) — low priority.
- [x] `decoders/masks.py` (TransformerMaskDecoder, PromptableMaskDecoder, native wrapper) →
      **keep** — cross-attention mask decoding over visual tokens; no library drop-in without
      changing the adapter contract.
- [x] `heads/classification.py` (linear/MLP/attention-pooling heads) → **keep** — `nn.Linear`
      glue + pooling from `heads/pooling.py`; nothing library-specific to add.
- [x] `heads/losses.py` → **keep** — re-exports `tasks.losses`; inherits that verdict.
- [x] `bridges/{coordinates,placement}.py` (coordinate encoders, boundary embeddings, causal
      label masking) → **keep** — core contract (`IGNORE_INDEX`, coordinate systems);
      `torch.cat`/embedding glue is library-native already.
- [x] `bridges/training.py`, `bridges/base.py` → **keep** — abstraction + validation.
- [x] `pathology/{aggregators,selectors,stores,distributed,encoders,adapters,pipeline}.py` →
      **keep** — HDF5 store + safetensors embedding reads already library-backed; encoder
      pipeline and aggregators are covered by `pathology/aggregation.md` and
      `pathology/selection.md`. No additional transfers.

## Result

Verified keep by design. Quick source reads confirmed decoder feature-map and adapter contracts, custom cross-attention mask decoding, classification-head/pooling glue, loss re-exports, bridge coordinate/placement/training contracts, and already library-backed pathology storage/encoder boundaries. No source or test files changed. Parity drift: not applicable (keeps).

Focused verification: `uv run --frozen pytest tests/phase_09/test_bridges.py tests/phase_08/test_pathology.py tests/phase_15/test_recipes.py` — 45 passed.

## Tests
`tests/phase_09/test_bridges.py`, `tests/phase_08/test_pathology.py`,
`tests/phase_11/test_heads_and_losses.py`, `tests/phase_13/test_recipes.py`.
