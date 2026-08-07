# transfer_plan/data/samplers/patches.md

Source: `medfm/data/samplers/patches.py` (deterministic 3D patch sampling: grid, random,
foreground-driven, with explicit PatchInfo metadata).
Wave: 1 (uses `medfm.data.transforms.base.make_generator`).

## Transfer checklist

- [ ] Grid/random/foreground samplers → **keep** — the contract requires sampling
      **exclusively from a seeded `torch.Generator`** (explicit determinism), clamping so every
      patch is full-shape, and recording `PatchInfo` (origin, padding, `target_positive`,
      `sampling_probability`, physical bounds). MONAI's `data.sampler`/`RandSpatialCropSamplesd`
      draw from the global RNG and emit no equivalent metadata → no drop-in. `scipy.ndimage` is
      already used where needed. Keep.
- [ ] `_clamp_origin` / `_padding_amounts` / `_randint` / `_rand_float` → **keep** (helpers).

## Tests
`tests/phase_04/test_patch_samplers.py`.
