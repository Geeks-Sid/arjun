# transfer_plan/data/samplers/patches.md

Source: `medfm/data/samplers/patches.py` (deterministic 3D patch sampling: grid, random,
foreground-driven, with explicit PatchInfo metadata).
Wave: 1 (uses `medfm.data.transforms.base.make_generator`).

## Transfer checklist

- [x] Grid/random/foreground samplers → **keep** — the contract requires sampling
      **exclusively from a seeded `torch.Generator`** (explicit determinism), clamping so every
      patch is full-shape, and recording `PatchInfo` (origin, padding, `target_positive`,
      `sampling_probability`, physical bounds). MONAI's `data.sampler`/`RandSpatialCropSamplesd`
      draw from the global RNG and emit no equivalent metadata → no drop-in. `scipy.ndimage` is
      already used where needed. Keep.
- [x] `_clamp_origin` / `_padding_amounts` / `_randint` / `_rand_float` → **keep** (helpers).

## Tests
`tests/phase_04/test_patch_samplers.py`.

## Result

- **Verdict: keep.** Source read confirms stochastic samplers use caller-owned or seeded
  `torch.Generator` draws, origins are clamped and padded to full shape, and `PatchInfo`
  records origin, padding, positivity, probability, and physical bounds. MONAI samplers do
  not preserve these metadata and RNG contracts, so no transfer is appropriate.
- **Parity drift:** not applicable; no library replacement.
- **Verification:** `uv run --frozen pytest tests/phase_04/test_patch_samplers.py` — PASS
  (24 passed); `uv run --frozen ruff check medfm/data/samplers/patches.py` — PASS;
  `uv run --frozen ruff format --check medfm/data/samplers/patches.py` — PASS;
  `uv run --frozen mypy medfm/data/samplers/patches.py` — PASS.
- **Files changed:** `transfer_plan/data/samplers/patches.md` only; no source or test files edited.
