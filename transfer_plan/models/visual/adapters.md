# transfer_plan/models/visual/adapters.md

Covers: `models/visual/{base,hf_generic,hoptimus0,medsiglip,raddino,medgemma_vision,ct_fm,triad,research_3d,native_tasks}.py`.
Wave: 2 — **keep by design**.

## Transfer checklist

- [x] `BaseVisualAdapter2D` / `AdapterPreprocess` / `LoraTargetSpec` / `LinearHead` →
      **keep** — encoder-adapter contract layer (EncoderOutput composition, capability gating,
      checkpoint provenance). No library analogue.
- [x] `GenericHFVisionAdapter` (family registry for SigLIP/DINOv2/ViT towers) → **keep** —
      already constructs backbones via `transformers` and timm; coordinates with the `hf`
      extra. Nothing to re-implement.
- [x] `HOptimus0Adapter` / `MedSigLIPAdapter` / `RADDINOAdapter` / `MedGemmaVisionAdapter` /
      `CTFMAdapter` / `FlexiCT3DAdapter` / `Triad*Adapter` / `MedSAM2Adapter` → **keep** -
      pinned-weight wrappers over upstream Hugging Face / timm / native checkpoints; the
      adapters' job is contract enforcement (preprocess hash, capability declaration), and the
      real backbone is already a library. The user note: we may borrow *their* reference code
      from `external_repos/{CONCH,CT-FM,Triad,MedSAM2,...}` when wrapping new variants — but
      licensing gates apply (see README).


## Tests
`tests/phase_06/test_other_adapters.py`, `tests/phase_06/test_raddino_medsiglip.py`,
`tests/phase_07/test_registry.py`, `tests/phase_14/test_recipes.py`.

## Result

All three visual adapter items verified as **keep**: base contract/provenance layers and
family-specific wrappers already delegate backbones to transformers, timm, MONAI/native, or
pinned checkpoints while retaining medfm preprocessing, capability, LoRA-target, and checkpoint
contracts. No transfer or parity drift measured. Source and test files were unchanged; only this
checklist was updated.

Validation (shared phase run): `uv run --frozen pytest tests/phase_02 tests/phase_03 tests/phase_04
tests/phase_05 tests/phase_06 tests/phase_07 tests/phase_09` — **PASS** (622 passed, 4 skipped,
1 warning). Scoped `ruff check` — **PASS**; `ruff format --check` — **PASS** (43 files);
scoped `mypy` — **PASS**.
