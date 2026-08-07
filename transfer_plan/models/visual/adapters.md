# transfer_plan/models/visual/adapters.md

Covers: `models/visual/{base,hf_generic,hoptimus0,medsiglip,raddino,medgemma_vision,ct_fm,triad,research_3d,native_tasks}.py`.
Wave: 2 — **keep by design**.

## Transfer checklist

- [ ] `BaseVisualAdapter2D` / `AdapterPreprocess` / `LoraTargetSpec` / `LinearHead` →
      **keep** — encoder-adapter contract layer (EncoderOutput composition, capability gating,
      checkpoint provenance). No library analogue.
- [ ] `GenericHFVisionAdapter` (family registry for SigLIP/DINOv2/ViT towers) → **keep** —
      already constructs backbones via `transformers` and timm; coordinates with the `hf`
      extra. Nothing to re-implement.
- [ ] `HOptimus0Adapter` / `MedSigLIPAdapter` / `RADDINOAdapter` / `MedGemmaVisionAdapter` /
      `CTFMAdapter` / `FlexiCT3DAdapter` / `Triad*Adapter` / `MedSAM2Adapter` → **keep** -
      pinned-weight wrappers over upstream Hugging Face / timm / native checkpoints; the
      adapters' job is contract enforcement (preprocess hash, capability declaration), and the
      real backbone is already a library. The user note: we may borrow *their* reference code
      from `external_repos/{CONCH,CT-FM,Triad,MedSAM2,...}` when wrapping new variants — but
      licensing gates apply (see README).

## Tests
`tests/phase_06/test_other_adapters.py`, `tests/phase_06/test_raddino_medsiglip.py`,
`tests/phase_07/test_registry.py`, `tests/phase_14/test_recipes.py`.
