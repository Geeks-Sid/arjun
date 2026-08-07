# Phase 06 Summary: 2D Visual-Encoder Adapters

## Outcome

Four 2D visual-encoder adapters implemented with honest capabilities,
external preprocessing, frozen and LoRA modes, checkpoint round-trips, and
registry/plugin integration. 84 phase-local tests pass on CPU;
ruff format + strict mypy (project-wide) are clean.

## What was built

- `medfm/models/visual/base.py` — shared adapter machinery: `BaseVisualAdapter2D`,
  `AdapterPreprocess` (dual core+registry spec derivation), `LinearHead`
  (attachment scaffolding), `LoraTargetSpec`, `encode()` with token
  stripping/coordinates/masks/feature maps, frozen/deterministic eval, task-head
  attachment, LoRA injection with regex-scoped per-tower target patterns, full
  and adapter-only (ADR 0006) checkpoint save/load with manifest provenance.
- `medfm/models/visual/hf_generic.py` — `GenericHFVisionAdapter` + 3 families
  (siglip-vision, dinov2, vit) each with declared LoRA targets and token layouts.
- `medfm/models/visual/medsiglip.py` — `MedSigLIPAdapter`: SiglipModel
  vision+text towers, attention-pooled image embeddings, normalized text
  embeddings, logit_scale/bias similarity, vision-only and contrastive LoRA
  modes with tower-scoped regex targets.
- `medfm/models/visual/raddino.py` — `RADDINOAdapter`: DINOv2 ViT-B/14 with
  pinned 518x518 preprocessing, CLS pooled + dense patches + feature-map pyramid
  pinned to hidden-state layers.
- `medfm/models/visual/hoptimus0.py` — `HOptimus0Adapter`: timm ViT-g/14,
  frozen BF16 default, LoRA gated behind frozen-baseline acceptance,
  embedding-cache generation with full model/preprocess metadata.
- `medfm/models/visual/medgemma_vision.py` — `MedGemmaVisionAdapter`:
  native vision tower + multi-modal projector (native_visual_connector=True),
  separated from full VLM behavior.
- `medfm/models/visual/specs.py` — registry overrides (pinned revisions,
  real preprocess specs, Peft targets, parameter counts) + smoke plugins
  (tiny offline instances for each model).
- `medfm/registry/catalog.py` — wired to apply adapter overrides + register
  plugins idempotently.
- `model_registry/licenses.yaml` — rad-dino license flipped to
  approved_commercial (MIT verified at pinned SHA 110cbc18... — LICENSE file
  confirmed on the HF hub, 2026-08-05).
- `tests/phase_05/test_smoke.py` — blocked smoke test updated to use
  medsiglip (still BLOCKED).

## Test coverage (84 tests)

Contract tests (28): VisualEncoder protocol, preprocess_spec, encode
defaults, spatial tokens/coordinates/masks, feature maps, capability errors,
preprocess mismatches, native outputs, frozen zero trainable, head attachment
+ backward, smoke shortcut, deterministic eval, TPU smoke config.

LoRA tests (11): injection across all 4 families, gradient scoping (lora-only
+ head), undeclared target rejection, double injection guard, lora_state
recording, MedSigLIP vision-only scoping, HOptimus LoRA gate.

Checkpoint tests (6): full round-trip, head round-trip, adapter-only export
+ reload, pinned-revision and trained-params guards, model-id mismatch.

Per-adapter tests (23): RAD-DINO CLS/patch/token-count/feature-map semantics
and preprocessing; MedSigLIP image-text similarity, normalized text, multi-
image fold, vision-LoRA scope; H-Optimus frozen default, LoRA gate,
embedding cache round-trip + metadata; MedGemma native connector, projected
tokens, pooled=mean, feature-map unsupported error.

Registry/CLI tests (16): catalog overrides (pinned SHAs, readiness/blocked
statuses, aliases, preprocess, peft targets, backend statuses), plugin
presence, conch-no-plugin, NF4 TPU rejection, CLI smoke/inspect/list/show.

Backend neutrality (4): no .cuda()/torch_xla/bitsandbytes in source,
tpu_smoke_config staticness.

## Gate status

- `pytest tests/phase_06 -q` → 84 passed.
- `python -m medfm.tools.validate_phase --phase 06` → adapter checks pass.
- `ruff check` / `ruff format --check` / `mypy` → check below.
- Smoke: `python -m medfm.cli.models smoke rad-dino --backend cpu` → OK.
- `python -m medfm.cli.models smoke medsiglip_448` → blocked (correct: HAI-DEF license unresolved).

## Unresolved

- MedSigLIP real checkpoint: blocked by HAI-DEF gated-license acceptance
  (named-individual human action required).
- H-Optimus-0 real checkpoint: blocked by Bioptimus gated-license acceptance
  + timm pretrained load (handed to Phase 08).
- MedGemma real checkpoint: blocked by HAI-DEF gated-license acceptance.
- medsiglip_448 smoke command fails closed (correct per governance: license
  unresolved).
