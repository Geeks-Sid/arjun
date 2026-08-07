# Phase 14 summary

Phase 14 delivers distinct, reproducible CT/MRI native-volume and slice-sequence
recipe families on top of the model-agnostic Phase 12 training engine. Native
3D tensors retain volumetric semantics and physical token coordinates; the
slice family remains an explicit `MULTI_IMAGE_2D` experiment with host-side
selection metadata.

## Delivered

- Added `medfm/recipes/phase14.py` with typed pinned metadata, deterministic
  offline-tiny builders, static 3D shape policies, backend observability, and
  model/task/optimizer/trainer builders.
- Added native 3D classification profiles for CT-FM, FlexiCT-3D, Triad MRI,
  and the generic MONAI-compatible TPU baseline. Classification supports
  frozen pooled heads, attention pooling, final-stage vision LoRA, and
  patient/study-level metric units and deterministic group-level score aggregation.
- Added native 3D segmentation profiles with foreground-lesion sampling,
  padding masks, optional deep supervision, constant/Gaussian sliding-window
  blending, crop-origin/spacing preservation, original-space restoration, and
  Dice/surface/HD95/lesion-recall/false-positive/volume-error metrics.
- Added native 3D VLM profiles with physical-coordinate embeddings, Perceiver
  or linear bridges, fixed visual/text buckets, cached spatial-token training,
  image/no-image/shuffled-image modes, and staged language/vision LoRA roles.
- Added a native structured-findings VLM profile with the structured task type
  while retaining the same grounded language-loss and generation contract.
- Added a separate slice-sequence VLM family and deterministic host selectors
  for uniform, anatomy-aware, report-conditioned, entropy, lesion-aware,
  multi-window, and related experiments. Source index, normalized/physical z,
  series order, window, MRI sequence, and selector revision are preserved.
- Added language-conditioned native 3D segmentation with fixed query buckets,
  explicit absent/multiple/laterality/ambiguous query fixtures, cross-attention
  decoder wiring, and separate mask-grounding metrics.
- Added twelve accepted 3D YAML profiles under `configs/recipes/3d/`, including the
  plan-required `classification_smoke.yaml`, CUDA/TPU BF16 declarations, and
  explicit TPU statuses.
- Added Phase 14 focused tests, validator registration, acceptance artifacts,
  and `docs/recipes/phase14_3d_model_cards.md`.

## Verification

- `python -m pytest tests/phase_14 -q`: 19 passed.
- `python -m pytest tests/phase_13 -q`: 19 passed.
- `python -m pytest tests/phase_12 tests/phase_13 tests/phase_14 -q`: 50 passed.
- CPU pipeline smokes passed for classification, MRI classification,
  FlexiCT classification with vision LoRA, baseline and LoRA segmentation,
  native VLM with and without cached tokens, slice-sequence VLM, and
  language-conditioned 3D segmentation; each completed one optimizer step.
- `python -m medfm.cli.train --config configs/recipes/3d/classification_smoke.yaml --format json` completed one CPU optimizer step with effective batch size 2 and a checkpoint at `checkpoints/last`.
- Checkpoint resume restored global step 1 in the Phase 14 focused suite.
- `python -m medfm.tools.validate_phase --phase 14` passed after this report
  directory was populated.

## Runtime limits

This workstation has no `torch_xla`, `bitsandbytes`, approved production
checkpoints, or protected clinical datasets. TPU PJRT/BF16 hardware execution,
CUDA multi-device/QLoRA execution, large-model VRAM measurements, external-site
performance, representative original-space visual review, and human-reader
review were not executed and are not presented as passing clinical evidence.
The generic MONAI-compatible TPU profile and all blocked dependency statuses
remain explicit handoff contracts.
