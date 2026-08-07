# Phase 07: Native 3D Visual-Encoder Adapters

## Objective

Support volumetric CT/MRI encoders and native task models without collapsing volumes into unrelated 2D samples or inventing unavailable spatial outputs.

## Dependencies

- [x] Phase 05 registry is accepted.
- [x] Phase 04 CT/MRI canonicalization and patching are accepted.
- [x] At least one CT and one MRI synthetic fixture are available.

## Scope boundaries

Allowed areas: `medfm/models/visual/` 3D/native task adapters, related registry/config files, and Phase 07 tests.

Do not implement full training recipes, generalized trainer logic, or slice-sequence VLMs.

## Delivery order

- [x] Implement `GenericMONAI3DAdapter` with a small local architecture.
- [x] Integrate CT-FM or FlexiCT-3D as the first real 3D encoder.
- [x] Integrate Triad as the first MRI encoder.
- [x] Add task-native NV-Segment-CTMR and MedSAM2 wrappers.
- [x] Add Merlin, remaining FlexiCT variants, M3D-CLIP, and M3D-LaMed as separately gated integrations.

## Shared native-3D checklist

- [x] Return pooled `[B,D]`, spatial `[B,N,D]`, feature-pyramid, and `[B,N,3]` coordinates where supported.
- [x] Return `None` with a declared limitation where unsupported.
- [x] Validate orientation, spacing, channel/sequence order, and shape before forward.
- [x] Generate token coordinates in the same order as flattened spatial tokens.
- [x] Preserve physical-coordinate transforms in outputs/metadata.
- [x] Support cropped-volume forward/backward and full-volume inference hooks.
- [x] Expose architecture-inspected LoRA targets; do not target every convolution.
- [x] Keep model-specific preprocessing represented in preprocess specs/configs.
- [x] Avoid device-specific allocation and CUDA-only interpolation/attention in the baseline adapter path.
- [x] Define a small fixed 3D patch shape for XLA compilation and a bounded production bucket set.
- [x] Record unsupported XLA operators and custom CUDA dependencies per model revision.
- [x] Keep sliding-window orchestration on the host with fixed-shape window batches.

## Adapter-specific checklist

### CT-FM / FlexiCT

- [x] Implement pinned checkpoint loading and native preprocessing compatibility.
- [x] Support embeddings, intermediate features, classification, segmentation attachment, and retrieval where available.
- [x] Register `flexict_2d`, `flexict_3d`, and `flexict_3d_vlm` separately.
- [x] Keep shared utilities internal without merging capability declarations.
- [x] Verify CLS and patch-token semantics against the pinned implementation.

### Merlin / Triad

- [x] Preserve Merlin image-text/phenotype/report capabilities only when checkpoint-compatible.
- [x] Reuse reviewed upstream preprocessing rather than approximating it.
- [x] Register Triad MAE and SimMIM variants separately.
- [x] Preserve MRI sequence channels and expose Swin intermediate features.
- [x] Test classification pooling and segmentation decoder attachment.

### Native segmentation and promptable models

- [x] Treat NV-Segment-CTMR as a native segmentation model first.
- [x] Preserve MONAI bundle metadata and preprocessing.
- [x] Add feature extraction or adapter injection only after architecture inspection.
- [x] Expose MedSAM2's initialize/encode/prompt/memory/decode lifecycle.
- [x] Keep MedSAM2 sequential memory semantics separate from native 3D token encoders.

### M3D research adapters

- [x] Keep M3D-CLIP retrieval/alignment separate from M3D-LaMed generation.
- [x] Mark research-only or gated constraints in registry outputs.
- [x] Require QLoRA/memory profile checks before language-component training.

## Tests and verification

- [x] Detect axis transposition using asymmetric synthetic volumes.
- [x] Verify spacing/orientation metadata survives adapter calls.
- [x] Verify token coordinate grids match token order and physical transforms.
- [x] Complete cropped-volume forward/backward.
- [x] Complete sliding-window reconstruction for a small synthetic segmentation model.
- [x] Verify LoRA targets only reviewed transformer/linear modules.
- [x] Save/reload adapter weights and compare outputs.
- [x] Record peak VRAM for representative patch sizes.
- [x] Verify unsupported full-volume sizes fail with a memory/config diagnostic.
- [x] Run fixed-shape forward/backward on CUDA and TPU for adapters declared supported.
- [x] Verify sliding-window batches do not trigger unbounded TPU recompilation.
- [x] Compare CUDA/TPU spatial token ordering and original-space reconstruction.
- [x] Save XLA compilation/fallback reports for real-model TPU smoke tests.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [CT-FM](https://github.com/project-lighter/CT-FM)
- [FlexiCT](https://github.com/ricklisz/FlexiCT)
- [Merlin](https://github.com/StanfordMIMI/Merlin)
- [Triad](https://github.com/wangshansong1/Triad)
- [NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
- [MedSAM2](https://medsam2.github.io/)
- [M3D](https://github.com/BAAI-DCAI/M3D)
- [MONAI sliding-window inferers](https://docs.monai.io/en/stable/inferers.html)

## Smoke command

```bash
uv run python -m medfm.tools.smoke --phase 07 --json
```

## Acceptance command

```bash
uv run pytest tests/phase_07 -q && uv run python -m medfm.tools.validate_phase --phase 07
```

## Exit criteria

- [x] Generic local 3D contract tests pass.
- [x] One real CT adapter passes inference and backward smoke tests.
- [x] One MRI adapter passes inference smoke or is blocked with a precise external reason.
- [x] Sliding-window output reconstructs the synthetic volume correctly.
- [x] At least one accepted 3D encoder exposes honest spatial tokens for Phase 09.
- [x] At least one real 3D path is TPU-accepted or the v1 TPU limitation is explicit with a tested generic MONAI fallback.

## Handoff

- [x] Identify the 3D encoder that unblocks Phase 09.
- [x] Publish crop/spacing/channel requirements and measured memory.
- [x] Publish token-coordinate and feature-pyramid semantics.
- [x] List native task wrappers that must bypass generic heads.
- [x] List fixed-shape buckets, unsupported XLA ops, and CUDA-only components.
