# Phase 07: Native 3D Visual-Encoder Adapters

## Objective

Support volumetric CT/MRI encoders and native task models without collapsing volumes into unrelated 2D samples or inventing unavailable spatial outputs.

## Dependencies

- [ ] Phase 05 registry is accepted.
- [ ] Phase 04 CT/MRI canonicalization and patching are accepted.
- [ ] At least one CT and one MRI synthetic fixture are available.

## Scope boundaries

Allowed areas: `medfm/models/visual/` 3D/native task adapters, related registry/config files, and Phase 07 tests.

Do not implement full training recipes, generalized trainer logic, or slice-sequence VLMs.

## Delivery order

- [ ] Implement `GenericMONAI3DAdapter` with a small local architecture.
- [ ] Integrate CT-FM or FlexiCT-3D as the first real 3D encoder.
- [ ] Integrate Triad as the first MRI encoder.
- [ ] Add task-native NV-Segment-CTMR and MedSAM2 wrappers.
- [ ] Add Merlin, remaining FlexiCT variants, M3D-CLIP, and M3D-LaMed as separately gated integrations.

## Shared native-3D checklist

- [ ] Return pooled `[B,D]`, spatial `[B,N,D]`, feature-pyramid, and `[B,N,3]` coordinates where supported.
- [ ] Return `None` with a declared limitation where unsupported.
- [ ] Validate orientation, spacing, channel/sequence order, and shape before forward.
- [ ] Generate token coordinates in the same order as flattened spatial tokens.
- [ ] Preserve physical-coordinate transforms in outputs/metadata.
- [ ] Support cropped-volume forward/backward and full-volume inference hooks.
- [ ] Expose architecture-inspected LoRA targets; do not target every convolution.
- [ ] Keep model-specific preprocessing represented in preprocess specs/configs.
- [ ] Avoid device-specific allocation and CUDA-only interpolation/attention in the baseline adapter path.
- [ ] Define a small fixed 3D patch shape for XLA compilation and a bounded production bucket set.
- [ ] Record unsupported XLA operators and custom CUDA dependencies per model revision.
- [ ] Keep sliding-window orchestration on the host with fixed-shape window batches.

## Adapter-specific checklist

### CT-FM / FlexiCT

- [ ] Implement pinned checkpoint loading and native preprocessing compatibility.
- [ ] Support embeddings, intermediate features, classification, segmentation attachment, and retrieval where available.
- [ ] Register `flexict_2d`, `flexict_3d`, and `flexict_3d_vlm` separately.
- [ ] Keep shared utilities internal without merging capability declarations.
- [ ] Verify CLS and patch-token semantics against the pinned implementation.

### Merlin / Triad

- [ ] Preserve Merlin image-text/phenotype/report capabilities only when checkpoint-compatible.
- [ ] Reuse reviewed upstream preprocessing rather than approximating it.
- [ ] Register Triad MAE and SimMIM variants separately.
- [ ] Preserve MRI sequence channels and expose Swin intermediate features.
- [ ] Test classification pooling and segmentation decoder attachment.

### Native segmentation and promptable models

- [ ] Treat NV-Segment-CTMR as a native segmentation model first.
- [ ] Preserve MONAI bundle metadata and preprocessing.
- [ ] Add feature extraction or adapter injection only after architecture inspection.
- [ ] Expose MedSAM2's initialize/encode/prompt/memory/decode lifecycle.
- [ ] Keep MedSAM2 sequential memory semantics separate from native 3D token encoders.

### M3D research adapters

- [ ] Keep M3D-CLIP retrieval/alignment separate from M3D-LaMed generation.
- [ ] Mark research-only or gated constraints in registry outputs.
- [ ] Require QLoRA/memory profile checks before language-component training.

## Tests and verification

- [ ] Detect axis transposition using asymmetric synthetic volumes.
- [ ] Verify spacing/orientation metadata survives adapter calls.
- [ ] Verify token coordinate grids match token order and physical transforms.
- [ ] Complete cropped-volume forward/backward.
- [ ] Complete sliding-window reconstruction for a small synthetic segmentation model.
- [ ] Verify LoRA targets only reviewed transformer/linear modules.
- [ ] Save/reload adapter weights and compare outputs.
- [ ] Record peak VRAM for representative patch sizes.
- [ ] Verify unsupported full-volume sizes fail with a memory/config diagnostic.
- [ ] Run fixed-shape forward/backward on CUDA and TPU for adapters declared supported.
- [ ] Verify sliding-window batches do not trigger unbounded TPU recompilation.
- [ ] Compare CUDA/TPU spatial token ordering and original-space reconstruction.
- [ ] Save XLA compilation/fallback reports for real-model TPU smoke tests.

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
python -m medfm.cli.models smoke ct_fm --input tests/fixtures/ct3d_small.nii.gz
```

## Acceptance command

```bash
pytest tests/phase_07 -q && python -m medfm.tools.validate_phase --phase 07
```

## Exit criteria

- [ ] Generic local 3D contract tests pass.
- [ ] One real CT adapter passes inference and backward smoke tests.
- [ ] One MRI adapter passes inference smoke or is blocked with a precise external reason.
- [ ] Sliding-window output reconstructs the synthetic volume correctly.
- [ ] At least one accepted 3D encoder exposes honest spatial tokens for Phase 09.
- [ ] At least one real 3D path is TPU-accepted or the v1 TPU limitation is explicit with a tested generic MONAI fallback.

## Handoff

- [ ] Identify the 3D encoder that unblocks Phase 09.
- [ ] Publish crop/spacing/channel requirements and measured memory.
- [ ] Publish token-coordinate and feature-pyramid semantics.
- [ ] List native task wrappers that must bypass generic heads.
- [ ] List fixed-shape buckets, unsupported XLA ops, and CUDA-only components.
