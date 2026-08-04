# Phase 06: 2D Visual-Encoder Adapters

## Objective

Implement standardized 2D visual encoders with honest capabilities, external preprocessing, frozen and LoRA modes, checkpoint round-trips, and task-head compatibility.

## Dependencies

- [ ] Phase 05 registry is accepted.
- [ ] Phase 04 can produce tensors from adapter preprocess specs.
- [ ] License approval exists before gated weights are loaded.

## Scope boundaries

Allowed areas: `medfm/models/visual/` 2D adapters, corresponding registry records/configs, and Phase 06 tests.

Do not implement trainer behavior, task-specific losses, or external LLM bridges.

## Delivery order

- [ ] First vertical slice: `GenericHFVisionAdapter` using a tiny local test model.
- [ ] First real model: `MedSigLIPAdapter`.
- [ ] Second radiology model: `RADDINOAdapter`.
- [ ] Pathology tile model: `HOptimus0Adapter` shared with Phase 08.
- [ ] Native visual pathway: `MedGemmaVisionAdapter` where supported.
- [ ] Add generic timm and optional CONCH adapters after core acceptance.

## Shared adapter checklist

- [ ] Implement adapter construction from an immutable `ModelSpec`.
- [ ] Expose `preprocess_spec()` without preprocessing inside `forward`.
- [ ] Return pooled embeddings, spatial tokens, hidden states, and feature maps only when genuinely available.
- [ ] Return token masks and documented coordinates for padded/spatial outputs.
- [ ] Preserve native output in a debug field without coupling tasks to it.
- [ ] Support frozen extraction and deterministic evaluation mode.
- [ ] Support task-head attachment without importing a concrete task.
- [ ] Expose modules eligible for LoRA and reasons for each target.
- [ ] Ensure unsupported requests fail with typed capability errors.
- [ ] Document expected input range, color space, channels, resolution, crop behavior, and patch size.
- [ ] Use input/model devices rather than `.cuda()` or CUDA-specific tensor creation.
- [ ] Prefer PyTorch SDPA or a pure-PyTorch attention path; register CUDA custom kernels as optional accelerations.
- [ ] Expose a fixed-resolution/static-batch smoke configuration for TPU compilation.
- [ ] Record per-adapter CPU/CUDA/TPU status and unsupported operators in the registry.

## Model-specific checklist

### MedSigLIP

- [ ] Support image and text embeddings and normalized image-text similarity.
- [ ] Expose patch tokens if the pinned revision supports them.
- [ ] Validate 448 x 448 preprocessing against the model processor.
- [ ] Support frozen, contrastive, classification-head, and vision-LoRA modes.
- [ ] Make external-VLM bridge attachment possible through shared spatial tokens.

### RAD-DINO

- [ ] Support pooled and dense patch features.
- [ ] Validate chest-X-ray channel and resize behavior.
- [ ] Expose feature representations suitable for classification, segmentation, retrieval, and bridging.
- [ ] Document any hidden-state hooks and pin them to the model revision.

### H-Optimus-0

- [ ] Support tile CLS, patch tokens, and intermediate hidden states.
- [ ] Default to frozen BF16 or cached embeddings.
- [ ] Add embedding-cache generation with complete model/preprocess metadata.
- [ ] Gate LoRA work behind an accepted frozen baseline and memory measurement.

### Other adapters

- [ ] Keep MedGemma native visual output separate from full native-VLM behavior.
- [ ] Implement generic HF/timm fallbacks only for architectures matching declared capabilities.
- [ ] Keep optional CONCH unavailable until license and repository behavior are reviewed.

## Tests and verification

- [ ] Run local tiny-model contract tests on CPU.
- [ ] For each real adapter, load one pinned checkpoint in the protected GPU environment.
- [ ] Run one synthetic/model-valid input and assert exact output semantics.
- [ ] Verify frozen mode has zero trainable backbone parameters.
- [ ] Attach a classification head and complete backward.
- [ ] Inject LoRA and verify gradients only on intended modules.
- [ ] Save/reload adapters and compare outputs within dtype-specific tolerance.
- [ ] Verify preprocess mismatches and unsupported spatial-token requests fail clearly.
- [ ] Record peak VRAM by loading mode.
- [ ] Run one fixed-shape optimizer step on CUDA and TPU for every adapter declared supported.
- [ ] Inspect XLA metrics for CPU fallbacks/recompilation on TPU.
- [ ] Compare deterministic CUDA and TPU embeddings/logits within model-specific tolerances.
- [ ] Verify optional CUDA attention kernels can be disabled without changing the public adapter contract.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [MedSigLIP model card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)
- [MedSigLIP repository](https://github.com/Google-Health/medsiglip)
- [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino)
- [H-Optimus](https://www.bioptimus.com/h-optimus)
- [MedGemma repository](https://github.com/google-health/medgemma)

## Smoke command

```bash
python -m medfm.cli.models smoke medsiglip_448
```

## Acceptance command

```bash
pytest tests/phase_06 -q && python -m medfm.tools.validate_phase --phase 06
```

## Exit criteria

- [ ] Generic contract tests pass without network access.
- [ ] MedSigLIP and RAD-DINO pass real-checkpoint inference smoke tests.
- [ ] At least one 2D adapter passes frozen backward, LoRA gradient, and reload tests.
- [ ] H-Optimus integration is accepted here or explicitly handed to Phase 08.
- [ ] Capabilities never imply unavailable spatial information.
- [ ] Every required adapter has an evidence-backed CUDA/TPU status; unsupported TPU models are blocked explicitly.

## Handoff

- [ ] Publish adapter output dimensions and token-coordinate semantics.
- [ ] Publish measured memory profiles.
- [ ] Identify the accepted 2D adapter that unblocks Phase 09.
- [ ] List revision-sensitive hooks and tests.
- [ ] List custom kernels, fallback paths, and backend parity tolerances.
