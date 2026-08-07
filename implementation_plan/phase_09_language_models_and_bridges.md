# Phase 09: Language Models and Vision-to-Language Bridges

## Objective

Support native VLMs and external 2D/3D/WSI encoders through explicit language adapters, coordinate-aware bridges, bounded visual tokens, and correct causal-language loss masking.

## Dependencies

- [x] Phase 06 has one accepted 2D adapter.
- [x] Phase 07 has one accepted native 3D adapter with usable tokens.
- [x] Phase 08 has one accepted WSI token/embedding path.
- [x] Phase 04 VLM text preparation and collators are accepted.

## Scope boundaries

Allowed areas: `medfm/models/language/`, `medfm/models/bridges/`, native VLM wrappers, bridge configs, and Phase 09 tests.

Do not implement generalized optimizer/trainer behavior or PEFT injection internals.

## Mode separation

- [x] Implement native VLM mode using the model's processor, tower, connector, and language model.
- [x] Implement external-encoder mode using framework visual tokens and a bridge.
- [x] Give modes separate registry capabilities and configuration names.
- [x] Prevent datasets from depending on model-specific placeholder syntax.

## Language adapter checklist

- [x] Implement `GenericHFCausalLMAdapter` with architecture checks.
- [x] Implement `GemmaCausalLMAdapter` and `MedGemmaAdapter`.
- [x] Add M3D-LaMed language integration behind research/license gates.
- [x] Verify `inputs_embeds` or official connector support before accepting external tokens.
- [x] Implement tokenization, embedding lookup, forward with visuals, and deterministic generation contracts.
- [x] Preserve model-specific chat templates and stop tokens in versioned configs.
- [x] Avoid logging raw clinical prompts/outputs by default.
- [x] Retie shared input/output embeddings after TPU placement when required by the architecture.
- [x] Expose fixed text and visual-token bucket configurations for XLA.
- [x] Keep CUDA FlashAttention optional and retain an SDPA/XLA-compatible attention path.
- [x] Record generation operations that cause XLA recompilation or host synchronization.

## Bridge checklist

- [x] Implement `VisionLanguageBridge` and `ProjectedVisualTokens` contracts.
- [x] Implement a linear bridge for smoke tests.
- [x] Implement the two-layer MLP bridge as the first practical default.
- [x] Implement a fixed-query Perceiver resampler with masks.
- [x] Defer Q-Former and spatial-pyramid variants; the accepted linear/MLP/Perceiver baseline is sufficient for Phase 09 recipes.
- [x] Validate input/output dimensions, token masks, and fixed token budgets.
- [x] Fully train bridges; do not add unnecessary LoRA to them.
- [x] Implement bridges with XLA-lowered PyTorch operations and static query counts.
- [x] Avoid data-dependent token pruning inside compiled forward; perform selection before collation or use fixed masked top-k.

## Coordinate and token placement checklist

- [x] Implement 2D normalized position, image index, view, and timepoint encodings.
- [x] Implement 3D normalized/physical positions, spacing, and series index encodings.
- [x] Implement WSI slide position, MPP, pyramid level, and slide index encodings.
- [x] Define model-specific visual boundary/token placement adapters around a common dataset representation.
- [x] Mask visual placeholders and prompt tokens from LM loss.
- [x] Preserve visual attention masks through projection and language forward.

## Training-stage support

- [x] Stage 1 configuration: train bridge/boundary embeddings; freeze vision and LLM.
- [x] Stage 2 configuration: train bridge plus language LoRA/QLoRA; freeze vision.
- [x] Stage 3 configuration: add late vision LoRA after Stage 2 evidence.
- [x] Stage 4 configuration: support weighted multitask batches without implementing scheduler internals here.
- [x] Expose trainable-module declarations for the Phase 10/12 gradient audit.

## Tests and verification

- [x] Use tiny local encoders/LMs to test bridge dimensions and masks on CPU.
- [x] Produce valid loss for external 2D, native 3D, and WSI visual tokens.
- [x] Produce valid loss through at least one native VLM path.
- [x] Verify only assistant output tokens contribute to loss.
- [x] Verify visual tokens receive bridge gradients.
- [x] Verify frozen encoders and LMs receive no Stage 1 gradients.
- [x] Verify padded visual tokens have no effect on outputs within tolerance.
- [x] Verify coordinate embeddings change the intended visual representation.
- [x] Verify generation respects stop tokens and output limits.
- [x] Measure VRAM at 32, 64, and 128 visual tokens on the target GPU.
- [x] Document TPU HBM/compile behavior as blocked: `torch_xla` is not installed on this workstation.
- [x] Document CUDA/TPU loss-mask, gradient, and update parity as blocked until a TPU runtime is available.
- [x] Define the zero-steady-state-recompilation XLA gate; runtime measurement is blocked with the unavailable TPU dependency.
- [x] Test tied weights and the local retie contract; XLA placement/reload execution is blocked with the unavailable TPU dependency.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [MedGemma documentation](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [MedGemma model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [M3D](https://github.com/BAAI-DCAI/M3D)
- [Accelerate TPU training](https://huggingface.co/docs/accelerate/basic_tutorials/tpu)
- [PyTorch/XLA troubleshooting](https://docs.pytorch.org/xla/master/debug.html)

## Smoke command

```bash
python -m medfm.tools.smoke --phase 09
```

## Acceptance command

```bash
pytest tests/phase_09 -q && python -m medfm.tools.validate_phase --phase 09
```

## Exit criteria

- [x] 2D, native 3D, and WSI external encoders each produce a valid LM loss.
- [x] One native VLM produces a valid loss.
- [x] Loss masking and visual gradient flow are proven by tests.
- [x] Frozen modules have no gradients in the relevant stages.
- [x] Visual token budgets stay within the target GPU envelope.
- [x] At least one language/bridge path passes protected TPU training smoke, or TPU support remains explicitly blocked.

## Handoff

- [x] Publish language/bridge trainable-module names for PEFT and trainer phases.
- [x] Publish token placement and masking semantics.
- [x] Publish measured memory by visual/text token lengths.
- [x] Identify deferred bridge variants and why they are not required for baseline recipes.
- [x] Publish backend-specific attention, token buckets, compile counts, and parity tolerances.
