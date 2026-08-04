# Phase 09: Language Models and Vision-to-Language Bridges

## Objective

Support native VLMs and external 2D/3D/WSI encoders through explicit language adapters, coordinate-aware bridges, bounded visual tokens, and correct causal-language loss masking.

## Dependencies

- [ ] Phase 06 has one accepted 2D adapter.
- [ ] Phase 07 has one accepted native 3D adapter with usable tokens.
- [ ] Phase 08 has one accepted WSI token/embedding path.
- [ ] Phase 04 VLM text preparation and collators are accepted.

## Scope boundaries

Allowed areas: `medfm/models/language/`, `medfm/models/bridges/`, native VLM wrappers, bridge configs, and Phase 09 tests.

Do not implement generalized optimizer/trainer behavior or PEFT injection internals.

## Mode separation

- [ ] Implement native VLM mode using the model's processor, tower, connector, and language model.
- [ ] Implement external-encoder mode using framework visual tokens and a bridge.
- [ ] Give modes separate registry capabilities and configuration names.
- [ ] Prevent datasets from depending on model-specific placeholder syntax.

## Language adapter checklist

- [ ] Implement `GenericHFCausalLMAdapter` with architecture checks.
- [ ] Implement `GemmaCausalLMAdapter` and `MedGemmaAdapter`.
- [ ] Add M3D-LaMed language integration behind research/license gates.
- [ ] Verify `inputs_embeds` or official connector support before accepting external tokens.
- [ ] Implement tokenization, embedding lookup, forward with visuals, and deterministic generation contracts.
- [ ] Preserve model-specific chat templates and stop tokens in versioned configs.
- [ ] Avoid logging raw clinical prompts/outputs by default.
- [ ] Retie shared input/output embeddings after TPU placement when required by the architecture.
- [ ] Expose fixed text and visual-token bucket configurations for XLA.
- [ ] Keep CUDA FlashAttention optional and retain an SDPA/XLA-compatible attention path.
- [ ] Record generation operations that cause XLA recompilation or host synchronization.

## Bridge checklist

- [ ] Implement `VisionLanguageBridge` and `ProjectedVisualTokens` contracts.
- [ ] Implement a linear bridge for smoke tests.
- [ ] Implement the two-layer MLP bridge as the first practical default.
- [ ] Implement a fixed-query Perceiver resampler with masks.
- [ ] Add Q-Former and spatial-pyramid variants only after baseline acceptance.
- [ ] Validate input/output dimensions, token masks, and fixed token budgets.
- [ ] Fully train bridges; do not add unnecessary LoRA to them.
- [ ] Implement bridges with XLA-lowered PyTorch operations and static query counts.
- [ ] Avoid data-dependent token pruning inside compiled forward; perform selection before collation or use fixed masked top-k.

## Coordinate and token placement checklist

- [ ] Implement 2D normalized position, image index, view, and timepoint encodings.
- [ ] Implement 3D normalized/physical positions, spacing, and series index encodings.
- [ ] Implement WSI slide position, MPP, pyramid level, and slide index encodings.
- [ ] Define model-specific visual boundary/token placement adapters around a common dataset representation.
- [ ] Mask visual placeholders and prompt tokens from LM loss.
- [ ] Preserve visual attention masks through projection and language forward.

## Training-stage support

- [ ] Stage 1 configuration: train bridge/boundary embeddings; freeze vision and LLM.
- [ ] Stage 2 configuration: train bridge plus language LoRA/QLoRA; freeze vision.
- [ ] Stage 3 configuration: add late vision LoRA after Stage 2 evidence.
- [ ] Stage 4 configuration: support weighted multitask batches without implementing scheduler internals here.
- [ ] Expose trainable-module declarations for the Phase 10/12 gradient audit.

## Tests and verification

- [ ] Use tiny local encoders/LMs to test bridge dimensions and masks on CPU.
- [ ] Produce valid loss for external 2D, native 3D, and WSI visual tokens.
- [ ] Produce valid loss through at least one native VLM path.
- [ ] Verify only assistant output tokens contribute to loss.
- [ ] Verify visual tokens receive bridge gradients.
- [ ] Verify frozen encoders and LMs receive no Stage 1 gradients.
- [ ] Verify padded visual tokens have no effect on outputs within tolerance.
- [ ] Verify coordinate embeddings change the intended visual representation.
- [ ] Verify generation respects stop tokens and output limits.
- [ ] Measure VRAM at 32, 64, and 128 visual tokens on the target GPU.
- [ ] Measure TPU HBM/compile behavior for fixed 32, 64, and 128 visual-token buckets where feasible.
- [ ] Compare CUDA/TPU loss masking, bridge gradients, and one-step updates.
- [ ] Assert no steady-state XLA recompilation for repeated identical buckets.
- [ ] Test tied weights and adapter reload after XLA placement.

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

- [ ] 2D, native 3D, and WSI external encoders each produce a valid LM loss.
- [ ] One native VLM produces a valid loss.
- [ ] Loss masking and visual gradient flow are proven by tests.
- [ ] Frozen modules have no gradients in the relevant stages.
- [ ] Visual token budgets stay within the target GPU envelope.
- [ ] At least one language/bridge path passes protected TPU training smoke, or TPU support remains explicitly blocked.

## Handoff

- [ ] Publish language/bridge trainable-module names for PEFT and trainer phases.
- [ ] Publish token placement and masking semantics.
- [ ] Publish measured memory by visual/text token lengths.
- [ ] Identify deferred bridge variants and why they are not required for baseline recipes.
- [ ] Publish backend-specific attention, token buckets, compile counts, and parity tolerances.
