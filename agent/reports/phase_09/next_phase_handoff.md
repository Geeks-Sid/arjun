# Phase 09 handoff

## Trainable module names

- `bridge`: linear/MLP/Perceiver bridge parameters; fully trainable, no LoRA.
- `boundary`: `VisualBoundaryEmbeddings.embeddings`; fully trainable.
- `language_lora`: reserved declaration for Phase 10/12 PEFT injection; Phase 09 does not inject adapters.
- `vision_lora`: reserved declaration for late Stage 3 vision adaptation; Phase 09 does not inject adapters.
- Stage 1 freezes `vision` and the language model while training `bridge` and `boundary`.
- Stage 2 adds `language_lora` and keeps vision frozen; Stage 3 adds `vision_lora`; Stage 4 requires explicit positive task weights.

## Placement and masking

External recipes use `TokenPlacementConfig(config_name="external-prefix-v1")`: `[boundary_begin, visual_tokens, boundary_end, text_tokens]`. A suffix policy is available for model-specific adapters. Visual placeholders are framework-owned spans, not dataset token IDs. `ProjectedVisualTokens.token_mask` becomes the language attention mask; padded visual positions and both boundary positions always use `-100` labels. Phase 04 assistant-only labels remain the only supervised text positions; optional `TokenizedText.metadata["prompt_token_mask"]` is masked defensively.

Native MedGemma is registered separately as `medgemma_native` with `native_visual_connector=True` and `accepts_inputs_embeds=False`. External generic/Gemma paths use `inputs_embeds=True`. M3D-LaMed is `m3d_lamed_external` and remains research/license gated.

## Static buckets and backend behavior

- Visual tokens: 32, 64, 128.
- Text tokens: 256, 512, 1024.
- CUDA default attention: SDPA; FlashAttention 2 is optional and never imported by CPU/XLA code.
- XLA attention: declared `xla`; no custom CUDA operations; Perceiver query count is static; no data-dependent pruning.
- Generation records operations with `causes_xla_recompilation` and `host_synchronization` flags.
- TPU compile/HBM/parity counts are explicitly blocked until a `torch_xla` runtime is available; target acceptance threshold is zero steady-state recompilation after warmup.

## Deferred variants

Q-Former and spatial-pyramid bridges are intentionally deferred. Linear, two-layer MLP, and fixed-query Perceiver bridges cover the accepted baseline recipes and provide the required mask/gradient/static-shape evidence without introducing model-internal cross-attention surgery.

## Memory evidence

On the NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), the tiny hidden-256/depth-2 training reference peaked at 33.204, 50.343, and 68.016 MiB allocated for 32, 64, and 128 visual tokens respectively (reserved: 40, 58, and 76 MiB). These are contract measurements, not production MedGemma 4B sizing.
