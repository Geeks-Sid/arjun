# Phase 09 summary

Phase 09 implements the language and vision-to-language contract without coupling datasets to model-specific placeholder syntax.

## Delivered

- External causal-LM adapters: `GenericHFCausalLMAdapter`, `GemmaCausalLMAdapter`, and gated `M3DLaMedAdapter`.
- Native `MedGemmaAdapter` mode with processor slot, connector verification, native capability declaration, deterministic generation, stop strings/token IDs, tied-weight retie, and attention/backend declarations.
- Linear, two-layer MLP, and fixed-query Perceiver bridges returning the core `ProjectedVisualTokens` contract.
- 2D, native 3D, and WSI coordinate encoders; learned boundary embeddings and prefix/suffix visual placement.
- Assistant-only causal loss masking, visual/prompt/padding attention masks, static text/visual buckets, and Stage 1-4 trainable-module declarations.
- CPU tests and smoke cover bridge shapes/masks, 2D/3D/WSI losses, native loss, gradient flow, frozen Stage 1 modules, coordinate changes, generation limits/stops, tied weights, and research gates.

## Verification

- `pytest tests/phase_09 -q`: 10 passed.
- `python -m medfm.tools.smoke --phase 09`: 1 check passed.
- `python -m medfm.tools.validate_phase --phase 09`: passed after report artifacts were written.
- Ruff checks for Phase 09 implementation, tests, and tools: clean.
- CUDA training-reference peak memory on NVIDIA GeForce RTX 4060 Laptop GPU (8,189,181,952 bytes): 32 visual tokens 33.204 MiB allocated / 40 MiB reserved; 64 tokens 50.343 / 58 MiB; 128 tokens 68.016 / 76 MiB. Reference model: tiny local LM (hidden 256, depth 2) plus MLP bridge; text input was short.

TPU compilation/HBM and CUDA/TPU parity remain explicitly blocked because `torch_xla` is not installed on this workstation. The implementation records the zero-steady-state-recompilation gate and preserves XLA-safe static operations, but no TPU result is fabricated.
