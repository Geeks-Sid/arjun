# ADR 0009: CUDA QLoRA vs TPU BF16 LoRA support policy

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

QLoRA (bitsandbytes NF4) keeps a 4B VLM trainable on a single 48 GB CUDA GPU (and on smaller dev devices), but bitsandbytes does not list TPU as a supported backend. PyTorch/XLA quantized operations are experimental. Precision policy must therefore differ per backend without forking model code (ADR 0007).

## Decision

- **CUDA:** BF16 preferred (native hardware support); FP16 with gradient scaling only where BF16 is unavailable or a model requires it; bitsandbytes NF4 QLoRA permitted only on backends explicitly supported by the installed bitsandbytes build; quantized base weights must be excluded from optimizer state; SDPA default, FlashAttention only after parity tests.
- **TPU:** XLA BF16 autocast for forward/loss; **no FP16-style gradient scaling**; **no bitsandbytes NF4 QLoRA** — never labeled TPU-supported; baseline TPU strategy is BF16 LoRA, frozen encoders, bridge/head/decoder training, and sharding; PyTorch/XLA quantized ops are experimental and outside baseline until a dedicated parity phase.
- Numerically sensitive reductions, losses, accumulators, calibration, and metrics stay FP32 on both backends.
- Quantization config is validated against backend capability **before model loading**; a QLoRA config targeting TPU fails fast with an actionable error.

## Alternatives considered

- **QLoRA everywhere:** impossible on TPU today. Rejected.
- **BF16 LoRA everywhere (drop QLoRA):** makes large-VLM recipes needlessly memory-bound on single GPUs and excludes smaller dev devices entirely. Rejected.
- **FP16 everywhere on TPU:** XLA BF16 has better dynamic range; gradient scaling adds complexity XLA handles poorly. Rejected.

## Consequences

- Recipe configs must declare per-backend precision/quantization; the registry capability matrix records supported precision per model/backend.
- Cross-backend parity tests compare CUDA QLoRA and TPU BF16-LoRA runs against declared tolerances, not bitwise equality.
- Adapter-only exports (ADR 0006) remain dtype-neutral CPU safetensors.

## Reversal conditions

Reverse the TPU quantization ban when bitsandbytes or PyTorch/XLA ships supported, parity-tested quantization on TPU; adopt via a new ADR with benchmark evidence.
