# CUDA QLoRA vs TPU BF16 LoRA guidance

Research release baseline; not for clinical use. Authority: ADR-0009
(`docs/architecture/adr_0009_cuda_qlora_vs_tpu_bf16_lora.md`).

## CUDA QLoRA (NF4, bitsandbytes)

- Enabled via the `cuda` extra; gated at runtime by
  `medfm.peft.quantization` (NF4 requires `torch.cuda.is_available()`).
- 4-bit base + frozen adapter checkpoints; train with the CUDA path; the
  portable artifact is the adapter-only safetensors bundle, not the quantized
  base.
- A bundle must declare CUDA-only tensors; those are **not** portable to TPU.

## TPU BF16 LoRA

- No bitsandbytes, no FlashAttention, no cuCIM on the TPU baseline
  (`uv sync --extra tpu` excludes them); the release gate enforces the config
  side (`medfm/tools/release.py::no_tpu_nf4`).
- Static-shape buckets are mandatory (ADR-0008) — pad or reject out-of-bucket
  requests; the release matrix requires compile/fallback counters from a
  protected TPU job before a TPU workflow is claimed.

## Which to choose

- Local interactive research on NVIDIA hardware → CUDA QLoRA.
- JAX-free TPU research / large replicated devices → BF16 LoRA.
- Anything that will be served on TPU later must never introduce CUDA-only
  kernels or NF4 into the trained adapter state.
