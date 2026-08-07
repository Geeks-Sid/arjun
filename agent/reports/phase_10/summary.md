# Phase 10 summary

Phase 10 delivers an auditable PEFT subsystem without adding task recipes or a generalized trainer loop.

## Delivered

- Versioned `LoRAConfig` and independent `QuantizationConfig` records with canonical hashes, aliases, validation, target confirmation, and backend policy plans.
- Architecture-aware target inspection/resolution for language models, 2D vision transformers, and reviewed late-stage 3D transformer targets. Patch embeddings, stems, normalizations, and broad/zero matches fail closed.
- Native named-adapter LoRA injection for visual and language adapters, deterministic freezing, modules-to-save/component attachment, optimizer-visible parameter groups, and trainability audits.
- CUDA NF4/QLoRA capability gates with lazy optional-dependency imports, k-bit preparation hooks, quantized-parameter markers, KV-cache disabling, and optimizer exclusion checks. TPU is explicitly BF16 LoRA/frozen-base and rejects bitsandbytes QLoRA.
- CPU safetensors adapter-only checkpoint export/load with base-model provenance, pinned revision, architecture, configuration hash, separate adapter/component files, multiple named-adapter reload, wrong-base rejection, and merge/unmerge equivalence checks.
- `python -m medfm.cli.peft inspect --model <id> [--format json]` and a Phase 10 validator contract check.
- Phase 10 behavioral tests cover configuration, policy, resolver boundaries, 2D/3D/language gradients, quantization safety, CLI reporting, checkpoint round-trips, merge equivalence, and failure-closed targeting.

## Backend evidence

The workstation exposes CUDA, and a tiny CUDA BF16 LoRA optimizer step completed on `cuda:0`. `bitsandbytes`, `transformers`, and `torch_xla` are not installed, so the protected 4B NF4 and TPU topology checks are reported as not applicable rather than fabricated. The local 8 GB development GPU is not the 48 GB target baseline for a real 4B run.
