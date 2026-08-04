# ADR 0006: Adapter-only checkpoints are canonical

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

PEFT-first training (ADR 0002) means trained state lives in adapters, heads, and bridges, not base weights. Checkpoints must be portable across CPU/CUDA/TPU, license-safe (no redistribution of gated base weights), and auditable.

## Decision

The **canonical checkpoint is an adapter-only export**: accelerator-neutral CPU safetensors containing only trained parameters (LoRA/QLoRA adapters, task heads, bridges/projectors, modules-to-save), plus a manifest with base-model id, pinned revision, configuration hash, and provenance metadata. Resumable distributed checkpoints (optimizer state, sharding) are a separate, non-canonical artifact. Full merged-weight exports are optional conveniences, never canonical, and prohibited where base-weight licenses disallow redistribution.

## Alternatives considered

- **Full merged checkpoints as canonical:** duplicates gigabytes per run, risks redistributing restricted weights, ties the artifact to one backend's dtype. Rejected.
- **Framework-specific formats (PEFT hub layout only):** good interchange but insufficient metadata discipline; we adopt the tensor layout discipline and add our manifest. Partially adopted.

## Consequences

- Export must load on CPU, CUDA, and TPU for any model declared portable (cross-backend test in the acceptance matrix).
- A checkpoint without a base-model reference and configuration hash is not exportable (`docs/reproducibility_policy.md`).
- Storage and deletion rules (`docs/data_governance.md`) apply to adapter checkpoints, which are small and cheap to retain.

## Reversal conditions

Reverse if a deployment target requires self-contained weights and licensing permits — then allow merged exports for that target via a new ADR, while keeping adapter-only as the source of truth.
