# Known limitations (research release)

This repository is **research software**; it is not validated for clinical use
and makes no diagnostic, treatment, or safety claim. See
`docs/clinical_safety_scope.md`.

## Blocked models and unsupported modalities/tasks

- `model_registry/v1_scope.yaml` + the generated
  `docs/release/support_matrix.md` are authoritative. Every model is
  `UNTESTED`, `BLOCKED_*` (with reason), `CPU_CONTRACT_ONLY`, or `NOT_APPLICABLE`
  on each backend until a protected hardware job records smoke evidence.
- Models with unresolved license terms (`status: blocked_unresolved`) cannot be
  downloaded or loaded for any purpose until the terms are reviewed and
  `acceptance` is recorded.
- Custom third-party CUDA models and their custom operators are **not
  TPU-compatible** unless the model card explicitly says otherwise.

## Hardware requirements

- CPU: any x86-64 with ≥ 8 GB RAM for contract/tiny workloads; medical/pathology
  extras required for those readers.
- CUDA: a local GPU plus the `cuda` extra (bitsandbytes for QLoRA). No
  hard-coded `.cuda()` path; placement follows `backend.py`.
- TPU: PyTorch/XLA (the `tpu` extra). NF4/bitsandbytes is never enabled on TPU;
  LoRA is BF16 (see ADR-0009 and `cuda_qlora_vs_tpu_bf16_lora.md`).

## Known nondeterministic kernels

Nondeterminism is recorded per kernel (`tests/phase_18/test_reliability.py`
pins the CPU deterministic contract). Accelerator kernels (cuBLAS reductions,
XLA op fusion) may differ from CPU within the documented cross-backend
tolerances; golden tests use those tolerances, not exact equality, on
non-CPU backends.
