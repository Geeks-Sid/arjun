# Phase 13 unresolved runtime and acceptance issues

These are explicit evidence gaps, not silent fallbacks:

- `torch_xla` is not installed. TPU PJRT/BF16 execution, no-recompile steady-state checks, replicated/SPMD reduction, TPU checkpoint/export, and TPU parity were not executable.
- `bitsandbytes` is not installed. CUDA NF4 QLoRA capability validation remains fail-closed; no NF4 result is claimed.
- Approved production checkpoints and de-identified clinical datasets are unavailable. All executed recipe runs use offline random contract adapters and synthetic data.
- No representative large-model VRAM/throughput, external-site evaluation, human review, or clinical-unit performance result is claimed.
- Visual-dependence ablation plumbing is exercised, but offline random weights are not accepted as grounding evidence. The report preserves image/no-image/shuffled deltas and pass criteria instead of hiding a failed shuffled condition.
- CUDA multi-device DDP/FSDP and topology-change resume are inherited Phase 12 hardware gaps and were not rerun for Phase 13.
- Repository-wide `python -m pytest tests -q` remains blocked by the unavailable optional `safetensors` package and duplicate test module basenames that trigger pytest import-mismatch collection errors. The focused Phase 13 and Phase 12 regression suites pass.
- The bare `pytest` executable in this workstation did not put the project package on `sys.path`; `python -m pytest tests/phase_13 -q` is the passing acceptance invocation.

No production or clinical acceptance is inferred from these offline contract results. The CUDA/TPU and approved-data checks remain prerequisites for later release acceptance.
