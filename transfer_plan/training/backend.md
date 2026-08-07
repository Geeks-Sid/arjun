# transfer_plan/training/backend.md

Source: `medfm/training/backend.py` (CPU/CUDA/XLA accelerator abstraction).
Wave: 2 — **keep by design**.

## Transfer checklist

- [ ] `AcceleratorBackend` (Cpu/Cuda/XlaTpu), `create_backend`, `resolve_attention`, memory
      snapshots, `parse_xla_metrics` → **keep** — this is the intentional CPU/CUDA/TPU
      abstraction (ADR-driven); `accelerate` (installed) does not cover the PyTorch/XLA path the
      same way. Do not replace. It already wraps `torch.cuda` / `torch_xla` APIs (library-backed).
- [ ] Distributed/parity helpers in `training/distributed.py`, `training/data.py`
      (`DeterministicDistributedSampler`) → **keep** — group-aware, resumable, static-shape
      semantics exceed `monai.data.DistributedSampler`/torch defaults.

## Tests
`tests/phase_10/test_config_and_backend.py`, `tests/phase_12/test_config_backend.py`.
