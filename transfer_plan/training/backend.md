# transfer_plan/training/backend.md

Source: `medfm/training/backend.py` (CPU/CUDA/XLA accelerator abstraction).
Wave: 2 — **keep by design**.

## Transfer checklist

- [x] `AcceleratorBackend` (Cpu/Cuda/XlaTpu), `create_backend`, `resolve_attention`, memory
      snapshots, `parse_xla_metrics` → **keep** — this is the intentional CPU/CUDA/TPU
      abstraction (ADR-driven); `accelerate` (installed) does not cover the PyTorch/XLA path the
      same way. Do not replace. It already wraps `torch.cuda` / `torch_xla` APIs (library-backed).
- [x] Distributed/parity helpers in `training/distributed.py`, `training/data.py`
      (`DeterministicDistributedSampler`) → **keep** — group-aware, resumable, static-shape
      semantics exceed `monai.data.DistributedSampler`/torch defaults.

## Result
- **Keep** `AcceleratorBackend`, backend construction/attention resolution, memory snapshots, and XLA metric parsing: the ADR-driven CPU/CUDA/XLA contract is broader than `accelerate`'s portable path, while the implementation already delegates device/runtime operations to PyTorch and optional torch_xla APIs.
- **Keep** distributed/data parity helpers: resumable group-aware sampling and static-shape bucket semantics are not a drop-in `monai.data.DistributedSampler`/torch default.
- **Parity drift:** none applicable; no library replacement was attempted.
- **Source reads:** `medfm/training/backend.py`, `distributed.py`, and `data.py` confirmed the backend-neutral public contracts and accelerator-specific hooks.
- **Verification:** `uv run --frozen pytest tests/phase_10/test_config_and_backend.py` (5 passed); `uv run --frozen pytest tests/phase_12/test_config_backend.py` (4 passed); `uv run --frozen ruff check` on all 12 assigned training sources (pass); `uv run --frozen ruff format --check` on all 12 assigned training sources (12 already formatted); `uv run --frozen mypy` on all 12 assigned training sources (pass). The checklist's `tests/phase_10/test_config_backend.py` path does not exist; its existing counterpart `test_config_and_backend.py` was run.

## Tests
`tests/phase_10/test_config_and_backend.py`, `tests/phase_12/test_config_backend.py`.
