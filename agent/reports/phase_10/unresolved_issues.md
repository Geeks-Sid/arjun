# Unresolved or environment-blocked checks

- The protected 4B NF4 QLoRA run was not executed: `bitsandbytes` and `transformers` are absent, and the local CUDA device is the 8 GB development GPU rather than the 48 GB target. The implementation fails with typed capability errors instead of silently falling back.
- TPU BF16 optimizer, XLA quantization parity, and cross-backend adapter reload were not executed because `torch_xla` is absent. TPU policy is explicit and reports BF16 LoRA, not QLoRA.
- Distributed DDP/FSDP/XLA wrapping was not exercised in this workstation run. Injection is required before wrapping, and adapter visibility/optimizer audits are available for the distributed integration phase.
- `pytest tests -q` without `PYTHONPATH=.` is not a valid repository-wide signal in this environment: pytest collection omits the checkout root for older phase conftests and fails with `ModuleNotFoundError: medfm`. The isolated Phase 09/10 regression run with `PYTHONPATH=.` passed 29 tests.
