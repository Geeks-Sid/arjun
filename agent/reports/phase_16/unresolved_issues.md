# Unresolved issues and explicit limitations

- The full repository pytest invocation is not green in this environment because `safetensors` is unavailable during Phase 03 collection and historical duplicate test module basenames cause Phase 06/08 import-mismatch collection errors. The focused Phase 16 suite and prior Phase 13–15 evaluation regression suite pass.
- No CUDA or TPU execution was claimed. The implementation provides deterministic backend tolerance and distributed-reduction contracts; accelerator parity requires hardware/runtime evidence.
- No external-site dataset, approved human-review dataset, or clinical validation study was supplied. Reports intentionally retain release limitations and set `clinically_validated` to false.
- Optional `jsonschema`, `PyYAML`, NumPy, SciPy, and Torch dependencies remain runtime requirements for the corresponding evaluation paths; the smoke path and focused tests have them installed.
