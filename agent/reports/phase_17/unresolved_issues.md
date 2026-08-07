# Unresolved issues and explicit limitations

- The full repository pytest invocation is not green in this environment for
  pre-existing reasons: `safetensors` is unavailable during Phase 03
  collection, and historical duplicate test module basenames cause Phase 06 /
  Phase 08 import-mismatch collection errors. The focused Phase 17 suite
  passes 15 tests.
- No CUDA or TPU execution was claimed. The implementation includes backend
  routing, fixed TPU bucket padding/rejection, and warmup contracts, but this
  environment supplied no accelerator evidence or XLA compilation counts.
- `highdicom` and `pydicom` were not available for a real DICOM SEG write/read
  in this environment. The exporter fails closed when the medical extra is
  missing and requires explicit reviewed policy before invoking highdicom.
- No external-site dataset, approved human-review dataset, production model
  weights, or legal deployment review was supplied. The license catalog and
  runtime matrix intentionally fail closed for unknown or blocked bundles.
- Real-checkpoint output parity, latency, VRAM, TPU compile counts, and
  capacity measurements remain deployment-specific evidence rather than
  framework-wide claims.
- The repository-wide mypy invocation remains non-zero because of existing
  diagnostics across training/recipe/model packages and missing optional
  dependency stubs. Phase 17 source is ruff-clean; this report does not claim
  a repository-wide typing gate.
