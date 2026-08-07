# Phase 12 unresolved runtime issues

These are environment or topology acceptance gaps, not silent fallbacks:

- `torch_xla` is not installed. TPU PJRT launch, BF16 autocast on XLA, replicated reduction, stable compilation measurement, XLA SPMD/FSDP multi-rank checkpoint planners, and portable CPU export were not executable here.
- `accelerate` is not installed. The direct backend path was exercised; the optional Accelerate placement/accumulation/clipping path was not runtime-tested.
- The workstation exposed one CUDA device. CUDA DDP/FSDP multi-rank wrapping, multi-rank failure synchronization, and topology-change reshard/resume were not executed. The DCP branch was verified with a fresh single-rank Gloo process group, model restore, and AdamW state restore.
- The optional `safetensors` package is not installed. The dependency-free standard safetensors writer was exercised by the adapter-export test; an upstream-reader interoperability run remains for an environment with the package installed.
- The available GPU is an 8 GB NVIDIA GeForce RTX 4060 Laptop GPU, so the planned 48 GB envelope was validated structurally and with tiny models, not with a representative large medical foundation model.
- Repository-wide `python -m pytest tests -q` collection is not clean in this environment: `tests/phase_03/test_caching.py` imports the unavailable `safetensors` package, and duplicate test module basenames cause pytest import-mismatch errors in phases 06 and 08. The focused Phase 12 suite, required validator, and targeted Phase 01/02 regression commands pass.

No TPU, multi-device, optional-dependency, or repository-wide result is presented as passing evidence.
