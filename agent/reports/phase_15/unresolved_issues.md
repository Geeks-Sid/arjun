# Phase 15 unresolved issues

- TPU PJRT/`torch_xla` is not installed on this workstation. The TPU BF16 profiles are static-shape contracts with explicit non-accepted hardware status; no TPU execution result is claimed.
- Protected, de-identified pathology cohorts and production checkpoint access were unavailable. The H-Optimus/GigaPath/TITAN/CONCH production paths remain registry/license gated; offline tiny and synthetic runs validate interfaces only.
- CUDA execution, multi-rank slide sharding, real host embedding-store input-stall measurements, VRAM/HBM peaks, compiler counts, throughput, and external-site performance were not exercised here. Metadata records these fields without inventing measurements.
- `ruff check` reports 55 diagnostics across the touched set, primarily existing long lines/import ordering/modernization warnings in the large recipe and smoke modules. Focused compilation, tests, smoke, and acceptance validation pass; lint cleanup was not used to suppress behavioral evidence.
- The repository-wide test suite was not used as the Phase 15 gate. The deterministic focused suite is the accepted verification surface for this phase.
