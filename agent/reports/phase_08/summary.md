# Phase 08: Pathology Tile and WSI Adapters

Implemented the bounded-memory two-stage pathology path: deterministic tile selection, fixed tile/token buckets, atomic per-slide HDF5 embedding stores, resumable chunk extraction, masked mean/attention MIL, and a fixed-token pathology VLM bridge.

## Delivered

- `medfm.models.pathology` exposes tile encoders, aggregators, selectors, HDF5 `EmbeddingStore`, deterministic rank sharding, and `PathologyVLMAdapter`.
- H-Optimus remains the shared Phase 06 tile adapter boundary; GigaPath tile/slide and TITAN boundaries have explicit offline fallbacks and registry plugins.
- Store rows preserve slide/tile IDs, coordinates, level, MPP, quality, encoder revision, preprocessing hash, layer, dtype, schema, chunks, and compression metadata.
- Tile reads and quality work remain host-side; cuCIM remains optional in the existing slide-reader boundary.
- HDF5 was selected in ADR 0011 for concurrent subset reads and atomic per-slide commits.

## Verification

- `python -m medfm.tools.smoke --phase 08 --json`: passed.
- `ruff check` on Phase 08 implementation/tests: passed.
- `python -m pytest tests/phase_08 -q`: 7 passed.

The local tiny encoder and cached embedding path are fully offline. Upstream H-Optimus, GigaPath, CONCH, and TITAN weights remain governed by the existing license records; no real checkpoint download or clinical capability is claimed.
