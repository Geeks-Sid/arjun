# ADR 0011: HDF5 per-slide pathology embedding stores

- **Status:** accepted for Phase 08
- **Decision:** Use one HDF5 file per slide, with chunked/gzip-compressed `embeddings` and aligned `tile_ids`, `coords`, `level`, `mpp`, and quality columns. A JSON sidecar carries the schema and encoder identity; a digest marker is written last.
- **Why:** HDF5 is available in the pathology extra, supports concurrent read-only subset access, bounded row reads, and explicit compression/chunk metadata. Per-slide files avoid opening a gigapixel pyramid or a whole cohort in memory.
- **Atomicity/resume:** Writers create a sibling temporary file and atomically replace the slide file. Completion markers are written after the sidecar. Extraction chunks are independently atomically persisted under `<slide>.chunks/` and finalized only after all healthy tiles are present.
- **Invalidation:** The store identity includes schema version, model revision, preprocessing hash, layer, and dtype. Any mismatch invalidates the existing complete store.
- **Corrupt tiles:** Reader failures are counted per requested tile. Extraction continues for `on_corrupt="skip"` until the configured failure threshold is exceeded; a store is never marked complete when the threshold is exceeded.
