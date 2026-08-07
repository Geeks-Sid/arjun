# Phase 09 handoff

- Use `EmbeddingStore.read_subset` plus `WSITokenSelector` to feed `PathologyVLMAdapter`; the fixed default is 256 pre-compression embeddings and 64 visual tokens, configurable within 128-1,024 and 32-128.
- Evidence is returned as `SelectedTokens.records`; each record carries level-0 x/y/width/height, level, MPP, tissue fraction, and quality values.
- Choose `MeanPoolingAggregator` for the deterministic baseline or `AttentionMILAggregator` for learned evidence weights. Both require and honor a real-tile mask.
- HDF5 store schema version is 1. Cache identity invalidates on model revision, preprocessing hash, layer, dtype, or schema changes.
- `extract_slide_embeddings` resumes committed `.chunks` files and applies the explicit corrupt-tile failure threshold. `DeterministicSlideSharder` assigns slides/chunks by stable SHA-256 rank.
- Real upstream pathology models remain license-gated; Phase 09 should consume the local contract or an approved checkpoint adapter, never bypass the registry.
