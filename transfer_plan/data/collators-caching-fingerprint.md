# transfer_plan/data/collators-caching-fingerprint.md

Covers: `data/collators/*`, `data/caching/*`, `data/fingerprint.py`, `data/manifests/*`,
`data/textprep/*`.
Wave: 2 — **keep by design**.

## Transfer checklist

- [ ] `collators/*` (bucket planning, padding, per-task collators) → **keep** — ADR-0008
      static-shape buckets + `pad_to_shape` are TPU-compilation contracts; `BucketPlan`/config
      hashing is contract logic. `pad_to_shape` already uses `torch.nn.functional.pad`.
- [ ] `caching/*` (DiskTensorCache, atomic rename, quarantine, LRU, rank partitioning) →
      **keep** — already library-backed at the payload level (safetensors reads/writes); the
      atomicity + corruption-detection + rank-safety policy is bespoke and test-pinned.
- [ ] `fingerprint.py` (deterministic manifest statistics + bucket recommendations) →
      **keep** — pure pandas aggregation (library-backed); percentile/bucket logic is policy.
- [ ] `manifests/*` (parquet/jsonl IO + schema validation) → **keep** — already uses
      `pandas`+`pyarrow` (+ `jsonschema` for schema validation).
- [ ] `textprep/*` (tokenize, prompts, sections, phi, unicode) → **keep** — tokenize wraps the
      `transformers`/`tokenizers` tokenizer contract; supervision masking + BOS/EOS layout are
      VLM-training contract logic; de-identification (phi) is custom regex policy.

## Tests
`tests/phase_03/test_caching.py`, `test_manifests.py`, `test_fingerprint.py`,
`tests/phase_04/test_collators.py`, `test_textprep.py`, `test_vlm_masking.py`.
