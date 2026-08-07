# transfer_plan/core/serialization.md

Source: `medfm/core/serialization.py` (canonical JSON/YAML, TensorMeta, dtype names, config_hash,
materialize_cpu).
Wave: 2 — **keep by design** (frozen Phase-02 contract; see `docs/core_contracts.md`).

## Transfer checklist

- [ ] `config_hash` (SHA-256 over canonical JSON) → **keep** — stdlib `hashlib` already;
      stable-sort canonicalization is the contract.
- [ ] `canonical_json`/`canonical_yaml`/`TensorMeta`/`materialize_cpu` → **keep** — the
      accelerator-neutral dtype-name mapping and inline-tensor policy are bespoke; no library
      matches the exact round-trip guarantees. Any change here is a `SCHEMA_VERSION` event.

## Tests
`tests/phase_02/*` (metadata roundtrip, device transfer, backend neutrality).
