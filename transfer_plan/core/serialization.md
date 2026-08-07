# transfer_plan/core/serialization.md

Source: `medfm/core/serialization.py` (canonical JSON/YAML, TensorMeta, dtype names, config_hash,
materialize_cpu).
Wave: 2 — **keep by design** (frozen Phase-02 contract; see `docs/core_contracts.md`).

## Transfer checklist

- [x] `config_hash` (SHA-256 over canonical JSON) → **keep** — stdlib `hashlib` already;
      stable-sort canonicalization is the contract.
- [x] `canonical_json`/`canonical_yaml`/`TensorMeta`/`materialize_cpu` → **keep** — the
      accelerator-neutral dtype-name mapping and inline-tensor policy are bespoke; no library
      matches the exact round-trip guarantees. Any change here is a `SCHEMA_VERSION` event.


## Tests
`tests/phase_02/*` (metadata roundtrip, device transfer, backend neutrality).

## Result

Both items verified as **keep**: serialization is the frozen Phase-02 contract, with deterministic
stdlib hashing/canonicalization, canonical dtype metadata, bounded inline tensors, and CPU
materialization. No transfer or parity drift measured. Source and test files were unchanged;
only this checklist was updated.

Validation (shared phase run): `uv run --frozen pytest tests/phase_02 tests/phase_03 tests/phase_04
tests/phase_05 tests/phase_06 tests/phase_07 tests/phase_09` — **PASS** (622 passed, 4 skipped,
1 warning). Scoped `ruff check` — **PASS**; `ruff format --check` — **PASS** (43 files);
scoped `mypy` — **PASS**.
