# transfer_plan/data/splits.md

Source: `medfm/data/splits.py` (patient/site/temporal split generation + leakage checks).
Wave: 2 — **keep by design**.

## Transfer checklist

- [x] `assign_split` / `_hash_bucket` / `check_split_leakage` / `assert_no_split_leakage` →
      **keep** — deterministic SHA-256 bucket assignment (row-order-free, seed-stable) and
      group-key leakage auditing are the ADR-0004 contract; `sklearn.model_selection` (not
      installed) can't reproduce the explicit group-hash or temporal/site policies. Already uses
      pandas/hashlib (library-backed).


## Tests
`tests/phase_03/test_fingerprint.py`, `tests/phase_04/test_collators.py` (indirect).

## Result

The split generation/audit item is **keep**: current `generate_split_assignment`/`_hash_bucket`
and leakage checks enforce deterministic SHA-256 group assignment and ADR-0004 policies that have
no drop-in equivalent. No transfer or parity drift measured. Source and test files were unchanged;
only this checklist was updated.

Validation (shared phase run): `uv run --frozen pytest tests/phase_02 tests/phase_03 tests/phase_04
tests/phase_05 tests/phase_06 tests/phase_07 tests/phase_09` — **PASS** (622 passed, 4 skipped,
1 warning). Scoped `ruff check` — **PASS**; `ruff format --check` — **PASS** (43 files);
scoped `mypy` — **PASS**.
