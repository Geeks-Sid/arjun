# transfer_plan/data/splits.md

Source: `medfm/data/splits.py` (patient/site/temporal split generation + leakage checks).
Wave: 2 — **keep by design**.

## Transfer checklist

- [ ] `assign_split` / `_hash_bucket` / `check_split_leakage` / `assert_no_split_leakage` →
      **keep** — deterministic SHA-256 bucket assignment (row-order-free, seed-stable) and
      group-key leakage auditing are the ADR-0004 contract; `sklearn.model_selection` (not
      installed) can't reproduce the explicit group-hash or temporal/site policies. Already uses
      pandas/hashlib (library-backed).

## Tests
`tests/phase_03/test_fingerprint.py`, `tests/phase_04/test_collators.py` (indirect).
