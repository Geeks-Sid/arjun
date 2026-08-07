# transfer_plan/inference/bundle.md

Source: `medfm/inference/bundle.py` (model bundle packaging: layout, manifest, migration, validation).
Wave: 1/2.

## Transfer checklist

- [x] `ModelBundle` layout/`load_bundle`/validation/migration → **keep** — BUNDLE_SCHEMA_VERSION,
      forbidden-resume-parts, and atomic packaging are bespoke contract logic. It already
      leans on `config_hash`, `safetensors`, and `yaml` (all library-backed). No transfer.
- [x] Checksum/Hash helpers → **keep** (hashlib already).

## Tests
`tests/phase_17/test_inference.py`, `tests/phase_05/test_model_registry.py`.


## Result
- `ModelBundle` layout/`load_bundle`/validation/migration and checksum/hash helpers: **keep**;
  bespoke schema, forbidden-resume checks, atomic packaging, and hashlib logic have no exact
  drop-in replacement while preserving the bundle contract.
- Parity drift: none observed in the focused bundle/registry tests.
- Files changed: this checklist only.
- Verification: `uv run --frozen pytest tests/phase_17/test_inference.py tests/phase_05/test_model_registry.py`
  (30 passed); `uv run --frozen ruff check medfm/inference/generation.py medfm/inference/pipeline.py
  medfm/inference/bundle.py` (passed); `uv run --frozen ruff format --check
  medfm/inference/generation.py medfm/inference/pipeline.py medfm/inference/bundle.py`
  (3 files already formatted); `uv run --frozen mypy medfm/inference/generation.py
  medfm/inference/pipeline.py medfm/inference/bundle.py` (passed).