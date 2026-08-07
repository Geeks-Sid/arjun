# transfer_plan/inference/bundle.md

Source: `medfm/inference/bundle.py` (model bundle packaging: layout, manifest, migration, validation).
Wave: 1/2.

## Transfer checklist

- [ ] `ModelBundle` layout/`load_bundle`/validation/migration → **keep** — BUNDLE_SCHEMA_VERSION,
      forbidden-resume-parts, and atomic packaging are bespoke contract logic. It already leans
      on `config_hash`, `safetensors`, and `yaml` (all library-backed). No transfer.
- [ ] Checksum/Hash helpers → **keep** (hashlib already).

## Tests
`tests/phase_17/test_inference.py`, `tests/phase_05/test_model_registry.py`.
