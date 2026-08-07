# Versioning and release policy

Applies to the research-release baseline. Not for clinical use.

## Semantic versioning

- **MAJOR**: breaks a documented public contract (adapter API, `MetricValue`
  JSON surface, bundle layout, `MedicalBatch` fields, ADR-level behavior).
- **MINOR**: adds a backward-compatible capability (new adapter, task, recipe,
  eval metric, backend evidence) or changes the support matrix.
- **PATCH**: bug fix or security fix that preserves behavior and published
  numbers; a numeric change in a parity-tested metric is a MINOR change.

The registry `schema_version`, cache-key hashes, and bundle
`BUNDLE_SCHEMA_VERSION` move independently of project version; they only bump on
a format change (see `medfm/core/versioning.py`).

## Migration

- Adapter-only bundles are the canonical artifact; a consumer migrates by
  replacing the bundle directory and re-registering (registered via
  `ModelRegistry`). Base-model revisions are pinned; a base bump requires a
  rebuild and re-recorded backend smoke evidence.
- Config/recipe files declare `schema_version`; a newer-than-supported schema is
  rejected, never silently downgraded (`medfm/core/versioning.py`).

## Deprecation

- `ModelSpec.deprecated` + `replaced_by` is the single source of truth; the
  release `support_matrix.md` lists deprecated ids.
- A deprecated model continues to load in the minor that marks it; removal only
  in the next MAJOR, with the model card updated one minor in advance.

## Rollback

See `docs/release/rollback.md`. Rollback is a supported release (the prior
bundle + checksummed artifacts), never a revert of the codebase's data layer.

## Release evidence

Every release regenerates `docs/release/checksums.txt` (SHA-256 over the
release docs and artifacts) and `docs/release/support_matrix.md`; the Phase 18
gate (`medfm release validate`) re-validates every phase report (00..18) and the
registry/license/backend-status invariants before the tag is cut.
