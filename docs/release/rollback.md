# Rollback runbook

Research release baseline; not for clinical use. Rollback targets a released
(-tagged, checksummed) artifact, not a code revert of the data layer.

## When to roll back

1. A release bundle fails post-deploy smoke in the serving environment.
2. A security incident in the `docs/security_policy.md` severity table makes the
   bundle unsafe to serve.
3. A distributed/TPU job shows recompilation or checkpoint/resume regressions
   that the release gate recorded as supported.

## Procedure

1. Identify the last good release from `docs/release/checksums.txt` +
   `docs/release/release_notes.md` (checksummed = immutable).
2. Redeploy that bundle: stop the service, replace the bundle directory, verify
   checksums with `sha256sum -c docs/release/checksums.txt` for the bundle
   artifacts, restart, and run the bundle smoke (`docs/inference/deployment_matrix.md`
   warmup/capacity runbook).
3. Record in the incident note (no raw payloads) the reason, affected bundle id,
   and the checksums honored.
4. Open a tracked follow-up; do not increment the registry `schema_version` for
   a rollback — it is a load-policy artifact, not a format change.

## Not supported

- Rolling back across a `schema_version` bump (reject, never silent downgrade).
- Rolling back to a merged base-weight convenience artifact as the canonical
  adapter source (conversion record required per `docs/inference/deployment_matrix.md`).
