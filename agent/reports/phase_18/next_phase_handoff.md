# Phase 18 → next-phase handoff

Phase 18 is the final phase in the v1 program; there is no Phase 19 code
contract. The handoff is **release readiness** plus scheduled operational
reviews.

## Immediate release actions

1. Cut the `v0.1.0-rc` tag at the current HEAD; `docs/release/checksums.txt` and
   `docs/release/support_matrix.md` are regenerated and committed with it.
2. Publish `docs/release/release_notes.md` + compatibility/rollback runbook to
   consumers.
3. Archive `agent/reports/phase_00..18` and the final validation evidence
   (acceptance.json statuses) with the tag.

## Tracked follow-ups (owners)

- Protected hardware evidence (GPU/TPU) → update registry `backend_support` +
  regenerate matrix (see `agent/reports/phase_18/unresolved_issues.md`).
- Model-license review for `BLOCKED` / `pending_review` records.
- First dependency/container audit report publication (nightly CI).

## Scheduled recurring reviews

- Monthly: model-license, dependency-security, and golden-fixture drift.
- Recurring per release: accelerator-runtime compatibility, TPU/GPU regression
  review, and re-link/review of references + licenses in
  `implementation_plan/references.md`.

## Operating notes

- Never weaken the acceptance gate to pass a release; extend parity fixtures
  instead (`scripts/generate_golden.py`, `tests/phase_18/`).
- Registry accelerator status is the source of truth for the release support
  matrix; protected jobs promote `UNTESTED` → `SUPPORTED_*` with smoke evidence
  only via `ModelRegistry.record_backend_result`.
- Security incidents follow `docs/security_policy.md`; waivers track expiries in
  `docs/release/waivers.md` and are enforced by the release gate.
