# Phase 18 unresolved issues and open follow-ups

All acceptance criteria passed; no release-blocking issue is unresolved. Open,
time-bound, tracked follow-ups (none requires a waiver today; `docs/release/waivers.md`
is empty):

## Scheduled reviews (owners: project maintainer)

- [ ] **Protected hardware evidence.** GPU/TPU/distributed jobs are declared
  (`.github/workflows/protected-hardware.yml`) but require labeled self-hosted
  runners; until they run, every registry model stays `UNTESTED` on all
  accelerator backends by design (honest, not an omission). When a protected
  job records smoke evidence, update backend statuses via
  `ModelRegistry.record_backend_result` and regenerate the support matrix.
- [ ] **Dependency and container audit reports.** `pip-audit` and Grype run on a
  nightly schedule in `ci.yml`; push the first report artifacts to
  `artifacts/audit/` after the runner is live.
- [ ] **Model-license review.** Several registry records are `BLOCKED` /
  `pending_review` (e.g. conch, medsiglip, ct-fm); resolve terms and record
  acceptance before those boards enter a supported release.
- [ ] **Deferred model/task coverage.** Custom-operator third-party CUDA models
  and research-only boards remain documented as blocked/untested rather than
  claimed.

## Environment-constrained (no code change required)

- A full clean-venv install-to-inference rehearsal runs as a CI job (the local
  rehearsal used a scratch `output_dir`, see `test_reliability.py`).
- GPU/TPU-only tests are env-guarded and skipped with reasons on this host
  (10 skipped), so the Level-1 CPU gate stays meaningful without hardware.

## Processes to schedule (per phase-18 handoff)

- Periodic model-license + dependency-security + regression reviews.
- Periodic TPU/GPU accelerator-runtime compatibility reviews.
- Archive phase 00-18 reports and final validation evidence for the release
  (release notes + checksums + support matrix already committed).
