# Phase 18 summary — CI, hardening, and release

Status: passed (2026-08-07)

Phase 18 closes the v1 program. It made the repository's gates executable in
CI without weakening any earlier acceptance contract, added protected-hardware
jobs with honest (explicitly-untested / blocked) registry evidence, added a
golden-regression + security + reliability test surface, produced a CPU-runnable
release gate (`medfm release validate`), and shipped the 0.1.0-rc research
release documentation and artifacts.

## What was delivered

- **Test matrix (L1-L4).** New pytest markers (`level1`..`level4`, `golden`,
  `real_checkpoint`, `security`, `reliability`) with `--strict-markers`; a root
  conftest guards protected real-checkpoint tests behind
  `MEDFM_RUN_REAL_CHECKPOINTS`. L1 = CPU contracts (full suite, 979 passed),
  L2 = synthetic-GPU tiny models, L3 = gated real-checkpoint smoke, L4 = golden
  regression with pinned fixtures and dtype-aware tolerances. `scripts/audit_skips.py`
  fails on JUnit skips without a reason; `scripts/audit_waivers.py` fails on
  expired waivers.
- **CI.** `.github/workflows/ci.yml` (Level 1: lint, typecheck, tests, coverage,
  junit, package build, secret scan, release gate, waiver audit; nightly
  dependency + container audit) and `.github/workflows/protected-hardware.yml`
  (L2-L4 + CUDA single/distributed + TPU replicated/SPMD-FSDP on labeled
  self-hosted runners).
- **Security and privacy.** `tests/phase_18/test_security.py` (manifest path
  traversal / unsafe URIs, gated-download refusal, malicious-checkpoint
  rejection, audit PHI redaction, prompt-injection isolation,
  `report_chars` non-echo); `scripts/scan_secrets.py`;
  `docs/security_policy.md` (reporting + incident response + severity SLA).
  Fixed a real PHI leak: manifest validation no longer echoes `report_chars`.
- **Reliability.** `tests/phase_18/test_reliability.py` (disk-full during
  checkpointing, training interruption -> complete resumable checkpoint,
  deterministic-mode loss reproducibility, documented cross-backend
  tolerances, scratch-dir install-to-inference rehearsal).
- **Golden regression.** `scripts/generate_golden.py` + pinned fixtures
  (shapes, preprocess statistics, logits, mask metrics, structured findings,
  memory envelope) with a SHA-256 manifest; `tests/phase_18/test_golden_regression.py`.
- **Release gate + artifacts.** `medfm/tools/release.py` +
  `medfm/cli/release.py` (registry per-backend status, license/scope
  consistency, no-eager-backend-import scan, TPU/NF4 policy scan,
  clinical-claims scan, all phase reports 00..18, support matrix + checksums
  generators). Release docs in `docs/release/` (versioning, rollback, known
  limitations, compatibility + support matrix, CUDA-QLoRA vs TPU-BF16,
  model/license/data summary, release notes, waivers).

## Gate results (acceptance command)

- `make lint` — passed
- `make typecheck` — passed (mypy strict, 195 files)
- `make test` — 979 passed, 10 skipped (skips are protected-hardware guards with
  reasons)
- `python -m medfm.tools.validate_phase --phase 18` — passed
- `make smoke` — passed (3 checks)

## Protected hardware

GPU/TPU/distributed jobs are declared and scheduled in
`.github/workflows/protected-hardware.yml` but are not executable in the
development environment. Registry and support-matrix statuses therefore
honestly remain `UNTESTED` (or `BLOCKED_*` with reason) until protected
hardware follows ups record smoke evidence — no unsupported accelerator claim
is made.

## Open follow-ups

Listed in `next_phase_handoff.md` (protected-hardware evidence, dependency and
container audit reports, model-license review for blocked records, deferred
model/task coverage, clean-venv CI rehearsal on a dedicated runner).
