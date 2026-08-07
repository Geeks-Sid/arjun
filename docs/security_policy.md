# Security Policy

Status: research-release baseline (Phase 18)
Applies to all code, container images, and release artifacts in this repository.

## Scope

In scope:

- Remote-code execution, arbitrary file read/write, path traversal, and pickle /
  `weights_only` violations in loading paths (bundles, manifests, checkpoints).
- Unsafe URI / manifest handling that can reach unintended storage or local paths.
- Credential or patient-data (PHI-shaped) content in committed artifacts.
- Backend-hostile imports that leak data across CPU/CUDA/TPU execution boundaries.
- Prompt-injection that escalates to tool/system behavior from untrusted report text.
- License-policy bypass in model download/load paths.

Out of scope (handled by the data layer, not this repo's threat model):

- PHI that already exists in external datasets; this repository only documents
  how it must be kept out of manifests/logs (see `docs/data_governance.md`).

## Reporting a vulnerability

1. Do **not** open a public issue for a live exploit. Report privately to the
   project maintainer (see `model_registry/*.yaml` `review_owner`).
2. Include: affected component + function, a minimal reproduction, the medfm
   version/bundle revision, the runtime (CPU/CUDA/TPU), and whether
   patient-adjacent data could be affected.
3. If the issue requires fixing a released bundle, reference the bundle id and
   checksum from `docs/release/checksums.txt`; do not attach logs containing
   report text or identifiers.

The maintainer acknowledges receipt within 3 business days and triages per the
severity table below.

## Severity and response SLAs

| Severity | Example | Acknowledge | Patch target | Notice |
| --- | --- | --- | --- | --- |
| Critical | RCE in a loading path; credential/PHI leak reachable from a public API | 1 business day | 5 business days | immediate |
| High | Path traversal; gated-license bypass; prompt-injection escalation | 3 business days | 14 business days | 30 days after patch |
| Medium | Unsafe host import; weak validation; dependency CVE | 5 business days | next release | next release |
| Low | Housekeeping / scan hardening | 10 business days | tracked follow-up | tracked |

Time-bound, named waivers for a non-fixable finding must record an expiry and a
deferral owner in `docs/release/waivers.md`; the Phase 18 gate fails on expired
waivers.

## Incident response process

1. **Triage** — confirm impact, affected versions/bundles, and whether data has
   left approved stores. Record a private incident note (no raw payloads).
2. **Contain** — for served bundles, pause the affected bundle tag; for
   repositories, rotate any exposed tokens and add the finding to
   `scripts/scan_secrets.py` as a regression pattern.
3. **Fix** — patch at the source; add a regression test in `tests/phase_18/test_security.py`
   that fails on the old behavior (mirrors the manifest `report_chars` non-echo
   test for every disclosure).
4. **Release** — bump the bundle/release version; regenerate
   `docs/release/checksums.txt`; update `docs/release/release_notes.md`.
5. **Disclose** — per the severity table, describing impact, affected versions,
   the fix, and mitigations, without quoting raw exploit or patient content.

## Automated scanning (CI)

- **Secrets / forbidden data:** `make security` runs `scripts/scan_secrets.py`
  over git-tracked files on every PR and the nightly `ci.yml` job. Secret
  patterns are never exempt; patient-data shapes are checked except on
  test/synthetic fixture surfaces (documented in the scanner).
- **Dependencies:** the `ci.yml` schedule runs `pip-audit` against `uv.lock`;
  any known-vulnerability advisory blocks the release tag.
- **Container images:** `docker/` images are scanned on the scheduled security
  job; a critical/high finding blocks release and opens a tracking issue.
- **License drift:** registry license records are validated on every release
  (`medfm release validate`) so a non-approved license cannot silently ship.

## Test and synthetic-data policy

Sample identifiers in tests (`tests/phase_*/...`, `tests/phase_03/synthetic.py`)
are synthetic placeholders (reserved example UID roots, obviously non-prod MRNs)
and exist to prove the runtime screening. They are exempt from the source-tree
patient-data scan but never exempt from the runtime screens
(`medfm/data/manifests/schema.py`, `medfm/data/textprep/phi.py`).
