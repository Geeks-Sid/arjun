# Release waivers (research-release baseline, Phase 18)

Time-bound, named waivers for findings that cannot be resolved before a release.
The Phase 18 gate (`scripts/audit_waivers.py`) fails on any row whose expiry is
in the past. No expiry means no waiver: the finding must be fixed.

Format (one row per waiver):

| id | expiry (YYYY-MM-DD) | finding | owner |

- Project status: research software, not for clinical use.
- Security incidents follow `docs/security_policy.md`.

| id | expiry | finding | owner |
| --- | --- | --- | --- |
| (none — no active waivers) | - | - | - |
