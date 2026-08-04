# Model Governance

Owner: Project Maintainer (acting model-governance owner)
Review date: 2026-11-02
Status: Binding for all phases

## 1. Model approval process

A model becomes loadable through the production registry only when **all** of the following hold:

1. A license record exists in `model_registry/licenses.yaml`, passes schema validation, and its `status` is `approved_research` or `approved_commercial` (see `docs/licensing_policy.md`).
2. `revision_policy` is pinned (exact revision/commit SHA) — no floating `main` references.
3. A per-backend accelerator status exists for every backend tier, with recorded evidence for any `SUPPORTED_*` claim (no blanket cross-accelerator claims — enforced by `tests/phase_00/test_accelerator_policy.py`).
4. Checkpoint provenance is recorded: base-model reference, configuration hash, source URI.

Until then the model is **disabled**: the registry must refuse to load it for anything other than governance review.

## 2. `trust_remote_code`, gated repositories, and untrusted checkpoints

- `trust_remote_code=True` is **prohibited by default**. It may be enabled only per model, in an explicit config field, after a human review of the remote code, and the pinned revision must be hash-locked. The review is recorded in the model's license record under `notes`.
- Gated repositories: access terms must be accepted by a named individual (`accepted_terms_date`); acceptance is per person, not per project. The framework must never commit, cache, or redistribute tokens.
- Untrusted checkpoints (anything not fetched from the recorded `repository` at the pinned revision, or supplied as a local file without provenance) load only in a sandboxed evaluation path with `weights_only=True`-style loading where the format permits; pickle-based formats from untrusted sources are rejected.
- Safetensors is the preferred interchange format; canonical deployment tensors are accelerator-neutral CPU safetensors (ADR 0006).

## 3. Review process

- Every license record has a named `review_owner` and `review_date`; overdue records automatically revert to `pending_review` and the model is disabled.
- Registry changes (add/enable/modify a model) are reviewed diffs: license record, scope entry, and accelerator status must change together or CI fails.

## 4. Deprecation process

1. Mark the model `deprecated` in `model_registry/v1_scope.yaml` with a successor or explicit "no successor".
2. Keep the license record for audit; set `status: rejected` for license-revoked models.
3. Existing checkpoints remain loadable with a loud deprecation warning; new runs are blocked.
4. Announce in the changelog and in the phase report of the phase performing the deprecation.

## 5. Incident process

A model incident is: license violation discovered, PHI leakage through a model artifact, safety-claim violation (`docs/clinical_safety_scope.md`), or a checkpoint provenance mismatch.

1. **Contain:** immediately set the model `status: blocked_unresolved` (disables loading) and quarantine affected checkpoints.
2. **Record:** open an incident note in the current phase's `unresolved_issues.md` and in `agent/reports/`.
3. **Assess:** review owner determines scope (which runs/checkpoints/datasets are affected).
4. **Remediate:** re-review license, retrain or delete affected artifacts, add a regression test.
5. **Close:** only with a named owner sign-off and a passing acceptance re-run.
