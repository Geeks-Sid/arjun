# Phase 05 Unresolved Issues

1. **No v1 model is production-loadable yet.** All 17 roster models register
   as BLOCKED because no license record in `model_registry/licenses.yaml` is
   `approved_*` (12 `blocked_unresolved`, 5 `pending_review` at phase start).
   Unblocking requires the license review checklist in
   `next_phase_handoff.md`: accept gated terms as a named individual, record
   acceptance via `medfm models accept-terms`, pin the exact revision SHA,
   then flip the license record status. Owner: Project Maintainer (Siddhesh),
   review due 2026-11-02 per the phase-00 records.
2. **Pinned SHAs are placeholders for blocked models.** ModelSpec allows a
   non-SHA revision only while BLOCKED; promoting any model to READY requires
   recording the exact commit SHA (revision_policy in the phase-00 records is
   `pinned_*_pending_acceptance` / `pinned_revision_required_unresolved`).
3. **Real adapter plugins do not exist yet.** `run_smoke` and
   `medfm accelerator validate-model` cover the dummy plugin end to end;
   real models raise `NoAdapterError` (exit 3) until phases 06-08 register
   their `ModelPlugin`s. CUDA/TPU smoke paths are implemented but only
   exercised on CPU in this tier; protected-hardware runs happen on the GPU/
   TPU runners (`MEDFM_RUN_GPU_TESTS=1` / `MEDFM_RUN_TPU_TESTS=1`).
4. **Memory estimates are conservative pre-adapter placeholders.** Parameter
   counts in `catalog._APPROX_PARAMS_B` are approximate; `measured_peak_bytes`
   is the refinement hook for real runs (Phase 12 records actuals).
5. **Hash verification requires an expected manifest.** `verify_file_hashes`
   takes an explicit {path: sha256} map; per-model expected manifests are
   recorded when each model's revision is pinned (they cannot be computed
   before license acceptance without downloading).
6. **brainiac is registered but deferred.** Phase-00 scope marks it `v1:
   false` with all backends NOT_APPLICABLE; it stays BLOCKED (license
   unresolved) and is excluded from v1 acceptance.
