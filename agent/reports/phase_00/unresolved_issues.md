# Phase 00 — Unresolved Issues

1. **Unresolved model licenses (models remain disabled).** The following models have unresolved terms or unconfirmed sources and are `blocked_unresolved` in `model_registry/licenses.yaml`; they must not be loadable until resolved (owner: Project Maintainer; review date 2026-11-02):
   - `medsiglip` — HAI-DEF terms unreviewed; weights URI unconfirmed.
   - `ct-fm` — source repository and license unconfirmed (preferred 3D CT backbone; **resolving this is a Phase 05 blocker**).
   - `flexict-3d` — repository and license unconfirmed.
   - `merlin` — code/weights licenses unconfirmed.
   - `m3d-lamed` — code/weights licenses unconfirmed.
   - `triad` — repository and license unconfirmed (preferred 3D MRI backbone; **Phase 05 blocker**).
   - `nv-segment-ctmr` — distribution channel (NGC/HF/MONAI bundle) and license unconfirmed.
   - `brainiac` — deferred post-v1; license unconfirmed.
   - `medsam2` — code/checkpoint licenses unconfirmed.
   - `gigapath-flash` — gated, custom terms unreviewed.
   - `medgemma-1.5-4b` — gated HAI-DEF terms; redistribution terms unresolved.
2. **Gated repositories with no accepted terms.** `h-optimus-0`, `conch`, `titan`, `gemma-generic` are `pending_review`; a named individual must accept terms and record `accepted_terms_date` before approval.
3. **Clinical safety officer role is unfilled.** The Project Maintainer is acting owner; a designated clinical safety officer is mandatory before any clinical-data use (`docs/clinical_safety_scope.md`).
4. **No hardware evidence yet.** All accelerator statuses are `UNTESTED` (or `NOT_APPLICABLE` for the deferred `brainiac`); first smoke evidence is expected from Phase 01 onward.

No other unresolved issues.
