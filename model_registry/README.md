# Model Registry (Phase 00 seed)

This directory holds governance data, not weights. **No model weights may be committed here.**

- `v1_scope.yaml` — machine-readable v1 scope: canonical modality/task enums, the modality × task disposition matrix, vertical slices, accelerator tiers, and the model roster with per-backend accelerator support status.
- `license_schema.json` — JSON Schema for license records (structure from `idea.md`).
- `licenses.yaml` — one preliminary license record per v1 checkpoint.

Rules (enforced by `tests/phase_00/` and `medfm.tools.validate_phase`):

- A model is not loadable through the production registry until its license record is populated and approved (`docs/model_governance.md`).
- Unresolved license terms are **blocking** (`status: blocked_unresolved`), never guessed.
- Backend support is per model/task/topology; a `SUPPORTED_*` status requires recorded smoke evidence. No blanket cross-accelerator claims.
- See `docs/licensing_policy.md` for the research-only vs commercial separation.
