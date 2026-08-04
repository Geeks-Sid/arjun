# Licensing Policy

Owner: Project Maintainer (acting model-license owner; per-record review owners in `model_registry/licenses.yaml`)
Review date: 2026-11-02
Status: Binding for all phases

## 1. Separation of research-only and commercially permitted artifacts

Every artifact (model checkpoint, dataset, derived embedding store) is classified into exactly one of:

- **research_only** — non-commercial terms (e.g. CC-BY-NC family, custom academic licenses, gated research agreements). Must be visibly marked; must never ship in a commercial distribution.
- **commercial_permitted** — permissive terms allowing commercial use (e.g. Apache-2.0, MIT) with all conditions recorded.
- **conditional** — commercial use allowed under conditions (e.g. health-specific use policies); conditions are recorded in the license record and treated as research_only until reviewed.
- **unresolved** — terms unknown or unreviewed. **Unresolved is blocking**: the artifact is disabled, never guessed.

The registry exposes this classification so a downstream packaging step can produce a commercial-safe bundle containing only `commercial_permitted` artifacts.

## 2. License record schema

The schema is defined in `model_registry/licenses.yaml` per the YAML structure in `idea.md`, validated by `medfm/tools/governance.py` against `model_registry/license_schema.json`. Required fields per record:

```yaml
model_id:            # unique slug matching v1_scope.yaml
provider:            # organization publishing the checkpoint
repository:          # source repository URL (HF/GitHub)
weights_uri:         # canonical weights location
revision_policy:     # pinned revision/commit SHA policy
code_license:        # SPDX or named license of the code
weights_license:     # SPDX or named license of the weights
commercial_use:      # permitted | prohibited | conditional | unresolved
derivative_models:   # permitted | prohibited | conditional | unresolved
redistribution:      # permitted | prohibited | conditional | unresolved
gated_access:        # true | false
accepted_terms_date: # ISO date or null (null only if not gated or not yet accepted)
approved_use_cases:  # list of strings
prohibited_use_cases:# list of strings
status:              # approved_research | approved_commercial | pending_review | blocked_unresolved | rejected
review_owner:        # named individual (role acceptable in v1 with a named deputy)
review_date:         # ISO date — overdue reverts to pending_review
notes:               # free text, includes trust_remote_code reviews
```

## 3. Blocking rules (enforced by tests)

- Any of `commercial_use`, `derivative_models`, `redistribution` equal to `unresolved` ⇒ `status` **must** be `blocked_unresolved`.
- `status: approved_*` requires a non-null `review_owner` and a `review_date`.
- Gated models with no `accepted_terms_date` may not be `approved_*`.
- A model must not be loadable through the production registry until its record is populated and approved (see `docs/model_governance.md`).

## 4. Code licenses

- Project code is authored under the repository's top-level LICENSE (added in Phase 01).
- Third-party code is vendored only with its license text and a notice; copyleft dependencies require an explicit ADR.
