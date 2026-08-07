# Phase 17 deployment license catalog

Deployment bundles are admitted only when their `base_models.json` references
an approved license entry and the manifest does not declare a prohibited
combination. This catalog is intentionally conservative: an absent entry is a
blocked deployment, not an implicit approval.

| License state | Bundle policy |
| --- | --- |
| `approved` | May be loaded when the pinned base revision, task, backend, and intended use match the entry. |
| `restricted` | May be loaded only in the documented environment and access-controlled workflow. |
| `blocked` | Must fail bundle registration and serving before model allocation. |
| `unknown` / absent | Must fail closed pending license review. |

The repository currently ships framework contracts and synthetic fixtures, not
third-party model weights or a legal approval for a specific production use.
Populate the deployment catalog from the governed model registry before
registering a real bundle. `licenses.yaml` and the registry manifest remain the
source of truth for license identifiers; this document defines the runtime
gate and does not grant rights.

Operational requirements:

- Pin the exact model revision and license identifier in `base_models.json`.
- Record task, modality, backend, and geographic/intended-use restrictions in
the deployment review.
- Keep resumable checkpoints and unreviewed model code out of deployment
  bundles.
- Retain the review record with the bundle checksum; revoke registration when
  the license or safety status changes.
