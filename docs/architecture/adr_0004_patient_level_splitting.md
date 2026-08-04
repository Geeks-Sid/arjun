# ADR 0004: Patient-level splitting

Status: Accepted (2026-08-04)
Deciders: Project Maintainer

## Context

Medical datasets leak easily: multiple studies per patient, slices per volume, tiles per slide, and derived/resampled copies can cross train/validation/test boundaries and inflate metrics.

## Decision

All splits are generated **patient-level first**, then site (for external validation), then time (for temporal validation). Split membership is keyed on `patient_id_hash`. The following must never cross splits: different studies of one patient; tiles of one slide; slides of one case; adjacent slices of one volume; derived or resampled copies. Split-leakage checks run at ingestion and a deliberately corrupted split must fail the check (Phase 03 acceptance).

## Alternatives considered

- **Sample-level random splitting:** standard in general ML, unacceptable leakage here. Rejected.
- **Study-level splitting:** still leaks across longitudinal studies of one patient. Rejected.
- **Group k-fold only:** allowed as a complement for cross-validation, not a replacement for a held-out patient-level test set. Partially adopted.

## Consequences

- Manifests require `patient_id_hash`, `site_id`, and `split` columns (Phase 03).
- Metrics are reported per split policy; cross-policy comparisons are invalid.
- Small datasets may need grouped CV recipes (Phase 16).

## Reversal conditions

None anticipated. Weakening this decision requires evidence that a specific public benchmark mandates a different official split, in which case the benchmark split is used for comparability **and** reported alongside a patient-level split; recorded in a new ADR per benchmark.
