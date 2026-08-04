# Data Governance

Owner: Project Maintainer (acting data-governance owner)
Review date: 2026-11-02
Status: Binding for all phases

## 1. De-identification

- Clinical data is de-identified **before** entering the training environment. The framework never performs primary de-identification; it only verifies it.
- All patient/study/series identifiers are stored as salted hashes (`patient_id_hash`, `study_id_hash`, `series_id_hash`). Raw identifiers, UIDs, names, MRNs, and dates (beyond configured buckets) must not appear in manifests, logs, caches, checkpoints, or reports.
- Burned-in pixel annotations and WSI label images are treated as PHI-bearing until explicitly cleared by the data owner.

## 2. PHI checks fail closed

- The ingestion layer (Phase 03) runs PHI checks on every sample: identifier fields, DICOM tags, free-text fields, and image metadata.
- **Fail closed:** any check that errors, times out, or returns "unknown" rejects the sample. There is no warn-and-continue mode for PHI.
- Rejected samples go to a quarantine log that itself contains only hashes and reason codes, never the flagged content.

## 3. Restricted report text

- Raw report text must not be embedded in general-purpose manifests. Manifests store `report_uri` references only.
- Restricted text may be stored **only** in: (a) the institution-approved access-controlled store referenced by `report_uri`, or (b) an encrypted local store with filesystem permissions `0600` and keys outside the repository.
- Tokenization caches of restricted text inherit the same access controls. Training logs must never echo report content.

## 4. Access control

- Dataset access is granted per dataset, per person, per purpose, and recorded in an access log.
- External tracking services (e.g. hosted experiment trackers) are opt-in only; medical metadata stays local-first.

## 5. Retention

| Artifact | Retention rule |
|---|---|
| Dataset manifests | Retained while the dataset version is referenced by any run or checkpoint |
| Preprocessing/embedding/tokenization caches | Deleted when invalidated (cache-key rules in Phase 03) or when the source dataset is deleted |
| Run metadata and artifacts | Per `docs/reproducibility_policy.md` |
| Quarantine logs | 90 days, then reviewed and purged |
| Checkpoints | Retained while referenced; adapter-only canonical form (ADR 0006) |

## 6. Audit

- Every manifest carries `dataset_name`, `dataset_version`, `license`, and `provenance_uri`.
- Every run records the dataset-manifest hash (mandatory run metadata, `docs/reproducibility_policy.md`), enabling audit from checkpoint back to exact data.
- Split-leakage checks (patient-level, ADR 0004) run at ingestion and are part of acceptance; a deliberately corrupted split must fail.

## 7. Deletion

- Data-subject or data-owner deletion requests propagate to: source store, manifests (row removal + version bump), all derived caches (invalidated by `source_file_hash`), and any checkpoint trained **solely** on the affected dataset version.
- Deletion events are recorded (hash of affected IDs, date, actor, scope) in an append-only deletion log.
