# Clinical Safety Scope

Owner: Project Maintainer (acting clinical-safety owner; a designated clinical safety officer is mandatory before any clinical-data use)
Review date: 2026-11-02
Status: Binding for all phases

## 1. Intended use

This framework is a **research engineering toolkit** for adapting medical foundation models. Its outputs are research artifacts only.

## 2. Intended-use exclusions (hard boundaries)

The system and its outputs must **not** be used for:

- Diagnosis, prognosis, triage, or treatment decisions for any real patient.
- Autonomous report signing or any workflow without a qualified human reviewer.
- Regulatory-cleared (FDA/CE/UKCA) clinical decision support — no such claim is made or implied.
- Clinical use of any generated text (reports, findings, VQA answers) without task-specific clinical validation and local governance approval.

## 3. Explicit claims policy

- The only permitted clinical statement is: **"Generated outputs require task-specific clinical validation and qualified human review before any clinical use."** This statement ships in model cards, README, and inference outputs metadata.
- No sensitivity/specificity/AUC number may be presented as a clinical performance claim; metrics are research metrics on stated datasets with stated splits.
- Any benchmark comparison must name the dataset, split policy (patient-level, see `docs/architecture/adr_0004_patient_level_splitting.md`), and known limitations.

## 4. Human-review requirements

- Every generated report, finding, or segmentation shown to a human in a research context must be visibly labeled as model output requiring expert review.
- Evaluation involving clinical interpretation must be performed or verified by a qualified clinician collaborator.
- Failure modes discovered during evaluation (hallucinated findings, missed pathology, PHI leakage) are safety incidents and follow `docs/model_governance.md` §incident process.

## 5. Safety-relevant engineering rules

- PHI checks **fail closed**: a sample that cannot be confirmed de-identified is rejected, never passed through with a warning (see `docs/data_governance.md` §PHI).
- `trust_remote_code`, gated repositories, and untrusted checkpoints are restricted per `docs/model_governance.md`.
- No patient data or model weights in Git.
- Modality is never inferred from tensor rank; mislabeled modality is a data-quality incident.
