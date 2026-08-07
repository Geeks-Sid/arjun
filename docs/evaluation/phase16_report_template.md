# Phase 16 Evaluation Report Template

This template describes a research evaluation report. It is not a clinical-validation report.

## Scope and data split

- Evaluation ID:
- Recipe/model revision:
- Task and modality:
- Clinical unit (`patient`, `study`, or `slide`):
- Threshold/calibration fit split: validation only
- Final test split:
- Internal/patient-disjoint/external/temporal/vendor/rare-class/missing-sequence/quality holdout:
- Dataset and preprocessing hashes:

## Runtime provenance

Record backend, precision, topology, static bucket, checkpoint format, seed, and model/data/preprocess hashes for every saved prediction artifact. Keep scientific model metrics separate from accelerator parity and performance metrics.

## Headline research metrics

Report discrimination, calibration, spatial quality, retrieval, structured-output, and generation metrics at the declared clinical unit. Include sample counts, per-class/macro/micro summaries, and patient/study/slide cluster-bootstrap intervals where appropriate.

BLEU and ROUGE are secondary context only. Clinical finding, contradiction, omission, hallucination, localization, and grounding analyses are primary evidence for generation and VLM tasks.

## Subgroups and generalization

Report scanner/site/organ, protocol/vendor, rare-class, missing-sequence, low-quality, temporal, external-site, and patient-disjoint results when data are available. Missing evidence must be recorded as a release limitation, not imputed.

## Baselines and ablations

List random/majority, frozen linear probe, LoRA, conventional task, frozen-encoder decoder, and nnU-Net/comparable 3D baselines as applicable. For VLMs list no-visual, shuffled-visual, bridge, token-budget, encoder-freezing, native-3D/slice-sequence, and coordinate ablations. A shuffled-visual result within the predeclared margin fails the visual-grounding gate.

## Human review

Use the protected review protocol: deterministic stratified sampling, reviewer blinding to model identity, the categories in `idea.md`, a declared disagreement-resolution process, and inter-rater agreement. Report major/potentially harmful errors separately from lexical mismatch. Store only access-controlled artifact references in this report; never embed reviewed clinical text or images.

## Known limitations and safety statement

- Generated outputs require task-specific clinical validation and qualified human review before any clinical use.
- Synthetic/offline fixtures are contract and regression evidence, not clinical validation.
- External-site, temporal, subgroup, and human-review evidence is absent unless explicitly listed above.
- Backend parity is only claimed for backends with deterministic fixture results within predeclared tolerances; larger divergence remains an investigation item.
