# Phase 16: Evaluation, Clinical Validation, and Regression Testing

## Objective

Build a clinical-unit-aware evaluation layer covering discrimination, calibration, spatial quality, retrieval, generation, grounding, subgroup behavior, uncertainty, baselines, ablations, and human review.

## Dependencies

- [x] Phase 11 task outputs and metric lifecycle are stable.
- [x] At least one recipe from Phases 13-15 can emit predictions.
- [x] Clinical validation language and prohibited claims from Phase 00 are available.

## Scope boundaries

Allowed areas: `medfm/evaluation/`, evaluation CLI/configs, reporting templates, golden fixtures, and Phase 16 tests.

Do not label any model clinically validated without a separate explicit validation study.

## Implementation checklist

### Evaluation data model

- [x] Define versioned prediction, target, group, and metric-result schemas.
- [x] Require patient/study/slide clinical unit declarations.
- [x] Separate threshold selection data from final test data.
- [x] Save prediction artifacts with model/data/preprocess hashes.
- [x] Support deterministic recomputation of metrics without rerunning inference.
- [x] Record backend, precision, topology, bucket, and checkpoint format with every prediction artifact.
- [x] Distinguish scientific model metrics from accelerator parity/performance metrics.
### Classification and calibration

- [x] Implement AUROC, AUPRC, sensitivity, specificity, precision, recall, F1, balanced accuracy, and confusion matrices.
- [x] Implement Brier score and expected calibration error.
- [x] Implement sensitivity at fixed specificity and the inverse.
- [x] Report per-class, macro, and micro metrics appropriately.
- [x] Compute patient-level bootstrap confidence intervals.
- [x] Never use slice-level confidence intervals when the clinical unit is patient/study.
- [x] Fit thresholds/calibration on validation data only.

### Segmentation and localization

- [x] Implement Dice, IoU, surface Dice, HD95, ASSD, lesion sensitivity, false-positive lesions/scan, and volume error.
- [x] Evaluate in original physical space where practical.
- [x] Handle empty-target and empty-prediction cases explicitly.
- [x] Report per-class and macro summaries.
- [x] Implement 2D/3D box IoU and physical localization error.

### Retrieval, VQA, and generation

- [x] Implement Recall@1/5/10, median/mean rank, and mAP in both directions.
- [x] Implement exact match, token F1, schema validity, and finding-level precision/recall.
- [x] Score negation, laterality, severity, and anatomy correctness.
- [x] Add RadGraph-style or equivalent clinical entity/relation evaluation where approved.
- [x] Add contradiction, omission, and hallucinated-finding analyses.
- [x] Keep BLEU/ROUGE secondary rather than headline metrics.

### 3D and pathology-specific evaluation

- [x] Evaluate native 3D and slice-sequence VLMs separately and comparatively.
- [x] Include small-lesion, anatomy localization, adjacent-slice consistency, and 3D boxes.
- [x] Report pathology tile, slide, and patient metrics.
- [x] Report scanner/site/organ results and evidence localization.
- [x] Sweep sampled tile count and magnification.

### Generalization, baselines, and ablations

- [x] Support internal, patient-disjoint, external-site, temporal, vendor/protocol, rare-class, missing-sequence, and low-quality holdouts.
- [x] Require random/majority, frozen linear probe, LoRA, and conventional task baselines where applicable.
- [x] Require frozen-encoder decoder and nnU-Net/comparable 3D segmentation baselines.
- [x] Add no-visual, shuffled-visual, frozen-bridge, bridge-type, token-budget, and coordinate ablations for VLMs.
- [x] Fail the visual-grounding gate if shuffled inputs perform similarly within a predeclared margin.
- [ ] Add CPU/CUDA/TPU parity evaluation on deterministic tiny and representative fixtures.
- [x] Predeclare backend-specific numerical tolerances and investigate, rather than normalize away, larger divergence.

### Distributed metric correctness

- [x] Reduce metric numerators/denominators by true valid counts across ranks.
- [x] Remove padded duplicate samples before patient/study/slide aggregation.
- [x] Gather variable metadata on the host without introducing shape changes in compiled TPU steps.
- [x] Verify thresholds, calibration models, and bootstrap seeds are identical across backends.
- [x] Ensure only the coordinator writes final reports while all ranks participate in reductions.

### Human review and reporting

- [x] Implement the error categories from `idea.md`.
- [x] Define reviewer instructions, sampling, blinding, and disagreement resolution.
- [x] Record inter-rater agreement where multiple reviewers participate.
- [x] Separate major/potentially harmful errors from lexical mismatch.
- [x] Generate a versioned evaluation report with examples and known limitations.
- [x] Keep reviewed clinical text/images access-controlled.

## Tests and verification

- [x] Compare metric implementations against small hand-computed fixtures.
- [x] Test confidence intervals and cluster bootstrap determinism.
- [x] Test empty segmentation and rare-class edge cases.
- [x] Verify thresholds cannot be fit on test data through the API.
- [x] Verify metric aggregation uses the configured clinical unit.
- [x] Test invalid structured generation and contradiction categories.
- [x] Test visual-dependence ablation gate behavior.
- [x] Recompute a golden report from saved predictions and compare checksums/tolerances.
- [x] Compare single-device and distributed metrics on the same prediction set.
- [ ] Compare CUDA and TPU predictions/metrics within predeclared tolerances.
- [x] Verify padded final TPU batches cannot change clinical-unit metrics.

## Implementation references

- [Accelerator training strategy](accelerator_training_strategy.md)
- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)
- [MONAI metrics](https://docs.monai.io/en/stable/metrics.html)
- [RadGraph](https://github.com/Stanford-AIMI/radgraph)

## Smoke command

```bash
python -m medfm.cli.evaluate --config configs/smoke/evaluation.yaml
```

## Acceptance command

```bash
pytest tests/phase_16 -q && python -m medfm.tools.validate_phase --phase 16
```

## Exit criteria

- [ ] Every accepted recipe has a complete metric suite.
- [x] Metrics and confidence intervals use the correct clinical unit.
- [x] VLM ablations demonstrate meaningful visual dependence.
- [ ] Release candidates include subgroup results and representative errors.
- [x] Reports contain no unsupported clinical-validation claim.
- [x] Backend parity and distributed aggregation reports pass for every claimed accelerator.

## Handoff

- [x] Publish evaluation schema and release-report template.
- [x] Publish metric thresholds and visual-grounding gate definitions.
- [x] Record missing external-site/human-review evidence as release limitations.
- [x] Provide calibration and metric artifacts to Phase 17.
- [x] Publish backend tolerances and unresolved CUDA/TPU divergences.
