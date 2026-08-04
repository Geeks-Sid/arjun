# Phase 16: Evaluation, Clinical Validation, and Regression Testing

## Objective

Build a clinical-unit-aware evaluation layer covering discrimination, calibration, spatial quality, retrieval, generation, grounding, subgroup behavior, uncertainty, baselines, ablations, and human review.

## Dependencies

- [ ] Phase 11 task outputs and metric lifecycle are stable.
- [ ] At least one recipe from Phases 13-15 can emit predictions.
- [ ] Clinical validation language and prohibited claims from Phase 00 are available.

## Scope boundaries

Allowed areas: `medfm/evaluation/`, evaluation CLI/configs, reporting templates, golden fixtures, and Phase 16 tests.

Do not label any model clinically validated without a separate explicit validation study.

## Implementation checklist

### Evaluation data model

- [ ] Define versioned prediction, target, group, and metric-result schemas.
- [ ] Require patient/study/slide clinical unit declarations.
- [ ] Separate threshold selection data from final test data.
- [ ] Save prediction artifacts with model/data/preprocess hashes.
- [ ] Support deterministic recomputation of metrics without rerunning inference.
- [ ] Record backend, precision, topology, bucket, and checkpoint format with every prediction artifact.
- [ ] Distinguish scientific model metrics from accelerator parity/performance metrics.

### Classification and calibration

- [ ] Implement AUROC, AUPRC, sensitivity, specificity, precision, recall, F1, balanced accuracy, and confusion matrices.
- [ ] Implement Brier score and expected calibration error.
- [ ] Implement sensitivity at fixed specificity and the inverse.
- [ ] Report per-class, macro, and micro metrics appropriately.
- [ ] Compute patient-level bootstrap confidence intervals.
- [ ] Never use slice-level confidence intervals when the clinical unit is patient/study.
- [ ] Fit thresholds/calibration on validation data only.

### Segmentation and localization

- [ ] Implement Dice, IoU, surface Dice, HD95, ASSD, lesion sensitivity, false-positive lesions/scan, and volume error.
- [ ] Evaluate in original physical space where practical.
- [ ] Handle empty-target and empty-prediction cases explicitly.
- [ ] Report per-class and macro summaries.
- [ ] Implement 2D/3D box IoU and physical localization error.

### Retrieval, VQA, and generation

- [ ] Implement Recall@1/5/10, median/mean rank, and mAP in both directions.
- [ ] Implement exact match, token F1, schema validity, and finding-level precision/recall.
- [ ] Score negation, laterality, severity, and anatomy correctness.
- [ ] Add RadGraph-style or equivalent clinical entity/relation evaluation where approved.
- [ ] Add contradiction, omission, and hallucinated-finding analyses.
- [ ] Keep BLEU/ROUGE secondary rather than headline metrics.

### 3D and pathology-specific evaluation

- [ ] Evaluate native 3D and slice-sequence VLMs separately and comparatively.
- [ ] Include small-lesion, anatomy localization, adjacent-slice consistency, and 3D boxes.
- [ ] Report pathology tile, slide, and patient metrics.
- [ ] Report scanner/site/organ results and evidence localization.
- [ ] Sweep sampled tile count and magnification.

### Generalization, baselines, and ablations

- [ ] Support internal, patient-disjoint, external-site, temporal, vendor/protocol, rare-class, missing-sequence, and low-quality holdouts.
- [ ] Require random/majority, frozen linear probe, LoRA, and conventional task baselines where applicable.
- [ ] Require frozen-encoder decoder and nnU-Net/comparable 3D segmentation baselines.
- [ ] Add no-visual, shuffled-visual, frozen-bridge, bridge-type, token-budget, and coordinate ablations for VLMs.
- [ ] Fail the visual-grounding gate if shuffled inputs perform similarly within a predeclared margin.
- [ ] Add CPU/CUDA/TPU parity evaluation on deterministic tiny and representative fixtures.
- [ ] Predeclare backend-specific numerical tolerances and investigate, rather than normalize away, larger divergence.

### Distributed metric correctness

- [ ] Reduce metric numerators/denominators by true valid counts across ranks.
- [ ] Remove padded duplicate samples before patient/study/slide aggregation.
- [ ] Gather variable metadata on the host without introducing shape changes in compiled TPU steps.
- [ ] Verify thresholds, calibration models, and bootstrap seeds are identical across backends.
- [ ] Ensure only the coordinator writes final reports while all ranks participate in reductions.

### Human review and reporting

- [ ] Implement the error categories from `idea.md`.
- [ ] Define reviewer instructions, sampling, blinding, and disagreement resolution.
- [ ] Record inter-rater agreement where multiple reviewers participate.
- [ ] Separate major/potentially harmful errors from lexical mismatch.
- [ ] Generate a versioned evaluation report with examples and known limitations.
- [ ] Keep reviewed clinical text/images access-controlled.

## Tests and verification

- [ ] Compare metric implementations against small hand-computed fixtures.
- [ ] Test confidence intervals and cluster bootstrap determinism.
- [ ] Test empty segmentation and rare-class edge cases.
- [ ] Verify thresholds cannot be fit on test data through the API.
- [ ] Verify metric aggregation uses the configured clinical unit.
- [ ] Test invalid structured generation and contradiction categories.
- [ ] Test visual-dependence ablation gate behavior.
- [ ] Recompute a golden report from saved predictions and compare checksums/tolerances.
- [ ] Compare single-device and distributed metrics on the same prediction set.
- [ ] Compare CUDA and TPU predictions/metrics within predeclared tolerances.
- [ ] Verify padded final TPU batches cannot change clinical-unit metrics.

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
- [ ] Metrics and confidence intervals use the correct clinical unit.
- [ ] VLM ablations demonstrate meaningful visual dependence.
- [ ] Release candidates include subgroup results and representative errors.
- [ ] Reports contain no unsupported clinical-validation claim.
- [ ] Backend parity and distributed aggregation reports pass for every claimed accelerator.

## Handoff

- [ ] Publish evaluation schema and release-report template.
- [ ] Publish metric thresholds and visual-grounding gate definitions.
- [ ] Record missing external-site/human-review evidence as release limitations.
- [ ] Provide calibration and metric artifacts to Phase 17.
- [ ] Publish backend tolerances and unresolved CUDA/TPU divergences.
