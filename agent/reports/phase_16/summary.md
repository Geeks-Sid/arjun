# Phase 16 completion summary

Phase 16 delivers a deterministic, clinical-unit-aware evaluation layer under `medfm/evaluation/` plus an offline evaluation CLI and versioned report/prediction schemas.

Implemented contracts include:

- versioned prediction, target, group, metric-result, runtime-provenance, and report schemas;
- patient/study/slide identity validation and padding-safe clinical-unit aggregation;
- offline prediction artifact persistence and metric recomputation;
- classification discrimination, calibration, operating points, fixed-target helpers, subgroup metrics, and cluster bootstrap intervals;
- 2D/3D segmentation Dice, IoU, surface Dice, HD95, ASSD, lesion sensitivity, false-positive lesions, volume error, box IoU, and physical localization;
- bidirectional retrieval, structured-generation validity, finding-level/attribute/contradiction/omission/hallucination analysis, and secondary BLEU/ROUGE;
- native-3D/slice-sequence comparison, adjacent-slice consistency, small-lesion sensitivity, pathology tile/slide/patient/site metrics, and tile-count/magnification sweeps;
- uncertainty/selective-risk summaries, baselines, holdout tables, visual-grounding gate, distributed count reductions, padded-sample removal, and backend parity tolerances;
- access-controlled, blinded human-review protocol with required error categories and inter-rater agreement;
- safety-checked versioned reports that separate scientific metrics from accelerator evidence and prohibit unsupported clinical-validation claims.

The smoke evaluation produced a saved prediction artifact and report from `configs/smoke/evaluation.yaml`. The report records `per_patient` metrics and the required clinical safety limitation.
