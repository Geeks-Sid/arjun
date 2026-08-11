# Selection OOF and freeze

This phase turns preregistered development comparisons into one immutable research artifact, permits exactly one lockbox use, and then creates a separately named all-label deployment artifact. Every API treats `dev`, `oof_holdout`, `lockbox`, and `participant_validation` as different roles. Participant validation is inference-only: its records may never carry targets, proxy scores, fit inputs, bootstrap rows, thresholds, or selection evidence.

All new modules live under the package created by SCH-01 and exported intentionally by SCH-09. Packaging acceptance must keep `medfm.challenges` in the inventory exercised by `tests/phase_01/test_packaging.py::{test_subpackages_importable,test_no_forbidden_top_level_imports}`; selection modules must be CPU-import-safe and lazily import accelerator libraries. Ordinary tests use deterministic tiny fixtures under `tests/challenges/medreason/`. Any real checkpoint/GPU test is separately marked `real_checkpoint` and `gpu`, is skipped unless both `MEDFM_RUN_REAL_CHECKPOINTS=1` and `MEDFM_RUN_GPU_TESTS=1`, and must also enforce the selected measured hardware profile. The local RTX 3090 and available disk do not satisfy the approved 31B/26B or 600 GiB gates.

## SEL-01 Freeze candidates prompts parsers transforms and seeds

**Depends on:** SCH-01, SCH-05, SCH-09, SPL-07, STR-05, EVA-05, EVA-09, MCQ-15, OPEN-15, RUN-09, RUN-10, RUN-11, RUN-12, RUN-13, RUN-14.

**Parallel safety and exclusive file ownership:** Sole owner of `configs/recipes/medreason/selection_preregistration.yaml`, `medfm/challenges/medreason/selection.py::{SelectionPreregistration,OptionalHypothesis,freeze_preregistration,load_preregistration}`, and `artifacts/runs/medreason/selection/preregistration/`. SEL-06 code may proceed concurrently, but SEL-02 through SEL-05 must consume this schema and must not define alternate constants.

**Target paths/symbols:** The paths/symbols above; tests in `tests/challenges/medreason/test_selection.py`.

**Inputs:** Frozen candidate manifests; exact prompt/rubric, parser, processor/chat-template, transform, split, judge, model, and runtime-component hashes. The finite candidate list includes only previously accepted components; metric results are prohibited inputs.

**Outputs and exact artifact schema:** Canonical `preregistration.json` plus `preregistration.sha256`. JSON is exactly `{schema_version:int,created_at_utc:str,split_manifest_sha256:sha256,candidates:[{candidate_id:str,parent_id:str|null,component_ids:[str],artifact_manifest_sha256:sha256}],hypotheses:[{hypothesis_id:str,candidate_id:str,parent_id:str,intended_metric:"mcq"|"gt"|"va",family_id:"optional_components",requires_shuffled_image_gate:bool}],frozen_hashes:{prompts:{str:sha256},parsers:{str:sha256},processors:{str:sha256},transforms:{str:sha256},judges:{str:sha256}},bootstrap:{seed:2026,resamples:1000,familywise_alpha:0.05,strata_weights:{clean:number,xray_stress:number,mri_stress:number}},non_inferiority:{mcq:-0.002,gt:-0.05,va:-0.05},ambitions:{mcq:0.975,gt:2.15,va:2.85},oof:{seed:2026,folds:3},selection:{metric_normalizers:{mcq:0.002,gt:0.05,va:0.05},tie_breaks:["all_ambitions","worst_normalized_margin","latency","component_count","candidate_id"]}}`. SHA-256 values are lowercase 64-hex; IDs are unique and sorted canonically.

**Ordered implementation:**
1. Validate every referenced file/hash; reject missing parents, cycles, duplicate IDs, mutable revisions, and any lockbox/participant-validation/hidden path.
2. Freeze the whole optional-test family before reading comparison metrics. Every non-root candidate names one parent and intended metric; visual components require the shuffled-image gate.
3. Select only audited weights: `0.50/0.25/0.25`, `0.50/0.50/0`, `0.50/0/0.50`, or `1/0/0`; never infer modality or invent a stratum.
4. Serialize with existing `medfm.core.serialization.canonical_json`, write atomically, hash exact bytes, and fail rather than overwrite.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection freeze-preregistration --config configs/recipes/medreason/selection_preregistration.yaml --output artifacts/runs/medreason/selection/preregistration`.

**Focused tests and exact command:** Deterministic two-candidate fixtures assert byte-identical repeats and reject reordered duplicates, a changed prompt hash, late hypothesis, cycle, protected path, and invented stratum. `uv run pytest tests/challenges/medreason/test_selection.py::test_preregistration_is_canonical_and_immutable tests/challenges/medreason/test_selection.py::test_preregistration_rejects_protected_inputs -q`.

**Acceptance evidence:** Fixture acceptance is canonical JSON/hash snapshots and focused test output. Protected acceptance requires real source hashes; fixture hashes do not establish licenses, judge access, or hardware readiness. Packaging inventory: `uv run pytest tests/phase_01/test_packaging.py::test_subpackages_importable tests/phase_01/test_packaging.py::test_no_forbidden_top_level_imports -q`.

**Non-goals/failure policy:** Never add candidates after metrics are visible, tune weights from results, or relabel participant validation as dev. Missing/drifted hashes fail closed and require a new preregistration lineage, never mutation.

**Handoff:** SEL-02 through SEL-16 consume the immutable path/hash, candidate DAG, hypotheses, tolerances, ambitions, seeds, and strata weights.

## SEL-02 Implement group-paired stratified bootstrap resampling

**Depends on:** SEL-01, SPL-07, STR-05, EVA-03, EVA-04.

**Parallel safety and exclusive file ownership:** Sole owner of `selection.py::{PairedBootstrapResult,group_paired_bootstrap,paired_metric_delta,combine_strata}`. It may run alongside SEL-03 implementation; each comparison exclusively writes its own `<hypothesis_id>/` directory.

**Target paths/symbols:** `medfm/challenges/medreason/selection.py`; `tests/challenges/medreason/test_selection.py`. Do not reuse `medfm.evaluation.advanced.cluster_bootstrap_ci`: the observed utility is unpaired and binary-only.

**Inputs:** Frozen preregistration and parent/child dev rows keyed by `(case_id,group_id,view_id,candidate_id)`, with immutable stress-pair IDs and computed MCQ/GT/VA values.

**Outputs and exact artifact schema:** `artifacts/runs/medreason/selection/comparisons/<hypothesis_id>/bootstrap.json`: `{schema_version:int,preregistration_sha256:sha256,hypothesis_id:str,seed:int,resamples:1000,group_count:int,strata:[{name:"clean"|"xray_stress"|"mri_stress",weight:number,group_ids_sha256:sha256}],point_delta:{mcq:number,gt:number,va:number},resample_deltas:{mcq:[number],gt:[number],va:[number]},pairing_sha256:sha256}`. Each array has exactly 1,000 finite values in generator order.

**Ordered implementation:**
1. Join child/parent by case, group, and view; fail on missing/duplicate/cross-group rows and verify clean/stress pairs preserve targets.
2. Sort group lists per stratum. Draw whole groups with replacement using the frozen local RNG and reuse identical sampled indices for parent, child, clean, and its stress view.
3. Compute case metrics, then group means, then stratum means, then the frozen weighted total. Never resample cases/views individually.
4. Record a sampled-index/pairing digest without questions, references, metadata, or paths. Reject `oof_holdout`, `lockbox`, `participant_validation`, and hidden roles for component promotion.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection bootstrap --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --hypothesis-id <ID> --predictions artifacts/runs/medreason/selection/dev_predictions --output artifacts/runs/medreason/selection/comparisons/<ID>`.

**Focused tests and exact command:** Unequal synthetic groups, paired views, and an absent stratum prove stable draws, whole-group duplication, shared indices, hand-computed weights, and rejection of one-case resampling/protected roles. `uv run pytest tests/challenges/medreason/test_selection.py::test_group_paired_bootstrap_is_deterministic_and_whole_group tests/challenges/medreason/test_selection.py::test_bootstrap_rejects_unpaired_or_protected_rows -q`.

**Acceptance evidence:** Fixture acceptance is exact vector/delta assertions. Protected acceptance requires hashes matching the real dev split/stress manifests; it does not establish lockbox or participant-validation access.

**Non-goals/failure policy:** Not an IID bootstrap or official interval. Empty/non-finite metrics, mismatched pairs, or protected roles stop the hypothesis rather than dropping rows.

**Handoff:** SEL-03/SEL-04 receive immutable candidate-minus-parent vectors and point deltas linked to preregistration.

## SEL-03 Apply Holm-Bonferroni across optional component tests

**Depends on:** SEL-01, SEL-02.

**Parallel safety and exclusive file ownership:** Sole owner of `selection.py::{HolmDecision,holm_bonferroni,holm_percentile_intervals}` and `family_decisions.json`. It reads the family only after all preregistered hypotheses settle.

**Target paths/symbols:** `medfm/challenges/medreason/selection.py`; `tests/challenges/medreason/test_selection.py`.

**Inputs:** Exact `family_id="optional_components"` membership and every SEL-02 bootstrap artifact. Failed hypotheses remain family members and are explicit non-promotions.

**Outputs and exact artifact schema:** `artifacts/runs/medreason/selection/family_decisions.json`: `{schema_version:int,preregistration_sha256:sha256,family_id:"optional_components",familywise_alpha:0.05,hypothesis_count:int,ordered:[{rank:int,hypothesis_id:str,raw_two_sided_p:number,holm_threshold:number,adjusted_p:number,reject_null:bool,adjusted_intervals:{mcq:[number,number],gt:[number,number],va:[number,number]}}]}` sorted by `(raw_two_sided_p,hypothesis_id)` with monotone adjusted p-values capped at 1.

**Ordered implementation:**
1. Require exactly complete frozen membership; missing, extra, or duplicate evidence fails the family.
2. Compute deterministic two-sided bootstrap p-values with frozen finite-sample convention `(extreme+1)/(resamples+1)`.
3. Apply Holm step-down `0.05/(m-rank+1)`, stop rejection after first failure, and calculate monotone adjusted p-values.
4. Produce adjusted two-sided percentile bounds for all three deltas at each hypothesis’s Holm alpha. Never shrink `m` and rerun.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection correct-family --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --comparisons artifacts/runs/medreason/selection/comparisons --output artifacts/runs/medreason/selection/family_decisions.json`.

**Focused tests and exact command:** Fixed synthetic p-values `[0.001,0.02,0.04]` and tied IDs verify order, thresholds, stop behavior, monotone adjustment, widened intervals, input-order invariance, and missing-member failure. `uv run pytest tests/challenges/medreason/test_selection.py::test_holm_step_down_and_intervals_are_deterministic tests/challenges/medreason/test_selection.py::test_holm_requires_complete_preregistered_family -q`.

**Acceptance evidence:** Fixture acceptance is exact hand-computed values. Protected acceptance is a complete family linked to every dev bootstrap hash, not an official judge result.

**Non-goals/failure policy:** No convenient subfamilies or uncorrected promotion intervals. Incomplete evidence rejects affected components without reducing multiplicity.

**Handoff:** SEL-04/SEL-05 consume adjusted intervals/p-values and ordered family membership.

## SEL-04 Enforce intended-metric and non-inferiority promotion gates

**Depends on:** SEL-02, SEL-03, EVA-03, EVA-04.

**Parallel safety and exclusive file ownership:** Sole owner of `selection.py::{PromotionDecision,apply_promotion_gate}` and append-free `promotion_decisions.json`; report writers are read-only.

**Target paths/symbols:** `medfm/challenges/medreason/selection.py`; `tests/challenges/medreason/test_selection.py`.

**Inputs:** SEL-03 adjusted intervals; intended metric; margins `mcq=-0.002`, `gt=-0.05`, `va=-0.05` (MCQ fraction, not display percentage).

**Outputs and exact artifact schema:** Each decision is `{schema_version:int,preregistration_sha256:sha256,hypothesis_id:str,intended_metric:str,metrics:{mcq:{point:number,adjusted_lower:number,adjusted_upper:number,margin:number,pass:bool},gt:{...},va:{...}},intended_improvement_pass:bool,all_non_inferiority_pass:bool,grounding_gate_required:bool,grounding_gate_pass:bool|null,promote:bool,reasons:[str]}`.

**Ordered implementation:**
1. Require intended metric adjusted lower bound strictly `>0`; equality fails.
2. Require every companion metric adjusted lower bound strictly greater than its frozen margin; equality fails. Check the intended metric margin too without weakening positive improvement.
3. Leave visual candidates pending SEL-05; nonvisual candidates set `grounding_gate_required=false`.
4. Evaluate every condition without short-circuiting and decide from unrounded values. Reject any evidence not role-tagged `dev` and hash-linked.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection gate --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --family-decisions artifacts/runs/medreason/selection/family_decisions.json --output artifacts/runs/medreason/selection/promotion_decisions.json`.

**Focused tests and exact command:** Synthetic boundaries cover `0`, next positive float, exact NI margins, NaN, and an intended gain with companion harm; protected roles fail. `uv run pytest tests/challenges/medreason/test_selection.py::test_promotion_gate_strict_boundaries_and_noninferiority tests/challenges/medreason/test_selection.py::test_promotion_gate_rejects_non_dev_evidence -q`.

**Acceptance evidence:** Fixture boundary matrix; protected acceptance needs corrected dev intervals from frozen judges/parsers. Lexical diagnostics cannot satisfy GT/VA gates.

**Non-goals/failure policy:** No scalar compensation or manual override. Missing/non-finite metric or unavailable required judge fails promotion.

**Handoff:** SEL-05/SEL-06 receive resolved or grounding-pending component decisions.

## SEL-05 Enforce shuffled-image grounding promotion gate

**Depends on:** SEL-03, SEL-04, EVA-11.

**Parallel safety and exclusive file ownership:** Sole owner of `selection.py::apply_shuffled_image_gate` and each `grounding_gate.json`; it may only attach a gate boolean/hash to a pending decision.

**Target paths/symbols:** New challenge-specific function; reuse only compatible semantics from observed `medfm.evaluation.advanced.visual_grounding_gate`, not its unpaired statistics; tests in `tests/challenges/medreason/test_selection.py`.

**Inputs:** Same-case real-image and group-safe shuffled-image dev predictions and preregistered visual hypotheses.

**Outputs and exact artifact schema:** `{schema_version:int,preregistration_sha256:sha256,hypothesis_id:str,split_role:"dev",real_prediction_sha256:sha256,shuffled_prediction_sha256:sha256,group_derangement_sha256:sha256,metric:"va",real_minus_shuffled:{point:number,adjusted_lower:number,adjusted_upper:number},required_minimum:0.25,pass:bool}`.

**Ordered implementation:**
1. Verify identical case IDs/targets and a deterministic derangement with no same-group image reassignment.
2. Compute paired real-minus-shuffled proxy VA with SEL-02 sampled groups and SEL-03 family correction.
3. For late open visual components require point gap `>=0.25` and corrected lower bound `>0`; apply only when preregistered.
4. Fail if shuffled execution became no-image fallback or changed prompt/parser/judge hashes. Never fit this on lockbox or participant validation.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection grounding-gate --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --real <REAL_JSONL> --shuffled <SHUFFLED_JSONL> --output <HYPOTHESIS_DIR>/grounding_gate.json`.

**Focused tests and exact command:** Synthetic 0.30 gap passes; 0.25 with interval crossing zero fails; same-group shuffles fail; input permutations remain canonical; nonvisual candidates bypass without a fake gap. `uv run pytest tests/challenges/medreason/test_selection.py::test_shuffled_image_gate_requires_paired_gap tests/challenges/medreason/test_selection.py::test_shuffled_image_gate_rejects_leaky_derangements -q`.

**Acceptance evidence:** Fixture acceptance proves state/statistics only. Protected acceptance requires real/shuffled dev artifacts from the frozen visual judge; no synthetic grounding claim.

**Non-goals/failure policy:** Does not establish causal clinical grounding or official VA. Invalid/missing evidence rejects the visual child and preserves its parent.

**Handoff:** SEL-06 receives a fully resolved component DAG with failed leaves removed.

## SEL-06 Create three group folds over development pool

**Depends on:** SEL-01, SEL-04, SEL-05, SPL-07.

**Parallel safety and exclusive file ownership:** Sole owner of `medfm/challenges/medreason/oof.py::{OOFFoldManifest,create_group_folds,validate_oof_folds}` and `artifacts/data/medreason/derived/oof_folds.json`.

**Target paths/symbols:** Paths above; `tests/challenges/medreason/test_oof.py`.

**Inputs:** Immutable 70% train plus 15% dev IDs/groups, released stratification tags, seed 2026, resolved candidates. Lockbox and participant-validation manifests are exclusion-only.

**Outputs and exact artifact schema:** `{schema_version:int,seed:2026,fold_count:3,source_split_manifest_sha256:sha256,development_pool_case_ids_sha256:sha256,excluded:{lockbox_case_ids_sha256:sha256,participant_validation_case_ids_sha256:sha256},folds:[{fold_id:0|1|2,train_group_ids:[str],holdout_group_ids:[str],train_case_ids_sha256:sha256,holdout_case_ids_sha256:sha256,stratum_counts:{str:int}}]}`. Every development group is held out once.

**Ordered implementation:**
1. Union only train/dev into the 85% pool and prove disjointness from protected roles; reject missing group IDs.
2. Deterministically stratify whole groups into three folds with seed 2026; never split a group to improve balance.
3. For fold `k`, train on the other folds and predict only `k`; roles are `oof_train` and `oof_holdout`.
4. Validate exactly-once coverage, pairwise group disjointness, canonical ID sorting, and stable serialization. Report sparse strata rather than duplicating groups.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof create-folds --split-manifest artifacts/data/medreason/derived/splits.json --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --output artifacts/data/medreason/derived/oof_folds.json`.

**Focused tests and exact command:** Uneven synthetic groups prove stable assignments/coverage and reject duplicate IDs, lockbox/participant-validation collisions, and missing groups. `uv run pytest tests/challenges/medreason/test_oof.py::test_three_folds_are_group_disjoint_and_deterministic tests/challenges/medreason/test_oof.py::test_oof_folds_reject_protected_overlap -q`.

**Acceptance evidence:** Fixture fold/hash assertions; protected acceptance requires the real split hash and proves no training/hardware capability.

**Non-goals/failure policy:** Do not recreate/tune the original split, rebalance after predictions, or expose protected labels. Any overlap blocks OOF.

**Handoff:** SEL-07/SEL-12 consume fold IDs and case/group/exclusion hashes.

## SEL-07 Persist every out-of-fold candidate prediction

**Depends on:** SEL-06, EVA-09, MCQ-15, OPEN-15, RUN-01, RUN-15.

**Parallel safety and exclusive file ownership:** Sole owner of `oof.py::{OOFPredictionRow,write_oof_predictions,load_oof_predictions}`. Fold/candidate jobs write unique directories; one consolidator exclusively writes `index.json`.

**Target paths/symbols:** `medfm/challenges/medreason/oof.py`; reuse deterministic persistence patterns from observed `medfm.evaluation.artifacts.save_prediction_artifact`; `tests/challenges/medreason/test_oof.py`.

**Inputs:** Fold checkpoints proven not trained on their holdout groups; raw/parsed predictions, option logits/open response/support fields, and sanitized telemetry.

**Outputs and exact artifact schema:** Per-fold JSONL row: `{schema_version:int,candidate_id:str,fold_id:int,split_role:"oof_holdout",case_id:str,group_id:str,task_type:"mcq"|"open",prediction:{mcq_label:str|null,option_log_likelihoods:{str:number}|null,open_response:{observations:[str],reasoning:str,answer:str}|null,support_probabilities:[number]|null,answer_log_likelihood:number|null},target_metrics:{mcq:number|null,gt:number|null,va:number|null,rvf:number|null},latency_ms:number,failure_class:str|null,fold_checkpoint_sha256:sha256,processor_sha256:sha256,prompt_sha256:sha256,parser_sha256:sha256}`. Index: `{schema_version:int,oof_fold_manifest_sha256:sha256,preregistration_sha256:sha256,candidates:{str:{fold_files:[{fold_id:int,path:str,sha256:sha256,row_count:int}],coverage_case_ids_sha256:sha256}}}`.

**Ordered implementation:**
1. Verify each checkpoint train hash equals its fold train hash and excludes holdout.
2. Emit only holdout rows, one per candidate/case, with inapplicable fields explicitly null.
3. Require exact coverage, shared judge/target provenance, finite scores, and no duplicates.
4. Canonically serialize `(candidate_id,fold_id,case_id)`, hash every file, and refuse overwrite. Reject dev-final/lockbox/participant-validation/hidden rows.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof index-predictions --fold-manifest artifacts/data/medreason/derived/oof_folds.json --prediction-root artifacts/runs/medreason/oof --output artifacts/runs/medreason/oof/index.json`.

**Focused tests and exact command:** Three-fold/two-candidate fixtures assert exact coverage, stable bytes, preserved fit fields, and rejection of in-fold checkpoints, duplicates, missing cases, non-finite values, and protected roles. `uv run pytest tests/challenges/medreason/test_oof.py::test_oof_index_has_exact_coverage_and_stable_hashes tests/challenges/medreason/test_oof.py::test_oof_persistence_rejects_training_and_split_leakage -q`.

**Acceptance evidence:** Fixture persistence proof; protected acceptance requires real fold checkpoint/prediction hashes. Synthetic judge values are not judge evidence.

**Non-goals/failure policy:** Do not regenerate only bad rows, average away failures, or persist sensitive text/paths. Incomplete coverage makes a candidate ineligible.

**Handoff:** SEL-08 through SEL-12 consume the immutable OOF index only.

## SEL-08 Fit temperatures from out-of-fold predictions only

**Depends on:** SEL-07.

**Parallel safety and exclusive file ownership:** Sole owner of `oof.py::{TemperatureFit,fit_oof_temperature}` and each candidate’s `temperature.json`; candidates may fit concurrently in disjoint directories.

**Target paths/symbols:** `medfm/challenges/medreason/oof.py`; `tests/challenges/medreason/test_oof.py`. Do not reuse observed `evaluation.calibration.fit_calibration`, which is validation histogram calibration rather than OOF temperature fitting.

**Inputs:** MCQ option log-likelihoods or declared binary support logits from complete `oof_holdout` rows only.

**Outputs and exact artifact schema:** `{schema_version:int,candidate_id:str,oof_index_sha256:sha256,fit_case_ids_sha256:sha256,input_kind:"mcq_options"|"support",method:"bounded_nll",bounds:[0.05,10.0],temperature:number,objective_before:number,objective_after:number,optimizer:{algorithm:"deterministic_scalar_grid_brent",tolerance:number,max_iterations:int},fold_counts:{"0":int,"1":int,"2":int}}`.

**Ordered implementation:**
1. Enforce role and complete lineage in the fit API; reject dev-final, lockbox, participant-validation, hidden, and in-fold rows.
2. Minimize pooled held-out NLL for one positive temperature at frozen bounds/tolerance; exact ties choose the smaller temperature.
3. Fit only preregistered score types, never GT/VA judge scores, and record fit digest/objectives/optimizer contract.
4. Treat `1.0` only as a valid optimum, never a fallback. Missing correct-option scores, non-finite logits, or incomplete folds fail.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof fit-temperature --oof-index artifacts/runs/medreason/oof/index.json --candidate-id <ID> --output artifacts/runs/medreason/selection/calibration/<ID>/temperature.json`.

**Focused tests and exact command:** Known overconfident logits yield a stable fit under row shuffle; deterministic ties and protected/in-fold rejection are covered. `uv run pytest tests/challenges/medreason/test_oof.py::test_temperature_fit_is_deterministic_and_oof_only tests/challenges/medreason/test_oof.py::test_temperature_fit_rejects_protected_and_in_fold_rows -q`.

**Acceptance evidence:** Fixture numeric fit and NLL non-increase; protected acceptance needs real complete OOF evidence and says nothing about lockbox calibration.

**Non-goals/failure policy:** No lockbox/dev-final refit or substitute calibrator. Invalid evidence makes candidate ineligible.

**Handoff:** SEL-09/SEL-10 receive calibrated OOF probabilities and immutable fit hashes.

## SEL-09 Fit non-negative fusion and support weights

**Depends on:** SEL-07, SEL-08, RUN-11, RUN-12, RUN-13.

**Parallel safety and exclusive file ownership:** Sole owner of `oof.py::{FusionFit,fit_nonnegative_fusion}` and `fusion.json`; RUN-13 remains owner of runtime fusion code.

**Target paths/symbols:** `medfm/challenges/medreason/oof.py`; `tests/challenges/medreason/test_oof.py`.

**Inputs:** Calibrated OOF-only answer likelihoods and atomic-claim support probabilities from preregistered, accessible routes. Unlicensed/unavailable components are absent, never faked.

**Outputs and exact artifact schema:** `{schema_version:int,candidate_id:str,oof_index_sha256:sha256,temperature_hashes:[sha256],feature_order:["gemma_answer_ll","medgemma_answer_ll","geometric_support"],constraints:{nonnegative:true,sum_to_one:true},weights:[number],objective:"mean_open_nll",objective_value:number,fit_case_ids_sha256:sha256,fold_counts:{"0":int,"1":int,"2":int}}`; order contains only available preregistered features.

**Ordered implementation:**
1. Construct features only from OOF rows; calculate geometric support using frozen epsilon and calibrated supported probabilities.
2. Solve the simplex objective deterministically from uniform initialization; ties prefer the simpler prefix feature set.
3. Reject negative/non-finite weights, missing cases, feature-order drift, in-fold/protected predictions, or any proxy judge runtime feature.
4. Persist zero weights rather than post-fit pruning and preserve exact temperature hashes.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof fit-fusion --oof-index artifacts/runs/medreason/oof/index.json --temperatures artifacts/runs/medreason/selection/calibration --candidate-id <ID> --output artifacts/runs/medreason/selection/calibration/<ID>/fusion.json`.

**Focused tests and exact command:** Synthetic convex optimum verifies simplex, input-order invariance, ties, and failures for a negative solution, judge features, protected roles, or missing specialist rows. `uv run pytest tests/challenges/medreason/test_oof.py::test_fusion_fit_obeys_simplex_and_is_deterministic tests/challenges/medreason/test_oof.py::test_fusion_fit_rejects_non_oof_or_judge_features -q`.

**Acceptance evidence:** Fixture optimizer/schema proof; protected acceptance requires licensed real OOF artifacts and does not justify unavailable MedGemma.

**Non-goals/failure policy:** No negative stacking, judge feature, lockbox refit, or invented specialist fallback. Failure removes fusion child and preserves parent.

**Handoff:** SEL-10/SEL-11/SEL-14 and RUN-13 consume feature order, weights, and fit hashes.

## SEL-10 Fit confidence view and support thresholds

**Depends on:** SEL-07, SEL-08, SEL-09, RUN-09, RUN-10.

**Parallel safety and exclusive file ownership:** Sole owner of `oof.py::{ThresholdFit,fit_oof_thresholds}` and `thresholds.json`; runtime view/support implementations are read-only inputs.

**Target paths/symbols:** New OOF fitter following split-check discipline of observed `medfm.evaluation.calibration.fit_threshold`; `tests/challenges/medreason/test_oof.py`.

**Inputs:** OOF confidence, paired base/extra-view outcomes, support probabilities, latency, and preregistered finite grids.

**Outputs and exact artifact schema:** `{schema_version:int,candidate_id:str,oof_index_sha256:sha256,fusion_sha256:sha256|null,grid_sha256:sha256,fit_case_ids_sha256:sha256,thresholds:{low_confidence:number,view_disagreement:number,support_keep:number},objective:{name:"preregistered_weighted_selection",value:number},constraints:{mcq_noninferiority:-0.002,gt_noninferiority:-0.05,va_noninferiority:-0.05,runtime_profile_id:str},tie_break:"fewer_views_then_higher_support_then_lexicographic"}`.

**Ordered implementation:**
1. Evaluate every frozen grid tuple on OOF rows only; runtime triggers cannot inspect targets.
2. Apply frozen clean/OOD weights, NI margins, and a measured runtime-profile ceiling when available.
3. Choose deterministically; ties prefer fewer views, stricter support, then lexicographic values.
4. If none clears gates, retain parent/no-extra-view state. Reject participant-validation distributions, lockbox/dev-final rows, judges as runtime features, and unmeasured hardware claims.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof fit-thresholds --oof-index artifacts/runs/medreason/oof/index.json --candidate-id <ID> --grid configs/recipes/medreason/selection_preregistration.yaml --output artifacts/runs/medreason/selection/calibration/<ID>/thresholds.json`.

**Focused tests and exact command:** Hand-computed synthetic grid covers deterministic ties, metric/runtime conflict, exact NI boundary, no-eligible parent state, missing folds, and protected roles. `uv run pytest tests/challenges/medreason/test_oof.py::test_threshold_grid_ties_and_gates_are_deterministic tests/challenges/medreason/test_oof.py::test_threshold_fit_never_uses_validation_distribution -q`.

**Acceptance evidence:** Fixture acceptance proves state/grid logic, not hardware. Protected acceptance requires complete OOF and measured profile evidence; otherwise runtime gate remains unproven.

**Non-goals/failure policy:** No participant-validation prevalence tuning, adaptive online fitting, or relaxed observed margins. Invalid evidence keeps parent.

**Handoff:** SEL-11/SEL-14 receive frozen thresholds/provenance; RUN-10/RUN-13 consume without fitting.

## SEL-11 Select system using normalized worst-margin rule

**Depends on:** SEL-08, SEL-09, SEL-10, EVA-13.

**Parallel safety and exclusive file ownership:** Sole owner of `selection.py::{SystemSelection,select_research_system}` and `system_selection.json`; Pareto inputs become read-only at invocation.

**Target paths/symbols:** `medfm/challenges/medreason/selection.py`; `tests/challenges/medreason/test_selection.py`.

**Inputs:** Complete OOF metrics; ambitions `0.975/2.15/2.85`; normalizers `0.002/0.05/0.05`; calibrated artifacts; measured comparable p95 latency when available; component count.

**Outputs and exact artifact schema:** `{schema_version:int,preregistration_sha256:sha256,oof_index_sha256:sha256,candidates:[{candidate_id:str,metrics:{mcq:number,gt:number,va:number},ambition_pass:{mcq:bool,gt:bool,va:bool},normalized_margins:{mcq:number,gt:number,va:number},worst_normalized_margin:number,latency_ms_p95:number|null,component_count:int,eligible:bool,exclusion_reasons:[str]}],winner_candidate_id:str|null,tie_break_trace:[{criterion:str,remaining_candidate_ids:[str]}],status:"selected"|"no_ambition_eligible_candidate"}`.

**Ordered implementation:**
1. Require all three ambitions. Compute `(metric-ambition)/normalizer` from unrounded values and maximize the minimum.
2. Exact ties prefer lower measured p95 latency, then fewer components, then candidate ID. Missing comparable latency is unknown, not zero; skip that tie-break.
3. Missing judge/metric evidence is ineligible. If none passes all ambitions, emit stop state and block SEL-13/lockbox rather than lowering thresholds.
4. Reject lockbox, participant-validation, hidden, final-full-pool, or unmeasured-hardware evidence.

**Execution command:** `uv run python -m medfm.challenges.medreason.selection select --preregistration artifacts/runs/medreason/selection/preregistration/preregistration.json --oof-index artifacts/runs/medreason/oof/index.json --calibration artifacts/runs/medreason/selection/calibration --output artifacts/runs/medreason/selection/system_selection.json`.

**Focused tests and exact command:** Synthetic candidates exercise ambition filtering, worst margin, every tie, exact boundaries, missing latency, no-eligible stop, and protected metrics. `uv run pytest tests/challenges/medreason/test_selection.py::test_system_selection_uses_worst_margin_and_ties tests/challenges/medreason/test_selection.py::test_system_selection_stops_without_three_ambitions -q`.

**Acceptance evidence:** Fixture tie trace; protected acceptance requires real OOF proxy judges and measured runtime. It supports only a local proxy claim, never an official win.

**Non-goals/failure policy:** No average rank/mean margin/manual preference/lockbox tie-break. No eligible candidate stops freeze.

**Handoff:** SEL-12/SEL-13 receive exactly one winner and artifact hashes or a stop state.

## SEL-12 Derive median best-step count from folds

**Depends on:** SEL-06, SEL-07, SEL-11, MCQ-10, OPEN-09.

**Parallel safety and exclusive file ownership:** Sole owner of `oof.py::{MedianStepRecord,derive_median_best_steps}` and `median_steps.json`; fold logs are immutable/read-only.

**Target paths/symbols:** `medfm/challenges/medreason/oof.py`; `tests/challenges/medreason/test_oof.py`.

**Inputs:** For the winner only, exactly three fold run manifests per selected route, with optimizer best-step chosen on that fold’s held-out groups using frozen precedence.

**Outputs and exact artifact schema:** `{schema_version:int,winner_candidate_id:str,oof_fold_manifest_sha256:sha256,routes:[{route_id:str,fold_runs:[{fold_id:0|1|2,run_manifest_sha256:sha256,best_step:int,selection_metric:str}],sorted_best_steps:[int,int,int],median_best_step:int}],early_stopping_disabled_for_full_pool:true}`.

**Ordered implementation:**
1. Require folds 0/1/2 exactly once per route and matching configs/data-order contracts.
2. Validate positive completed optimizer steps; reject epochs, microsteps, or filename-derived values.
3. Sort three values and take the middle independently for MCQ/open/specialist routes.
4. Disable full-pool/all-label best-step reselection and reject dev-final, lockbox, participant-validation, incomplete, or smoothed-log steps.

**Execution command:** `uv run python -m medfm.challenges.medreason.oof median-steps --selection artifacts/runs/medreason/selection/system_selection.json --fold-manifest artifacts/data/medreason/derived/oof_folds.json --run-root artifacts/runs/medreason/oof --output artifacts/runs/medreason/selection/median_steps.json`.

**Focused tests and exact command:** `[250,400,300] -> 300` under reorder; reject missing/duplicate folds, zero/float/microstep, mismatched hashes, and protected-role logs. `uv run pytest tests/challenges/medreason/test_oof.py::test_median_steps_uses_three_optimizer_steps tests/challenges/medreason/test_oof.py::test_median_steps_rejects_incomplete_or_leaky_runs -q`.

**Acceptance evidence:** Fixture median/state proof. Protected acceptance needs three real completed fold manifests per route and does not prove local hardware fit.

**Non-goals/failure policy:** No averaging/rounding, full-pool early stop, or lockbox step selection. One missing fold blocks SEL-13.

**Handoff:** SEL-13/SEL-16 consume per-route median steps and source hashes.

## SEL-13 Train selected system once on development pool

**Depends on:** SEL-11, SEL-12, MCQ-15, OPEN-15, MOD-13, MOD-14, GOV-09.

**Parallel safety and exclusive file ownership:** Sole owner of `medfm/challenges/medreason/freeze.py::{DevelopmentTrainingRequest,train_selected_development_system}` and `artifacts/runs/medreason/research_full_pool/`. Routes use disjoint subdirectories; no completed route is replaced under one run ID.

**Target paths/symbols:** New `freeze.py`; reuse observed `medfm.training.checkpoint.CheckpointManager`; `tests/challenges/medreason/test_freeze.py` and protected hardware cases in `tests/challenges/medreason/test_freeze_hardware.py`.

**Inputs:** Winner, exact 85% pool, median steps, frozen configs/seeds, base/processor hashes, OOF calibration, protected-exclusion digests, and measured hardware preflight evidence.

**Outputs and exact artifact schema:** `research_training_run.json`: `{schema_version:int,artifact_role:"research_evaluation",winner_candidate_id:str,development_pool_case_ids_sha256:sha256,excluded_lockbox_case_ids_sha256:sha256,excluded_participant_validation_case_ids_sha256:sha256,selection_sha256:sha256,median_steps_sha256:sha256,hardware_preflight_sha256:sha256,routes:[{route_id:str,config_sha256:sha256,seed:int,optimizer_steps:int,adapter_path:str,adapter_sha256:sha256,run_metadata_sha256:sha256,completed:bool}],calibration_hashes:[sha256]}`.

**Ordered implementation:**
1. Preflight licenses, hashes, exact winner, disk, attention parity, and measured 100-batch memory profile before model loading; enforce required profile rather than merely marking a test.
2. Prove training IDs equal train+dev and exclude lockbox/participant validation at loader boundary.
3. Train each route once for fixed median optimizer steps with no evaluation callback, early stopping, search, or calibration refit.
4. Export adapter-only safetensors and run metadata; retain OOF fits byte-for-byte. Preflight failure consumes nothing; a post-start failure is immutable and cannot be retried from lockbox feedback.

**Execution command:** `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.freeze train-development --selection artifacts/runs/medreason/selection/system_selection.json --median-steps artifacts/runs/medreason/selection/median_steps.json --split-manifest artifacts/data/medreason/derived/splits.json --output artifacts/runs/medreason/research_full_pool`.

**Focused tests and exact command:** Tiny trainers assert exact steps, one call, no evaluator, exact exclusions, unchanged fits, and reject overlap, second completion, or config drift. `uv run pytest tests/challenges/medreason/test_freeze.py::test_development_training_runs_once_for_median_steps tests/challenges/medreason/test_freeze.py::test_development_training_excludes_protected_splits -q`. Protected command, not run by default: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -m "gpu and real_checkpoint" tests/challenges/medreason/test_freeze_hardware.py::test_selected_profile_preflight_and_hundred_batches -q`.

**Acceptance evidence:** Fixture acceptance proves orchestration only. Protected acceptance requires real adapters, run metadata, and measured profile evidence; the 24,576 MiB local card cannot establish 31B/26B or 48/96 GB feasibility.

**Non-goals/failure policy:** No extra epochs, checkpoint cherry-pick, participant-validation training, or lockbox-guided retry. Hardware/disk/access failure blocks real training.

**Handoff:** SEL-14 receives completed adapters/run hashes, hardware evidence, and unchanged OOF fits.

## SEL-14 Freeze research artifact and manifest hash

**Depends on:** SEL-01, SEL-09, SEL-10, SEL-13, SCH-08, SCH-09.

**Parallel safety and exclusive file ownership:** Sole owner of `freeze.py::{ResearchArtifactManifest,freeze_research_artifact,verify_research_artifact}` and `artifacts/models/medreason/research_evaluation/`; later tasks may only read it.

**Target paths/symbols:** `medfm/challenges/medreason/freeze.py`; intentional exports in the SCH-09-owned package inventory; `tests/challenges/medreason/test_freeze.py`.

**Inputs:** Completed training run; exact base/processor/chat-template, adapters, runtime config, OOF fits, prompts/parsers/transforms, environment lock, exclusions, and measured profile evidence if available.

**Outputs and exact artifact schema:** `manifest.json`: `{schema_version:int,artifact_id:str,artifact_role:"research_evaluation",created_at_utc:str,preregistration_sha256:sha256,selection_sha256:sha256,research_training_run_sha256:sha256,split_manifest_sha256:sha256,training_case_ids_sha256:sha256,excluded_lockbox_case_ids_sha256:sha256,excluded_participant_validation_case_ids_sha256:sha256,files:[{path:str,sha256:sha256,size_bytes:int,kind:"base"|"processor"|"adapter"|"calibration"|"threshold"|"runtime_config"|"environment"}],runtime_profile:{profile_id:str,measured:bool,evidence_sha256:sha256|null},immutable:true}` plus `manifest.sha256` over canonical bytes.

**Ordered implementation:**
1. Stage winner files only; reject symlinks, escapes, mutable remote references, missing OOF artifacts, or hash drift.
2. Validate role and exact 85% training digest; include both protected-exclusion hashes and measured status honestly.
3. Hash sorted contained files, atomically rename complete directory, reopen/verify, then make immutable/read-only.
4. Exclude lockbox references/results, participant-validation data/predictions, judges, optimizer states, and failed candidates.

**Execution command:** `uv run python -m medfm.challenges.medreason.freeze freeze-research --training-run artifacts/runs/medreason/research_full_pool/research_training_run.json --output artifacts/models/medreason/research_evaluation`.

**Focused tests and exact command:** Synthetic files prove canonical order, containment, atomic completion, add/remove/change detection, role/exclusions, and calibration tamper failure. `uv run pytest tests/challenges/medreason/test_freeze.py::test_research_manifest_is_canonical_and_immutable tests/challenges/medreason/test_freeze.py::test_research_manifest_detects_tampering_and_protected_content -q`.

**Acceptance evidence:** Fixture tamper proof; protected acceptance requires every real file hash and measured evidence only when `measured=true`. Packaging inventory must pass the command stated at phase top.

**Non-goals/failure policy:** No judges/protected data/caches in runtime. Hash drift invalidates the artifact; never repair in place.

**Handoff:** SEL-15 receives the sole manifest hash and lockbox exclusion digest; SEL-16 references but never mutates it.

## SEL-15 Enforce single-use frozen lockbox evaluation

**Depends on:** SEL-02, SEL-14, EVA-03, EVA-04, EVA-08.

**Parallel safety and exclusive file ownership:** Sole owner of `freeze.py::{LockboxState,claim_lockbox_once,evaluate_frozen_lockbox}` and `artifacts/runs/medreason/lockbox/`. OS-level exclusive creation/locking must make concurrent claims mutually exclusive.

**Target paths/symbols:** `medfm/challenges/medreason/freeze.py`; evaluator wiring; `tests/challenges/medreason/test_freeze.py`.

**Inputs:** Exactly one verified research manifest, immutable 15% lockbox capability, fixed evaluator/judge hashes, and bootstrap contract. Non-reference preflight happens before capability open.

**Outputs and exact artifact schema:** State: `{schema_version:int,research_manifest_sha256:sha256,lockbox_manifest_sha256:sha256,state:"ready"|"claimed"|"completed"|"failed_after_claim",claim_id:str|null,claimed_at_utc:str|null,completed_at_utc:str|null,attempt_count:0|1,result_sha256:sha256|null,sanitized_error_class:str|null}`. Result: `{schema_version:int,artifact_role:"research_evaluation",research_manifest_sha256:sha256,lockbox_manifest_sha256:sha256,point_estimates:{mcq:number,gt:number,va:number},group_paired_intervals:{mcq:[number,number],gt:[number,number],va:[number,number]},bootstrap:{seed:2026,resamples:1000},proxy_labels:{gt:true,va:true}}`.

**Ordered implementation:**
1. Verify artifacts, judges, parser, resources, and output without reading lockbox cases/references.
2. Atomically transition `ready -> claimed` before access. Claim is irrevocable; any post-open error becomes `failed_after_claim`, never ready.
3. Accept only one exact manifest; prohibit candidate arrays, fit/train callbacks, method/config changes, or comparisons.
4. Evaluate once, persist points/paired intervals without per-case references, then transition `claimed -> completed`.
5. Reject second/concurrent invocation, changed manifest, participant-validation substitution, and any use of results as selection input. Sanitize logs.

**Execution command:** `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.freeze evaluate-lockbox --research-manifest artifacts/models/medreason/research_evaluation/manifest.json --split lockbox --bootstrap-resamples 1000 --state-dir artifacts/runs/medreason/lockbox`.

**Focused tests and exact command:** Injectable clock/capability prove legal transitions, exactly one concurrent winner, no evaluator call on second use, crash-after-open consumption, non-consuming preflight failure, manifest drift, fit/train callback denial, and participant-validation denial. `uv run pytest tests/challenges/medreason/test_freeze.py::test_lockbox_claim_is_atomic_irrevocable_and_single_use tests/challenges/medreason/test_freeze.py::test_lockbox_cannot_change_system_or_accept_validation -q`.

**Acceptance evidence:** Fixture concurrency/state proof is synthetic. Protected acceptance is one `completed` or `failed_after_claim` receipt linked to real hashes; scores belong only to the research artifact and remain proxy-labeled.

**Non-goals/failure policy:** No retry after access, per-case inspection, candidate comparison, recalibration, or lockbox retraining. Post-claim infrastructure failure consumes the use.

**Handoff:** SEL-16 receives only research/result hashes and terminal status, never per-case lockbox feedback.

## SEL-16 Create distinct all-label deployment artifact

**Depends on:** SEL-12, SEL-14, SEL-15, DAT-03, DAT-05, GOV-09.

**Parallel safety and exclusive file ownership:** Sole owner of `freeze.py::{AllLabelDeploymentManifest,train_all_label_deployment,freeze_all_label_deployment}` and `artifacts/models/medreason/all_label_deployment/`; it never writes research or lockbox roots.

**Target paths/symbols:** `medfm/challenges/medreason/freeze.py`; `tests/challenges/medreason/test_freeze.py` and protected `test_freeze_hardware.py`.

**Inputs:** Already selected architecture/config; verified released labeled manifest (reported plan total 17,722 only when locally evidenced); median steps; unchanged OOF temperatures/fusion/thresholds; terminal lockbox state. The 2,532 reported participant-validation cases are exclusion-only and not assumed locally present.

**Outputs and exact artifact schema:** `manifest.json`: `{schema_version:int,artifact_id:str,artifact_role:"all_label_deployment",derived_from_research_manifest_sha256:sha256,lockbox_result_sha256:sha256|null,architecture_selection_sha256:sha256,all_label_case_ids_sha256:sha256,excluded_participant_validation_case_ids_sha256:sha256,median_steps_sha256:sha256,calibration_hashes:[sha256],hardware_preflight_sha256:sha256,routes:[{route_id:str,optimizer_steps:int,adapter_sha256:sha256,run_metadata_sha256:sha256}],files:[{path:str,sha256:sha256,size_bytes:int,kind:str}],research_lockbox_scores_attributable:false,immutable:true}` plus distinct `manifest.sha256`.

**Ordered implementation:**
1. Require terminal SEL-15 before all-label training, but consume no per-case result.
2. Verify all and only released labeled records; reject participant-validation, hidden, unapproved external, pseudo-trace, private annotation, generated preference, or missing-label sources.
3. Re-run hardware/disk preflight and train selected routes once from same initialization/config for median steps, without new selection, early stopping, or fit.
4. Reuse OOF temperatures, weights, and thresholds byte-for-byte; never fit on in-sample all-label outputs.
5. Freeze under a distinct root/role/hash; set `research_lockbox_scores_attributable=false`. Never attach research lockbox estimates to deployment.

**Execution command:** `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.freeze train-all-label --research-manifest artifacts/models/medreason/research_evaluation/manifest.json --lockbox-state artifacts/runs/medreason/lockbox/lockbox_state.json --median-steps artifacts/runs/medreason/selection/median_steps.json --train-manifest artifacts/data/medreason/derived/train_manifest.json --output artifacts/models/medreason/all_label_deployment`.

**Focused tests and exact command:** Synthetic manifests prove released-label inclusion, validation exclusion, exact steps, one call, unchanged calibration, distinct IDs/hashes, false score attribution, and failures for unlabeled/validation rows, config drift, lockbox-derived threshold, retraining, or research overwrite. `uv run pytest tests/challenges/medreason/test_freeze.py::test_all_label_artifact_is_distinct_and_reuses_oof_fits tests/challenges/medreason/test_freeze.py::test_all_label_training_rejects_validation_and_lockbox_feedback -q`. Protected command, not run by default: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -m "gpu and real_checkpoint" tests/challenges/medreason/test_freeze_hardware.py::test_all_label_profile_preflight -q`.

**Acceptance evidence:** Fixture acceptance proves schema/provenance/state only. Protected acceptance requires verified released labels, real completed runs, storage/model access, and measured hardware; absent evidence forbids claims of row count, 48/96 GB support, official eligibility, or hidden performance. Packaging inventory must pass before handoff.

**Non-goals/failure policy:** This is not the lockbox-scored research artifact and cannot inherit its scores. Never train on participant validation, infer labels, refit in-sample, add components, or retry from hidden/lockbox outcomes.

**Handoff:** Docker/release consumes the all-label deployment manifest; research reports consume SEL-14 plus SEL-15. Publish both hashes side by side with non-interchangeable roles.
