# Grouping splits and stress

This phase turns the audited released training package into leakage-aware groups, a frozen development split, diagnostics, and deterministic robustness views. It extends rather than replaces `medfm/data/splits.py`: the existing `generate_split_assignment`, `check_split_leakage`, and `assert_no_split_leakage` behavior remains intact, and the official-eligible MedReason path never uses `ResearchOverride`. Unless a card says otherwise, canonical JSON uses `medfm.core.serialization.canonical_json`, artifact hashes use `config_hash`, the fixed seed is `2026`, unit fixtures are CPU-only with CUDA hidden, and protected released-data checks require explicit `MEDFM_RUN_MEDREASON_DATA=1` opt-in. The normalized released examples remain the source of labels and metadata; participant validation, hidden data, paths, filenames, and model-inferred metadata never supply grouping or stress truth.

Shared derived artifacts are under `artifacts/data/medreason/derived/`: `question_fingerprints.jsonl`, `grouping.jsonl`, `grouping_report.json`, `splits.json`, `split_leakage_report.json`, `diagnostics.json`, `text_baseline/`, `stress_manifest.jsonl`, and `stress_report.json`. Every JSON artifact records its schema and algorithm version, seed (including when no RNG is used), ordered input-manifest hashes, configuration hash, and payload hash. Raw questions, answers, identifiers, and paths must not enter summary logs or exception text.

## SPL-01 Fingerprint normalized question and option tuples

- **Depends on:** SCH-02, SCH-06, DAT-03.
- **Parallel safety and exclusive file ownership:** May run alongside SPL-05 design work, DAT-08, and DAT-10 after the normalized schema is frozen. The implementer exclusively owns new `medfm/challenges/medreason/fingerprints.py` and `tests/challenges/medreason/test_fingerprints.py`; do not edit `medfm/data/splits.py`, grouping code, or package exports owned by SCH-09.
- **Target paths/symbols:** `medfm/challenges/medreason/fingerprints.py`: `QuestionOptionFingerprint`, `normalize_fingerprint_text`, and `fingerprint_question_options`; `tests/challenges/medreason/test_fingerprints.py`.
- **Inputs:** `MedReasonExample.question` and the ordered option objects defined by SCH-02. Use `normalize_unicode(..., form="NFKC")` on fingerprint-only copies, Unicode `casefold`, and deterministic horizontal-whitespace collapse; never mutate the preserved NFC source strings. Configuration is `fingerprint_version="medreason-question-options-v1"`, `seed=2026` (recorded but no RNG is used).
- **Outputs:** One `question_fingerprints.jsonl` row per case with `case_id`, normalized-question SHA-256, ordered full-tuple SHA-256, option count, algorithm version, and configuration hash. The full tuple is canonical JSON `{"question": normalized_question, "options": [{"label": normalized_label, "text": normalized_text}, ...]}` in original option order; labels and texts are length-delimited by JSON rather than concatenated.
- **Implementation:**
  1. Validate that the question is non-empty and that MCQ options satisfy SCH-02; an open case uses an empty option list, not a fabricated option.
  2. Normalize only the fingerprint copies with the fixed policy. Preserve punctuation and option order so reordered choices do not collide.
  3. Hash the normalized question and complete ordered tuple with SHA-256 over UTF-8 canonical JSON. Do not use Python `hash()`.
  4. Sort output rows by `case_id`, reject duplicate IDs, and hash the complete canonical row sequence plus the ordered normalized-input manifest hashes.
- **Focused Tests:** Synthetic Unicode-equivalent strings must collide after normalization; different punctuation, option text, labels, or order must not. Test delimiter ambiguity, open cases, duplicates, and row-order-independent output. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_fingerprints.py`.
- **Acceptance evidence:** Fixture acceptance is the passing CPU command and a golden canonical tuple/hash whose value is stable after input row shuffling. Protected-package acceptance is a gated run proving one fingerprint row per released case and no raw text in logs; it is not available without authorized data. No GPU or 48/96 GB evidence is required or claimed.
- **Non-goals/failure policy:** This is not semantic similarity, template detection, or an option permutation. Fail closed on malformed options. An exact tuple collision is grouping evidence, not patient identity and never supports a patient-disjoint claim.
- **Handoff:** SPL-04 and SPL-05 consume the normalization version and hashes; SPL-08 consumes exact tuple hashes. Downstream manifests pin the fingerprint configuration and payload hashes.

## SPL-02 Union groups by authoritative source identifiers

- **Depends on:** SCH-02, DAT-10, SPL-01.
- **Parallel safety and exclusive file ownership:** May run alongside SPL-05 and the stress-transform tasks. It starts the shared `medfm/challenges/medreason/grouping.py` and `tests/challenges/medreason/test_grouping.py`; SPL-03 and SPL-04 must follow serially and re-read these files before editing.
- **Target paths/symbols:** `medfm/challenges/medreason/grouping.py`: `GroupingEvidence`, `TransitiveGroupBuilder`, `union_authoritative_identifiers`, and `finalize_groups`; `tests/challenges/medreason/test_grouping.py`.
- **Inputs:** Normalized examples and DAT-10's normalized, hashed source/study/patient/article identifiers. Only non-empty released identifier values are accepted. The typed evidence key is `(identifier_kind, normalized_identifier_hash)` so equal strings of different kinds never collide. Version is `medreason-group-union-v1`; seed `2026` is recorded, but unioning uses no RNG.
- **Outputs:** Deterministically sorted evidence edges and preliminary `grouping.jsonl` rows with `case_id`, `group_id_hash`, sorted `group_basis` evidence kinds, and `has_authoritative_identifier`. No raw identifier is emitted.
- **Implementation:**
  1. Create one disjoint-set node per case and reject duplicate case IDs.
  2. Iterate identifier kinds in fixed order `patient`, `study`, `article`, `source`, and values lexicographically. Union all case nodes sharing the same typed value. Null, blank, invalid, or unavailable values add no edge.
  3. Preserve full transitivity: if A shares a study with B and B an article with C, A/B/C become one component regardless of row order.
  4. Derive `group_id_hash` as SHA-256 of canonical JSON containing the version and sorted member case IDs; select roots only by deterministic lexical rank, never union-find insertion order.
  5. Emit evidence counts and configuration/payload hashes without exposing raw IDs.
- **Focused Tests:** Cover chains, cycles, typed-key collision isolation, missing identifiers, duplicate cases, shuffled rows/edges, and the case where a patient link bridges two source groups. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_grouping.py -k 'identifier or transitive or authoritative'`.
- **Acceptance evidence:** Fixture acceptance includes identical group hashes for permuted inputs and one component for a three-hop chain. Protected acceptance requires only completeness/count checks under the gated audit, not a claim that every released record has a patient identifier. No accelerator is used.
- **Non-goals/failure policy:** Never derive identifiers from filenames, paths, question text, or reverse source-article search. `source` means a released normalized identifier, not a generic modality/dataset label. Missing patient metadata must remain missing; these groups are leakage groups, not asserted patient groups.
- **Handoff:** SPL-03/SPL-04 extend the same disjoint-set evidence graph. SPL-06 consumes `has_authoritative_identifier` and basis kinds; SPL-07 consumes finalized `group_id_hash`.

## SPL-03 Union exact and near-duplicate image groups

- **Depends on:** DAT-09, SPL-02.
- **Parallel safety and exclusive file ownership:** May run alongside SPL-05, SPL-09, and STR-01, but is serialized with SPL-02/SPL-04 because it edits `grouping.py` and `test_grouping.py`. Those files are exclusive during this task.
- **Target paths/symbols:** `medfm/challenges/medreason/grouping.py`: `NearDuplicatePolicy` and `union_image_duplicates`; `tests/challenges/medreason/test_grouping.py`.
- **Inputs:** DAT-09 decoded-pixel SHA-256 and perceptual hashes, including their algorithm/version and bit width. The initial frozen policy accepts exact decoded-pixel equality and, within matching perceptual-hash algorithm/version/width partitions, Hamming distance `<=4` for 64-bit hashes. Version `medreason-image-union-v1`, seed `2026` (no RNG), and threshold are hashed.
- **Outputs:** Added typed edges `image_exact` or `image_near`, near-edge Hamming distances, recomputed transitive group hashes, and counts by evidence type in `grouping.jsonl`. A case with multiple images joins on any qualifying image edge while preserving its original image order elsewhere.
- **Implementation:**
  1. Union all cases sharing a non-null decoded-pixel SHA-256, even if filenames or encoded bytes differ.
  2. Partition perceptual hashes by declared algorithm/version/width; reject malformed hashes and never compare incompatible partitions.
  3. Enumerate pairs deterministically with a BK-tree or equivalent exact Hamming-radius search, sort candidate pairs by case/image identity, and union every pair at distance at most four. Approximate nearest-neighbor randomness is prohibited.
  4. Feed image edges into the existing transitive builder, then regenerate component hashes from sorted case membership.
  5. Persist policy/configuration and edge-set hashes so a threshold change invalidates every downstream split.
- **Focused Tests:** Verify encoded-byte differences with identical decoded pixels union, Hamming distances 4/5 fall on opposite sides, incompatible hash versions never compare, multiple-image transitivity works, and shuffled pair enumeration is stable. Include malformed hash failure. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_grouping.py -k 'image or hamming or near_duplicate'`.
- **Acceptance evidence:** CPU fixtures prove exact threshold and transitivity. Protected data can later report counts under the gated audit, but fixture success does not establish real near-duplicate recall or patient disjointness. No GPU/hardware claim is made.
- **Non-goals/failure policy:** Do not use filenames, file stems, path proximity, or an unversioned visual embedding. Do not silently relax the threshold to improve balance. Any unknown perceptual-hash contract fails the near-duplicate stage while exact decoded-pixel grouping remains available and the failure is explicit.
- **Handoff:** SPL-04 uses finalized image components as one permissible similarity gate. SPL-07/SPL-08 pin the image policy and edge-set hashes.

## SPL-04 Gate high-similarity question grouping with evidence

- **Depends on:** SPL-01, SPL-02, SPL-03.
- **Parallel safety and exclusive file ownership:** May run alongside SPL-05 and STR-02/STR-03. It is the final serialized editor of `grouping.py` and `test_grouping.py`; SPL-06 and SPL-07 consume its frozen output rather than editing those files.
- **Target paths/symbols:** `medfm/challenges/medreason/grouping.py`: `QuestionSimilarityPolicy`, `question_shingles`, and `union_supported_question_similarity`; `tests/challenges/medreason/test_grouping.py`.
- **Inputs:** SPL-01 normalized questions, SPL-02 released source identifier edges, and SPL-03 exact/near image components. Policy `medreason-question-similarity-v1` uses word-token 3-shingle Jaccard `>=0.90`; questions shorter than three tokens qualify only by exact normalized equality. Seed `2026` is recorded; exhaustive deterministic candidate generation uses no RNG.
- **Outputs:** Accepted `question_similar_supported` edges with similarity and support kind, rejected-candidate diagnostic counts, finalized transitive groups, and hashes of policy, accepted edges, and grouping payload.
- **Implementation:**
  1. Tokenize normalized question copies on Unicode alphanumeric runs, preserving token sequence; construct a set of consecutive three-token shingles.
  2. Generate exact candidates from an inverted shingle index, then compute exact Jaccard; sort pairs lexicographically. Do not use randomized MinHash.
  3. Require both similarity threshold and independent released evidence: the cases already share an exact/near image component or a typed authoritative source identifier. Question similarity alone never unions cases.
  4. Add accepted edges to the transitive builder and recompute membership-derived group hashes. Record which gate admitted each edge.
  5. Hash policy and accepted-edge set; report threshold-near rejected counts without raw text.
- **Focused Tests:** Cover similarity above threshold with and without supporting evidence, below-threshold pairs sharing an image, short questions, punctuation/Unicode normalization, and a bridge that proves transitive closure. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_grouping.py -k 'question_similarity or similarity_gate'`.
- **Acceptance evidence:** Fixtures must show that a generic near-identical prompt with no image/source support remains in a different group. Protected audit may quantify accepted edges, but cannot validate patient identity or hidden-data overlap. CPU only.
- **Non-goals/failure policy:** No language model, embedding service, source-article reverse matching, participant-validation lookup, or inferred source label. Threshold changes require a new version and complete split invalidation; do not tune it against dev or lockbox scores.
- **Handoff:** SPL-06 receives finalized group evidence; SPL-07 consumes immutable group hashes; SPL-08 audits every accepted edge across assigned splits.

## SPL-05 Report template reuse without collapsing groups

- **Depends on:** SPL-01.
- **Parallel safety and exclusive file ownership:** Safe in parallel with SPL-02 through SPL-04 and stress work. Exclusive new files are `medfm/challenges/medreason/templates.py` and `tests/challenges/medreason/test_templates.py`; it must not call the group builder or change `group_id_hash`.
- **Target paths/symbols:** `medfm/challenges/medreason/templates.py`: `template_fingerprint` and `build_template_reuse_report`; `tests/challenges/medreason/test_templates.py`.
- **Inputs:** SPL-01 normalized question copies only. Policy `medreason-template-report-v1` deterministically replaces Unicode numeric runs with `<NUM>`, option-label-only tokens with `<OPTION>`, and runs of horizontal whitespace with one space; no RNG, seed `2026` recorded.
- **Outputs:** `grouping_report.json` template section containing template SHA-256, count, task-type counts, and capped case-ID examples; configuration and payload hashes. Raw template text is excluded.
- **Implementation:**
  1. Apply the fixed substitutions to fingerprint-only question copies without touching stored questions or options.
  2. Hash canonical normalized templates and aggregate counts in sorted hash order.
  3. Report reuse frequency and concentration separately by task type; case examples are capped and sorted.
  4. Keep this output diagnostically separate from all grouping edges and assert that input/output `group_id_hash` values are unchanged.
- **Focused Tests:** Numeric variants should share a template; clinically different words should not; repeated generic prompts must increase a report count without joining cases. Verify deterministic hashes, privacy-safe serialization, and unchanged group IDs. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_templates.py`.
- **Acceptance evidence:** CPU fixtures prove report-only behavior and a stable golden hash. Protected acceptance is frequency coverage only, gated on authorized data. It is not leakage-free or patient-disjoint evidence.
- **Non-goals/failure policy:** Do not strip arbitrary medical entities, learn templates, or group by template. Template frequency is a shortcut diagnostic, not a semantic label or private-shift signal.
- **Handoff:** SPL-09 and SPL-10 can cite the template-report hash when interpreting shortcut diagnostics; no downstream splitter consumes template IDs.

## SPL-06 Report cases lacking defensible grouping signals

- **Depends on:** SPL-02, SPL-03, SPL-04, SPL-05.
- **Parallel safety and exclusive file ownership:** May run alongside diagnostics and stress transforms after grouping freezes. Exclusive files are `medfm/challenges/medreason/grouping_report.py` and `tests/challenges/medreason/test_grouping_report.py`; do not edit grouping algorithms.
- **Target paths/symbols:** `medfm/challenges/medreason/grouping_report.py`: `GroupingCoverageReport`, `build_grouping_coverage_report`, and `patient_disjoint_status`; `tests/challenges/medreason/test_grouping_report.py`.
- **Inputs:** Final grouping rows and evidence edges, plus released-case manifest hash. Version `medreason-grouping-coverage-v1`, seed `2026` (no RNG).
- **Outputs:** `grouping_report.json` with total cases/groups, component-size histogram, evidence-kind coverage, sorted/capped `cases_without_defensible_signal`, `singleton_fallback_count`, `patient_identifier_coverage`, and `patient_disjoint_status` equal to either `verified_from_complete_released_patient_identifiers` or `not_verifiable_from_released_metadata`.
- **Implementation:**
  1. Define defensible signal as at least one authoritative released identifier, exact/near image link, exact full question-option link to another case, or supported high-similarity edge. Template reuse and filename stems never qualify.
  2. Retain signal-less cases as deterministic singleton components for assignment, but mark `singleton_fallback=true`; never synthesize a patient ID.
  3. Set verified patient status only if every released case has a valid released patient identifier and no patient hash crosses components/splits. Otherwise emit `not_verifiable_from_released_metadata`, regardless of group-disjointness.
  4. Reconcile all counts to the normalized-case manifest and hash the canonical report.
- **Focused Tests:** Cover no-signal singleton cases, template-only cases, partially missing patient metadata, full patient coverage, count reconciliation, stable row ordering, and a deliberately conflicting patient edge. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_grouping_report.py`.
- **Acceptance evidence:** Fixture acceptance proves honest status transitions and count invariants. Protected acceptance may establish the actual released metadata coverage only when the gated audit runs; until then, `not_verifiable_from_released_metadata` is required. No GPU evidence applies.
- **Non-goals/failure policy:** Group-disjoint across observed evidence is not synonymous with patient-disjoint. Do not suppress, impute, or infer missing identifiers to improve the status. A count mismatch fails the report and blocks splitting.
- **Handoff:** SPL-07 records this report hash in `splits.json`; every training/evaluation report displays its patient-disjoint status rather than constructing a stronger claim.

## SPL-07 Generate deterministic stratified group-disjoint splits

- **Depends on:** SPL-01, SPL-04, SPL-06, SCH-05.
- **Parallel safety and exclusive file ownership:** This is the only card in this phase allowed to edit existing `medfm/data/splits.py` and `tests/phase_03/test_splits.py`. It also exclusively owns new `medfm/challenges/medreason/splits.py` and `tests/challenges/medreason/test_splits.py`. Do not overlap it with other generic split work; SPL-08/SPL-09 wait for its artifact.
- **Target paths/symbols:** Add `generate_stratified_group_assignment` and `StratifiedSplitReport` to `medfm/data/splits.py` without altering `generate_split_assignment`, `check_split_leakage`, `assert_no_split_leakage`, defaults, temporal exemptions, or `ResearchOverride`. Add `build_medreason_splits` in challenge `splits.py`.
- **Inputs:** Final non-null `group_id_hash` for every released case; ratios `TRAIN=0.70`, `VAL=0.15`, `TEST=0.15`; seed `2026`; task type plus only supplied released modality/body-system/reasoning tags. Missing optional tags become explicit `__MISSING__` strata; no tag is inferred. Input/grouping/config hashes are mandatory.
- **Outputs:** `splits.json` maps framework `TRAIN/VAL/TEST` to challenge names `train/dev/lockbox`, records every case/group assignment, target and observed case/group/stratum counts, indivisible-group deviations, grouping coverage status/hash, algorithm version `medreason-stratified-groups-v1`, seed, and report/payload hashes.
- **Implementation:**
  1. Add a new generic atomic-group API; leave the patient-first API and all existing leakage checks byte-semantically unchanged. Reject null group IDs, duplicate case IDs, invalid ratios, and any group whose rows are preassigned inconsistently. Do not route MedReason through `generate_split_assignment`, whose patient-key hash bucketing is not transitive stratification and whose role names are not the challenge contract.
  2. Build each group's count vector over task type and every available tag dimension/value. Compute per-stratum target counts from normalized ratios.
  3. Order groups by descending maximum inverse-frequency-weighted stratum contribution, then descending group size, then SHA-256 of `"medreason-stratified-groups-v1:2026:<group_id_hash>"`.
  4. For each atomic group, choose the split minimizing the post-assignment sum of squared normalized deficits across stratum counts plus normalized total-case deficit; compare exact rational/integer cross-products where possible. Break score ties with SHA-256 of `(seed, group_id_hash, split_name)`. Never split a group to hit a ratio.
  5. Emit deviation diagnostics instead of promising exact 70/15/15 counts. Re-running or shuffling input must reproduce assignments.
  6. Run both the MedReason edge audit and existing `assert_no_split_leakage(..., temporal_policy=False)` on a compatible frame before writing. The official path forbids `ResearchOverride`.
- **Focused Tests:** Preserve all existing phase-03 tests, then test atomic multi-case groups, multi-label stratification, missing optional tags, large-group ratio deviation, deterministic row-order/seed behavior, split-name mapping, and cross-split patient/image leakage refusal. Exact commands: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/phase_03/test_splits.py` and `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_splits.py`.
- **Acceptance evidence:** CPU fixtures prove old-policy compatibility and new deterministic group stratification. Protected acceptance requires the gated real manifest, observed distribution table, and clean leakage gate; fixture balance is not evidence of released-data balance or patient disjointness. No accelerator claim.
- **Non-goals/failure policy:** Do not stratify on answers, answer positions, participant validation, generated labels, or lockbox outcomes. Do not weaken temporal/patient leakage policy, silently split oversized groups, or describe `TEST` as repeatedly tunable; it is the one-use lockbox.
- **Handoff:** Training consumes `train`; iteration consumes `dev`; SEL-15 alone may open `lockbox`. All downstream artifacts pin split report/payload hashes and seed.

## SPL-08 Audit image question and source overlap

- **Depends on:** SPL-07.
- **Parallel safety and exclusive file ownership:** May run with SPL-09/SPL-10 and stress implementation after `splits.json` freezes. Exclusive files are `medfm/challenges/medreason/leakage.py` and `tests/challenges/medreason/test_leakage.py`; use, do not edit, generic split functions.
- **Target paths/symbols:** `medfm/challenges/medreason/leakage.py`: `MedReasonLeakageReport`, `audit_medreason_split_overlap`, and `assert_medreason_split_clean`; `tests/challenges/medreason/test_leakage.py`.
- **Inputs:** Split assignments; every authoritative-identifier, exact/near-image, exact-tuple, and supported-question edge; decoded-pixel hashes; grouping and split hashes. Version `medreason-overlap-audit-v1`, seed `2026` (no RNG).
- **Outputs:** `split_leakage_report.json` with per-evidence overlap counts, capped hashed/case-ID examples, rows checked, `ok`, generic `LeakageReport.to_dict()`, patient metadata coverage/status, config hash, and payload hash.
- **Implementation:**
  1. Verify every recorded evidence edge has endpoints in one split and every transitive component maps to exactly one split.
  2. Independently scan exact decoded-pixel and question-option hashes, near-image edges, high-similarity supported edges, and typed source/study/patient/article hashes for cross-split occurrences.
  3. Materialize the compatible columns required by `check_split_leakage`; call `assert_no_split_leakage` with no override. Preserve its existing patient/study/series/group/image checks.
  4. Fail before training on any known cross-split overlap, unknown case, missing assignment, grouping hash mismatch, or audit count mismatch.
  5. Report patient disjointness only under SPL-06's complete-metadata rule; otherwise report group disjointness across observed evidence and `not_verifiable` patient status.
- **Focused Tests:** Inject one violation of each edge/hash kind; test transitive bridge leakage, unknown IDs, capped privacy-safe output, clean fixtures, and a missing-patient clean grouping that remains `not_verifiable`. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_leakage.py`.
- **Acceptance evidence:** Fixtures prove detection coverage, not real-data cleanliness. Protected acceptance is a gated audit whose report has `ok=true`, reconciled counts, and matching input hashes; only that artifact supports claims about the released split. No hardware evidence.
- **Non-goals/failure policy:** No `ResearchOverride` in official-eligible execution. Do not inspect participant validation, hidden cases, filenames, or source articles. Never convert absence of a detected overlap into an unsupported patient-disjoint assertion.
- **Handoff:** Every trainer preflight consumes `ok`, report hash, split hash, and patient status; GOV-10/CLI-08 fail closed when they are absent or mismatched.

## SPL-09 Measure answer-position and option-length bias

- **Depends on:** SPL-07, SPL-08.
- **Parallel safety and exclusive file ownership:** May run with stress tasks. It starts shared `medfm/challenges/medreason/diagnostics.py` and `tests/challenges/medreason/test_diagnostics.py`; SPL-10 follows serially and re-reads them.
- **Target paths/symbols:** `medfm/challenges/medreason/diagnostics.py`: `answer_position_report` and `option_length_bias_report`; `tests/challenges/medreason/test_diagnostics.py`.
- **Inputs:** Labeled MCQ examples from `train` and `dev` only, original ordered options, split/leakage hashes. Policy `medreason-mcq-bias-v1`, seed `2026` (no RNG). Lockbox and participant validation are rejected by default.
- **Outputs:** `diagnostics.json` sections for correct-position counts/frequencies, label counts, option-count strata, correct/incorrect option Unicode-code-point and whitespace-token lengths, longest/shortest-option heuristic accuracy with deterministic first-index tie policy, and manifest/config/payload hashes.
- **Implementation:**
  1. Resolve the released correct label to exactly one original option index; fail on missing/non-unique mappings.
  2. Aggregate counts separately for train and dev and by option count. Denominators and missing/unsupported-case counts are explicit.
  3. Measure length on the same normalized fingerprint copy using both code points and whitespace-delimited tokens; do not tokenize with a model checkpoint.
  4. Evaluate fixed longest/shortest heuristics without fitting, with ties resolved by lowest original index and tie counts reported.
  5. Serialize aggregate statistics only; keep case-level diagnostic rows in ignored artifacts if needed and never log question/answer text.
- **Focused Tests:** Use uneven label positions, variable option counts, equal-length ties, Unicode, and malformed correct labels. Verify denominator arithmetic, split rejection, deterministic hash, and that changing option order changes position statistics. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_diagnostics.py -k 'position or option_length or heuristic'`.
- **Acceptance evidence:** CPU fixtures prove formulas. Protected gated output can later describe released train/dev diagnostics, but no fixture or aggregate predicts official/private performance. No GPU is involved.
- **Non-goals/failure policy:** Do not use lockbox labels before SEL-15, infer answers for participant validation, or turn diagnostics into training labels. High heuristic accuracy is a leakage warning, not a system result.
- **Handoff:** SPL-10 includes these fixed heuristics as controls; MCQ-02 consumes the finding only as motivation for permutation, not as a label source.

## SPL-10 Run train-only text baseline diagnostic

- **Depends on:** SPL-09.
- **Parallel safety and exclusive file ownership:** Serialized after SPL-09 because it finishes `diagnostics.py` and `test_diagnostics.py`. It may run alongside all stress tasks and must not touch evaluator, model, or split code.
- **Target paths/symbols:** `medfm/challenges/medreason/diagnostics.py`: `TextBaselineConfig`, `run_text_only_baseline`, and `score_option_candidates`; `tests/challenges/medreason/test_diagnostics.py`.
- **Inputs:** Question and original option text from leakage-clean `train`, evaluation on `dev`, never pixels. Fixed configuration: candidate text `question + " [OPTION] " + option_text`; word TF-IDF 1-2 grams; binary `sklearn.linear_model.LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=2026)`; train each option candidate as correct/incorrect. Seed/config/split/input hashes are mandatory.
- **Outputs:** `text_baseline/config.json`, fitted diagnostic artifact, per-dev-case option scores/prediction in ignored storage, and aggregate dev MCQ accuracy in `diagnostics.json`; every file has a SHA-256. Predictions preserve case ID but no raw text.
- **Implementation:**
  1. Expand only train cases to option-candidate rows and fit TF-IDF vocabulary/statistics and classifier on train; reject any dev/lockbox ID in fit inputs.
  2. Score every original dev option, choose maximum positive-class decision score, and break exact ties by lowest original index.
  3. Compare against labels only after scoring; report accuracy, failures, and SPL-09 fixed heuristic controls. Never use dev to fit vocabulary, hyperparameters, thresholds, or retry behavior.
  4. Verify train/dev group sets are disjoint and pin the SPL-08 leakage-report hash before fit.
  5. Fail clearly if fixtures contain only one binary class or no evaluable MCQ cases; do not fabricate a score.
- **Focused Tests:** Assert vocabulary excludes a dev-only token, train IDs alone reach `.fit`, group overlap blocks execution, same seed is byte-stable, tie behavior is stable, and image mutations cannot change predictions. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_diagnostics.py -k 'text_baseline'`.
- **Acceptance evidence:** Fixture acceptance proves isolation and deterministic execution. Protected acceptance requires an authorized train/dev run with artifact hashes; it is a diagnostic local baseline, not official MCQ accuracy and not a winning/private-shift claim. CPU only; no real checkpoint or GPU gate.
- **Non-goals/failure policy:** No image, foundation model, pseudo-label, participant-validation output, lockbox label, or hyperparameter search. Do not promote or train the submission model from this artifact.
- **Handoff:** EVA-12 may display the baseline as a separately labeled diagnostic; selection tasks must not treat it as an official-compatible model candidate.

## STR-01 Select stress parameters from stable case hashes

- **Depends on:** SCH-02, DAT-03.
- **Parallel safety and exclusive file ownership:** Safe in parallel with grouping and split work. Exclusive files are `medfm/challenges/medreason/stress_params.py` and `tests/challenges/medreason/test_stress_params.py`; STR-02/STR-03 consume this API without editing it.
- **Target paths/symbols:** `stress_params.py`: `STRESS_SEED`, `StressSpec`, `select_stress_spec`, `stress_variant_id`; test mirror.
- **Inputs:** Canonical released `case_id`, supported stress stratum, base seed `2026`, and version `medreason-stress-params-v1`. X-ray Cartesian tuples are photon count `{500,1000,2000,5000}` × gamma `{0.75,1.25}` with Gaussian sigma `0.01` and blur sigma `0.8`. MRI tuples are Rician sigma `{0.02,0.05}` × gamma `{0.75,1.25}` × Gibbs strength `{0.2,0.4}` with bias coefficient `0.3`.
- **Outputs:** One immutable `StressSpec` for an eligible case, containing stratum, complete parameter tuple, CPU RNG seed, selector digest, version/config hash, and `variant_id`; no image or label is changed here.
- **Implementation:**
  1. Compute SHA-256 of canonical JSON `{"version":...,"seed":2026,"case_id":...,"stratum":...}`; reject blank/noncanonical IDs and unsupported strata.
  2. Interpret the first eight digest bytes as an unsigned big-endian integer modulo the lexicographically enumerated Cartesian product length. Use the next eight bytes, masked to 63 bits, as the CPU generator seed.
  3. Derive `variant_id` from the complete canonical spec hash. Never use Python hash, global RNG, row index, process ID, epoch, or worker ID.
  4. Persist the full lookup-table order and configuration hash in `stress_manifest.jsonl`; always retain the clean case separately.
- **Focused Tests:** Golden cases must map to fixed tuples/seeds; input row order and process restart must not matter; different case IDs should exercise tuple variation; invalid strata/IDs fail. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_params.py`.
- **Acceptance evidence:** CPU golden digests and complete Cartesian-bound checks establish deterministic selection. Protected data only adds coverage counts; it cannot establish resemblance to the private split. No GPU/hardware claim.
- **Non-goals/failure policy:** These tuples are approved synthetic robustness proxies, not estimates of real dose, 7T acquisition, or organizer private cases. Do not tune selection hashes or parameters on dev/lockbox outcomes.
- **Handoff:** STR-02/STR-03 receive the exact spec and RNG seed; STR-05 pins the selector/config hash; STR-06 validates one spec per eligible case.

## STR-02 Implement deterministic low-dose X-ray stress transforms

- **Depends on:** DAT-08, STR-01.
- **Parallel safety and exclusive file ownership:** Safe in parallel with STR-03 and grouping work. Exclusive files are `medfm/challenges/medreason/stress_xray.py` and `tests/challenges/medreason/test_stress_xray.py`; it must not edit generic radiology transforms or modality gating.
- **Target paths/symbols:** `stress_xray.py`: `XRayStressRecord` and `apply_xray_stress`; test mirror. Reuse seeded-generator conventions from `medfm/data/transforms/base.py` but keep challenge stress isolated.
- **Inputs:** A decoded finite 2D X-ray tensor copy and an X-ray `StressSpec`. Algorithm version `medreason-xray-stress-v1`; randomness comes only from the recorded CPU generator seed.
- **Outputs:** Float32 stressed image with unchanged shape/channel order, transform record containing photon count, Gaussian/gamma/blur parameters, RNG seed, input/output pixel SHA-256, algorithm/config hash, and paired case/variant IDs. The clean image remains unchanged.
- **Implementation:**
  1. Convert a copy to CPU float32 and deterministically min-max normalize finite intensities to `[0,1]`; reject empty, non-finite, constant, or non-2D spatial inputs rather than emit a misleading view.
  2. In fixed order, draw Poisson counts from `x * photon_count` and divide by the count, add Gaussian noise with sigma `0.01`, clamp, apply `x ** gamma`, apply separable Gaussian blur sigma `0.8` with explicit fixed boundary policy, then clamp to `[0,1]`.
  3. Draw all random values from one local CPU `torch.Generator` seeded by STR-01; forbid global RNG and device-dependent CUDA kernels.
  4. Preserve dimensions and channel order; do not crop, resize, rotate, translate, or flip. Record operation order and hashes.
- **Focused Tests:** Same case/spec must be bit-identical in the pinned environment; different specs differ; shape/order stay fixed; clean input is not mutated; a coordinate phantom is never flipped; non-finite/constant/wrong-rank inputs fail; global RNG state is unchanged. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_xray.py`.
- **Acceptance evidence:** Synthetic CPU fixtures prove implementation and non-geometric operation history. A gated released-image smoke check is needed to prove artifact compatibility, not clinical low-dose fidelity. No GPU, private-shift, or hardware-support claim.
- **Non-goals/failure policy:** This does not estimate exposure, simulate a scanner, reproduce private low-dose X-rays, or augment training by default. Transform errors keep the clean case and mark the stress report invalid; they do not make the stratum “unavailable.”
- **Handoff:** STR-04 invokes this transform only after metadata gating; STR-06 validates hashes, unchanged labels, geometry, and pairing.

## STR-03 Implement deterministic brain-MRI stress transforms

- **Depends on:** DAT-08, STR-01.
- **Parallel safety and exclusive file ownership:** Safe in parallel with STR-02 and grouping work. Exclusive files are `medfm/challenges/medreason/stress_mri.py` and `tests/challenges/medreason/test_stress_mri.py`; do not alter generic MRI preprocessing or claim volumetric support.
- **Target paths/symbols:** `stress_mri.py`: `BrainMRIStressRecord`, `percentile_normalize_2d`, and `apply_brain_mri_stress`; test mirror.
- **Inputs:** A decoded finite 2D brain-MRI image copy and MRI `StressSpec`. Algorithm version `medreason-brain-mri-stress-v1`; fixed 0.5/99.5 percentiles, bias coefficient `0.3`, and the STR-01 RNG seed.
- **Outputs:** Float32 stressed image with unchanged dimensions/channel order; record of percentile bounds, parameters, RNG seed, operation order, input/output pixel hashes, paired IDs, and config hash. Original retained unchanged.
- **Implementation:**
  1. On CPU float32, compute deterministic linear-interpolation 0.5th/99.5th percentiles over finite pixels per image plane, clip, and min-max normalize. Reject non-finite/constant/unsupported-rank inputs.
  2. Add Rician noise as `sqrt((x+n1)^2+n2^2)` with independent seeded zero-mean Gaussian fields at selected sigma, then clamp.
  3. Multiply by a fixed smooth centered quadratic field normalized to `[-1,1]`, `1 + 0.3*field`; clamp and apply the selected gamma.
  4. Apply Gibbs proxy deterministically in 2D Fourier space by retaining the centered rectangular fraction `1-strength` on both spatial-frequency axes, zeroing the remainder, inverse transforming the real component, and clamping. Record exact FFT normalization and odd/even index rules.
  5. Use no geometric operation and no CUDA kernel. Preserve each released 2D image independently; never combine slices or claim volume reasoning.
- **Focused Tests:** Golden percentile/odd-even FFT fixtures, fixed-seed equality, Rician parameter variation, unchanged shape/order/input, no global RNG mutation, and explicit failures for constant/non-finite/3D-volume inputs. A coordinate phantom must preserve axes and operation history must contain no spatial transform. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_mri.py`.
- **Acceptance evidence:** CPU fixtures prove exact numerical contract and absence of configured geometric operations. A gated released-image smoke check proves compatibility only; it cannot prove 7T, scanner, pathology, or private-shift fidelity. No GPU claim.
- **Non-goals/failure policy:** This is not a 7T acquisition simulator, bias correction, volumetric transform, or training augmentation by default. Do not infer MRI sequence/field strength. Eligible-case transform failure blocks weighted reporting rather than being silently dropped.
- **Handoff:** STR-04 gates invocation; STR-06 validates pair identity, hashes, labels, and structural geometry.

## STR-04 Gate stress application on released modality metadata

- **Depends on:** DAT-03, DAT-10, STR-02, STR-03.
- **Parallel safety and exclusive file ownership:** Runs after both transform APIs freeze and may run alongside diagnostics. Exclusive files are `medfm/challenges/medreason/stress.py` and `tests/challenges/medreason/test_stress.py`; it consumes but does not edit X-ray/MRI modules.
- **Target paths/symbols:** `stress.py`: `StressEligibility`, `classify_released_stress_stratum`, and `build_stress_views`; test mirror.
- **Inputs:** Only normalized modality/body-region fields explicitly supplied by released metadata, through the DAT-03/DAT-10 adapter. Frozen alias table/version `medreason-stress-modality-gate-v1`, seed `2026`, and transform config hashes. Paths, pixels, filenames, questions, and classifiers are forbidden gate inputs.
- **Outputs:** `stress_manifest.jsonl` has one clean row for every selected case and at most one stress row per eligible case, with `eligibility` (`xray`, `brain_mri`, or `unavailable`), source metadata-field names (not values), reason code, paired case/variant ID, spec/transform hashes, and payload hash.
- **Implementation:**
  1. Normalize only released metadata values through a frozen casefolded alias map. X-ray requires an explicit radiography/X-ray modality value. Brain MRI requires explicit MRI modality plus explicit brain body region, or one explicit released value already normalized as brain MRI.
  2. Treat missing, ambiguous, conflicting, generic MRI without brain region, or inferred values as unavailable. Never fall back to pixel/path/text classification.
  3. Always emit and retain the clean view. For eligible cases select exactly one STR-01 tuple and call exactly one matching transform.
  4. Emit reason/count diagnostics for unavailable cases and hash the alias table, inputs, and manifest. Changing metadata normalization invalidates the manifest.
- **Focused Tests:** Explicit X-ray and brain-MRI metadata route correctly; MRI without brain, path-only “xray”, image-looking pixels, conflicts, missing fields, and unsupported modalities remain clean-only. Verify one stress maximum, stable hashes, and no access to forbidden fields via sentinel objects. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress.py -k 'gate or eligibility or modality'`.
- **Acceptance evidence:** Fixtures prove fail-closed gating. Only an authorized gated audit can establish how many released records carry qualifying metadata; until then no OOD stratum availability is claimed. CPU only.
- **Non-goals/failure policy:** Never train a modality classifier or call inferred modality ground truth. Absence of released modality evidence yields missing robustness evidence, not a private-shift conclusion.
- **Handoff:** STR-05 uses eligibility and reason counts to select declared weights; STR-06 checks clean/stress cardinality and identity.

## STR-05 Build weighted clean and stress selection reports

- **Depends on:** STR-04, SCH-05.
- **Parallel safety and exclusive file ownership:** May run alongside SPL-10 after the stress-manifest schema freezes. Exclusive files are `medfm/challenges/medreason/stress_report.py` and `tests/challenges/medreason/test_stress_report.py`; selection/bootstrap code remains owned by SEL-02.
- **Target paths/symbols:** `stress_report.py`: `StressWeights`, `build_weighted_stress_report`, and `resolve_available_weights`; test mirror.
- **Inputs:** Per-case clean metric rows and paired stress metric rows on the allowed dev/OOF scope, stress manifest/config hashes, and metric provenance. Version `medreason-weighted-stress-report-v1`, seed `2026` (aggregation has no RNG). Proxy GT/VA rows are accepted only when their judge provenance says available/proxy; missing judges do not receive substitute scores.
- **Outputs:** `stress_report.json` with per-metric clean/X-ray/MRI means and denominators, resolved weights, weighted mean, missing-evidence reasons, transform failure counts, paired-case coverage, input/config/payload hashes, and explicit `synthetic_robustness_proxy=true` / `private_shift_reproduced=false` labels.
- **Implementation:**
  1. Join on `(case_id, metric_name)` and require stress rows to reference exactly one clean parent. Compute each stratum mean before weighting; never weight individual cases in a way that changes the declared stratum contribution.
  2. With both strata available, use clean `0.50`, X-ray `0.25`, brain MRI `0.25`. With exactly one metadata-eligible stratum, use clean `0.50` and available stress `0.50`. With neither, use clean `1.00` and report missing robustness evidence.
  3. Resolve availability separately for each metric/task scope. A stratum is unavailable only when there are zero metadata-eligible cases; missing predictions or transform errors in a non-empty eligible stratum invalidate that metric report and are never renormalized away.
  4. Reject lockbox candidate-comparison rows and participant-validation rows. Preserve full-precision component values and compute the weighted sum in fixed order before display rounding.
  5. Hash ordered case/metric inputs, resolved weights, transform policies, and output.
- **Focused Tests:** Exact 50/25/25 arithmetic, either one-stratum fallback, clean-only fallback, task-specific availability, missing paired rows, transform failures, duplicate pairs, unavailable judges, and input-order stability. Exact command: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_report.py`.
- **Acceptance evidence:** CPU fixtures prove weighting and fail-closed denominators. Protected reports require authorized released cases and real prediction provenance; fixture numbers are not performance, hardware, judge, or private-shift evidence.
- **Non-goals/failure policy:** Do not bootstrap or promote here, impute absent strata, substitute lexical metrics for GT/VA proxies, or describe synthetic scores as official/private performance. SEL-02 owns paired uncertainty.
- **Handoff:** SEL-02 consumes per-case pair identities, component metrics, resolved weights, and report hashes; optional-component gates cite this report but recompute paired bootstrap intervals.

## STR-06 Verify stress preserves labels anatomy and pairing

- **Depends on:** STR-02, STR-03, STR-04, STR-05, SPL-07.
- **Parallel safety and exclusive file ownership:** Final phase-integrity card; run after all stress files freeze. Exclusive files are `medfm/challenges/medreason/stress_validation.py`, `tests/challenges/medreason/test_stress_validation.py`, and phase-specific `tests/challenges/medreason/test_packaging_inventory.py`. It does not edit shared `tests/phase_01/test_packaging.py` or package exports owned by SCH-09.
- **Target paths/symbols:** `stress_validation.py`: `StressPairValidation`, `validate_stress_pair`, and `validate_stress_manifest`; test mirrors. Packaging inventory imports every new module from this phase in a CPU-only subprocess and verifies no CUDA initialization or forbidden top-level accelerator dependency.
- **Inputs:** Clean examples/images, stress manifest and images, normalized label-bearing fields, split/group IDs, operation histories, and all policy hashes. Version `medreason-stress-validation-v1`, seed `2026` (no RNG).
- **Outputs:** Validation section in `stress_report.json` with pair counts, label/metadata/hash equality checks, geometry checks, clean-input immutability, split/group identity, packaging inventory result, config/payload hashes, and explicit distinction between fixture, protected-artifact, and hardware evidence.
- **Implementation:**
  1. Require every stress row to have one clean parent with identical `case_id`, task type, original option order, answer, reasoning trace, metadata, `group_id_hash`, and split. Compare canonical hashes of all non-image fields rather than reserializing selectively.
  2. Require exactly one deterministic stress variant for each metadata-eligible case and none for ineligible cases; reject duplicate/orphan/cross-split pairs and selector/spec/hash mismatches.
  3. Prove the structural anatomy-preservation contract: same image count/order, rank, shape, channels, and spatial metadata; clean pixel hash unchanged; transform history contains only the approved intensity/noise/blur/2D-frequency operations and no crop/resize/flip/rotation/translation/slice reordering. Do not claim semantic anatomy or pathology preservation beyond these observable invariants.
  4. Recompute output hashes and rerun transforms from recorded specs on CPU; require byte-identical outputs within the pinned dependency environment.
  5. Add phase packaging inventory coverage and an import subprocess with `CUDA_VISIBLE_DEVICES=""`; imports must not initialize CUDA/XLA or require protected data/network.
- **Focused Tests:** Deliberately mutate a label, option order, case/group/split ID, clean image, shape, image order, transform history, spec hash, and pairing cardinality; each must fail. Verify deterministic replay and CPU-safe import of `fingerprints`, `grouping`, `templates`, `grouping_report`, `splits`, `leakage`, `diagnostics`, all stress modules, and reports. Exact commands: `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_validation.py` and `CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_packaging_inventory.py`.
- **Acceptance evidence:** Fixture-level acceptance is both CPU commands plus golden replay/hash evidence. Protected-artifact acceptance additionally requires `MEDFM_RUN_MEDREASON_DATA=1 CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/challenges/medreason/test_stress_validation.py -m protected_data` against authorized immutable artifacts; until that succeeds, no released-stratum coverage is claimed. This phase needs no GPU and supplies no 48/96 GB, clinical-fidelity, or private-shift evidence.
- **Non-goals/failure policy:** Label preservation does not validate label correctness, and no-geometric-operation checks do not prove clinical equivalence. Never flip anatomy, alter labels, repair failed stress rows, or drop eligible failures to make a report pass. Clean views always remain available; invalid stress evidence blocks stress-based promotion.
- **Handoff:** SEL-01 freezes every policy/hash; SEL-02 uses verified pair keys for same-group resampling; training consistency tasks may consume stress views only after their own declared gate and can never replace clean examples.
