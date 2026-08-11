# Open training and export

This phase implements the open adapter without adding semantic supervision beyond the released package. CPU fixture tests live under the mirrored `tests/challenges/medreason/` tree. Any real model/GPU test must be marked `gpu`, `level3`, and `real_checkpoint`, and must require both `MEDFM_RUN_GPU_TESTS=1` and `MEDFM_RUN_REAL_CHECKPOINTS=1`; ordinary commands must skip it without downloading. The measured local RTX 3090 (24,576 MiB), 364 GiB free storage, and lack of remote hosts cannot establish 31B/26B training, protected judge, or 48/96 GB acceptance. All randomness derives from seed `2026` and stable case/group identifiers. Participant validation, lockbox, hidden data, private annotations, generated semantic labels, pseudo-traces, and proxy-judged preferences are never training inputs.

## OPEN-01 Normalize only released evidence supervision targets

**Depends on:** `GOV-02`, `SCH-02`, `SCH-03`, `DAT-03`, `DAT-04`, `DAT-11`.

**Parallel safety and exclusive file ownership:** May run with `OPEN-03`; must finish before `OPEN-02` and `OPEN-05`. Exclusively owns new `medfm/challenges/medreason/open_targets.py` and `tests/challenges/medreason/test_open_targets.py` while active.

**Target paths/symbols:** `open_targets.py::{EvidenceProvenance, OpenSupervisionTarget, normalize_released_evidence}` and its mirrored test module.

**Inputs:** Validated train-origin `MedReasonExample` records, released-field provenance from `DAT-03`, and immutable archive/extracted-manifest hashes. A field is eligible only when the released adapter identifies it as a released reasoning trace, caption, or curation-evidence value; unknown Unicode metadata is preserved but is not supervision.

**Outputs:** Immutable `OpenSupervisionTarget` with normalized released text, released answer, provenance class/source hash, `target_mode="released_evidence"`, and canonical target hash.

**Implementation:**
1. Accept only open train examples whose evidence field is tied to the immutable source manifest.
2. Normalize line endings and boundary/structural whitespace deterministically while retaining Unicode, source ordering, and wording; never paraphrase, summarize, classify claims, or infer observations.
3. Preserve whether content came from released trace, caption, or curation evidence even when later serialized.
4. Reject empty, conflicting, duplicate, generated, judge-produced, unprovenanced, participant-validation, hidden, dev-target, or lockbox evidence.
5. Emit only hashes/counts in diagnostics; never write generated text back to normalized examples.

**Tests:** Prove byte-identical targets/hashes, Unicode retention, exact source ordering, typed cap rejection rather than silent truncation, and sanitized failures. Negative fixtures cover every forbidden origin, pseudo-trace, preference, and missing provenance; the target contains no thought channel or invented schema content. Command: `uv run pytest -q tests/challenges/medreason/test_open_targets.py -k 'released_evidence or provenance or leakage or deterministic or token_cap'`.

**Acceptance evidence:** Fixture JSON shows source artifact hash, provenance class, normalization version, target hash, mode, and denied-origin matrix. Protected package acceptance additionally requires the real audit manifest.

**Non-goals/failure policy:** No semantic transformation, pseudo-trace, preference pair, private annotation, or assumption that an unavailable release field exists. Missing eligible evidence hands off to answer-only; malformed/conflicting evidence fails closed.

**Handoff:** `OPEN-02/03/05` consume the target mode, provenance, canonical text/hash, and source archive hash.

## OPEN-02 Build answer-only targets when evidence is absent

**Depends on:** `OPEN-01`, `SCH-02`, `DAT-03`, `DAT-04`.

**Parallel safety and exclusive file ownership:** Serial with `OPEN-01` because it extends `open_targets.py` and its tests; may run with `OPEN-03` after interfaces freeze. Owns `build_answer_only_target` and its test class.

**Target paths/symbols:** `medfm/challenges/medreason/open_targets.py::{OpenTargetMode, build_answer_only_target}`; `tests/challenges/medreason/test_open_targets.py::TestAnswerOnlyTargets`.

**Inputs:** Released open train examples with a validated answer and an explicit “no eligible released evidence” result—not malformed evidence.

**Outputs:** A target whose only supervised content is the released answer, `target_mode="answer_only"`, provenance/hash, and aggregate mode counts.

**Implementation:**
1. Select answer-only only after `OPEN-01` establishes genuine absence of released evidence.
2. Preserve the answer except deterministic line-ending/boundary normalization; expose exact assistant target spans.
3. Do not insert empty JSON keys, observation boilerplate, a rationale prefix, model completion, or hidden thought.
4. Keep answer-only data eligible for the same length-bucket rejection and sampling audit.
5. Count modes without logging case content or metadata.

**Tests:** Assert supervised text is exactly the answer plus declared assistant terminator; schema keys, prompt/image tokens, thought delimiters, trace text, pseudo-traces, and preferences are absent. Train origin passes; participant validation, hidden/lockbox, judge/generated content, and malformed-evidence downgrades fail. Output is seed-independent and over-cap answers reject rather than truncate. Command: `uv run pytest -q tests/challenges/medreason/test_open_targets.py -k 'answer_only or no_pseudo or leakage or deterministic or token_cap'`.

**Acceptance evidence:** Target snapshots and count manifest prove every accepted case is answer-only and every denied origin remains excluded; real coverage needs protected train manifests.

**Non-goals/failure policy:** No placeholder evidence, teacher rationale, preference optimization, or pseudo-trace. Missing train answer remains a data error.

**Handoff:** `OPEN-05` receives target spans, `OPEN-06` mode counts, and `OPEN-10` only case IDs/mode flags—not a writable training-target path.

## OPEN-03 Define structured observations reasoning answer response schema

**Depends on:** `SCH-04`, `SCH-07`, `SCH-08`, `MOD-02`.

**Parallel safety and exclusive file ownership:** May run with `OPEN-01/02`; serial with `OPEN-04`. Exclusively owns new `medfm/challenges/medreason/open_response.py` and `tests/challenges/medreason/test_open_response.py` while active.

**Target paths/symbols:** `open_response.py::{OpenResponse, parse_open_response, serialize_open_response}`.

**Inputs:** Internal text using exactly `{"observations":["directly visible claim"],"reasoning":"short inference","answer":"concise final answer"}` and released-evidence targets.

**Outputs:** Validated `OpenResponse`, canonical UTF-8 JSON, and privacy-safe typed parse failures.

**Implementation:**
1. Require exactly three keys, ordered string observations, string reasoning, and non-empty string answer; reject extras, duplicate JSON keys, wrong types, deep/oversized inputs, and thought-channel keys.
2. Keep syntactic validity separate from evidence provenance/grounding.
3. Serialize stable key order and Unicode deterministically.
4. Expose tokenizer-aware budget validation hooks without hard-coding a tokenizer.
5. Never include raw response text in errors or logs.

**Tests:** Round-trip Unicode/escapes byte-identically; reject duplicate/extra keys, invalid types, empty answer, oversized/deep payloads, and private-thought fields. Fake-tokenizer cap boundaries are exact and seed-independent; log capture has no prompt, trace, answer, metadata, or path. Assert no pseudo-trace/preference API exists. Command: `uv run pytest -q tests/challenges/medreason/test_open_response.py -k 'schema or round_trip or cap or leakage or thought'`.

**Acceptance evidence:** Canonical fixture vectors, schema version, error codes, cap boundaries, and byte scans prove fixture-level schema behavior only.

**Non-goals/failure policy:** No semantic judge, generated training trace, preference construction, or repair loop. Schema validity is not grounding evidence.

**Handoff:** `OPEN-04/10/12/14` and runtime parsing consume this exact schema/version and errors.

## OPEN-04 Export bounded trace and concise answer fields

**Depends on:** `OPEN-03`, `EVA-02`, `MOD-12`.

**Parallel safety and exclusive file ownership:** Serial with `OPEN-03`; may run alongside targets/sampling. Owns export symbols and `TestOpenExport` in the same two files.

**Target paths/symbols:** `medfm/challenges/medreason/open_response.py::{OpenExport, count_generated_tokens, export_open_prediction}`; `tests/challenges/medreason/test_open_response.py::TestOpenExport`.

**Inputs:** Valid `OpenResponse`, exact checkpoint tokenizer/processor, and official prediction schema from `EVA-02`.

**Outputs:** `reasoning_trace` formed by ordered non-empty observations followed by reasoning, capped at 160 generated tokens; `answer` copied and capped at 48; cap counts/truncation status retained outside submission fields.

**Implementation:**
1. Count with the exact tokenizer and exclude prompt/visual tokens.
2. Remove structural JSON, join with one fixed separator, and preserve observation order.
3. Shorten only at deterministic token boundaries; never corrupt Unicode or semantically rewrite an answer. Empty-after-cap returns a typed runtime-fallback error.
4. Emit only official fields; keep telemetry private.
5. Reject/strip private thought before export, including defense-in-depth marker checks.

**Tests:** Cover trace 159/160/161 and answer 47/48/49 tokens, Unicode, empty combinations, deterministic bytes, official fixture parsing, and over-cap failure/shortening policy. Byte scans prove no JSON wrapper, prompt, reference, path, telemetry, thought, pseudo-trace, or preference leaks. Command: `uv run pytest -q tests/challenges/medreason/test_open_response.py -k 'export or token_budget or thought_stripping or official_schema or leakage'`.

**Acceptance evidence:** Boundary snapshots with exact token counts and official-parser success; protected tokenizer acceptance requires immutable real artifacts and guarded tests.

**Non-goals/failure policy:** No semantic summarization, answer invention, recursive repair, or private-thought submission.

**Handoff:** Runtime receives `OpenExport`; evaluation stores content only in access-controlled prediction artifacts and logs only hashes/counts.

## OPEN-05 Mask all non-assistant multimodal training tokens

**Depends on:** `OPEN-01`, `OPEN-02`, `OPEN-03`, `MOD-02`, `MOD-03`, `MOD-12`.

**Parallel safety and exclusive file ownership:** May run with sampling after target spans freeze; serial with processor-collation changes. Exclusively owns new `medfm/challenges/medreason/open_collator.py` and `tests/challenges/medreason/test_open_collator.py`.

**Target paths/symbols:** `open_collator.py::{OpenMultimodalBatch, build_open_supervised_example, OpenMultimodalCollator}`; reuse only the `IGNORE_INDEX` and assistant-only invariants from `medfm.data.textprep.tokenize`, not its left-truncating execution path.

**Inputs:** Exact processor/chat-template output, every native Gemma 4 multimodal tensor, and released-evidence or answer-only target spans.

**Outputs:** Challenge-local mapping batch passed as native model keyword arguments; labels are `-100` for system/user/prompt, image placeholders, padding, and every non-target position.

**Implementation:**
1. Derive assistant boundaries from template/processor metadata, never decoded substring search.
2. Supervise full released-evidence response or only the answer span by mode, plus the declared terminator.
3. Mask visual/prompt/padding positions even when token IDs overlap text IDs.
4. Preserve `pixel_values`, attention masks, token-type IDs, image-position IDs, and any processor-created field. Do not use generic `MedicalBatch`, which drops native fields.
5. Reject zero-supervision and processor lengths outside 2,048/4,096/8,192/16,384 buckets; never invoke existing silent/left truncation.

**Tests:** Real-shaped fakes prove position labels, answer-only masking, repeated-text safety, padded sample masking, native-field preservation, direct keyword forwarding, and typed overflow rejection. Include seed independence, sanitized errors, forbidden-origin leakage, and thought/pseudo/preference denial. Command: `uv run pytest -q tests/challenges/medreason/test_open_collator.py -k 'mask or answer_only or native_fields or overflow or leakage'`.

**Acceptance evidence:** Position fixtures, supervised-token counts by mode, native-key inventory, and overflow errors. Real processor tests are `gpu level3 real_checkpoint` and doubly environment-gated.

**Non-goals/failure policy:** No generic batch coercion, silent truncation, image supervision, decoded-text span search, pseudo-trace, preference, or thought training.

**Handoff:** `OPEN-06/08/09` receive validated challenge-local batches and mode/token counts.

## OPEN-06 Mix open and MCQ retention examples equally

**Depends on:** `SPL-07`, `MCQ-01`, `MCQ-03`, `OPEN-02`, `OPEN-05`.

**Parallel safety and exclusive file ownership:** Begins after collator contracts; serial with `OPEN-07`. Exclusively owns new `medfm/challenges/medreason/open_sampling.py` and `tests/challenges/medreason/test_open_sampling.py`.

**Target paths/symbols:** `open_sampling.py::{RetentionMixSampler, RetentionMixState, build_retention_mix}` and challenge training-checkpoint integration.

**Inputs:** Train-only group-disjoint open and MCQ retention pools, seed `2026`, epoch, rank/world size, accumulation window.

**Outputs:** Resumable deterministic stream with exactly 8 open and 8 MCQ examples per 16-example accumulation window; manifest of requested/observed ratios and split/order hashes.

**Implementation:**
1. Stable-sort separate train queues before seeded epoch/rank shuffling.
2. Interleave equal task counts per complete even accumulation window; reject odd geometry.
3. Cycle the shorter queue only at deterministic boundaries and record reuse; never refill from dev, lockbox, or participant validation.
4. Implement `state_dict/load_state_dict` on the challenge batch sampler and explicitly restore it through the challenge checkpoint builder; the generic trainer currently does not restore `batch_sampler` state.
5. Persist only IDs/hashes/counts, not text.

**Tests:** Assert every window is 8:8, changed seed changes order not ratio/membership, rank shards are deterministic, and interruption/resume produces the uninterrupted sequence exactly. Forbidden split IDs, missing sampler state, cap replacement from protected data, thoughts, pseudo-traces, and preferences fail. Command: `uv run pytest -q tests/challenges/medreason/test_open_sampling.py -k 'retention or ratio or resume or seed or leakage'`.

**Acceptance evidence:** Fixture manifest records split hash, seed/epoch, 0.5/0.5 ratios, reuse counts, sampler-state hash, and uninterrupted/resumed sequence hashes.

**Non-goals/failure policy:** No task-ratio tuning, semantic relabeling, validation retention, or silent non-resumable fallback.

**Handoff:** `OPEN-07` layers group weighting without changing task ratio; `OPEN-08/09` consume sampler and state hashes.

## OPEN-07 Oversample modalities only through source groups

**Depends on:** `SPL-02`, `SPL-03`, `SPL-06`, `SPL-07`, `STR-04`, `OPEN-06`.

**Parallel safety and exclusive file ownership:** Serial with `OPEN-06` on sampling code/tests; may run with pilot schema after interfaces freeze. Owns group-sampling symbols.

**Target paths/symbols:** `medfm/challenges/medreason/open_sampling.py::{GroupSamplingConfig, GroupModalitySampler, summarize_group_sampling}`; `tests/challenges/medreason/test_open_sampling.py::TestGroupModalitySampling`.

**Inputs:** Defensible `group_id`, released modality metadata, train split, explicit X-ray/MRI group weights, seed `2026`. Unknown modality remains unknown.

**Outputs:** Deterministic group-first, no-replacement-within-window case selection; requested/realized group and case ratios plus reuse audit.

**Implementation:**
1. Weight source/duplicate groups, never individual cases; mixed/unknown groups remain explicit.
2. Select a group then a case without replacement, prohibiting one case twice in an accumulation window.
3. Bound/configure cross-window group reuse and report it.
4. Preserve exact open/MCQ 50/50 windows independently inside task queues.
5. Mark unavailable requested strata instead of guessing modality or fabricating evidence.

**Tests:** Unequal-size synthetic groups prove ratios follow group weights, not case counts; 50/50 task ratio, no duplicate case, deterministic seed/resume, and missing-stratum behavior hold. Unknown modality is not inferred; split leakage, overflow replacement, thought/pseudo/preference entry, and case-level weighting reject. Command: `uv run pytest -q tests/challenges/medreason/test_open_sampling.py -k 'group_modality or oversample or no_repeat or ratio or seed or leakage'`.

**Acceptance evidence:** Requested/observed ratios, group reuse histogram, sequence/state hashes, and unavailable-stratum status. Real modality coverage requires released audit evidence.

**Non-goals/failure policy:** No modality classifier, anatomy/label transformation, speculative OOD claim, or generated supervision.

**Handoff:** `OPEN-08` freezes sampling hashes; `OPEN-13` reuses stable case/group stress pairing.

## OPEN-08 Add open-adapter learning-rate pilot configurations

**Depends on:** `EVA-08`, `EVA-13`, `MOD-09`, `MOD-10`, `MOD-12`, `MOD-13`, `MOD-14`, `OPEN-05`, `OPEN-06`, `OPEN-07`, `CLI-03`.

**Parallel safety and exclusive file ownership:** May run while optional modules are developed; serial with `OPEN-09`. Exclusively owns new `medfm/challenges/medreason/open_training.py` and `tests/challenges/medreason/test_open_training.py`; `CLI-06` owns YAML.

**Target paths/symbols:** `open_training.py::{OpenPilotConfig, build_open_pilot_configs, build_linear_warmup_scheduler, build_open_component_builders, validate_open_pilot_pair}`; reuse `medfm.training.pipeline::{ComponentBuilders, TrainingPipeline}` and checkpoint hash patterns, never `medfm.recipes.phase13`.

**Inputs:** Advanced immutable model record, base recipe, identical sampler sequence, seed `2026`, buckets, model-family role, and measured gate references.

**Outputs:** Two 250-step configs: `{2e-5,5e-5}` for advanced large Gemmas or `{5e-5,1e-4}` for MedGemma; all non-LR semantics identical.

**Implementation:**
1. Fix NF4 double quantization/BF16, microbatch 1, accumulation 16, rank 16/alpha 32/dropout .05, fused AdamW, max grad .3, no KV cache, frozen vision, and discovered language attention/MLP targets.
2. Implement explicit linear decay with warmup ratio `.03`; do not inherit the existing optimizer’s cosine scheduler.
3. Freeze data/split/sampler/processor/template/initial-adapter hashes, 250 steps, and evaluation cadence; only LR/run ID/output differ.
4. Validate references to 100-real-batch memory and attention parity without claiming local success. Hardware tests use triple markers and both env guards.
5. Preregister winner rule: exact judges available; dev-only group-paired evidence; GT/VA margins at least `-0.05`; maximize worst normalized GT/VA margin, then lower dev loss, then lower LR.

**Tests:** Assert LR sets, exact linear/warmup curve boundaries, identical non-LR hashes/order, 50/50/group sampling, caps, masking, and frozen vision. Reject cosine/default scheduler, wrong seed/steps, forbidden origins, pseudo/preferences/thought supervision, missing hardware evidence, or mutable promotion rules. Command: `uv run pytest -q tests/challenges/medreason/test_open_training.py -k 'pilot_config or linear_warmup or invariants or leakage or promotion_rule'`.

**Acceptance evidence:** Canonical config diff, LR curve vector, builder-stage list, cap/sampler/target hashes, promotion rule, and scoped hardware status; fixture success is code acceptance only.

**Non-goals/failure policy:** Do not run pilots, auto-recover OOM, use Phase 13 synthetic builders, unfreeze vision, or substitute judges.

**Handoff:** `OPEN-09` receives config/initial-state/data-order/scheduler hashes and fixed promotion rule.

## OPEN-09 Select and train winning open pilot

**Depends on:** `OPEN-08`, `EVA-06`, `EVA-07`, `EVA-08`, `STR-05`, `SEL-02`, `SEL-03`, `SEL-04`.

**Parallel safety and exclusive file ownership:** Serial with `OPEN-08` on training code/tests. Independent MCQ work may proceed, but protected models/judges load sequentially. Owns selection, continuation, and evaluation-cadence symbols.

**Target paths/symbols:** `medfm/challenges/medreason/open_training.py::{OpenPilotResult, select_open_pilot, OpenContinuationConfig, OpenEvaluationCadence, build_open_continuation}`.

**Inputs:** Comparable pilot manifests, dev GT/VA and available-OOD reports, corrected group-paired intervals, dev loss, sampler/checkpoint state, and resource telemetry.

**Outputs:** Deterministic winner/rejection decision and, only for an eligible winner, two-epoch continuation with fixed evaluations/early stopping and best-step metadata.

**Implementation:**
1. Reject partial or incomparable ancestry/config/order/evaluator hashes.
2. Apply frozen precedence on dev only; unavailable exact judges block proxy promotion.
3. Implement a challenge-local evaluation callback at every declared optimizer-step boundary and record calls; do not rely on generic `Trainer.eval_every_steps`, which is currently unused.
4. Continue the winner for exactly two epochs, restore challenge batch-sampler/optimizer/scheduler state, and early-stop only at fixed evaluations.
5. Record mode/task/modality/stress ratios, caps/rejections, seed, sanitized failures, and best clean language-only parent.

**Tests:** Cover winner/tie/margin/correction/missing-judge/mismatched-sequence/seed/leakage cases; assert exact evaluation calls, resume equivalence, two epochs, 50/50 ratios, 160/48 references, thought stripping, and no pseudo/preferences. Command: `uv run pytest -q tests/challenges/medreason/test_open_training.py -k 'pilot_selection or continuation or evaluation_cadence or resume or leakage'`.

**Acceptance evidence:** Decision JSON includes candidates, intervals, precedence, evaluation-step list, sampler/scheduler state hashes, observed ratios, selected checkpoint, and best step. Protected promotion remains separately gated.

**Non-goals/failure policy:** No third LR, post-hoc cadence/rule, lockbox comparison, judge substitution, or hardware claim.

**Handoff:** `OPEN-10–14` receive frozen parent hash; `OPEN-15` receives the complete decision/checkpoint chain.

## OPEN-10 Prompt frozen-base evidence extraction for answer-only data

**Depends on:** `OPEN-02`, `OPEN-03`, `OPEN-04`, `OPEN-09`, `EVA-07`, `EVA-11`, `SEL-02`, `SEL-04`, `SEL-05`.

**Parallel safety and exclusive file ownership:** May run with optional-candidate implementation after parent freeze; protected visual-judge runs are serial. Exclusively owns new `medfm/challenges/medreason/evidence_extraction.py` and `tests/challenges/medreason/test_evidence_extraction.py`.

**Target paths/symbols:** `evidence_extraction.py::{EvidenceExtractionPrompt, EvidenceExtractionCandidate, extract_frozen_base_evidence}`.

**Inputs:** Answer-only inference cases, frozen base/processor/template/images, frozen prompt/seed/caps. Reference answers are never inserted during dev inference.

**Outputs:** Schema-valid image-conditioned response marked `inference_only`; zero extracted observations enter training targets.

**Implementation:**
1. Use one fixed prompt requesting concise visible observations and answer.
2. Run frozen base greedily and bounded; parse/export through `OPEN-03/04` using challenge-local native tensors.
3. Type-block `inference_only` responses from target/collator conversion.
4. Evaluate real/no/shuffled images on paired dev groups without text logging.
5. Promote only when corrected VA lower bound improves, GT is non-inferior by `-0.05`, shuffled grounding and runtime gates pass.

**Tests:** Fakes prove images/native fields are passed, references never enter prompts, outputs are deterministic/capped/thought-free, and inference artifacts cannot enter samplers. Deny participant/lockbox leakage, pseudo/preferences, missing controls/judges/hardware, and failed promotion intervals; training ratios remain unchanged. Command: `uv run pytest -q tests/challenges/medreason/test_evidence_extraction.py -k 'frozen_base or answer_leakage or inference_only or grounding_gate or deterministic'`.

**Acceptance evidence:** Prompt/base/processor hashes, response hash, zero-training-target count, controls, corrected decision, and scoped hardware/judge status.

**Non-goals/failure policy:** No teacher-answer prompting, model update, pseudo-trace storage, preference pair, agent loop, or grounding claim from schema alone.

**Handoff:** Accepted prompt/candidate hash goes to runtime/freeze; rejected route contributes decision metadata only.

## OPEN-11 Gate late vision projector and block LoRA

**Depends on:** `OPEN-09`, `MOD-10`, `MOD-11`, `MOD-13`, `MOD-14`, `EVA-07`, `EVA-11`, `STR-05`, `SEL-02`, `SEL-04`, `SEL-05`.

**Parallel safety and exclusive file ownership:** Begins after language adapter freeze; candidate development may overlap conceptually, but shared code and protected GPU/judges are serial. Exclusively owns new `medfm/challenges/medreason/open_candidates.py` and `tests/challenges/medreason/test_open_candidates.py` until its symbols land.

**Target paths/symbols:** `open_candidates.py::{VisionLoRAConfig, discover_late_vision_targets, evaluate_vision_candidate}`.

**Inputs:** Frozen language parent, exact native topology, released train groups, clean/available-OOD dev metrics, shuffled controls, and measured resource references.

**Outputs:** One rank-8 candidate on only projector/final two vision blocks at `0.1 ×` language LR; decision and accepted adapter-only tensors.

**Implementation:**
1. Discover/record exact target modules and fail on ambiguity.
2. Reload/freeze base and language adapter; initialize only late-vision LoRA with seed `2026`.
3. Preserve released-only targets, native batch fields, masking, caps, 50/50 mix, and group oversampling.
4. Require clean VA and every available X-ray/MRI OOD VA stratum improve, GT drop ≤`.05`, and real-minus-shuffled VA ≥`.25`, with corrected paired tests and runtime gate.
5. Exclude rejected tensors from freeze/deployment.

**Tests:** Fake topology proves targets/rank/LR/freeze; promotion fixtures fail each clean/OOD/GT/shuffle/correction/runtime criterion, ambiguity, leakage, cap/mask/ratio drift, seed drift, and thought/pseudo/preference input. Real tests are triple-marked/doubly guarded. Command: `uv run pytest -q tests/challenges/medreason/test_open_candidates.py -k 'vision_lora or target_discovery or shuffled_gate or ood or frozen_parent'`.

**Acceptance evidence:** Trainable-module manifest, parent/candidate hashes, strata intervals, shuffle gap, resource evidence reference, and decision; fake topology proves fixture acceptance only.

**Non-goals/failure policy:** No full vision tune, earlier blocks, guessed modules/modality, lockbox selection, or unmeasured hardware claim.

**Handoff:** `OPEN-15` receives accepted tensors/targets or rejection; runtime never dynamically discovers modules.

## OPEN-12 Gate internal thinking while stripping private thoughts

**Depends on:** `OPEN-04`, `OPEN-09`, `EVA-06`, `EVA-07`, `EVA-10`, `SEL-02`, `SEL-04`.

**Parallel safety and exclusive file ownership:** Serial for edits to `open_candidates.py`/tests and protected execution. Owns thinking symbols/tests.

**Target paths/symbols:** `medfm/challenges/medreason/open_candidates.py::{ThinkingCandidateConfig, ThinkingResult, strip_private_thought, evaluate_thinking_candidate}`; mirrored `TestThinkingCandidate`.

**Inputs:** Frozen parent, exact template thinking controls, one bounded greedy thought pass, seed `2026`, caps, paired proxy/runtime evidence.

**Outputs:** Private in-memory thought and separately generated concise `OpenResponse`; zero thought bytes in submitted fields, logs, caches, predictions, or persisted raw artifacts.

**Implementation:**
1. Require checkpoint-supported channel separation; unsupported/ambiguous templates disable the candidate.
2. Generate one greedy bounded private channel, then a separate final structured response.
3. Strip on token/channel boundaries before decode/persistence; reject thought markers in final schema/export.
4. Promote only when corrected GT and VA lower bounds are both `>0`, remaining margins pass, and measured latency/VRAM meet selected runtime gate.
5. Persist hashes/counts and sanitized failures only.

**Tests:** Channel-aware fakes place nested JSON/markers in thought and byte-scan response/export/logger/exception/cache/artifacts for total absence. Assert greedy repeat determinism, cap/schema failures, unchanged 50/50 ratios, forbidden split denial, and rejection for either metric/runtime failure; no pseudo/preferences can be derived. Command: `uv run pytest -q tests/challenges/medreason/test_open_candidates.py -k 'thinking or strip_private or no_persist or runtime_gate or deterministic'`.

**Acceptance evidence:** Byte-scan report, thought/final token counts, corrected intervals, hardware evidence status, and decision. Fixture stripping does not prove real template support.

**Non-goals/failure policy:** Never supervise/submit/persist thought or convert it into traces/preferences; no sampling, debate, or repair loop.

**Handoff:** Freeze/runtime receive accepted config/hash or rejection and only the final-response interface.

## OPEN-13 Gate deterministic stress consistency training candidate

**Depends on:** `OPEN-09`, `STR-01`, `STR-02`, `STR-03`, `STR-04`, `STR-05`, `STR-06`, `SEL-02`, `SEL-03`, `SEL-04`.

**Parallel safety and exclusive file ownership:** Starts after clean adapter lock; serial on `open_candidates.py`/tests and shared GPU. Owns consistency symbols/tests.

**Target paths/symbols:** `open_candidates.py::{ConsistencyConfig, teacher_forced_open_kl, build_consistency_pilots, evaluate_consistency_candidate}`.

**Inputs:** Frozen clean parent, paired released clean/stress cases, stable case-hash transforms, released target tokens, lambdas `{0.1,0.2}`, 250 steps, seed `2026`.

**Outputs:** Two comparable `L_task + lambda*KL(p_clean||p_stress)` pilots with exactly 50% stressed occurrences; at most one accepted adapter.

**Implementation:**
1. Compute teacher-forced KL only at released supervised target positions using `OPEN-05` masks.
2. Pair each original with one deterministic stress view without changing group, anatomy, label, or response target.
3. Hold every field/order fixed except lambda; preserve open/MCQ retention while realizing 50% clean/50% stress.
4. Keep a lambda only when clean/available-OOD weighted selection improves under corrected pairing and clean GT/VA remain within `.05` of parent.
5. Hash parent, transform, target, mask, sequence, scheduler, and candidate.

**Tests:** KL direction/equality/masking, exact stress/task ratios, seed/resume reproducibility, unchanged targets/caps/thought stripping, and each promotion failure are observable. Reject generated targets, preferences, leakage, mismatch, unavailable judges, and extra lambdas/steps. Command: `uv run pytest -q tests/challenges/medreason/test_open_candidates.py -k 'consistency or teacher_forced_kl or stress_ratio or lambda or promotion'`.

**Acceptance evidence:** Pair/sequence hashes, ratio reports, loss summaries, clean/weighted intervals, and decision. Synthetic stress is only a proxy.

**Non-goals/failure policy:** No model-generated consistency targets, label/anatomy changes, pseudo-traces/preferences, or private-shift claim.

**Handoff:** `OPEN-15` receives accepted adapter/hash or rejection; selection consumes frozen identity.

## OPEN-14 Gate four-sample self-consistency inference candidate

**Depends on:** `OPEN-03`, `OPEN-04`, `OPEN-09`, `EVA-06`, `EVA-07`, `SEL-01`, `SEL-02`, `SEL-03`, `SEL-04`.

**Parallel safety and exclusive file ownership:** May be developed alongside training-only candidates after SFT freeze; protected evaluation is serial. Exclusively owns new `medfm/challenges/medreason/self_consistency.py` and `tests/challenges/medreason/test_self_consistency.py`.

**Target paths/symbols:** `self_consistency.py::{SelfConsistencyConfig, derive_case_sample_seeds, select_supported_response, run_self_consistency_candidate}`.

**Inputs:** Frozen adapter; exactly four samples at temperature `1.0`, top-p `.95`, top-k `64`; four case-hash seeds; fixed non-judge support selector; caps and paired dev evidence.

**Outputs:** Four parsed/capped inference-only candidates and one deterministic selection; decision metadata, never training/preference records.

**Implementation:**
1. Derive four distinct seeds from case ID, `2026`, and sample index; remain batch-order/resume invariant.
2. Generate exactly four bounded samples and independently strip thoughts, parse, and export.
3. Apply one preregistered selector/hash; ties use sample index. No judge, debate, adaptive sampling, or agent loop.
4. Promote only when corrected GT and VA lower bounds both exceed zero, remaining non-inferiority/runtime gates pass, and every selected output satisfies schema/caps.
5. Type-block samples from dataset/collator conversion.

**Tests:** Repeated/rebatched runs yield identical seeds/samples/tie result; malformed, over-cap, and thought-bearing samples cannot leak. Assert exact parameters/count, reference-free selector, unchanged training ratios, denied participant/lockbox tuning, no DPO/pseudo/preferences, and rejection for either metric/runtime/correction failure. Command: `uv run pytest -q tests/challenges/medreason/test_self_consistency.py -k 'four_sample or seed or selector or thought or promotion'`.

**Acceptance evidence:** Config/selector hashes, seed vectors, four response hashes, official schema/token counts, latency multiplier, intervals, decision, and hardware scope.

**Non-goals/failure policy:** No fifth/adaptive sample, preference mining, DPO, pseudo-trace training, judge shipping, debate, or repair loop.

**Handoff:** Freeze/runtime receive accepted immutable config/selector or rejection; training receives nothing.

## OPEN-15 Freeze accepted open adapter and metadata

**Depends on:** `OPEN-09`, `OPEN-10`, `OPEN-11`, `OPEN-12`, `OPEN-13`, `OPEN-14`, `MOD-11`, `CLI-09`, `SCH-09`, `SEL-01`.

**Parallel safety and exclusive file ownership:** Terminal task after every candidate decision; no candidate artifact may mutate concurrently. Exclusively owns new `medfm/challenges/medreason/open_freeze.py`, `tests/challenges/medreason/test_open_freeze.py`, and the `challenges` entry/import assertion in `tests/phase_01/test_packaging.py` while active.

**Target paths/symbols:** `open_freeze.py::{OpenFreezeManifest, validate_open_freeze_inputs, freeze_open_adapter}`; reuse `medfm.peft.checkpoint::{save_adapter_checkpoint, load_checkpoint_manifest}` tensor/hash conventions; update `tests/phase_01/test_packaging.py::{SUBPACKAGES, test_subpackages_importable}` inventory coverage.

**Inputs:** Selected language adapter, accepted optional tensors/configs, every candidate decision, exact base/processor/template/config/data/split/sampler/target/scheduler hashes, seed/caps/masking/ratios, and scoped legal/judge/hardware evidence.

**Outputs:** Adapter-only safetensors and canonical manifest under content-addressed `artifacts/models/medreason/open/<artifact-hash>/`; no base/optimizer/raw thought/training text/judge output/rejected tensors.

**Implementation:**
1. Verify an unbroken released-source-to-decision ancestry and reject missing/mutable/mismatched inputs.
2. Record exact LoRA targets/config, seed `2026`, buckets, 160/48 caps, masking, 50/50 task ratio, group modality policy, thought-strip version, evaluation cadence/sampler/scheduler state, and promotion decisions.
3. Record zero pseudo-traces, generated labels, preferences, participant-validation/lockbox/hidden training records, and persisted private thoughts.
4. Work around overwriteable generic exports: write to a private staging directory, hash/reload, atomically finalize to a create-new content-addressed path, refuse an existing destination, and make the manifest immutable/read-only. Never overwrite an accepted artifact.
5. Separate `fixture_code`, `protected_artifact`, `judge_proxy`, `gpu_24gb`, `hardware_48gb`, and `hardware_96gb` statuses; only measured evidence may pass a scope. Ensure the new challenge package is included in wheel/import inventory without eager CUDA/bitsandbytes/flash-attn imports.

**Tests:** Fail on every ancestry/hash/seed/ratio/group/cap/mask/scheduler/cadence/promotion mismatch, leaked origin, pseudo/preference count, thought byte/key, existing destination, partial staging, or forbidden file. Verify deterministic manifest/hash, fake adapter reload, adapter-only inventory, no-clobber atomicity, CPU-safe import, and package inclusion. Commands: `uv run pytest -q tests/challenges/medreason/test_open_freeze.py -k 'freeze or manifest or adapter_only or no_clobber or leakage or thought_scan or promotion_chain'` and `uv run pytest -q tests/phase_01/test_packaging.py -k 'subpackages_importable or forbidden_top_level or cpu_import'`.

**Acceptance evidence:** Frozen SHA-256 inventory, reload report, ancestry graph, zero-leakage counters, decision table, schema/cap/mask/sampling invariants, packaging inventory result, and explicit evidence scopes. Real checkpoint/hardware acceptance remains gated.

**Non-goals/failure policy:** No base merge, retroactive selection, lockbox run, all-label deployment retraining, overwrite, pseudo-trace/preference, thought persistence, or unsupported license/model/hardware claim. Any missing prerequisite aborts without a partial final artifact.

**Handoff:** `SEL-06` through `SEL-16`, runtime, CLI, and Docker tasks receive immutable manifest/artifact hashes, exact base/processor/template revisions, accepted inference-config hashes, and evidence-scope statuses.
