# MCQ training and scoring

The default MCQ route is supervised option learning plus conditional likelihood scoring. It never free-generates the submitted label. Implement it inside `medfm/challenges/medreason/`, using a challenge-local multimodal batch that preserves rank-3 `pixel_values`, `image_position_ids`, token-type IDs, image masks, and future processor fields; generic `MedicalBatch`, `MultiImageVLCollator`, and the existing generation wrapper cannot preserve the required native fields. Reuse `build_supervised_example` masking semantics, but not its left-truncation behavior: reject an oversized processed example. Reuse trainer/backend/checkpoint primitives where their contracts fit, while explicitly filling the currently missing fixed-interval evaluation, resumable batch-sampler state, linear warmup/decay, and immutable final-export behavior. Never route production through synthetic `medfm.recipes.phase13` VLM builders.

All ordinary tests use tiny fakes with production field names under `tests/challenges/medreason/`. Every real-checkpoint test must skip unless `MEDFM_RUN_REAL_CHECKPOINTS=1`, an immutable local snapshot is present, CUDA is available, the declared device-memory preflight passes, and required terms/access are satisfied; tests never download. The local RTX 3090 (24,576 MiB), 364 GiB free storage, and lack of remote compute cannot establish protected 31B/26B training, 48/96 GB support, or full artifact staging. New modules require intentional export/private status, package-inventory coverage, and the CPU import-safety test in `tests/phase_01/test_packaging.py`.

## MCQ-01 Build exact label and option assistant targets

**Depends on:** SCH-02, SCH-09, DAT-03, DAT-04, MOD-02, MOD-03, MOD-12.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/mcq_targets.py` symbols `MCQOptionView`, `format_mcq_assistant_target`, and `build_mcq_supervised_example`, with their target tests. Serialize this file with MCQ-02/03. Do not edit schema/processor modules owned by prerequisites.

**Target paths/symbols:** New `mcq_targets.py` symbols above; `tests/challenges/medreason/test_mcq_targets.py`; masking anchor `medfm/data/textprep/tokenize.py::build_supervised_example` and `IGNORE_INDEX`; package inventory/export coverage through SCH-09.

**Inputs:** Validated `MedReasonExample`; immutable supplied `(label, text)` options; explicit `display_to_original`; exact checkpoint processor/chat template; challenge-local processor batch from MOD-03; declared length bucket.

**Outputs:** A processed supervised example whose only assistant content is exactly `<display label>: <option text>`, with canonical/display indices and labels, mapping, supervised count, target hash, and every non-label processor tensor preserved.

**Implementation:**

1. Represent each option by original index, exact supplied label, and text; never recover identity by text equality.
2. Present options in `display_to_original` order, using the original ordered label alphabet for display positions. Map the gold original index through the inverse mapping.
3. Format exactly `f"{display_label}: {option_text}"`; append no rationale, confidence, pseudo-trace, or semantic label.
4. Apply the exact checkpoint chat template and processor. Preserve image placeholders and all multimodal fields in the challenge-local batch.
5. Reproduce `build_supervised_example`'s observable assistant-only contract: assistant content and its terminal assistant/EOS token supervised; prompt, image, padding, and non-assistant template tokens masked. Reject, rather than truncate, output beyond its declared bucket.
6. Record only hashes/indices in metadata; logs contain case ID and sanitized error class, never target content.

**Focused tests and exact commands:** Test a nontrivial label alphabet, repeated option text, Unicode, swapped mappings, malformed/duplicate mappings, and overflow. A label-only or text-only remap bug must fail. Assert rank-3 pixels and all multimodal fields survive. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_targets.py -k 'target or mapping or unicode or overflow'`. Inventory command: `uv run pytest -q tests/phase_01/test_packaging.py tests/challenges/medreason/test_package_inventory.py`.

**Acceptance evidence:** Focused fixture output and serialized hashes/mappings prove code behavior; inventory/CPU-import tests prove the module ships without eagerly initializing CUDA. Real processor acceptance remains under real-checkpoint/hardware guards and proves neither access nor quality.

**Non-goals/failure policy:** No free-generation label path, rationale SFT, pseudo-label, participant-validation answer, hidden-data tuning, GRPO, phase-13 production builder, or silent truncation. Ambiguous identity/boundary is a hard sanitized error.

**Handoff:** MCQ-02 consumes option identity/mapping; MCQ-03 consumes assistant-span metadata; MCQ-04 reuses the exact formatter so train/inference targets cannot drift.

## MCQ-02 Deterministically permute options for every epoch

**Depends on:** MCQ-01, SCH-08, SPL-07, SPL-09.

**Parallel safety and exclusive file ownership:** Own `mcq_targets.py::epoch_option_order`, `remap_gold_to_display`, and the challenge dataloader epoch-state adapter; serialize `mcq_targets.py` with MCQ-01/03. Do not modify generic split logic.

**Target paths/symbols:** `mcq_targets.py::epoch_option_order`, `remap_gold_to_display`, `MedReasonMCQDataLoaderState`; `tests/challenges/medreason/test_mcq_permutation.py`; existing `Trainer::_set_epoch`, `_loader_state`, `_loader_state_target` are anchors, not assumed sufficient because they omit a bare `batch_sampler` state.

**Inputs:** Seed `2026`, exact case ID bytes, zero-based epoch, validated option count/order, original gold index, resumed epoch/batch cursor.

**Outputs:** Bijective `display_to_original` and inverse tuples; remapped gold display index/label; version `medreason-mcq-epoch-order-v1`; resumable loader state with epoch, cursor, mapping hash.

**Implementation:**

1. For every original index, SHA-256 hash canonical bytes containing version, seed, case ID, epoch, and index; sort by `(digest, original_index)`. Never use Python `hash()`, global RNG, worker/rank, or iteration order.
2. Validate a complete bijection, derive its inverse, and remap label plus option text together.
3. Expose `set_epoch`, `state_dict`, and `load_state_dict` on the challenge loader wrapper so checkpoint discovery sees state even with a `batch_sampler`.
4. Restore epoch and batch cursor before materializing resumed mappings; do not repeat/skip the interrupted permutation.
5. Persist version/seed/epoch/mapping digest only.

**Focused tests and exact commands:** Golden mappings across fresh processes; inverse/bijection; label-and-text remap; worker/iteration invariance; uninterrupted versus interrupted/resumed equality. These catch Python hash use and missing batch-sampler restoration. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_permutation.py`.

**Acceptance evidence:** Golden vectors plus resume/worker tests and `permutation.algorithm`, `seed`, `mapping_digest`. Fixture acceptance does not validate protected labels.

**Non-goals/failure policy:** No participant-validation answer transformation, label-frequency balancing, or requirement that adjacent epochs always differ. Invalid/stale resume state fails before forward.

**Handoff:** MCQ-03 builds remapped labels; MCQ-05/06 reuse mapping algebra; MCQ-11 stores original indices.

## MCQ-03 Mask prompt image and padding tokens

**Depends on:** MCQ-01, MCQ-02, MOD-02, MOD-03, MOD-12.

**Parallel safety and exclusive file ownership:** Own `mcq_targets.py::apply_mcq_assistant_mask` and `validate_mcq_supervised_batch`; serialize with MCQ-01/02. Do not change generic `build_supervised_example`.

**Target paths/symbols:** Symbols above; `tests/challenges/medreason/test_mcq_masking.py`; anchors `build_supervised_example`, `validate_supervised_batch`, `IGNORE_INDEX`.

**Inputs:** Challenge-local processor batch, exact assistant boundary, `input_ids`, `attention_mask`, visual positions/placeholder IDs, pad ID, token-type/image-position fields.

**Outputs:** `labels` supervising only candidate assistant content plus terminal assistant/EOS; positive supervised counts; all other fields byte-identical.

**Implementation:**

1. Derive assistant boundary from checkpoint template output, never token-text search/fixed offset; assert it selects MCQ-01 target.
2. Initialize labels to `IGNORE_INDEX`, copy target IDs only at valid assistant positions.
3. Force prompt/system/user/template, image placeholders/positions, `attention_mask == 0`, and padding to `IGNORE_INDEX`, even if a malformed assistant mask overlaps.
4. Support left/right padding; reject zero supervision, ambiguous/noncontiguous spans, shape mismatch, and overflow. Do not use generic left truncation.
5. Validate batch supervision and preserve rank-3 pixels, `image_position_ids`, token types/masks, and unknown fields.

**Focused tests and exact commands:** Parity with `build_supervised_example` on a compatible fixture; left/right padding; image/prompt leakage; repeated label-like prompt; overlapping masks; zero supervision; field preservation. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_masking.py`.

**Acceptance evidence:** Exact supervised-position/count and field-equality tests. Gated command: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -q tests/challenges/medreason/test_mcq_masking_real.py`; it also guards local snapshot, CUDA, access, and declared memory.

**Non-goals/failure policy:** No prompt/image/pad/thought supervision, silent truncation, or guessed boundary. Hardware skip is not protected acceptance.

**Handoff:** MCQ-10 uses labels for CE; MCQ-12 adds contrastive loss; MCQ-04 reuses target masks.

## MCQ-04 Batch candidate conditional likelihood scoring

**Depends on:** MCQ-01, MCQ-03, MOD-01, MOD-02, MOD-03, MOD-12, EVA-09.

**Parallel safety and exclusive file ownership:** Own new `medfm/challenges/medreason/mcq_scoring.py` core scorer/tests. Serialize with MCQ-05/06/07; RUN-01/04 supply an atomic adapter session and call this scorer rather than duplicate it.

**Target paths/symbols:** `CandidateBatch`, `build_candidate_batch`, `conditional_token_log_probs`, `score_candidate_batch`; `tests/challenges/medreason/test_mcq_scoring.py`.

**Inputs:** Challenge-local prompt/image batch, candidate targets/masks, model next-token logits, atomic active-adapter scoring session from RUN-01.

**Outputs:** Per-candidate FP32 summed conditional likelihood, count, mean, display/original indices, and EVA-09 audit records.

**Implementation:**

1. Build identical prompts with differing assistant targets. Repeat/interleave every native processor field; never convert through `MedicalBatch`, `MultiImageVLCollator`, or existing generation pipeline.
2. Pad within one bucket and issue one candidate-batch forward. Deterministic measured chunks are allowed only with parity; never truncate.
3. Keep adapter selection and the entire forward inside RUN-01's atomic adapter session because existing `AdapterManager` locking ends before forward.
4. Apply causal shift (`t-1` logits score token `t`); gather only MCQ-03 target positions including terminal EOS.
5. Accumulate log-softmax FP32; reject zero/non-finite results. Never call `generate()`.
6. Persist sanitized IDs/sums/counts/hashes/scores, not content.

**Focused tests and exact commands:** Analytic logits catch unshifted gathers; batch-loop equality; rank-3 pixels/image positions/token types repetition; prompt/image exclusion; one scoring forward; concurrent fake adapter switches cannot change active adapter mid-forward; fake `generate()` raises. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_scoring.py -k 'conditional or batch or causal_shift or fields or atomic_adapter or no_generate'`.

**Acceptance evidence:** CPU fixture showing one forward, exact sums/counts, native shapes, atomic adapter identity. Real tests use common access/CUDA/memory guards; fixture is not 31B/26B evidence.

**Non-goals/failure policy:** No generation, beam/parser, prior, GRPO, private `_uniform_positions`, or non-atomic adapter switch. Missing fields/mask, overflow, or non-finite score fails.

**Handoff:** MCQ-05 normalizes/remaps; MCQ-06 expands orders; MCQ-11 mines; RUN-04 owns production scheduling around this scorer.

## MCQ-05 Length-normalize and map candidate scores back

**Depends on:** MCQ-02, MCQ-04.

**Parallel safety and exclusive file ownership:** Own `mcq_scoring.py::normalize_candidate_scores`, `map_scores_to_original_labels`, tests; serialize with adjacent scorer cards.

**Target paths/symbols:** `NormalizedCandidateScore`, `normalize_candidate_scores`, `map_scores_to_original_labels`, `select_original_label`; normalization cases in `test_mcq_scoring.py`.

**Inputs:** FP32 summed likelihoods/counts, `display_to_original`, exact supplied labels.

**Outputs:** Mean per original option, complete label-score map, deterministic exact supplied label, mapping audit.

**Implementation:**

1. Compute `sum_logp / supervised_target_token_count`; never padded/prompt/character length and never omit label/EOS.
2. Require every display/original index once; map by index, never text.
3. Attach exact original label; select greatest finite mean.
4. Resolve exact ties by lowest original index and record `tie=true`; never tune tie rule.
5. Persist sum/count/mean.

**Focused tests and exact commands:** Long/short example distinguishes raw sum from mean; nonidentity mapping/repeated text; duplicate/missing map, zero/NaN/Inf; tie determinism. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_scoring.py -k 'normalize or map_back or tie or invalid_score'`.

**Acceptance evidence:** Fixture JSON with indices/labels/sums/counts/means/winner; arithmetic only.

**Non-goals/failure policy:** No learned penalty, calibration, label prior, text mapping, generation, or GRPO.

**Handoff:** MCQ-06 averages original scores; RUN-04 consumes result.

## MCQ-06 Ensemble original cyclic and reverse option orders

**Depends on:** MCQ-02, MCQ-04, MCQ-05.

**Parallel safety and exclusive file ownership:** Own `inference_option_orders`, `build_three_order_candidate_batch`, `score_three_orders`; serialize `mcq_scoring.py`; coordinate RUN-04.

**Target paths/symbols:** Symbols above; `tests/challenges/medreason/test_mcq_order_ensemble.py`.

**Inputs:** Canonical options, challenge batch, atomic adapter session. Fixed mappings: original `(0..n-1)`, left cyclic `(1..n-1,0)`, reverse `(n-1..0)`.

**Outputs:** Exactly three named records, mapped normalized scores, arithmetic means, selected supplied label, mapping digest.

**Implementation:**

1. Generate exactly three bijections; no confidence search.
2. Re-label each order and concatenate all `3*n` candidates into one forward/order set inside one atomic adapter session (or deterministic parity-proven memory chunks), preserving native fields.
3. Normalize before mapping, require full original coverage, arithmetic-mean three scores per original option.
4. Retain all three even if orders coincide for small `n`.
5. Return exact supplied label; retain audit scores.

**Focused tests and exact commands:** Position-biased fake catches displayed-label averaging; assert `3*n`, one batched forward/order set, stable/chunk parity, arbitrary-label/repeated-text remap, tie determinism, atomic adapter identity, no label generation. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_order_ensemble.py`.

**Acceptance evidence:** Fixture order mappings/scores/counts/forward count. Protected latency/VRAM needs guarded measurement.

**Non-goals/failure policy:** No all-permutation/confidence weighting, parser/generation, private image-selection helper, or incomplete order.

**Handoff:** MCQ-07 wraps corrupt-image path; MCQ-13 follows thought; RUN-04 batches production.

## MCQ-07 Implement corrupt-image text-only candidate fallback

**Depends on:** DAT-06, SCH-07, MCQ-04, MCQ-05, MCQ-06, RUN-07.

**Parallel safety and exclusive file ownership:** Own `score_text_only_fallback` and MCQ hook in `inference.py`; serialize inference hook with RUN-02/07.

**Target paths/symbols:** `mcq_scoring.py::score_text_only_fallback`, `inference.py::predict_mcq`; `tests/challenges/medreason/test_mcq_fallback.py`.

**Inputs:** Runtime case with missing/corrupt image, question/options without answer, processor/model, sanitized decode error.

**Outputs:** Exact supplied label selected by text-only three-order conditional scoring; sanitized route/error telemetry and scores.

**Implementation:**

1. Catch only declared missing/decode-corrupt errors; do not relabel OOM/programming failure.
2. Build prompt without image content/placeholders, retaining question/options.
3. Use same formatter/scorer/normalization/remapping/ensemble inside atomic adapter session; never position prior.
4. Return `mcq_text_only_corrupt_image`; log only ID/error class.
5. If scoring fails, raise distinct sanitized error to RUN-07, not first label.

**Focused tests and exact commands:** Missing/corrupt fixtures return model-preferred non-first label; changing text changes winner at same positions; assert no image fields, three orders, one batched order-set forward, original-label remap/tie rule, no generation, privacy logs, OOM propagation. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_fallback.py`.

**Acceptance evidence:** Official-style synthetic result/route/scores/caplog; container acceptance belongs RUN/Docker.

**Non-goals/failure policy:** No imputation, first/majority prior, generation, or broad exception swallowing.

**Handoff:** RUN-07 calls it; EVA-10 consumes telemetry; MCQ-15 freezes enablement.

## MCQ-08 Add large-model learning-rate pilot configurations

**Depends on:** GOV-03, GOV-04, GOV-06, GOV-09, DAT-11, SPL-07, EVA-12, EVA-13, MOD-09, MOD-10, MOD-11, MOD-12, MOD-13, MOD-14, MCQ-01, MCQ-03, MCQ-06.

**Parallel safety and exclusive file ownership:** Own new `mcq_training.py` pilot specs/builder/tests. CLI-05 owns YAML; candidate directories disjoint.

**Target paths/symbols:** `MCQPilotSpec`, `large_model_pilot_specs`, `validate_mcq_pilot_pair`, `build_mcq_linear_scheduler`, `build_mcq_training_step`; `tests/challenges/medreason/test_mcq_pilots.py`; reuse trainer/backend/checkpoint/tracker/PEFT primitives, not phase-13 builders or `MedicalBatch`.

**Inputs:** Advanced exact revision, immutable splits/hashes, LoRA targets, NF4 hash, challenge processor/batch, model family.

**Outputs:** Paired 250-step specs: Gemma 4 `{2e-5,5e-5}`, MedGemma `{5e-5,1e-4}`, identical pair invariants, directories `artifacts/runs/medreason/mcq/pilots/<model>/<lr>/`.

**Implementation:**

1. Specify NF4 double quantization/BF16 compute-storage, microbatch 1, accumulation 16, LoRA 16/32/0.05, frozen vision, discovered language targets, checkpointing, no KV cache, fused AdamW, grad norm 0.3, seed 2026.
2. Implement linear decay with warmup ratio `0.03`; do not silently use existing generic cosine scheduler.
3. Vary only LR; require equal initialization/data/permutation/bucket/evaluation/targets and exactly 250 steps.
4. Use challenge-native batch/forward while reusing infrastructure; never `phase13_builders()`.
5. Require MOD-13 100-real-batch peak `< min(85 GiB, 0.90*device memory)` before protected pilots, plus env/local snapshot/CUDA/access/storage/hardware guards.
6. Record hashes, targets, loss/dev scores, allocated/reserved peak, failure class.

**Focused tests and exact commands:** Exact LR/steps/invariants; scheduler trace catches cosine; rank-3 native batch catches `MedicalBatch`; phase-13 rejection; fake SFT never generates/RL. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_pilots.py`. Protected, not default: `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train --config configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml --max-steps 250`.

**Acceptance evidence:** Fixture specs/scheduler tests. Protected acceptance requires both paired real runs and gate evidence; current machine is blocked.

**Non-goals/failure policy:** No full/vision tuning, unadvanced model, OOM mutation, validation-output tuning, default generation/GRPO, synthetic builder.

**Handoff:** MCQ-09 compares; CLI-05 serializes; OPS-10 executes after gates.

## MCQ-09 Select pilot winner with declared precedence

**Depends on:** MCQ-08, EVA-03, STR-05.

**Parallel safety and exclusive file ownership:** Own `select_pilot_winner`; read immutable pilots/write new summary; serialize with MCQ-08/10.

**Target paths/symbols:** `PilotMetrics`, `PilotSelection`, `select_pilot_winner`; `tests/challenges/medreason/test_mcq_pilot_selection.py`.

**Inputs:** Two comparable 250-step summaries with common dev count/denominator, available clean/OOD MCQ, dev loss, LR, invariant hashes; no lockbox/participant validation.

**Outputs:** `pilots/<model>/selection.json` with ordered keys, missing strata, selected hash or blocked reason.

**Implementation:**

1. Require equal invariants and complete steps.
2. Lexicographic precedence: highest exact dev MCQ; exact tie prefers clean/available-OOD non-inferiority; then lower dev loss; then lower LR.
3. Follow STR-05 availability; report absent evidence.
4. Keep full precision and first differing key; loss never overrules accuracy.
5. Hash selection for MCQ-10.

**Focused tests and exact commands:** Accuracy over loss; tie stages; close floats/different counts; wrong steps/hash/denominator/lockbox block; missing strata report. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_pilot_selection.py`.

**Acceptance evidence:** Golden fixture; real selection requires paired protected artifacts.

**Non-goals/failure policy:** No judge, lockbox, manual override, averaged precedence.

**Handoff:** MCQ-10 receives hash; MCQ-15 records it.

## MCQ-10 Train two epochs with fixed early stopping

**Depends on:** MCQ-02, MCQ-03, MCQ-09, MOD-11, MOD-12, MOD-13.

**Parallel safety and exclusive file ownership:** Own `MCQTrainingController` and fixed-evaluation/resume state; serialize `mcq_training.py`. Do not assume generic `eval_every_steps` works—it is parsed but unused.

**Target paths/symbols:** `MCQTrainingController`, `MCQEarlyStoppingState`, `run_selected_mcq_sft`; `tests/challenges/medreason/test_mcq_training.py`; reuse trainer primitives. Any generic callback hook is narrowly owned here with regressions.

**Inputs:** Selected pilot hash, train/dev, challenge batch/model, positive preregistered interval/patience, restored loader state, hardware evidence.

**Outputs:** At most two epochs, fixed-step dev history, resumable checkpoints, epoch-1 mining checkpoint/event, best step, deterministic stop reason.

**Implementation:**

1. Continue pilot only on matching optimizer/scheduler/data hashes; else declared fresh run.
2. Implement actual optimizer-boundary callbacks each fixed `eval_every_steps`; end-only validation is insufficient.
3. Persist/restore loader epoch/cursor/batch-sampler/permutation and evaluation cadence.
4. Best by dev MCQ, then loss, then earliest. Require epoch 1 and checkpoint before mining; stop only in epoch 2 after fixed patience; never third epoch.
5. Use challenge linear scheduler/assistant CE; export best, not last.
6. Protected run requires all preflight/memory gates.

**Focused tests and exact commands:** Exact eval steps/best/no pre-mining stop/no third epoch; resume equivalence includes sampler/permutation/eval/scheduler; missing interval/patience/stale hash block; no generation/GRPO. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_training.py -k 'two_epochs or interval or early_stop or resume or sft_only'`.

**Acceptance evidence:** Fixture histories/checkpoint hashes; protected acceptance requires real gate, epoch-1 checkpoint, interval history, best adapter, measured memory.

**Non-goals/failure policy:** No adaptive cadence, lockbox, pseudo-label, generation, default GRPO, OOM degradation, dropped sampler state.

**Handoff:** MCQ-11 mines epoch 1; MCQ-12 runs epoch 2; MCQ-15 consumes best metadata.

## MCQ-11 Mine hard negatives after first epoch

**Depends on:** MCQ-04, MCQ-05, MCQ-10.

**Parallel safety and exclusive file ownership:** Own `mine_hard_negatives`/records/tests; reads frozen epoch 1, separate artifact; serialize with MCQ-10/12.

**Target paths/symbols:** `HardNegativeRecord`, `mine_hard_negatives`, `load_hard_negative_manifest`; `tests/challenges/medreason/test_mcq_hard_negatives.py`.

**Inputs:** Epoch-1 adapter/base/processor hashes, train-only MCQs, mappings, all option scores.

**Outputs:** `hard_negatives_epoch1.jsonl` plus hash: ID, correct original index, up to two highest wrong original indices/scores/counts, mapping/checkpoint hashes; no text.

**Implementation:**

1. Deterministically score every train option with epoch-1 adapter in atomic session.
2. Exclude gold by original index; sort wrongs by score then original index.
3. Top two; fewer wrongs means available only and `contrastive_eligible=false`, never invent.
4. Serialize originals before epoch-2 permutation.
5. Reject non-train/stale/incomplete/non-finite inputs; hash ordered JSONL.

**Focused tests and exact commands:** Permuted labels map original negatives; gold excluded even highest; ties/repeated text/fewer options/stale hash/split leakage; complete scoring. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_hard_negatives.py`.

**Acceptance evidence:** Tiny JSONL/manifest; real acceptance requires complete count/hashes and no content.

**Non-goals/failure policy:** No generated/semantic negatives, external/proxy/dev/lockbox mining, random filling.

**Handoff:** MCQ-12 consumes indices/hash; MCQ-15 records it.

## MCQ-12 Add batched three-candidate contrastive objective

**Depends on:** MCQ-03, MCQ-04, MCQ-10, MCQ-11.

**Parallel safety and exclusive file ownership:** Own `MCQContrastiveTrainingStep`/loss tests; serialize training source; generic loop remains shared.

**Target paths/symbols:** `build_contrastive_candidate_batch`, `MCQContrastiveTrainingStep`, `contrastive_candidate_loss`; `tests/challenges/medreason/test_mcq_contrastive.py`; `TrainingStep`/`LossOutput` contracts.

**Inputs:** Epoch-2 supervised challenge batch, assistant labels, correct/two wrong originals, mapping, native fields.

**Outputs:** `L_total=L_assistant_ce+L_contrastive`; three-way CE over length-normalized scores from one forward; scalar/count diagnostics.

**Implementation:**

1. Resolve original identities into current display labels/text; assert mapping.
2. Stable `[correct, hardest_wrong, second_wrong]`, all native fields, one forward.
3. FP32 normalized scores, class zero CE, fixed weight 1.0 plus assistant CE; report components.
4. Validate epoch-1 hash; ineligible cases CE-only/count, no fabricated candidate.
5. Audit only LoRA gradients; reject duplicates/gold/stale/non-finite.

**Focused tests and exact commands:** Analytic losses/gradients catch raw sum/wrong class; new permutation preserves identities; one forward/native fields; frozen base/vision; invalid/ineligible. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_contrastive.py`.

**Acceptance evidence:** CPU analytic loss/audit; protected epoch 2 requires hard-negative hash/component history/adapter audit.

**Non-goals/failure policy:** No generated negatives, proxy preferences, dev mining, base tuning, generated scoring, default RL.

**Handoff:** MCQ-13/14 compare to clean parent; MCQ-15 records objective.

## MCQ-13 Gate bounded think-then-score inference candidate

**Depends on:** MCQ-06, MCQ-10, MCQ-12, SEL-02, SEL-03, SEL-04.

**Parallel safety and exclusive file ownership:** Own new `mcq_gates.py` think path/tests; coordinate inference hook with RUN-01/02/04; serialize with MCQ-14.

**Target paths/symbols:** `ThinkThenScoreConfig`, `run_think_then_score`, `decide_think_then_score`; `tests/challenges/medreason/test_mcq_think_gate.py`.

**Inputs:** Locked clean parent; one deterministic bounded Gemma 4 thought config; paired dev clean/OOD/latency; preregistered tolerances.

**Outputs:** Labels still from MCQ-06; private transient thought; immutable decision JSON.

**Implementation:**

1. Default direct scoring/thinking off. One optional greedy bounded thought, then unchanged three-order scoring in one atomic adapter session.
2. Enforce token/time bounds; never submit/persist/log thought; clear it.
3. Require `+0.3 pp` dev, OOD loss no worse `0.2 pp`, latency `<=1.5x`.
4. Also require 1,000 group-paired/Holm lower-bound promotion/non-inferiority; no lockbox.
5. Missing measured evidence rejects to direct parent.

**Focused tests and exact commands:** Thought alters logits but output supplied score-selected label; default zero generation; exact boundaries/stat failures; adapter atomicity; no thought in JSON/log/error; lockbox/missing latency block. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_think_gate.py`.

**Acceptance evidence:** Fixture gate/privacy; promotion needs immutable paired real predictions/measured hardware latency.

**Non-goals/failure policy:** Not default free-generation answer, submitted reasoning, agent loop, self-consistency. Rejected path does not ship.

**Handoff:** MCQ-15 consumes decision; RUN enables only accepted hash.

## MCQ-14 Gate optional MCQ consistency and GRPO candidates

**Depends on:** STR-01, STR-02, STR-03, STR-04, STR-05, MCQ-10, MCQ-12, SEL-02, SEL-03, SEL-04.

**Parallel safety and exclusive file ownership:** Own `mcq_gates.py` consistency/GRPO/tests; serialize with MCQ-13; isolated directories.

**Target paths/symbols:** `ConsistencyCandidateSpec`, `GRPOCandidateSpec`, `build_consistency_loss`, `grpo_reward`, `optional_mcq_candidates`, `decide_optional_mcq_candidate`; `tests/challenges/medreason/test_mcq_optional_gates.py`.

**Inputs:** Locked parent; paired clean/stress train/dev; remapped option distributions; plateau evidence; preregistered family; no pseudo/lockbox.

**Outputs:** Separate artifacts/decisions under `gates/{consistency_lambda_0.1,consistency_lambda_0.2,grpo}/`; all default disabled.

**Implementation:**

1. Only lambda `{0.1,0.2}`, 250 steps, 50% stress, `L_task+lambda*KL(p_clean||p_stress)` on correctly mapped distributions.
2. Keep only with weighted clean/OOD improvement, clean within `0.2 pp`, paired/Holm pass; report missing strata.
3. GRPO absent by default; one candidate only on recorded SFT plateau: LR `5e-6`, batch 3, accumulation 4, four generations, max completion 128, max 1,000 steps.
4. Reward `1.0` correct, `0.05` exactly one valid supplied label (wrong included), `-0.05` invalid/multiple. Deployment remains score based.
5. Test one at a time; delete rejected tensors, retain decision record.

**Focused tests and exact commands:** Analytic KL catches direction/remap; exact constants; reward correct/wrong-valid/invalid/multiple; default no RL/generation allocation; no plateau no GRPO; clean/OOD/Holm/lockbox failures. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_optional_gates.py`.

**Acceptance evidence:** CPU analytic/gate tests. Protected acceptance needs candidate hashes, paired predictions, intervals, measured runtime; unavailable locally.

**Non-goals/failure policy:** Optional only—no default GRPO/free generation, DPO, pseudo-preferences, hidden reward, unbounded completion, lockbox choice.

**Handoff:** MCQ-15 consumes decisions; SEL-01 survivors; OPS-12 execution.

## MCQ-15 Freeze accepted MCQ adapter and metadata

**Depends on:** MOD-11, MCQ-09, MCQ-10, MCQ-11, MCQ-12, MCQ-13, MCQ-14.

**Parallel safety and exclusive file ownership:** Own new `mcq_artifacts.py`, freeze tests, content-addressed destination. Do not overlap SEL-14. Never mutate runs/overwrite final.

**Target paths/symbols:** `MCQAdapterManifest`, `freeze_mcq_adapter`, `verify_frozen_mcq_adapter`; `tests/challenges/medreason/test_mcq_artifacts.py`; `save_adapter_checkpoint`, `load_checkpoint_manifest`; inventory/import coverage.

**Inputs:** Best checkpoint; exact base/processor/template/NF4/LoRA/data/split/permutation/pilot/negative/objective hashes; best history; scoring/fallback; decisions; measured references if present.

**Outputs:** Immutable `artifacts/models/medreason/mcq/<candidate_hash>/` with adapter-only safetensors, manifest, scoring profile, decisions, SHA-256 inventory; distinct research/all-label IDs.

**Implementation:**

1. Select best plus accepted optional only; rejected/missing resolves to clean direct scorer.
2. Export adapter-only; verify exact base, CPU safetensors, no base/optimizer tensors.
3. Record all provenance/training hashes and source run.
4. Freeze target/mask, normalized three orders, mapping/tie, atomic switch+score requirement, direct default, corrupt-image scoring, optional hashes. GRPO never changes score-based submission.
5. Stage, hash, reload, atomically rename content-addressed destination, refuse differing existing destination, apply immutability—fixing current overwrite/unlocked export.
6. Separate research/all-label artifacts and metrics; measured hardware fields only with linked evidence, else unavailable.
7. Cover intentional module inventory and CPU import safety; CUDA dependencies lazy.

**Focused tests and exact commands:** Tiny reload/hash; adapter-only/no optimizer/base; wrong revision; missing/mismatched hashes; rejected optional yields direct profile; accepted requires hash; atomic collision refuses overwrite; research/deployment separation; CPU import/inventory. Command: `uv run pytest -q tests/challenges/medreason/test_mcq_artifacts.py tests/challenges/medreason/test_package_inventory.py tests/phase_01/test_packaging.py`. Protected after gates: `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train --config configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml --export-adapter`.

**Acceptance evidence:** Fixture reloadable adapter, canonical hashes, collision protection, package tests. Protected acceptance additionally requires real safetensors/provenance/decisions/measured hardware; configs/fixtures/estimates are insufficient.

**Non-goals/failure policy:** No base weights, merged checkpoint, validation/lockbox tuning, private thoughts, rejected tensors, official/winning claim, overwrite, or unlocked final. Hash mismatch preserves source and blocks freeze.

**Handoff:** RUN-01 loads component hash; SEL-14 includes research hash; SEL-16/OPS-17 separate all-label artifact; DOC-04 vendors selected license-approved verified component.
