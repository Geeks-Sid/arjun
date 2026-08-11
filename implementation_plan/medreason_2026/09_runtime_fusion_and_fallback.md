# Runtime fusion and fallback

Phase root: `medfm/challenges/medreason/`; mirrored tests: `tests/challenges/medreason/`. Existing reusable anchors are `medfm.inference.generation::{GenerationConfig,generate,validate_json_output,require_valid_output}`, `medfm.inference.errors::{InferenceError,StructuredOutputError}`, `medfm.inference.server::AdapterManager`, `medfm.inference.schemas::InferenceLimits`, `medfm.inference.audit::AuditLogger`, `medfm.data.textprep.tokenize::{IGNORE_INDEX,build_supervised_example}`, `medfm.data.transforms.base::derive_seed`, and the endpoint-retaining pattern in `medfm.recipes.slice_selectors._uniform_positions`. The generic `VLMPipeline` and `MultiImageVLCollator` are not production substitutes: they do not preserve arbitrary native processor fields or enforce the complete 16,384-token processor length. `InferenceService` also returns `ok=False` and does not lock adapter activation plus the full forward, so the challenge predictor must wrap—not silently alter—it.

All CPU fixture commands below are intentionally unguarded and use tiny fakes. GPU tests carry `@pytest.mark.gpu` and require `MEDFM_RUN_GPU_TESTS=1`; real-checkpoint tests additionally carry `@pytest.mark.real_checkpoint` and require `MEDFM_RUN_REAL_CHECKPOINTS=1`. A protected command must set both variables, must find an exact local immutable checkpoint, and must skip rather than download. `tests/challenges/medreason/conftest.py` mirrors the hardware guard in `tests/phase_17/conftest.py`. RUN-01 adds `challenges` to `tests/phase_01/test_packaging.py::SUBPACKAGES`; all challenge modules remain CPU-importable, lazily import CUDA-only packages, and are covered by `test_no_forbidden_top_level_imports` and `test_cpu_import_does_not_initialize_cuda_or_xla`.

For every optional component, promotion uses only three-fold OOF predictions over train-plus-dev. The preregistered 1,000 group-paired bootstrap with Holm-Bonferroni family-wise `alpha=0.05` must have adjusted lower bound `>0` for the intended metric and greater than `-0.2 pp` MCQ, `-0.05` proxy GT, and `-0.05` proxy VA for every other metric. Every optional visual route also requires real-image proxy VA minus shuffled-image proxy VA `>=0.25`. All routes must meet the preregistered selected-hardware latency/VRAM ceiling; no absolute limit is invented before measurement. A failed optional component is removed from route/config/bundle manifests and vendored artifacts, and the core Gemma route must import and run without it.

## RUN-01 Load one base and switch task adapters

**Depends on:** GOV-03, MOD-01, MOD-11, MCQ-15, OPEN-15, SCH-01, SCH-05.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/inference.py::{TaskAdapterRouter,AdapterBinding}`, `tests/challenges/medreason/test_predictor.py` adapter-switch nodes, `tests/challenges/medreason/conftest.py`, and the `challenges` entry in `tests/phase_01/test_packaging.py::SUBPACKAGES`. RUN-03/04/05/08/09/14 may proceed concurrently in different files; serialize with RUN-02/07/15 and any task editing those shared tests.

**Target, inputs, outputs:** Input one immutable MOD-01 base and MCQ/open adapter-only safetensors manifests/hashes. `TaskAdapterRouter.activate(task_type)` outputs the active `AdapterBinding` while retaining the same base object. Support native HF `set_adapter` and the reviewed custom activation path; never accept a request-supplied path.

**Implementation:** (1) verify base revision and adapter hashes before allocation; (2) register an explicit task-enum-to-adapter map; (3) hold one re-entrant lock across activation and the complete processor/forward/decode transaction exposed as `router.run_with_adapter`; (4) make repeated activation idempotent and clear stale caches; (5) load no specialist here and never use Phase-13 fixture builders.

**Tests:** Tiny fake asserts one base load, alternating MCQ/open activation, identical base identity, hash/base mismatch before load, idempotence, and two-thread non-interleaving. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_predictor.py::test_one_base_switches_adapters_without_interleaving -q`. Packaging command: `uv run --frozen pytest tests/phase_01/test_packaging.py -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_predictor.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Record cold load separately from switch-plus-forward p95 and peak VRAM; fixture timing is not hardware evidence. Adapter task dispatch has no fitted threshold. Preserve no-image/shuffled route identity; this task cannot waive the visual grounding gate. Any cache/co-residence choice is OOF-only and is removed if it misses quality or resource gates.

**Acceptance evidence:** Fixture output proves one base and serialized switching; packaging output proves CPU-safe inventory/imports. Protected JSON must name exact base/adapter hashes, hardware, switch latency, and peak VRAM. Fixture acceptance does not prove 48/96 GB support.

**Non-goals and failure policy:** No downloads, duplicate bases, arbitrary adapters, model training, or license claim. Core adapter mismatch blocks freeze; failed optional cache/co-residence code and artifacts are removed.

**Handoff:** RUN-02 receives the locked `run_with_adapter` interface and immutable binding identifiers; deployment receives CPU import proof and, later, protected measurements.

## RUN-02 Dispatch deterministic per-case prediction routes

**Depends on:** RUN-01, SCH-02, SCH-04, DAT-06, MCQ-07, OPEN-04.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/inference.py::{MedReasonPredictor,predict_case,_dispatch}` and dispatch nodes in `test_predictor.py`. Serialize with RUN-01/07/15; other component files may proceed against the declared callable interfaces.

**Target, inputs, outputs:** Input a normalized runtime case, frozen route manifest, router, `score_mcq_orders`, and `generate_open`; output one `MedReasonPrediction` for the same ID plus sanitized diagnostics. The only route key is normalized `task_type`, never question/options/filename/model output.

**Implementation:** (1) validate ID/task/options before model work while preserving broken image references; (2) decode inside the per-case boundary; (3) activate the exact task adapter and execute deterministic MCQ or greedy open; (4) invoke only optional routes enabled by hash-verified frozen manifest; (5) map recoverable failures to RUN-07 and reserve process failure for corrupt manifests/invariants.

**Tests:** Interleaved task fixtures, unknown task before model call, optional-disabled spies, corrupt middle case with healthy neighbors, and byte-identical repeated dispatch. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_predictor.py::test_dispatch_is_task_typed_deterministic_and_case_local -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_predictor.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Collect per-route p95/VRAM using privacy-safe diagnostics. Any confidence/routing threshold is loaded only from three-fold OOF artifacts; core task dispatch is threshold-free. Maintain real/no-image/shuffled tags and reject optional routes below the `0.25` grounding gap or resource ceiling.

**Acceptance evidence:** Fixture call trace proves deterministic route order and one prediction after recoverable errors. Protected evidence adds exact artifact hashes and measured end-to-end route costs; fixtures do not establish checkpoint or hardware acceptance.

**Non-goals and failure policy:** No request-driven feature flag, runtime judge, network, retry loop, or heuristic task inference. Failed optional callables are deleted from manifest/import graph; core dispatch failure blocks release.

**Handoff:** RUN-07 receives the exception boundary, RUN-08 the sanitized event inputs, and RUN-15 a total per-case contract.

## RUN-03 Preserve image order and bound visual length

**Depends on:** SCH-02, DAT-06, MOD-02, MOD-03, MOD-12.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/multimodal.py::{PreparedInputs,prepare_multimodal_inputs,select_image_indices}` and `test_multimodal.py`; serialize with RUN-09 if it extends these types. Consumers do not edit this file.

**Target, inputs, outputs:** Input ordered `image_paths`, decoded images, native processor/chat template, and case text. Output `PreparedInputs(model_inputs,selected_image_indices,token_count)` preserving every processor key (`input_ids`, masks, pixel tensors, token-type/image-position/grid fields when emitted).

**Implementation:** (1) process all images in input order and measure complete processor length; (2) retain all when `<=16,384`; (3) otherwise select deterministic approximately uniform indices retaining first/last, using `_uniform_positions` only as a pattern and remeasuring actual length; (4) preserve ascending original order and all native fields; (5) reject an unsatisfiable single-image/text case rather than truncate, and log counts/indices only.

**Tests:** Sentinel processor fields, row/image alignment, zero/one/two/many images, exactly 16,384/one over, endpoint retention, deterministic uniform subset, and impossible length. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_multimodal.py::test_prepare_preserves_native_fields_and_bounds_complete_length -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_multimodal.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** The 16,384 cap/endpoint rule is fixed. Alternative selectors are OOF-only optional candidates and need corrected quality, `>=0.25` grounding gap, and selected-hardware latency/VRAM. Fixture token counts are not real-processor evidence.

**Acceptance evidence:** Fixture `PreparedInputs` proves no silent truncation or dropped keys. Protected report records exact processor hash, lengths, selected indices, p95, and VRAM; it does not imply volumetric reasoning or 48/96 GB support.

**Non-goals and failure policy:** No modality guessing, saliency model, path logging, or 3D claim. Remove failed learned selectors and retain the fixed safe selector.

**Handoff:** RUN-04/05/09 receive ordered native model inputs and bounded selection diagnostics.

## RUN-04 Batch MCQ candidates across all orderings

**Depends on:** RUN-01, RUN-03, MCQ-04, MCQ-05, MCQ-06, MCQ-15.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/scoring.py::{MCQScoreResult,conditional_log_likelihood,score_mcq_orders}` and MCQ nodes in `test_mcq_scoring.py`; no overlap with MCQ scoring interface writers.

**Target, inputs, outputs:** Input original labels/text, native prepared inputs, frozen MCQ adapter, and original/cyclic/reverse permutations. Output mapped finite per-label scores, exact supplied winning label, deterministic tie metadata, and offline logits.

**Implementation:** (1) form every `<label>: <option text>` candidate for all three bijections; (2) create one `3N` candidate batch when the frozen memory profile permits, preserving prompt and assistant target masks using `build_supervised_example`/`IGNORE_INDEX`; (3) compute length-normalized conditional log-likelihood, excluding prompt/pad, map to original labels, and average; (4) tie-break by original option order; (5) delegate corrupt images to MCQ-07 text-only conditional scoring, never a position prior or free generation.

**Tests:** Assert `3N` in one forward, masking/EOS, length normalization, remap, Unicode/non-contiguous labels, stable tie, chunk equivalence, and text-only fallback. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_mcq_scoring.py::test_three_order_candidates_score_in_one_mapped_batch -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_mcq_scoring.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Equal order averaging is parent. Temperature/non-negative weights are OOF-only. Think-then-score needs `>=0.3 pp` gain, OOD loss `<=0.2 pp`, and `<=1.5x` latency plus overall resource gate. Preserve visual controls; MCQ accuracy alone is not grounding evidence.

**Acceptance evidence:** Fixtures prove each candidate exactly once and exact-label output; protected evidence records artifact hashes, p95/VRAM, scores, and controls. Fixture success is not quality acceptance.

**Non-goals and failure policy:** No generated submitted label/thought or learned tie-break. Remove failed thinking/weighted variants; direct equal-weight scoring remains.

**Handoff:** RUN-02 gets a deterministic scorer; EVA/SEL get mapped scores and route hashes.

## RUN-05 Parse structured open responses deterministically

**Depends on:** RUN-01, RUN-03, OPEN-03, OPEN-04, OPEN-15, SCH-04.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/open.py::{generate_open,parse_open,export_open}` and core nodes in `test_open_output.py`; serialize with RUN-06.

**Target, inputs, outputs:** Use `schemas.py::{OPEN_RESPONSE_SCHEMA,OpenStructuredResponse}` with observations, reasoning, answer. Input greedy model output/tokenizer; output strict parsed response and official non-empty `reasoning_trace`/`answer`, capped at 160/48 generated tokens, with private thought absent.

**Implementation:** (1) call bounded `generate` with `do_sample=false`; decode completion only, not prompt; (2) parse exactly one JSON object via `validate_json_output`; (3) reject wrong/empty fields, trailing objects, NaN, wrappers, and thought channels; (4) deterministically join ordered observations then reasoning; (5) cap at tokenizer boundaries and revalidate official fields.

**Tests:** Valid Unicode, key order, wrong types, empty answer/trace, prompt-prefix decoding, trailing JSON, private thought, caps, deterministic bytes. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_open_output.py::test_open_parse_and_export_are_strict_bounded_and_thought_free -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_open_output.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Parser/schema/caps freeze before OOF and are not lockbox-tuned. Optional thinking/four-sample routes must improve both GT and VA under corrected OOF, meet grounding and p95/VRAM gates. Greedy fixture timing is not hardware proof.

**Acceptance evidence:** Fixture output proves strict bounded schema and no thought leakage; protected report proves real tokenizer behavior/costs, not official judge quality.

**Non-goals and failure policy:** No pseudo-trace, semantic repair, runtime judge, or default sampling. Remove failed thinking/self-consistency artifacts and retain greedy single sample.

**Handoff:** RUN-06 gets typed parse failures; RUN-11 gets ordered observations; RUN-15 gets official fields.

## RUN-06 Permit exactly one deterministic format repair

**Depends on:** RUN-05, SCH-07.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/open.py::{repair_once,parse_with_one_repair}` and repair nodes in `test_open_output.py`; serialize with RUN-05.

**Target, inputs, outputs:** Input one bounded raw generation and strict error category. Output valid `OpenStructuredResponse` with `repair_count=0|1`, or `StructuredOutputError`; never another generation.

**Implementation:** (1) strict parse first; (2) only for allowlisted syntax, remove one fence/extract one balanced object/normalize declared punctuation without changing string values; (3) run the same validator once; (4) reject missing/empty/type/private-thought/multiple-object cases; (5) enforce the one-call invariant in state/API.

**Tests:** Fence, balanced extraction, value preservation, malformed escape, missing answer, malicious wrapper, two objects, and validator spy proving at most one repair. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_open_output.py::test_format_repair_is_value_preserving_and_runs_once -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_open_output.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Repair adds no model call; allowlist freezes before OOF. Any repair-rate routing threshold is OOF-only. Measure overhead; repaired strings retain control identity and cannot manufacture grounding or bypass `0.25`.

**Acceptance evidence:** Fixtures prove unchanged values, bounded count, and static error; protected evidence reports repair frequency/overhead separately from grounding.

**Non-goals and failure policy:** No semantic completion, repeat generation, or model/judge repair. If repair harms corrected metrics/resources, remove it and use strict fallback.

**Handoff:** RUN-07 receives terminal failures; EVA receives only repair count/error class, not raw generation.

## RUN-07 Emit schema-valid corrupt-image fallback predictions

**Depends on:** RUN-02, RUN-04, RUN-06, DAT-06, MCQ-07, SCH-04, SCH-07.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/inference.py::{fallback_prediction,_recoverable_error}` and fallback nodes in `test_predictor.py`; serialize with RUN-01/02/15.

**Target, inputs, outputs:** Input validated case, sanitized error enum, supplied options, optional text-only scorer. Output same-ID schema-valid prediction plus non-submission fallback class.

**Implementation:** (1) allowlist missing/decode/processor/structured/backend failures and discard raw messages; (2) MCQ uses text-only conditional option score; if unavailable use only a preregistered deterministic supplied-option catastrophic policy, never learned prior/randomness; (3) open returns fixed non-empty processing-failure trace/answer making no diagnosis; (4) validate fallback with official schema; (5) continue subsequent cases.

**Tests:** Missing/corrupt image, secret-bearing decoder exception, processor overflow, invalid JSON after repair, scorer failure, Unicode IDs/options, consecutive failures. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_predictor.py::test_corrupt_cases_emit_valid_private_fallbacks_and_continue -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_predictor.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Fallback is deterministic; proactive confidence thresholds, if tested, are OOF-only. Include fallback cases in coverage/metrics and record latency. Open fallback makes no grounding claim and is not dropped from shuffled controls.

**Acceptance evidence:** Fixtures prove one valid output and no leaked secret for each failure; protected evidence measures real decoder/text-scorer costs. No quality/hardware claim follows from fixtures.

**Non-goals and failure policy:** No fabricated diagnosis, random label, raw exception, skipped case, or retries. Core fallback failure blocks freeze; rejected proactive fallback logic is removed.

**Handoff:** RUN-08 gets safe classes; RUN-15 gets per-case totality.

## RUN-08 Sanitize runtime errors logs and telemetry

**Depends on:** SCH-07, EVA-10, RUN-02.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/telemetry.py::{RuntimeEvent,RuntimeTelemetry,sanitize_exception}` and `test_privacy.py`; callers pass only typed allowlisted values.

**Target, inputs, outputs:** Input case ID, static route/component/error IDs, finite duration/VRAM/counts, artifact hashes. Output deterministic JSONL outside results. Challenge logs contain only case ID plus sanitized error class; richer offline diagnostic fields must remain hashes/bounded numerics—never question, answer, options, trace, metadata, prompt, path, output, raw exception, or traceback.

**Implementation:** (1) allowlist serialize rather than recursively redact; (2) bound/control-check IDs; (3) map unknown errors without `str/repr`; (4) omit unavailable/NaN measurements rather than estimate; (5) secret-scan JSONL/stdout/stderr.

**Tests:** Plant secrets in every forbidden field/nested exception, test control IDs, unknown errors, NaN, exact key order, and captured streams. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_privacy.py::test_runtime_logs_allowlist_only_safe_case_events -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_privacy.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Count logging in end-to-end p95; no sampling threshold may hide failures or be label-tuned. Retain only control IDs needed to audit OOF provenance and shuffled grounding; logging cannot change the `0.25` gate.

**Acceptance evidence:** Fixture secret scan proves code-level allowlist; protected complete-run stream scan and overhead prove deployed acceptance separately.

**Non-goals and failure policy:** No raw payload hash dump, clinical store, remote telemetry, or verbose profiling. Logging failure blocks release audit but not case output; leaking/slow optional profiler is removed.

**Handoff:** EVA/SEL receive safe timings/errors/hashes; Docker receives stream-scan contract.

## RUN-09 Add optional contrast views and labeled crops

**Depends on:** RUN-03, STR-06, SCH-05.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/views.py::{ViewSet,make_contrast_view,make_labeled_crops}` and view-builder nodes in `test_optional_routes.py`; serialize with RUN-10.

**Target, inputs, outputs:** Input selected decoded images; output originals first, one deterministic contrast-normalized view, and labeled top-left/top-right/bottom-left/bottom-right 2x2 crops with preserved source order and remeasured processor length.

**Implementation:** (1) never mutate original; (2) finite deterministic normalization including constant image; (3) complete non-overlapping grid with deterministic odd boundaries; (4) label every crop in prompt; (5) reapply RUN-03 bounds and hide all work behind disabled-by-default capability.

**Tests:** Pixel coverage, odd sizes, order/labels, constant image, no mutation, overflow, and disabled zero-call. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_extra_views_are_deterministic_labeled_bounded_and_optional -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Promote only corrected OOF gain/non-inferiority, `>=0.25` real-minus-shuffled VA, and full-route p95/VRAM; no lockbox transform tuning. Fixtures prove geometry only.

**Acceptance evidence:** Fixture arrays/indices prove deterministic views; protected per-stratum metrics, controls, processor lengths, costs, and hashes prove candidate acceptance.

**Non-goals and failure policy:** No flips, saliency, modality guessing, or unconditional views. On failure delete builder route/config/artifacts and verify no optional import.

**Handoff:** RUN-10 receives deterministic views/costs; SEL receives paired route predictions.

## RUN-10 Gate extra views using OOF confidence

**Depends on:** RUN-09, SEL-01, SCH-05.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/views.py::{ViewPolicy,should_add_views}` and gate nodes in `test_optional_routes.py`; serialize with RUN-09. SEL-10 owns fitting artifacts, not runtime policy code.

**Target, inputs, outputs:** Input original-route calibrated confidence/disagreement and immutable OOF policy; output boolean and static reason. Original runs first; views run once only on disagreement or confidence strictly below threshold.

**Implementation:** (1) validate artifact names three development folds and excludes lockbox/participant-validation; (2) apply frozen calibration/threshold with explicit equality; (3) disagreement overrides confidence; (4) reject NaN/missing provenance; (5) disabled policy neither imports nor builds views.

**Tests:** Below/equal/above, disagreement, NaN, bad provenance, disabled spy, repeat determinism. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_view_gate_uses_only_frozen_oof_policy -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Threshold/temperature are OOF-only. Count original plus optional rerun in p95/VRAM and require corrected metrics and `>=0.25` grounding gap. Synthetic fixture thresholds never become production defaults.

**Acceptance evidence:** Fixture proves boundary/provenance/zero-work behavior; protected artifact hash, trigger rate, metrics, grounding, and costs prove route acceptance.

**Non-goals and failure policy:** No online tuning, judge, keyword/modality rule, or always-on views. Failed route removes policy/threshold/view artifacts entirely.

**Handoff:** RUN-02 gets a pure policy; SEL-10 later supplies the frozen artifact.

## RUN-11 Atomize independent Gemma and MedGemma observations

**Depends on:** RUN-05, MOD-01, MOD-06, OPEN-15, GOV-04.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/fusion.py::{AtomicClaim,OpenCandidate,atomize_observations}` and atomization nodes in `test_optional_routes.py`; serialize with RUN-12/13.

**Target, inputs, outputs:** Input two independently generated valid structured responses and hashes. Output ordered atomic claims retaining source model/observation/claim indices and exact normalized text. MedGemma is lazy and optional.

**Implementation:** (1) generate both from the same ordered images before cross-exposure; (2) split with a frozen deterministic syntax grammar only; (3) preserve negation/order/provenance, drop empties, reject unsafe splits; (4) exact-deduplicate only within candidate; (5) Gemma-only startup succeeds without specialist files.

**Tests:** Punctuation/conjunction/negation/Unicode, duplicates, provenance, independence spies, specialist absent. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_candidates_are_independent_and_claim_atomization_preserves_provenance -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Grammar freezes before OOF. Specialist requires corrected OOF gates, `>=0.25` grounding, accepted terms, and measured co-resident/sequential p95/VRAM. Fixtures prove no medical truth.

**Acceptance evidence:** Fixture claims prove deterministic provenance and Gemma-only operation; protected hashes/license evidence/controls/costs prove specialist eligibility separately.

**Non-goals and failure policy:** No LLM atomizer, debate, paraphrase, or fabricated claim. Failed/unredistributable specialist removes model, fusion config, route, and imports.

**Handoff:** RUN-12 gets atomic claims; RUN-13 gets intact candidates.

## RUN-12 Score cross-model atomic visual support

**Depends on:** RUN-11, MOD-03, SEL-01.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/fusion.py::{SupportScore,score_cross_support}` and support nodes in `test_optional_routes.py`; serialize with RUN-11/13.

**Target, inputs, outputs:** Each model scores the other candidate's atomic claims against ordered images; output calibrated finite supported probabilities and their log-space geometric mean with artifact provenance. No proxy judge ships.

**Implementation:** (1) bounded fixed support prompt/batch preserving native processor fields; (2) normalize supported/unsupported likelihood and apply only OOF temperature; (3) validate `[0,1]`, use epsilon only for numerical stability; (4) require both cross-model scores, never sole self-support; (5) invalid/unavailable score disables specialist for the case without confidence imputation.

**Tests:** Calibration, 0/1, underflow, NaN/inf, cross-source enforcement, ordering, missing model, no judge import. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_cross_support_is_calibrated_geometric_and_fail_closed -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Temperatures/support threshold are OOF-only; dual-model claim scoring must pass corrected metrics, `>=0.25` grounding, and full p95/VRAM. Co-residence is never inferred from fixtures.

**Acceptance evidence:** Fixtures prove numeric/source invariants and primary fallback; protected fold provenance, distributions, controls, and hardware measurements prove acceptance.

**Non-goals and failure policy:** No lexical proxy, external API, judge, or imputation. Failure removes scorer/calibration/specialist artifacts and runtime references.

**Handoff:** RUN-13 gets ordered claims/support; SEL-09/10 fit immutable weights/thresholds.

## RUN-13 Fuse supported claims with answer likelihood

**Depends on:** RUN-11, RUN-12, OPEN-04, SCH-05.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/fusion.py::{FusionCalibration,score_candidate,fuse_open_candidates}` and fusion nodes in `test_optional_routes.py`; serialize with RUN-11/12.

**Target, inputs, outputs:** Input valid/repaired candidates, support scores, length-normalized answer likelihood, and non-negative OOF weights/threshold. Output one schema-valid bounded open prediction, chosen source, retained claim provenance, and offline score components.

**Implementation:** (1) validate finite non-negative weights and three-fold provenance; (2) remove claims below threshold with explicit equality, preserving order; (3) rebuild without paraphrase/new generation and allow only RUN-06's single repair; (4) combine aggregate log support and answer likelihood; (5) choose higher valid score, tie to primary Gemma, and fall back primary then RUN-07.

**Tests:** Threshold boundary, bad/negative weights/provenance, all claims removed, invalid candidates, repair count, ties, order, primary fallback. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_fusion_uses_nonnegative_oof_weights_and_primary_fallback -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** All temperatures/weights/thresholds are OOF-only; corrected metrics, `>=0.25` grounding, and full-route p95/VRAM are mandatory. Fixture arithmetic is not quality evidence.

**Acceptance evidence:** Fixture proves deterministic valid choice/fallback; protected OOF hashes, intervals, controls, license status, and costs prove promotion.

**Non-goals and failure policy:** No negative weights, debate, judge, semantic rewrite, or lockbox refit. Failure removes fusion/calibration/specialist files and ships primary only.

**Handoff:** RUN-02 gets optional pure fusion callable; RUN-15 gets one bounded result.

## RUN-14 Add fold-local contamination-safe retrieval candidate

**Depends on:** SPL-03, SPL-07, SPL-08, RUN-05, SCH-05.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/retrieval.py::{RetrievalHit,RetrievalIndex,build_fold_index}` and retrieval nodes in `test_optional_routes.py`; split manifests remain read-only.

**Target, inputs, outputs:** Input group/fold manifests, decoded/perceptual hashes, approved immutable BiomedCLIP artifact, train-only examples. Output separate hashed fold-local index and at most three ordered safe hits; disabled by default.

**Implementation:** (1) index only current fold training cases; (2) reject held-out/group/source overlap and exact/near perceptual duplicates before ranking; (3) stable similarity with case-ID hash tie-break and `k=3`; (4) never generate annotations and never index participant validation/lockbox; (5) no retriever/index load when disabled.

**Tests:** Cross-fold leakage, exact/near-image and group exclusions, stable ties, fewer than three, prohibited split, disabled spy. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py::test_retrieval_is_fold_local_deduplicated_top_three_and_optional -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_optional_routes.py -m 'gpu and real_checkpoint' -q`.

**Latency, grounding, OOF gate:** Retrieval ships only if every official metric is non-inferior and one materially improves under corrected OOF; it must retain `>=0.25` grounding, 16,384 length, and p95/VRAM. Fixture indices are not real contamination evidence.

**Acceptance evidence:** Fixture manifests prove locality/exclusions; protected retriever/index hashes, zero-overlap reports, paired metrics, controls, and costs prove acceptance.

**Non-goals and failure policy:** No global OOF index, reverse article match, network, pseudo-label/trace, or hidden data. Any contamination/license/metric/resource failure deletes retriever/index/config/manifest references.

**Handoff:** RUN-02 gets only enabled frozen hits; SEL gets per-fold index hashes and OOF predictions.

## RUN-15 Emit one ordered result per input case

**Depends on:** RUN-02, RUN-04, RUN-06, RUN-07, RUN-08, RUN-10, RUN-13, RUN-14, SCH-04, SCH-08, EVA-02.

**Parallel safety and exclusive file ownership:** Own `medfm/challenges/medreason/inference.py::{MedReasonPredictor.predict_all,write_results}` and final-output nodes in `test_predictor.py`; serialize with RUN-01/02/07. Vendored official schema/evaluator remains EVA-owned and byte-for-byte unchanged.

**Target, inputs, outputs:** Input ordered `/input/cases.json` parsed by official-compatible wrapper and frozen predictor. Output atomic `/output/results.json` official v1.0 payload (`name`, `type`, `answers`, `version`) with exactly one same-order record per ID; MCQ exact supplied label, open non-empty bounded trace/answer.

**Implementation:** (1) reject duplicate input IDs before inference; (2) append each success/fallback without sorting; (3) validate ID/task fields and final count/order/set; (4) serialize with SCH-08 canonical JSON and atomic replace, never partial final output; (5) keep telemetry outside payload and ensure removed optional components have no final manifest/bundle files.

**Tests:** Mixed tasks, Unicode/non-sortable IDs, failed middle case, duplicate ID, missing/extra/swapped output, invalid label, empty open fields, atomic-write fault, byte-identical repeats. Exact CPU command: `uv run --frozen pytest tests/challenges/medreason/test_predictor.py::test_predict_all_emits_exactly_one_ordered_atomic_result_per_case -q`. Protected command: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest tests/challenges/medreason/test_predictor.py -m 'gpu and real_checkpoint' -q`; later Docker tasks separately run official two-case and participant-package validators.

**Latency, grounding, OOF gate:** Assembly has no fitted threshold and cannot change predictions. End-to-end p95/VRAM includes fallbacks, optional routes, logging, and serialization. Preserve external control/route hashes so OOF-only provenance and `>=0.25` grounding are auditable; lockbox/participant output cannot alter routing or bytes.

**Acceptance evidence:** Fixture output proves exact count/order/schema, atomicity, and duplicate bytes. Protected evidence separately records all input IDs, official validator result, artifact hashes, complete-run latency/VRAM, duplicate bytes, and privacy scan. Fixtures do not prove official judge or hardware acceptance.

**Non-goals and failure policy:** No sorting, dropped case, partial final file, post-hoc judge, network, or hidden-data tuning. Core output failure blocks Docker freeze. Every rejected optional component must be absent from manifest/image while core Gemma plus mandatory fallbacks still completes.

**Handoff:** EVA-02 and Docker receive deterministic results, count/order invariants, privacy-safe telemetry location, exact enabled-component hashes, packaging inventory proof, and explicit fixture-versus-protected status.
