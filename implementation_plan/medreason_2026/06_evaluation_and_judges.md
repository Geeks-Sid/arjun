# Evaluation and judges

Authority: approved MedReason plan and official Docker commit `05748c0341b72dc08132bd108208b78dc14a2f0b`. The public scorer only aggregates organizer-provided case scores; it contains no judge prompts or calls. Therefore the Llama and Qwen processes below are **local proxies**, never official judges. Every fixture test is CPU-only/offline and uses synthetic `tests/challenges/medreason/fixtures/evaluation/` cases covering exact MCQ labels and all RVF caps. Real 70B/72B work is separately guarded by `MEDFM_RUN_REAL_JUDGES=1`, accepted licenses, immutable artifacts, CUDA/storage/memory preflight, and exclusive GPU ownership. The measured local RTX 3090 (24,576 MiB) and 364 GiB free disk support code/fixture acceptance only, not real-judge or 48/96 GB acceptance.

Reuse `medfm.core.serialization.canonical_json`/`config_hash`, `medfm.training.run_metadata`, redaction patterns from `medfm.training.tracking`, and `medfm.evaluation` artifact/report primitives. Do **not** reuse its case-folding generation exact match as official MCQ, unpaired binary-only bootstrap as a paired selector, histogram calibration as score calibration, or loss-level ablation as a persisted-prediction control. Restricted raw artifacts may contain evaluation payloads; telemetry and console output may contain only case ID, stage, numeric timing/memory, hashes, process state, and sanitized error class.

## EVA-01 Vendor official evaluator files byte-for-byte

**Depends on:** GOV-07, GOV-08.

**Parallel safety and exclusive file ownership:** Run after GOV-07 freezes acquisition inputs. EVA-01 exclusively owns evaluator bytes under `docker/medreason/vendor/MedReason-Evaluation/`, their entries in `docker/medreason/vendor/SHA256SUMS.json`, `tests/challenges/medreason/test_official_vendor.py`, and evaluator inventory assertions in `tests/challenges/medreason/test_package_inventory.py`. Docker tasks consume these files read-only; nobody reformats or normalizes them.

**Target paths/symbols:** Verbatim `scoring.py`, `validate_submission.py`, `docs/{evaluation.md,submission_format.md}`, and `test/{cases.json,ground_truth.json,judge_scores.json,metrics.json}`. Add only an out-of-tree resource locator/hash verifier under `medfm/challenges/medreason/official_source.py`.

**Inputs:** GOV-07's commit-pinned source. Cross-check upstream Git blob OIDs: scorer `cb8b249ad2a599c3ae370e5974271d565f444ad0`, validator `d18d069a77090564d267ce9633c5b0d108dc2e1b`, docs `f150a94fc22fddd50e445ca19d296dbbee002dfc`/`23ce20748336395e31ccc6788e3514b54ff3265e`, fixtures `215f0db355e6f22458fb1089cd93f28874e3eb8a`, `8e3e8019b0f66cf0e6ff1d5131e274281076a8a9`, `b6397a442e71aa513e9db0887f1d773f6a9f6671`, `25ecb38082cb7bfd8877ff04c3f98f09d60fdd41`. Git OIDs are not SHA-256 substitutes; record complete measured SHA-256 values (the independently reported scorer prefix is `b260a6fb`).

**Outputs:** Byte-identical tree plus sorted manifest fields `commit`, `upstream_path`, `git_blob_oid`, `bytes`, `sha256`, `vendored_path`.

**Ordered implementation:**
1. Copy bytes only from the immutable commit artifact; verify Git object ID, then SHA-256 after copy and at each load.
2. Keep challenge-owned toy rubric fixtures outside the vendor tree.
3. Never import vendor code via global `sys.path` mutation or edit it to satisfy style tools.
4. Inventory every file in the Docker build context/package; editable-worktree presence is not packaging proof.
5. Verification is CPU-only and starts no judge; log relative path and sanitized hash error only.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_official_vendor.py`; `uv run pytest -q tests/challenges/medreason/test_package_inventory.py -k official_evaluation`. The upstream two-case fixture must reproduce MCQ `1.0`, GT `2.0`, VA `3.0`, counts `2/1/1`, and ordered per-case values. Test one-byte mutation, CRLF conversion, wrong commit, omitted package file, process/CUDA isolation, and privacy.

**Acceptance evidence:** Complete hash manifest, toy parity record, and packaging inventory. This is fixture/source acceptance, not proxy access, hardware fit, or organizer-judge equivalence.

**Non-goals and failure policy:** No vendor edits, inferred prompts, or model calls. Any mismatch blocks official-parity claims. “Official” applies only to the public aggregator.

**Handoff:** EVA-02–04 consume tree/manifest hash; Docker release consumes the exact inventory.

## EVA-02 Wrap official case serializer and prediction parser

**Depends on:** EVA-01, SCH-02, SCH-04, SCH-08.

**Parallel safety and exclusive file ownership:** May run with judge work. Exclusively owns `medfm/challenges/medreason/official_io.py` and `tests/challenges/medreason/test_official_io.py`; does not edit runtime serializers or vendor bytes.

**Target paths/symbols:** `OfficialSubmissionSerializer`, `OfficialPredictionParser`, `OfficialGroundTruthParser`, `OfficialJudgeScoreParser`, `OfficialSchemaError`.

**Inputs:** Version-1.0 normalized cases/predictions, vendor parser/validator, synthetic exact-label and rubric fixtures.

**Outputs:** Canonical `{name,type,answers,version:{major:1,minor:0}}`, typed records, and wrapper/vendor parity matrix.

**Ordered implementation:**
1. Serialize exactly one input-ordered record per ID; MCQ uses one supplied label; open has non-empty trace and answer.
2. Run schema-first validation before immutable scorer calls. Reject JSON null/non-string required fields, NaN/Infinity, non-discrete or out-of-range judge values, duplicates, missing/extra cases, and task mismatches.
3. Document raw scorer quirks in parity tests: it stringifies some nulls, casts arbitrary floats/NaN, ignores extra judge-score IDs, and accepts a bare answers list; production wrapper remains deliberately stricter according to official discrete 0–4 docs.
4. Preserve task-alias differences between scorer and submission validator; record entry point instead of silently reconciling them.
5. Organizer score files and local proxy files produce different types; no filename/flag can relabel proxy data.
6. Use canonical serialization; parsing stays CPU-only and logs no payload text.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_official_io.py`. Compare upstream and toy fixtures against vendor subprocess behavior, then assert wrapper rejects documented unsafe permissiveness. Cover whitespace/case labels, aliases, Unicode, null/NaN/Infinity, extra judge IDs, duplicates/missing/extra, empty trace, process isolation, privacy, and `evaluation_kind="local_proxy"` preservation.

**Acceptance evidence:** Golden bytes and an explained raw-vendor-versus-strict-wrapper matrix. Fixture acceptance only.

**Non-goals and failure policy:** No fuzzy labels, repair, protected exception text, or proxy coercion. Unexplained divergence blocks metrics.

**Handoff:** EVA-03/04 receive typed validated records; EVA-09 receives bytes/parser hashes.

## EVA-03 Implement exact official MCQ accuracy aggregation

**Depends on:** EVA-02.

**Parallel safety and exclusive file ownership:** May run with EVA-04 after report-shape agreement. Exclusively owns MCQ symbols in `medfm/challenges/medreason/metrics.py` and `tests/challenges/medreason/test_official_mcq.py`.

**Target paths/symbols:** `OfficialMetricAggregator.score_mcq`, `MCQCaseResult`, `OfficialMetricReport`; reuse `medfm.evaluation.schemas.MetricResult` for reporting only.

**Inputs:** Strict parsed predictions/truth, vendor manifest, exact-label toy fixtures.

**Outputs:** Per-case `mcq_correct`, key exactly `MCQ Accuracy`, exact counts.

**Ordered implementation:**
1. Compute literally `float(pred.answer.strip() == truth.answer.strip())`; never case-fold, match option text, remap, round, or give partial credit.
2. Reject missing/extra/duplicate/task mismatch and absent MCQ reference. Return JSON `null` for an empty MCQ stratum.
3. Preserve truth iteration order and arithmetic mean.
4. Keep pure CPU aggregation independent of proxy availability and processes.
5. Mark fixture/dev results `official_public_aggregation`, not official leaderboard results.
6. Store labels only in restricted artifacts; telemetry contains IDs/counts/numbers/hashes.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_official_mcq.py`. Vendor-parity test upstream mixed fixture plus toy rubric cases; exact/whitespace/lowercase/numeric labels; empty stratum; missing reference; duplicate/missing/extra/task mismatch; no judge/CUDA import; privacy.

**Acceptance evidence:** Zero-difference per-case/aggregate parity. Hidden-label acceptance remains organizer-only.

**Non-goals and failure policy:** No semantic/fuzzy scoring or denominator omission. MCQ continues without proxies but cannot pass three-metric gates.

**Handoff:** EVA-12 gets exact metric; EVA-13 gets unrounded fraction; EVA-14 gets provenance.

## EVA-04 Implement official RVF-gated visual accuracy aggregation

**Depends on:** EVA-02.

**Parallel safety and exclusive file ownership:** May run with EVA-03. Exclusively owns open symbols and `compute_va_final` in `medfm/challenges/medreason/metrics.py`, plus `tests/challenges/medreason/test_official_open_metrics.py`.

**Target paths/symbols:** Exact fields `GT_final`, `VA_answer`, `RVF_trace`, `VA_final`, `Open-ended GT`, `Open-ended VA`.

**Inputs:** Strict organizer score records or separately typed local proxy records; toy 0–4 rubric boundaries.

**Outputs:** Separate `official_organizer_aggregate` and `local_proxy_aggregate` reports.

**Ordered implementation:**
1. Implement exactly: RVF `<=1` → `min(VA,1)`; RVF `==2` → `min(VA,3)`; otherwise VA unchanged.
2. Strict wrapper accepts only finite discrete `0..4`, as public docs specify. Keep a raw-vendor parity test documenting that upstream casts arbitrary floats and exact equality controls its middle branch.
3. Require one score per open case; reject missing and, at schema boundary, unexpected IDs. Mean GT and capped VA in truth order; empty open stratum is JSON `null`.
4. Organizer aggregation is CPU/file-only and never launches a model. Proxy aggregation requires proxy record type/hashes.
5. Every local display/machine field says proxy and `organizer_equivalent=false`.
6. Logs contain IDs/numbers/hashes only.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_official_open_metrics.py`. Vendor fixture and toy `(VA=4,RVF=0/1)->1`, `(4,2)->3`, `(4,3/4)->4`; strict rejection of negative/fractional/NaN/Infinity; raw-vendor `2.000001` documentation; empty/missing/extra scores; no worker/GPU; privacy and immutable proxy labels.

**Acceptance evidence:** Exact legal-domain parity and explicit permissiveness-difference tests. Judge assignment/tie behavior remains unavailable.

**Non-goals and failure policy:** No clamp/imputation/tie inference/model call. Missing proxy yields no open proxy aggregate; lexical metrics never populate these fields.

**Handoff:** EVA-07 supplies proxy fields; EVA-12/13 consume typed aggregate; EVA-14 reports limitations.

## EVA-05 Freeze public rubric prompts and hashes

**Depends on:** EVA-01, GOV-05, GOV-08, MOD-08.

**Parallel safety and exclusive file ownership:** May run with metrics. Exclusively owns `medfm/challenges/medreason/judges/prompts/`, `judges/contracts.py`, rubric tests, and prompt package inventory.

**Target paths/symbols:** `gt_proxy_v1.txt`, `va_proxy_v1.txt`, `rvf_proxy_v1.txt`, `RubricPromptManifest`, `RUBRIC_SHA256SUMS.json`, strict response schema.

**Inputs:** Public 0–4 rubric: GT receives question/reference/answer; VA image/question/answer; RVF image/question/trace. Exact organizer prompts, intermediate anchors, quantization, and caption/tie policy are unpublished.

**Outputs:** Immutable local-proxy prompts and raw-byte/combined hashes with source references.

**Ordered implementation:**
1. Transcribe only published criteria/endpoints. Added deterministic JSON instructions are explicitly local protocol.
2. Keep GT text-only; VA/RVF image-conditioned; no caption context by default.
3. Freeze roles, field order, whitespace, integer parser, temperature zero, exact revisions, quantization hash, and failure policy.
4. Refuse changed prompt bytes before loading any model.
5. Inventory prompts/manifests/toy assets in packaged source/wheel.
6. Never log populated prompts/responses; rows carry `organizer_prompt_known=false`.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_rubrics.py`; `uv run pytest -q tests/challenges/medreason/test_package_inventory.py -k rubric`. Test hashes, public inputs/anchors, integer `0..4`, malformed prose/multiple values, fake-worker isolation, privacy, and EVA-04 cap parity. Never assert organizer-prompt parity.

**Acceptance evidence:** Prompt manifest/inventory/parser matrix and visible limitations. Fixture acceptance only.

**Non-goals and failure policy:** No invented intermediate rubric, examples, tie logic, or organizer-equivalence claim. Hash mismatch blocks proxy promotion, not MCQ.

**Handoff:** EVA-06/07 consume prompt schema; EVA-09 persists hashes; EVA-14 exposes limitations.

## EVA-06 Run sequential Llama GT proxy judge

**Depends on:** GOV-05, GOV-06, GOV-09, GOV-10, MOD-08, MOD-09, EVA-05, EVA-08, EVA-15.

**Parallel safety and exclusive file ownership:** Code may run with EVA-07 after supervisor freeze. Protected execution excludes Qwen, candidate inference, training, and memory measurements. Exclusively owns `medfm/challenges/medreason/judges/gt.py` and `tests/challenges/medreason/test_gt_proxy.py`.

**Target paths/symbols:** `GTProxyRequest`, `GTProxyResult`, `run_gt_worker`; `meta-llama/Llama-3.1-70B-Instruct@1605565b47bb9346c5515c34102e054115b4f98b`; immutable judge root.

**Inputs:** Accepted Meta license/access, exact snapshot, frozen prompt, hashed 4-bit config, restricted request. Revision exists but is `gated:manual`; team acceptance is unknown.

**Outputs:** Restricted JSONL with ID, integer `proxy_gt_final`, status, content/model/prompt/config hashes, latency, sanitized failure; proxy manifest.

**Ordered implementation:**
1. Require `MEDFM_RUN_REAL_JUDGES=1`, accepted gated access, hashes, CUDA/storage/capacity preflight, and exclusive lease before child imports.
2. Spawn one child; pass file paths/hashes, never protected text via argv/stdout; verify GPU exclusivity.
3. Load only exact 4-bit revision in child, temperature zero, parse one `0..4` integer. Never substitute IDs/models.
4. Atomically persist raw/parsed response, synchronize, record peaks, release/cache-clear, exit; EVA-15 verifies baseline.
5. Access/hash/license/OOM/timeout/invalid/nonzero-exit means unavailable, never score zero.
6. Allowlisted telemetry; always label `GT local proxy`.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_gt_proxy.py` uses spawned fake/toy rubric cases for parsing, ordering, atomicity, hashes, crash/OOM/timeout, exclusivity, privacy, and EVA-04 arithmetic. Protected: `MEDFM_RUN_REAL_JUDGES=1 CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests/challenges/medreason/test_gt_proxy.py -k real_checkpoint`; guard honestly. No organizer score parity claim.

**Acceptance evidence:** Fake-worker artifacts/timeline/privacy; protected exact-revision and measured-memory manifest only if prerequisites exist.

**Non-goals and failure policy:** No download/API/fallback/pseudo-label/preference/concurrency. Llama alone cannot promote without Qwen.

**Handoff:** EVA-08 state, EVA-09 artifacts, EVA-15 lifecycle, EVA-12 complete bundles.

## EVA-07 Run sequential Qwen VA and RVF judge

**Depends on:** GOV-05, GOV-06, GOV-09, GOV-10, MOD-08, MOD-09, EVA-05, EVA-08, EVA-15.

**Parallel safety and exclusive file ownership:** Code may run with EVA-06; protected execution starts only after Llama exit/baseline. Exclusively owns `judges/visual.py` and its tests.

**Target paths/symbols:** `VisualProxyRequest`, `VisualProxyResult`, `run_visual_worker`; `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5`; immutable judge root.

**Inputs:** Accepted custom Qwen License, verified ungated revision, prompts, processor/chat template, hashed 4-bit config, restricted image/question/answer/trace. Model card gives no authoritative 4-bit suitability; parity is measured.

**Outputs:** Restricted raw/parsed `proxy_va_answer`, `proxy_rvf_trace`, `proxy_va_final`, hashes/timing/memory/status.

**Ordered implementation:**
1. Require licenses/hashes/CUDA/storage/capacity/lease and post-Llama baseline.
2. Load exact revision/processor in fresh child; preserve `input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values`, `image_grid_thw` and hash processor/template/config.
3. Run temperature-zero raw-image VA/RVF; no default caption/tie context; parse one integer each.
4. Cap only through EVA-04, persist atomically, release/exit, verify EVA-15 baseline.
5. Decode/OOM/timeout/invalid/hash/license/4-bit-parity failure means unavailable; no text-only or alternate VLM.
6. Logs omit payloads/paths; labels say local proxy and unknown organizer prompt.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_visual_proxy.py` uses synthetic images/fake `0..4` outputs for tensor fields, cap parity, parsing, isolation/order, failures/privacy. Protected: `MEDFM_RUN_REAL_JUDGES=1 CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests/challenges/medreason/test_visual_proxy.py -k real_checkpoint`. Judge assignment parity is not claimed.

**Acceptance evidence:** Fixture artifacts/isolation/privacy; protected evidence only with accepted license, exact snapshot, measured fit/numerical parity.

**Non-goals and failure policy:** No alternate VLM/caption/text fallback/API/pseudo-annotation/concurrency. Missing Qwen blocks promotion.

**Handoff:** EVA-08 state; EVA-04 fields; EVA-09/11 artifacts; EVA-15 lifecycle.

## EVA-08 Block proxy promotion when judges are unavailable

**Depends on:** GOV-05, GOV-09, GOV-10, EVA-04, EVA-05.

**Parallel safety and exclusive file ownership:** Implement before workers. Exclusively owns `judges/availability.py`, `ProxyBundleStatus`, and `test_judge_availability.py`; gates cannot reinterpret status.

**Target paths/symbols:** `JudgeStatus`, `require_complete_proxy_bundle`, closed block-reason enum.

**Inputs:** License/artifact/prompt/config/preflight/worker/parse/release state; toy combinations.

**Outputs:** Status JSON with individual reportability, `promotion_allowed`, `winning_caliber_claim_allowed`, sanitized reasons.

**Ordered implementation:**
1. Available means exact/frozen hashes, full case coverage, legal scores, clean exit, and memory release; file presence is insufficient.
2. If either exact judge is inaccessible/unlicensed/unverified/unrunnable, both promotion and winning-caliber claim are false.
3. MCQ and lexical diagnostics may continue; one available proxy is incomplete diagnostic only.
4. Never zero/impute/substitute/reuse stale candidate scores or accept partial coverage.
5. Enforce state before thresholds in data model/CLI.
6. State handling is CPU-only, private, and always proxy-labeled.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_judge_availability.py`. Matrix both/either states, stale hashes, partial toy cases, OOM/timeout/release failure; verify MCQ vendor parity continues, promotion fails early, lexical cannot fill GT/VA, no GPU/process, private logs.

**Acceptance evidence:** Exhaustive transition matrix and blocked command result; not real-judge proof.

**Non-goals and failure policy:** No fallback/bypass/official naming. Missing prerequisite is `unavailable`, not zero.

**Handoff:** EVA-06/07/15 publish evidence; EVA-12/13 require it; EVA-14 displays it.

## EVA-09 Persist raw parsed predictions and option logits

**Depends on:** SCH-05, SCH-07, SCH-08, EVA-02, EVA-05.

**Parallel safety and exclusive file ownership:** May run with workers after schema freeze. Exclusively owns `evaluation_artifacts.py` and tests; generic evaluation files remain read-only.

**Target paths/symbols:** `MedReasonEvaluationBundle`, raw/parsed/option records, `save_evaluation_bundle`; reuse `PredictionArtifact`, `RuntimeProvenance`, `save_prediction_artifact`.

**Inputs:** Raw/parsed outputs, original option order, all conditional scores/logits, trace/answer/failure, processor/chat/base/adapter/prompt/config/split/control hashes.

**Outputs:** `artifacts/runs/medreason/<run>/evaluation/<candidate>/<condition>/` with raw/parsed JSONL, `option_scores.jsonl`, generic prediction artifact, case scores, metrics, telemetry, SHA manifest.

**Ordered implementation:**
1. Atomically save raw before parse and parsed after, input ordered; retain losing option scores.
2. Record trace/answer separately, route/failure, exact revisions/hashes, decoding/seed/split/group/control.
3. Mirror compatible rows into full `RuntimeProvenance`; challenge fields stay in versioned companions.
4. Use canonical JSON/config hash, reject duplicate IDs, hash every file, recompute without inference/judges.
5. Restrict raw files; telemetry is allowlisted/sanitized.
6. Separate organizer/proxy bundles; proxy manifest fixes `organizer_equivalent=false`.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluation_artifacts.py`. Round-trip toy MCQ/open/rubric, vendor recomputation, losing logits, Unicode, duplicate/truncation/mutation/stale hash/fallback, deterministic bytes, no CUDA/process, privacy, immutable proxy labels.

**Acceptance evidence:** Golden bundle/hashes/`PredictionArtifact.content_hash()`/parity/privacy. Protected content requires protected runs.

**Non-goals and failure policy:** No raw tracker/console content, omitted failures, or reconstructed missing logits. Incomplete bundle is invalid.

**Handoff:** EVA-10 telemetry; EVA-11/12 bundles; selection immutable manifests.

## EVA-10 Record sanitized latency VRAM and failure telemetry

**Depends on:** SCH-07, SCH-05, EVA-09.

**Parallel safety and exclusive file ownership:** May run with controls. Exclusively owns `evaluation_telemetry.py` and tests; coordinate runtime hooks with RUN-08.

**Target paths/symbols:** `EvaluationTelemetryEvent`, `TelemetryRecorder`, `FailureClass`, latency/CUDA adapter; reuse tracking redaction/run metadata.

**Inputs:** Monotonic timestamps, case/route/condition, sanitized enum, CUDA peaks when measured, PID/state/hashes.

**Outputs:** Allowlisted telemetry JSONL and summary with latency quantiles, memory, failures, evidence/method.

**Ordered implementation:**
1. Synchronize and separate model load, warm inference, and judge timing.
2. Read/reset CUDA peaks only in owner child; CPU fixture uses null GPU fields, not zero.
3. Map exception to closed enum immediately; raw message never reaches telemetry.
4. Separate candidate/Llama/Qwen segments; release stays unproven until EVA-15.
5. Hash/join telemetry and mark fixture/protected plus organizer/proxy.
6. Instrumentation cannot create availability or scores.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluation_telemetry.py` covers toy fake workers, timing, null/mock VRAM, failures, segments, deterministic summary, score parity, privacy. Guarded: `MEDFM_RUN_REAL_JUDGES=1 CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests/challenges/medreason/test_evaluation_telemetry.py -k real_cuda`.

**Acceptance evidence:** Fixture goldens/parity/privacy and honest hardware status. Fixture VRAM is not acceptance.

**Non-goals and failure policy:** No raw payload/error/path/secret or estimated hardware claims. Missing measurement is null/unproven.

**Handoff:** EVA-12/13 measurements; EVA-15 segments; EVA-14 evidence class.

## EVA-11 Implement no-image and shuffled-image controls

**Depends on:** SPL-07, STR-06, SCH-05, EVA-09, EVA-10.

**Parallel safety and exclusive file ownership:** Run after base candidate freeze. Exclusively owns `controls.py` and tests; runtime owns model execution.

**Target paths/symbols:** `ControlCondition`, no-image builder, deterministic shuffled assignment, pairing validator; reuse `visual_grounding_gate`.

**Inputs:** Frozen split/group/candidate, images, released modality metadata, seed `2026`, primary predictions.

**Outputs:** Mapping hash, paired bundles/deltas, grounding report.

**Ordered implementation:**
1. No-image preserves nonvisual fields and uses declared missing-visual path, not arbitrary blank image.
2. Build case-hash derangement within split and compatible image-count/modality strata only when metadata exist; report singleton strata.
3. Freeze mapping before run; only image assignment changes.
4. Run conditions in separate candidate processes, then sequential proxies after release.
5. Use exact MCQ/RVF metrics and preregistered `visual_grounding_gate` strict `>` margin.
6. Restrict mappings/paths/content; telemetry is allowlisted and proxy-labeled.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluation_controls.py`. Synthetic images test deterministic derangement/no self/strata/singletons, invariant fields, no-image route, strict margin, vendor parity, toy caps, process nonoverlap/privacy.

**Acceptance evidence:** Mapping/pair/gate/timeline/privacy fixtures. Not real grounding evidence.

**Non-goals and failure policy:** Not OOD reconstruction/annotation. Never shuffle labels/questions, infer modality, use validation answers, or promote incomplete proxies.

**Handoff:** EVA-12 controls; EVA-13/SEL-05 gate; EVA-14 report.

## EVA-12 Run fixed zero-shot model tournament

**Depends on:** MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, MOD-06, MOD-07, MOD-12, MOD-13, EVA-03, EVA-04, EVA-06, EVA-07, EVA-08, EVA-09, EVA-10, EVA-11, SPL-07, STR-05.

**Parallel safety and exclusive file ownership:** Planning can parallelize; protected GPU execution serializes. Exclusively owns `tournament.py`, tests, and zero-shot run manifests; frozen models/prompts/splits are read-only.

**Target paths/symbols:** `ZeroShotCandidate`, `TournamentPlan`, `TournamentRunner`, `TournamentReport`; config handoff remains CLI-owned.

**Inputs:** Exactly Gemma 4 31B, conditional 26B-A4B, MedGemma 1.5 4B, frozen MedGemma 27B diagnostic; optional older controls non-gating; frozen dev/stress/control hashes.

**Outputs:** Bundle per candidate/condition and clean/stress/control table with exact MCQ, proxy GT/VA, failures, latency/VRAM, status/evidence.

**Ordered implementation:**
1. Freeze rows/order/revisions/processors/chat/prompts/parsers/decoding/seeds/controls/tolerances/hashes.
2. Use identical official-style I/O and persist raw/parsed/options before scoring.
3. Candidate process exits/releases, then Llama and Qwen run separately/sequentially.
4. Run primary/no-image/shuffled and declared stress weights; report missing strata.
5. If either judge unavailable, finish MCQ/lexical/control but block open promotion.
6. Fakes by default. Real models need `MEDFM_RUN_REAL_CHECKPOINTS=1`; proxies also need `MEDFM_RUN_REAL_JUDGES=1`, artifacts/licenses/preflight. Current workstation is not accepted.
7. Restrict artifacts and separate proxy/organizer namespaces.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_zero_shot_tournament.py` for toy candidates/rubrics, fixed-plan drift, vendor parity, controls, unavailable semantics, nonoverlap/privacy/artifacts. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_REAL_JUDGES=1 CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests/challenges/medreason/test_zero_shot_tournament.py -k protected`. Eventual run: `CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.evaluate --config configs/recipes/medreason/zero_shot_tournament.yaml --split-manifest artifacts/data/medreason/derived/splits.json`.

**Acceptance evidence:** Fixture manifest/table/parity/timeline/privacy and guarded status. Real advancement needs protected evidence.

**Non-goals and failure policy:** No added candidates, validation/lockbox tuning, training/fusion/win claim. Missing proxies block open promotion.

**Handoff:** EVA-13 report hashes; OPS-08 protected execution; training advanced IDs only.

## EVA-13 Enforce candidate advancement and ambition thresholds

**Depends on:** EVA-08, EVA-11, EVA-12, MOD-13.

**Parallel safety and exclusive file ownership:** Run after tournament freeze. Exclusively owns `promotion.py` and tests; selection owns paired bootstrap and lockbox access.

**Target paths/symbols:** `BaselineAdvancementDecision`, `AmbitionThresholdDecision`, advancement/ambition functions.

**Inputs:** Immutable tournament, complete proxy status, exact metrics/telemetry, 100-real-batch evidence, and only SEL-15-authorized one-system lockbox.

**Outputs:** Hash-linked decisions with rule/version/value/unit/evidence/blockers.

**Ordered implementation:**
1. 31B advances unless real 100-batch memory gate fails. 26B advances if within `0.5` percentage points MCQ, `0.10` GT, `0.10` VA **and** `>=1.5x` faster, or 31B misses training/48 GB/runtime gate. 4B is optional open route unless it wins MCQ. 27B never advances. Maximum three trainable candidates.
2. Require exact hashes, complete conditions, measured coverage/hardware, both proxies; unproven never passes.
3. Separately test only one frozen once-opened lockbox system against all three: MCQ `>=0.975`, local proxy GT `>=2.15`, local proxy VA `>=2.85`.
4. Label as ambition, never official win; fixture arithmetic cannot pass protected claim.
5. No system/calibration change after lockbox and no multiple candidates.
6. Gate is CPU-only/private/proxy-labeled.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluation_gates.py`. Exact boundaries/units/speed/fallback/max/4B-27B/missing/stale/single-lockbox; toy RVF cap and vendor parity; no process/GPU/privacy.

**Acceptance evidence:** Golden branch table, hash links, blocked statuses, `official_win_claim=false`. Protected acceptance requires measured artifacts.

**Non-goals and failure policy:** No relaxed/partial/lexical/rank/lockbox-tuned claim. Missing evidence is `not_eligible`, not zero.

**Handoff:** OPS-09/training IDs; selection ambitions; EVA-14 decisions.

## EVA-14 Label proxy results and diagnostic metrics honestly

**Depends on:** EVA-03, EVA-04, EVA-08, EVA-09, EVA-10, EVA-13.

**Parallel safety and exclusive file ownership:** May render before protected runs. Exclusively owns `evaluation_report.py` and tests; generic report module is reused read-only.

**Target paths/symbols:** `MedReasonEvaluationReport`, `EvidenceClass`, `MetricProvenance`, renderer.

**Inputs:** Vendor/split/candidate/judge/prompt/config hashes/status, bundles, telemetry, controls, decisions, lexical diagnostics.

**Outputs:** JSON/text rows declaring kind (`official_public_aggregation`, `local_proxy`, `lexical_diagnostic`), split, evidence (`fixture`, `protected_local`, `organizer`), availability/equivalence, count/value/hash.

**Ordered implementation:**
1. Organizer/unqualified official judge naming requires organizer case scores. Local rows say `Local proxy GT/VA/RVF`, `organizer_equivalent=false`, `organizer_prompt_known=false`.
2. State unpublished prompt/quantization/caption/tie details.
3. Lexical diagnostics cannot fill GT/VA, pass gates, or mask availability.
4. Show fixture/protected/organizer and hardware/storage limits without 48/96 GB inference.
5. Include controls/failures/counts/hashes/method. Winning-caliber wording needs protected all-three lockbox pass and remains ambition.
6. Render only allowlisted summaries; raw content stays restricted.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluation_report.py`. Snapshot complete/incomplete/lexical/organizer/toy-cap/control/ambition reports; assert vendor parity, forbidden unqualified proxy wording, no process/GPU, privacy.

**Acceptance evidence:** Golden JSON/text, provenance schema, wording/privacy guards; fixture reports visibly say fixture/proxy.

**Non-goals and failure policy:** No marketing alias, rank extrapolation, omitted limitation, or fallback official naming. Missing provenance is unreportable.

**Handoff:** Selection/OPS/Docker consume report/hash; papers consume evidence-qualified rows only.

## EVA-15 Verify judge process releases GPU memory

**Depends on:** GOV-09, GOV-10, EVA-10.

**Parallel safety and exclusive file ownership:** Implement before workers; checks globally serialize on evaluation GPU. Exclusively owns `judges/runner.py`, `judges/gpu_lock.py`, and `test_judge_process_isolation.py`.

**Target paths/symbols:** `JudgeProcessSupervisor`, `ExclusiveGpuLease`, `GpuMemoryBaseline`, `run_isolated_worker`, `verify_memory_return`.

**Inputs:** Worker/request/output paths, hashes, GPU process inventory, synchronized allocated/reserved/device-used samples, frozen tolerance/timeout, fake workers.

**Outputs:** Restricted lifecycle JSONL and safe release evidence with PID/role/state/exit, before/peak/after memory, method/tolerance, lock/hash/pass.

**Ordered implementation:**
1. Acquire OS lease and ensure no training/candidate/judge process; record role/PID, not sensitive command line.
2. Spawn short-lived children; supervisor stays CPU-only; exactly one model role owns GPU.
3. Establish synchronized baseline using declared Torch/NVML method; sample peak and post-exit use; require process disappearance and return within preregistered tolerance.
4. Fsync output, require clean exit, synchronize/cleanup, verify teardown before next role.
5. Timeout terminates process tree, retains failure artifact, verifies cleanup, and marks unavailable even with partial score. Release failure blocks next stage/promotion.
6. Toy fake probes/rubrics by default. Real checks require exact artifacts, `MEDFM_RUN_REAL_JUDGES=1`, CUDA/capacity/storage/licenses; no local 70B/72B claim.
7. Lifecycle telemetry is allowlisted and proxy-labeled.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_judge_process_isolation.py` verifies lease exclusion, Llama-exit/baseline-before-Qwen, candidate/training exclusion, crash/timeout/tree cleanup, partial rejection, tolerance, toy score preservation, vendor parity, privacy. Protected: `MEDFM_RUN_REAL_JUDGES=1 CUDA_VISIBLE_DEVICES=0 uv run pytest -q tests/challenges/medreason/test_judge_process_isolation.py -k real_cuda_release`.

**Acceptance evidence:** Fake lifecycle/release matrix; protected proof requires measured samples and zero worker PIDs. Fixture cleanup is not hardware acceptance.

**Non-goals and failure policy:** No in-process loads, concurrency, estimates, sleeps as proof, continuation after release failure, or official-judge claim. Baseline failure makes proxy unavailable.

**Handoff:** EVA-06/07 consume supervisor; EVA-08 release status; EVA-10/14 sanitized evidence; operations guarded result.
