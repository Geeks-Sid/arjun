# Experiment execution

This phase starts only after implementation, fixture tests, protected-hardware guards, and release code are ready. The commands below are prescribed for later execution; none were executed while writing this plan.

## Mandatory operator controls

- Run from the `arjun/` repository root. Every operation creates a new append-only attempt at `artifacts/runs/medreason/operations/<OPS-ID>/<UTC-attempt-id>/` containing `operation.json`, `commands.log`, `stdout.log`, `stderr.log`, `sha256sums.txt`, and exactly one of `COMPLETE.json` or `STOP.json`. Record prerequisite hashes, operator/reviewer, UTC times, source commit, `uv.lock` SHA-256, host profile, literal command, exit code, and output hashes. Never overwrite an attempt.
- A file hash is lowercase 64-hex SHA-256. A directory is accepted only through a canonical sorted manifest of relative path, byte size, and file SHA-256, plus the manifest SHA-256. Mutable aliases are not evidence.
- Logs contain only case IDs, counts, aggregate metrics, timings, hashes, and sanitized error classes. Never log questions, answers, references, metadata, prompts, traces, image paths, credentials, terms text, or access URLs. Keep original legal/access receipts restricted; expose only hashes and redacted decisions.
- `STOP.json` is the required fail-closed outcome for an unmet predicate. Resume only as a new attempt after the missing evidence exists; do not weaken a gate.
- **Hard blockers measured 2026-08-09:** local GPU NVIDIA GeForce RTX 3090, 24,576 MiB; repository-local free space 364 GiB versus at least 600 GB required; no configured SSH compute hosts. Local code/fixture acceptance remains possible, but real 31B/26B/70B/72B runs, full immutable staging, 100-batch memory gates, and 48/96 GB claims are blocked until OPS-06 produces replacement evidence. A 24 GB run is never evidence for either target profile.
- Official eligibility, judge parity, model availability, redistribution, 48/96 GB support, and winning-caliber metrics are never inferred. GT/VA are local proxies because official prompts, revisions, quantization, and tie policy are unpublished. Participant validation is never used for tuning or scoring. Official hardware assignment, wall time, and archive limits are also unpublished and require written evidence.

## OPS-01 Obtain written late-submission organizer exception

**Depends on:** GOV-01, GOV-08, GOV-10.

**Parallel safety and exclusive file ownership:** May run with OPS-02 and OPS-06. Exclusive root: `artifacts/runs/medreason/governance/organizer_exception/`; no other operation may replace an accepted receipt.

**Target paths/symbols:** Restricted `receipt.*`, `redacted_summary.json`, `evidence_manifest.json`, and GOV-10 late-submission preflight predicate.

**Inputs:** Published July 22, 2026 closure; actual team/registration facts; organizer response; official runtime commit `05748c0341b72dc08132bd108208b78dc14a2f0b`.

**Outputs:** Immutable receipt and summary fields `decision`, `scope`, `team`, `received_at_utc`, `submission_deadline`, `hardware_contract_received`, and `official_contact`; unknowns remain `null`.

**Hard prerequisites and no-skip stop conditions:** A response through an organizer-published channel must identify this team and explicitly authorize late submission. Silence, FAQ text, post-challenge research permission, or archival-paper eligibility is not approval. It must also provide or acknowledge unpublished hardware, time, and archive limits before an artifact is labelled submission-ready. Absent an explicit exception, write `STOP.json`; research may continue, but OPS-17 and submission-mode OPS-18 stop. This does not clear the 24 GB/364 GiB/no-host blockers.

**Implementation:**
1. Save original response bytes without conversion and restrict permissions.
2. Produce a redacted summary without strengthening ambiguous language.
3. Hash the original, summary, and official-contact evidence.
4. Require independent second review; disagreement resolves to `ambiguous` and stops.
5. Feed only decision and evidence hashes to preflight; never copy private correspondence into Docker or logs.

**Focused tests and exact commands:** Later run `uv run pytest -q tests/challenges/medreason/test_preflight.py -k organizer_exception`. Failures: wrong team, expired/ambiguous scope, paper-track-only wording, missing limits, missing hash, leaked correspondence. Verify durable bytes with `sha256sum artifacts/runs/medreason/governance/organizer_exception/receipt.* artifacts/runs/medreason/governance/organizer_exception/redacted_summary.json`.

**Acceptance evidence:** `COMPLETE.json` has two-person approval, receipt/summary hashes, explicit scope/deadline/contract fields, and preflight pass. Otherwise only `STOP.json` is honest evidence. No score or hardware metric is produced.

**Non-goals and failure policy:** Never claim an official win or a paper-track back door. Preserve denial/ambiguity and continue only labelled post-challenge research.

**Handoff:** OPS-17 and submission-mode OPS-18 consume decision/deadline/contract fields and receipt hash; publications consume the official-versus-research label.

## OPS-02 Obtain approved Synapse challenge data access

**Depends on:** GOV-02, GOV-08, GOV-10.

**Parallel safety and exclusive file ownership:** May run with OPS-01/06. Exclusive root: `artifacts/runs/medreason/governance/synapse_access/`; it never writes source archives.

**Target paths/symbols:** Restricted access and human-accepted terms receipts for Synapse `syn74403682`, visible package inventory, `redacted_summary.json`, `evidence_manifest.json`.

**Inputs:** Authorized registered account, actual data-use terms, source page, and package inventory visible to that account.

**Outputs:** Receipt/terms hashes and summary of account/team, acceptance UTC, package identifiers, restrictions, and approval state.

**Hard prerequisites and no-skip stop conditions:** Human acceptance evidence under the correct identity is mandatory; no automatic terms click or inferred permission. Require access to released training and participant-facing validation packages. Pending/revoked/wrong-identity access or unverifiable package identity writes `STOP.json` and blocks OPS-03. Public counts are 17,722 train and 2,532 validation (2,057 MCQ/475 open), but train supervision keys are unpublished until authorized archive access and must not be invented. Never claim participant-validation labels. Storage remains blocked until OPS-06.

**Implementation:**
1. Authenticate interactively outside captured logs; disable tracing and never persist tokens.
2. Human-review and accept actual terms; save receipt and visible identifiers exactly.
3. Reconcile restrictions with GOV-02 and second-review scope.
4. Hash restricted evidence/redacted summary and pass only hashes/state to preflight.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_preflight.py -k synapse_access`. Failures: absent human acceptance, mismatched identity, unavailable package, leaked token, invented supervision key, validation described as labelled. Verify with `sha256sum artifacts/runs/medreason/governance/synapse_access/redacted_summary.json artifacts/runs/medreason/governance/synapse_access/evidence_manifest.json`.

**Acceptance evidence:** `COMPLETE.json` records receipt/terms/package hashes, second reviewer, and zero-secret scan. Dataset hashes do not exist until OPS-03.

**Non-goals and failure policy:** No scraping, borrowed account, source-article matching, inferred schema, or inferred labels. Ambiguous terms stop download.

**Handoff:** OPS-03 receives approved package identifiers and receipt/terms hashes; later provenance embeds those hashes.

## OPS-03 Download immutable released data archives

**Depends on:** OPS-02, OPS-06, DAT-01, DAT-02, DAT-11, GOV-08, GOV-10.

**Parallel safety and exclusive file ownership:** Sole writer of `artifacts/data/medreason/source/`. Download to unique `.partial` files and atomically promote; no extraction/audit/pruning overlaps.

**Target paths/symbols:** `artifacts/data/medreason/source/medreason2026_train.zip`, `medreason2026_validation_participant_facing.zip`, and `source_manifest.json`.

**Inputs:** Approved package identifiers/receipts, authorized Synapse mechanism, OPS-06 capacity reservation.

**Outputs:** Exact archive bytes, sizes, source identifiers, retrieval UTC, credential-free downloader command/version record, file hashes, and canonical source-manifest hash.

**Hard prerequisites and no-skip stop conditions:** Require at least the 600 GB reservation; current 364 GiB is insufficient. Stop on error-page downloads, interruption, wrong/missing package, repeat hash mismatch, corrupt archive, or an answer field in participant validation. No mirror is substituted.

**Implementation:**
1. Use only the authorized downloader/identifiers exposed after access; record the exact redacted command. Do not invent unavailable Synapse URLs.
2. Download once to `.partial`, fsync, verify package identity/type/size/hash.
3. Atomically rename, set read-only, create sorted manifest.
4. Rehash after promotion; quarantine and `STOP` on difference.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_data_archives.py -k 'immutable or provenance or secure_extract'`. Later real verification:

```bash
sha256sum artifacts/data/medreason/source/medreason2026_train.zip
sha256sum artifacts/data/medreason/source/medreason2026_validation_participant_facing.zip
```

Failures: path traversal/symlink members, corrupt ZIP, HTML body, changed bytes, answer leakage.

**Acceptance evidence:** Two matching 64-hex hashes/sizes in `source_manifest.json`, `sha256sums.txt`, and post-promotion check; credential-free logs; cited OPS-02/06 hashes. Counts wait for OPS-07.

**Non-goals and failure policy:** No hidden/private/external data or outputs. Never repair in place; quarantine and redownload as a new attempt.

**Handoff:** OPS-07 consumes exact paths/archive hashes; all later runs consume source-manifest hash.

## OPS-04 Accept and download selected model snapshots

**Depends on:** GOV-03, GOV-04, GOV-06, GOV-08, GOV-10, MOD-04 through MOD-07, OPS-06.

**Parallel safety and exclusive file ownership:** May overlap OPS-03/05 only with disjoint quotas. Sole writer of `artifacts/models/medreason/base/<model_id>/<40-char-revision>/`; trainers/Docker read only.

**Target paths/symbols:** Exact `google/gemma-4-31B-it@419b2efe421994fdfd3394e621983d4cc511cd4f`, `google/gemma-4-26B-A4B-it@47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, conditional `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, and optional frozen-only `google/medgemma-27b-it@2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f`.

**Inputs:** Verified remote SHAs, registry/license decisions, actual model terms, processor/template allowlist, storage profile.

**Outputs:** Authorized local snapshots and sorted per-file manifests, registry/license receipt hashes, processor/chat-template hashes. Mutable cache refs are not evidence.

**Hard prerequisites and no-skip stop conditions:** All exact SHAs exist, but access does not equal license. Gemma 4 is ungated yet still requires recorded human terms/license review. MedGemma is gated (`auto`) and requires human acceptance and explicit redistribution decision; no automated click or inferred redistribution. Unresolved MedGemma shipping stops that route only. Stop on revision mismatch, absent required processor/template, unexpected code, manifest drift, or insufficient storage. Download never proves 48/96 GB loadability; local 24 GB cannot do so.

**Implementation:**
1. Resolve each registry ID to exact revision before transfer; record authorized credential-free fetch command.
2. Human-review terms and hash acceptance/license evidence.
3. Fetch to partial revision directory; preserve release processor/chat-template files.
4. Reject symlinks/path escape/mutable refs/unreviewed custom code; hash every file.
5. Validate manifest before read-only promotion. Keep 27B out of training/fusion/deployment; exclude unapproved MedGemma from Docker.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_model_assets.py -k 'revision or manifest or license or offline'`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -q tests/challenges/medreason/test_model_assets.py -m real_checkpoint`; when enabled, missing assets/access is failure, never skip. Failures: moving revision, missing template, altered shard, automatic terms action, gated use without receipt.

**Acceptance evidence:** Manifest hash per snapshot with exact ID/revision/file hashes/bytes, registry and human-license evidence hashes, processor/template hashes, optional/blocked status. No hardware claim.

**Non-goals and failure policy:** No Gemma 3/latest substitution, automatic acceptance, or inferred redistribution. MedGemma failure leaves Gemma 4 route after its own review.

**Handoff:** OPS-08 gets immutable manifests and availability set; OPS-18 gets only explicitly redistribution-approved artifacts.

## OPS-05 Accept and download exact judge snapshots

**Depends on:** GOV-05, GOV-06, GOV-08, GOV-10, MOD-08, EVA-05, EVA-08, OPS-06.

**Parallel safety and exclusive file ownership:** May overlap other downloads under disjoint quota. Sole writer of `artifacts/judges/medreason/<model_id>/<revision>/`; judges never enter training or Docker roots.

**Target paths/symbols:** `meta-llama/Llama-3.1-70B-Instruct@1605565b47bb9346c5515c34102e054115b4f98b` and `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5`, file manifests, BitsAndBytes config hash, rubric hashes.

**Inputs:** Actual access/license terms, registry revisions, public rubrics, approved quantization config, provisioned judge storage/GPU.

**Outputs:** Two judge-only authorized local snapshots, receipt/license hashes, tokenizer/processor and file-manifest hashes, availability verdict.

**Hard prerequisites and no-skip stop conditions:** Llama is gated (`manual`) and anonymously protected bytes return 401; require authorized access and human acceptance. Qwen is ungated but uses custom license terms, not Apache; require human review/acceptance and do not infer redistribution. No auto terms clicks. Both exact judges must be available/runnable sequentially for GT/VA promotion or winning-caliber claims. Missing either writes `STOP.json`; lexical metrics remain diagnostics only. Current 24 GB/364 GiB/no-host blocks real acceptance.

**Implementation:**
1. Human-review terms under correct identity; save restricted receipts/hashes without tokens.
2. Resolve/fetch exact revisions into partial judge-only roots.
3. Hash files, tokenizer/processor, quantization config, rubrics; freeze read-only.
4. Verify separate short-lived processes and baseline VRAM recovery; never co-reside with training.
5. Exclude judges from Docker inventories.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_judges.py -k 'license or revision or sequential or unavailable'`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_judges.py -m 'real_checkpoint and gpu'`; enabled-but-missing CUDA/assets/access fails, not skips. Failures: one judge absent, wrong SHA, simultaneous residency, prompt drift, nonzero temperature, VRAM not recovered.

**Acceptance evidence:** Two manifest/human-acceptance hashes; custom/gated license decisions; quantization/rubric hashes; protected logs with PIDs, temperature 0, peak allocated/reserved, baseline recovery. Otherwise blocked availability, not proxy metrics.

**Non-goals and failure policy:** No substitute judge, automatic acceptance, inferred redistribution, private labels, or official-score claim. Missing exact judge disables promotion.

**Handoff:** OPS-08/09/12 consume availability/hashes; OPS-18 proves judges absent.

## OPS-06 Provision required storage and GPU profiles

**Depends on:** GOV-09, GOV-10, MOD-13, MOD-14, DOC-02, DOC-15.

**Parallel safety and exclusive file ownership:** May overlap governance, but precedes large downloads/runs. Exclusive `artifacts/runs/medreason/provisioning/`; benchmark jobs own GPUs/storage reservations.

**Target paths/symbols:** `host_inventory.json`, `storage_reservation.json`, `gpu_96gb_quality.json`, `gpu_48gb_compatibility.json`, `judge_process_profile.json`, local base-image digest and ABI inventory.

**Inputs:** At least 600 GB reserved SSD; measured 96 GB quality and physical 48 GB compatibility GPUs; sequential judge requirements; digest-pinned base/wheelhouse needs.

**Outputs:** Hashed raw inventories, reservations/quotas, GPU UUID/model/total memory, driver/CUDA/kernel/container versions, reachability, availability booleans.

**Hard prerequisites and no-skip stop conditions:** Present RTX 3090 24,576 MiB, 364 GiB free, no SSH hosts means current `STOP`. Provision authorized compute and at least 600 GB reservation. Training 100-batch peak must be `< min(85 GiB, 0.90 * total_device_memory)`. A 96 GB run cannot prove 48 GB; require matching 48 GB end-to-end evidence, with approved 45 GB peak-allocation ceiling when selected. Base digest must be local before no-pull build.

**Implementation:**
1. Inventory host/filesystem/GPU/driver/CUDA/container/network; redact secrets.
2. Reserve disjoint quotas for assets, judges, runs/checkpoints, wheelhouse/build/export. Prune only superseded optimizer checkpoints under policy.
3. Measure 96 GB training/runtime guards; never estimate activation/KV/container overhead.
4. Measure 48 GB compatibility on physical matching hardware.
5. Run judges sequentially and verify GPU baseline recovery.
6. Hash inventories; host unavailable until repository path, artifact mount, GPU query work.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_hardware_guards.py`. Capture:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
python -c "import shutil; print(shutil.disk_usage('.'))"
docker image inspect --format '{{index .RepoDigests 0}}' medreason-base:locked
```

Protected: `MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_hardware_guards.py -m gpu`; enabled missing CUDA/capacity must fail. Failures: 24 GB mislabeled 48 GB, free not reserved space, stale host, missing digest, VRAM not recovered.

**Acceptance evidence:** At least 600 GB reserved, separate 96/48 profile hashes, measured totals, host reachability, base digest, guard outputs. Until then `STOP.json` repeats `24576 MiB`, `364 GiB`, `no configured SSH hosts`.

**Non-goals and failure policy:** No parameter-count estimates as proof or full fine-tuning. On resource loss, preserve checkpoints/logs and invalidate affected host only.

**Handoff:** Every protected operation consumes provisioning/profile hashes; OPS-18 consumes separate measured 48/96 evidence.


## OPS-07 Run strict released-data audit and split

**Depends on:** OPS-03, OPS-06, DAT-01 through DAT-12, SPL-01 through SPL-10, STR-01 through STR-06, CLI-01.

**Parallel safety and exclusive file ownership:** Sole writer of `artifacts/data/medreason/derived/`; no extraction, audit, split, or derived-data mutation may overlap. Source archives stay read-only.

**Target paths/symbols:** `MedReasonExample`, image inventory/hashes, transitive grouping, `splits.json`, audit/bias/overlap reports, and deterministic stress manifests.

**Inputs:** Two archive hashes, seed 2026, supplied tags/identifiers, normalized question-option/image hashes, and approved stress protocol. Actual train supervision keys come only from the authorized archives.

**Outputs:** Valid normalized data; image/group manifests; group-disjoint 70/15/15 train/dev/lockbox; participant runtime fixture without answers; deterministic stress manifests; aggregate derived manifest.

**Hard prerequisites and no-skip stop conditions:** Stop on duplicate IDs, missing/decode-failing released image, invalid/nonunique labels, missing train answer, any validation answer, path escape, unexplained group overlap, or source hash mismatch. Report cases lacking defensible signals rather than claim patient-disjointness. Public totals are 17,722 train and 2,532 validation (2,057 MCQ/475 open); any observed mismatch stops. Real audit requires the OPS-06 storage reservation, so current 364 GiB blocks it.

**Implementation:**
1. Execute exactly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.data audit \
  --train artifacts/data/medreason/source/medreason2026_train.zip \
  --validation artifacts/data/medreason/source/medreason2026_validation_participant_facing.zip \
  --output artifacts/data/medreason/derived
```

2. Decode every released image once; record dimensions/mode, decoded-pixel SHA-256, perceptual hash, and supplied modality/identifiers.
3. Form transitive groups and seed-2026 stratified split; verify zero source/image/question group overlap.
4. Emit answer-position, option-length, text-only, template, and source diagnostics without participant-validation training.
5. Build modality-gated deterministic X-ray/MRI stress manifests, retaining originals and labels.
6. Hash/freeze all outputs and seal the lockbox selector.

**Focused tests and exact commands:** Later run `uv run pytest -q tests/challenges/medreason/test_data_audit_cli.py tests/challenges/medreason/test_splits.py tests/challenges/medreason/test_stress.py`. Failures include Unicode loss, corrupt images, filename-only grouping, template mega-groups, overlap, unstable stress, false modality, and answer leakage.

**Acceptance evidence:** Command exit 0; source/normalized/image/group/split/stress hashes; exact public counts and task breakdown when observed; ratios, zero-overlap counters, unavailable strata, ungroupable count, and sanitized logs. Counts or schema differences are investigated, never patched.

**Non-goals and failure policy:** No inferred schema/modality/patient IDs, pseudo-labels, private/hidden data, or private-split claim. Quarantine a derived failure; never mutate source.

**Handoff:** OPS-08 onward consume frozen manifests/selectors; lockbox stays sealed until OPS-16.

## OPS-08 Run three-model zero-shot tournament controls

**Depends on:** OPS-04, OPS-05, OPS-06, OPS-07, EVA-01 through EVA-12, CLI-02, CLI-04, MOD-13, MOD-14.

**Parallel safety and exclusive file ownership:** Models may run on separate exclusive GPUs, but all use the same frozen serializer, parser, split, judges, controls, and config. Judges run sequentially. Exclusive root: `artifacts/runs/medreason/zero_shot_tournament/`.

**Target paths/symbols:** `configs/recipes/medreason/zero_shot_tournament.yaml`, official commit `05748c0341b72dc08132bd108208b78dc14a2f0b`, 31B/26B-A4B/4B rows, no-image and shuffled-image controls, predictions/logits/telemetry.

**Inputs:** Frozen split/model/judge/rubric/processor/template/quantization hashes and preregistered candidate list.

**Outputs:** Raw/parsed predictions, option logits, trace/answer, controls, exact MCQ, proxy GT/VA/RVF, latency/failures/VRAM, and tournament manifest.

**Hard prerequisites and no-skip stop conditions:** Stop on candidate/config drift, official wrapper/hash mismatch, unavailable exact judge for proxy promotion, attention-parity failure, or hardware failure. Official judge prompts, quantization, and tie policy are unpublished, so local GT/VA remain proxies even when exact local checkpoints run. Current 24 GB/no-host state blocks the tournament. Missing optional 4B is recorded unavailable; never substitute.

**Implementation:**
1. Freeze every input hash and execute exactly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.evaluate \
  --config configs/recipes/medreason/zero_shot_tournament.yaml \
  --split-manifest artifacts/data/medreason/derived/splits.json
```

2. Use the identical processor/serializer/parser for clean, no-image, and shuffled-image runs.
3. Run the 70B judge, terminate and prove VRAM recovery, then run the 72B judge.
4. Persist predictions before aggregation; label proxies and apply the exact RVF cap.
5. Hash all outputs without lockbox or participant-validation scoring.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_evaluate_cli.py tests/challenges/medreason/test_zero_shot_tournament.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_zero_shot_tournament.py -m 'real_checkpoint and gpu'`; enabled missing CUDA/assets/access must fail, never skip. Failures: invalid label, parser drift, nonzero judge temperature, missing control, leaked text, or co-resident judges.

**Acceptance evidence:** Candidate/config/split/judge hashes; complete counts; exact MCQ and proxy GT/VA/RVF/control metrics; latency percentiles, failure counts, peak allocated/reserved VRAM, and judge baseline-recovery logs. Every proxy field persists `proxy=true`.

**Non-goals and failure policy:** No advancement, lockbox/participant scoring, or official parity claim. Missing exact judge retains lexical/schema diagnostics but blocks proxy promotion and winning-caliber claims.

**Handoff:** OPS-09 consumes only the immutable tournament manifest and availability flags.

## OPS-09 Advance candidates under fixed baseline gates

**Depends on:** OPS-08, EVA-13, EVA-14, SEL-01, GOV-10.

**Parallel safety and exclusive file ownership:** One serialized decision; no tournament edit/rerun overlaps. Exclusive `artifacts/runs/medreason/baseline_advancement/decision.json`.

**Target paths/symbols:** Fixed advancement predicates, selected/rejected IDs, downstream config hashes, and literal approved commands.

**Inputs:** Tournament evidence. The 31B advances unless its 100-real-batch memory gate fails; 26B advances if within `0.5 pp MCQ / 0.10 GT / 0.10 VA` of 31B and at least `1.5x` faster, or if 31B misses a training, 48 GB, or runtime gate; 4B trains only as optional open route unless it wins MCQ directly; 27B never advances.

**Outputs:** At most 31B, conditional 26B, and 4B, with exact unrounded predicates, reasons, config hashes, and commands.

**Hard prerequisites and no-skip stop conditions:** Require every preregistered row/control, exact judges for proxy comparisons, and measured latency/memory. Stop on a missing metric, unit, or config. The 100-batch evidence must come from provisioned hardware; an RTX 3090 OOM neither rejects the 96 GB candidate nor proves the 48 GB fallback.

**Implementation:** Verify tournament/preregistration hashes; evaluate unrounded predicates in declared precedence; emit every pass/fail reason; bind each advanced row to a reviewed `approved_command`; second-review and freeze before training.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_advancement.py`. Failures include boundary rounding, inverted speed ratio, more than three candidates, 27B promotion, unavailable judge treated as zero, and unconditional 4B fusion.

**Acceptance evidence:** Input metric hashes, exact booleans, selected/rejected set, config/command hashes, reviewer identity, and sanitized decision log. This task allocates no model and produces no new metric.

**Non-goals and failure policy:** No cherry-picking, new checkpoint, threshold relaxation, or substitute model. Missing evidence produces `STOP.json`.

**Handoff:** OPS-10/11 consume the exact candidate/config/command set.

## OPS-10 Train and lock advanced MCQ adapters

**Depends on:** OPS-09, OPS-06, MCQ-01 through MCQ-15, MOD-09 through MOD-14, CLI-03, CLI-05, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** One process per exclusive GPU; pilots may parallelize only with identical data-order manifests and disjoint outputs. Run roots are candidate-specific under `artifacts/runs/medreason/`; adapters are `artifacts/models/medreason/adapters/<run_id>/<adapter_name>/`.

**Target paths/symbols:** `configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml`, 100-batch `memory_gate.json`, LR pilots, hard-negative/contrastive stage, adapter-only `adapter.safetensors` and `manifest.json`.

**Inputs:** Advanced candidates, train/dev only, deterministic permutations, exact model/processor/template/split/config hashes, seed 2026.

**Outputs:** Pilot/final metadata, dev predictions, memory/latency logs, exact LoRA targets, and accepted adapter-only artifacts.

**Hard prerequisites and no-skip stop conditions:** Run 100 real batches below `min(85 GiB, 0.90 * total_device_memory)` before long training. Current 24 GB/no-host is blocked. Stop on silent truncation, undeclared bucket overflow, masking/remap error, nonfinite loss, provenance drift, memory failure, or full-base export. Never full-fine-tune.

**Implementation:**
1. Verify target `<label>: <option text>`, assistant-only masking, per-epoch remap, buckets 2,048/4,096/8,192/16,384, NF4 double quantization with BF16, rank 16/alpha 32/dropout 0.05, accumulation 16, and discovered target list.
2. Run 100 batches and persist `artifacts/runs/medreason/<run_id>/memory_gate.json` with device identity/total, batch count, allocated/reserved peak, threshold, and pass.
3. Run the primary exactly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train \
  --config configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml
```

   Another advanced candidate runs only its literal hash-bound OPS-09 command; absence is a hard stop.
4. Pilot approved learning rates for 250 steps on identical order; select by dev MCQ, clean/OOD non-inferiority, then dev loss. Continue two epochs with fixed evaluation intervals.
5. Mine two wrong options after epoch 1 and batch the three-candidate contrastive objective.
6. Evaluate original/cyclic/reverse conditional scores; optional thinking waits for OPS-12.
7. Export/reload adapter safetensors. Manifest requires `kind=adapter_only`, exact base ID/revision, architecture, config hash, and tensor file hashes.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_mcq_training.py tests/challenges/medreason/test_mcq_scoring.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_mcq_training.py -m 'real_checkpoint and gpu'`; enabled missing prerequisites fail. Plausible bugs are remap, mask, normalization, truncation, missing target list, base-weight export, resume drift, and OOM at batch 99.

**Acceptance evidence:** Memory-gate hash/values; run/config/data/base/processor/template hashes; LR precedence metrics; dev clean/OOD/control metrics; adapter/manifest/reload hashes; steps/checkpoints/failure counts. Fixture evidence is labelled fixture and cannot satisfy protected adapter/hardware acceptance.

**Non-goals and failure policy:** No lockbox, participant validation, pseudo-label, full tune, or default GRPO. Resume only a hash-identical checkpoint; preserve/reject failure without silently reducing quality.

**Handoff:** OPS-12/13 consume clean adapter manifest hash, step metadata, dev predictions, and parent ID.

## OPS-11 Train and lock advanced open adapters

**Depends on:** OPS-09, OPS-06, OPEN-01 through OPEN-15, MOD-09 through MOD-14, CLI-03, CLI-06, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** Exclusive GPU; may overlap OPS-10 only on disjoint GPU/output. Primary run root `artifacts/runs/medreason/gemma4_31b_open_qlora/`; adapters use the same adapter-only contract.

**Target paths/symbols:** `configs/recipes/medreason/gemma4_31b_open_qlora.yaml`, structured schema/answer-only path, 50/50 retention, group sampling, bounded export, memory gate, adapter artifacts.

**Inputs:** Advanced candidates, released-only supervision, train/dev groups, exact provenance/asset hashes, seed 2026.

**Outputs:** Pilots, clean open adapters, answer-only evidence-extraction diagnostics, proxy metrics/controls, memory logs, immutable manifests.

**Hard prerequisites and no-skip stop conditions:** Same 100-batch gate; current 24 GB/no-host is blocked. Stop on pseudo/private trace, evidence without released provenance, retention imbalance, repeated group within a batch, trace beyond 160 or answer beyond 48 generated tokens, private-thought leakage, truncation, or unavailable judges for proxy promotion.

**Implementation:**
1. Hash target provenance as released evidence or answer-only; never write prompted evidence back as supervision.
2. Validate schema/masking/native processor/buckets, 50/50 retention, and group-only oversampling.
3. Run and persist the 100-batch memory gate.
4. Run the primary exactly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train \
  --config configs/recipes/medreason/gemma4_31b_open_qlora.yaml
```

   Other routes require the literal OPS-09 command.
5. Pilot approved rates for 250 steps and train chosen route two epochs with fixed stopping; evaluate frozen-base evidence extraction for answer-only cases.
6. Evaluate clean/OOD/no-image/shuffled-image with sequential proxies and RVF caps.
7. Export/reload adapter-only safetensors with exact base/config/tensor manifest.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_open_training.py tests/challenges/medreason/test_open_export.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_open_training.py -m 'real_checkpoint and gpu'`. Enabled missing prerequisites fail. Failures include pseudo-trace, thought leak, invalid/unbounded schema, retention/group error, non-grounding, and base weights in export.

**Acceptance evidence:** Memory gate; supervision counts/hashes; LR decision; proxy-labelled clean/OOD/control/RVF metrics; schema failure rate; adapter/manifest/reload and provenance hashes. Fixture proof remains distinct from real judge/hardware acceptance.

**Non-goals and failure policy:** No generated supervision, DPO, agent loop, lockbox, or validation tuning. Missing judges blocks proxy lock, not schema diagnostics.

**Handoff:** OPS-12/13 receive adapter hashes, evidence mode, bounds, prediction hashes, and parent IDs.

## OPS-12 Test optional components one at time

**Depends on:** OPS-10, OPS-11, RUN-09 through RUN-14, MCQ-13, MCQ-14, OPEN-11 through OPEN-14, SEL-01 through SEL-05, STR-05.

**Parallel safety and exclusive file ownership:** Candidate runs may parallelize only with a frozen parent, disjoint GPU/output, and no mutable index. Decisions serialize under `artifacts/runs/medreason/selection/{preregistration,comparisons}/`.

**Target paths/symbols:** Consistency, MCQ thinking/GRPO, open vision/thinking/four-sample, specialist, views/crops, retrieval; `configs/recipes/medreason/selection_preregistration.yaml`; paired bootstrap/Holm artifacts.

**Inputs:** Clean parents, frozen hypotheses/order, dev/stress pairs, 1,000 group resamples, intended metrics, margins, grounding/runtime gates.

**Outputs:** One child-parent report each, adjusted intervals, decisions, accepted route graph, and rejected audit hashes.

**Hard prerequisites and no-skip stop conditions:** Freeze candidates, prompts, parsers, transforms, seeds, and tolerances before first result. Missing preregistered result/judge stops the family. Promote only when intended adjusted lower bound is `> 0` and all others exceed `-0.2 pp MCQ / -0.05 GT / -0.05 VA`; visual options also pass shuffled-image grounding. Use the same 1,000 group resamples and Holm family-wise alpha 0.05. Current hardware remains blocked.

**Implementation:**
1. Run the manifest-bound `python -m medfm.challenges.medreason.selection freeze-preregistration --config configs/recipes/medreason/selection_preregistration.yaml` after implementation supplies required upstream SHA-256 arguments.
2. Change exactly one component per child and enforce approved component-specific pilots/gates.
3. Persist paired predictions before sequential judges.
4. Run manifest-bound `bootstrap`, `correct-family`, `gate`, and `grounding-gate` subcommands; never omit required input SHA-256.
5. Exclude failed components from downstream manifests while retaining audit artifacts; combine only preregistered compatible passes.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_optional_gates.py tests/challenges/medreason/test_selection.py`. Failures: multi-change child, case rather than group resampling, mismatched stress samples, uncorrected decision, equality-at-zero acceptance, grounding omission, retrieval leakage, or thought leak.

**Acceptance evidence:** Registry/resample hashes; per-child prediction hashes; raw/adjusted intervals; all margins, grounding, runtime, and VRAM fields; exact boolean/reason; accepted graph hash. Protected/proxy labels remain explicit.

**Non-goals and failure policy:** No post-result hypotheses, failed combinations, agent loop, shipped judges, or lockbox/participant fitting. Missing family row means no promotions.

**Handoff:** OPS-13 consumes the surviving graph and decision hashes.

## OPS-13 Generate three-fold out-of-fold predictions

**Depends on:** OPS-12, SEL-01, SEL-06, SEL-07, RUN-01 through RUN-15, OPS-06.

**Parallel safety and exclusive file ownership:** Fold/routes may run on exclusive GPUs with disjoint outputs and one immutable fold manifest. No calibrator reads partial outputs. Exclusive `artifacts/runs/medreason/oof/`.

**Target paths/symbols:** `artifacts/data/medreason/derived/oof_folds.json`, fold-local training/retrieval, held-out predictions/features/best steps, and aggregate OOF manifest.

**Inputs:** Frozen 85% train-plus-dev pool, surviving route graph, recipes/model/judge hashes, and no lockbox or participant-validation data.

**Outputs:** Exactly one held-out row per eligible case/route with fold/group ID, logits or structured response, support/confidence/view features, failure class, latency/VRAM, and train-group-manifest hash.

**Hard prerequisites and no-skip stop conditions:** Each group must occur in exactly one held-out fold and never its training fold; retrieval is fold-local with perceptual-hash exclusions. Stop on duplicate/missing OOF cases, group leakage, mismatched config/asset hash, incomplete route, forbidden ID, or any calibrator reading partial data. Provisioned real GPUs are required; current 24 GB/no-host blocks this operation.

**Implementation:**
1. Create folds with manifest-bound `python -m medfm.challenges.medreason.oof create-folds`; freeze `oof_folds.json` and three literal train/evaluate command hashes before execution.
2. For each fold, train every surviving route on the other two folds with identical policy and record its best step.
3. Predict the held-out fold exactly once, including raw/parsed outputs and all calibration/fusion/view features; run controls and sequential proxy judges.
4. Validate group exclusion and join by case ID without averaging duplicate predictions.
5. Seal each fold, then run manifest-bound `python -m medfm.challenges.medreason.oof index-predictions`; publish aggregate only after all routes/folds pass.

**Focused tests and exact commands:** Later run `uv run pytest -q tests/challenges/medreason/test_oof.py -k 'fold or leakage or completeness or persistence'`. Protected runs must use the exact literal commands stored in the OOF manifest and compare upstream SHA-256 before execution. Failures: source-group overlap, duplicate row, absent fallback row, global retrieval index, partial calibration read, or fold config drift.

**Acceptance evidence:** Fold hash; three train/held-out group hashes; per-route prediction hashes/counts; one-row-per-case completeness; zero leakage counters; fold best steps; proxy/control/latency/VRAM/failure summaries; literal command/config hashes. No lockbox result exists.

**Non-goals and failure policy:** OOF is not a new search. No final-dev, lockbox, or participant fitting. Re-run only the exact failed fold and invalidate any fit that read partial data.

**Handoff:** OPS-14 consumes the complete sealed OOF manifest and three best-step values.

## OPS-14 Fit calibration and select one system

**Depends on:** OPS-13, SEL-08 through SEL-12, RUN-10 through RUN-13.

**Parallel safety and exclusive file ownership:** Independent fits may use isolated working directories; final decision is serialized. Exclusive `artifacts/runs/medreason/selection/calibration/`.

**Target paths/symbols:** OOF temperatures, non-negative fusion/support weights, confidence/view/support thresholds, median steps, normalized worst-margin selection, chosen system/profile, Pareto table.

**Inputs:** Complete OOF predictions; ambition thresholds `97.5% MCQ`, `2.15/4` proxy GT, `2.85/4` proxy VA; latency/VRAM/profile evidence.

**Outputs:** Hash-addressed calibrations, exactly one selected route graph/profile, median step count, and exact development-pool config/literal command under `artifacts/runs/medreason/research_full_pool/`.

**Hard prerequisites and no-skip stop conditions:** Fit only OOF rows and require non-negative weights. First require all ambitions, then maximize worst normalized margin, then prefer faster/simpler. If none meet ambitions, report that honestly and do not claim winning-caliber. Stop on incomplete OOF, forbidden IDs, negative/nonfinite fit, or unmeasured hardware profile. The local 24 GB card cannot be selected as 48/96 GB proof.

**Implementation:**
1. Verify OOF/case-universe hashes and reject forbidden IDs.
2. Run manifest-bound `python -m medfm.challenges.medreason.oof fit-temperature`, `fit-fusion`, and `fit-thresholds`, each requiring upstream SHA-256.
3. Run `python -m medfm.challenges.medreason.oof median-steps` and freeze the exact median of three.
4. Run manifest-bound `python -m medfm.challenges.medreason.selection select` using the fixed rule and exact precision.
5. Emit/hash one selected system, calibration bundle, Pareto table, full-pool training config, and literal approved command.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_calibration.py tests/challenges/medreason/test_system_selection.py`. Failures: final-dev fitting, negative weights, rounding rank reversal, incorrect median, two selected systems, or unsupported profile label.

**Acceptance evidence:** Input OOF hash; calibration values/hashes; non-negativity checks; three best steps/median; ambitions and normalized margins; selected ID/profile; Pareto/config/command hashes. Every GT/VA field persists `proxy=true`.

**Non-goals and failure policy:** No lockbox peeking, new component, shipped judge, unconstrained weights, or hand-tuned threshold. A fit failure stops selection.

**Handoff:** OPS-15 receives one selected system/calibration/step/profile/config/command bundle.

## OPS-15 Train and freeze development-pool research system

**Depends on:** OPS-14, SEL-13, SEL-14, CLI-03, CLI-08, CLI-09, OPS-06.

**Parallel safety and exclusive file ownership:** Exactly one 85%-pool final run; no competing candidate may train or mutate calibrations. Exclusive `artifacts/runs/medreason/research_full_pool/` and `artifacts/models/medreason/research_evaluation/`.

**Target paths/symbols:** Selected adapters/routes, complete 85% pool, frozen median steps/calibration, canonical research manifest, unused lockbox authorization token.

**Inputs:** OPS-14 hashes and literal command, train-plus-dev group manifest, exact base/processor/template hashes, seed and hardware profile.

**Outputs:** One trained adapter set, offline-reload proof, frozen research manifest/inference config, and bound one-use token.

**Hard prerequisites and no-skip stop conditions:** Architecture, data, transforms, steps, seed, calibrations, and thresholds must match OPS-14 exactly. Current 24 GB/no-host is blocked. Stop on config drift, early/extra steps, lockbox access, refit, provenance drift, nonfinite loss, reload mismatch, or incomplete manifest. There is no choose-the-best rerun.

**Implementation:**
1. Verify all selected hashes and create a single-use training authorization.
2. Execute only the frozen literal command, implemented by manifest-bound `python -m medfm.challenges.medreason.freeze train-development`.
3. Train the complete 85% once to the median optimizer-step count; use lockbox/participant validation zero times.
4. Export adapter-only artifacts, reload offline, and run development-only schema/control checks.
5. Run `python -m medfm.challenges.medreason.freeze freeze-research` with required upstream manifest SHA-256; include every base/processor/adapter/config/data/calibration/hardware hash.
6. Freeze the research manifest and create an unused one-shot token bound to its hash.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_research_freeze.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_research_freeze.py -m 'real_checkpoint and gpu'`; when enabled, missing prerequisites fail, never skip. Failures: double authorization, changed step count, base weights in adapter, mutable calibration, lockbox training path, or manifest omission.

**Acceptance evidence:** Exact command/config/input hashes; fixed step count; training/checkpoint and memory logs; adapter/calibration/reload hashes; canonical research-manifest SHA-256; unused token. Fixture proof is not protected artifact acceptance.

**Non-goals and failure policy:** No rerun selection, new early stopping, architecture change, or lockbox metric. Resume only the same hash-bound checkpoint; irrecoverable failure requires a new research cycle before lockbox.

**Handoff:** OPS-16 receives only the frozen research manifest/config and unused token.

## OPS-16 Evaluate frozen lockbox exactly once

**Depends on:** OPS-15, SEL-15, CLI-02, CLI-07, EVA-01 through EVA-15, OPS-06.

**Parallel safety and exclusive file ownership:** No parallel candidate or mutable research process. One operator owns sealed input and `artifacts/runs/medreason/lockbox/`; judges remain sequential.

**Target paths/symbols:** `configs/recipes/medreason/frozen_system.yaml`, research manifest, atomic one-shot token, sealed 15% selector, predictions, and 1,000 group-bootstrap intervals.

**Inputs:** Exact research/split/judge/rubric hashes and one system only.

**Outputs:** One lockbox report with exact MCQ, proxy GT/VA/RVF, intervals, controls, latency/VRAM/failures, and permanently consumed token.

**Hard prerequisites and no-skip stop conditions:** Token must be unused and bound to exact hashes. Stop before inference on mismatch, prior use, judge failure, or nonexclusive access. After any partial result is visible, no method, calibration, threshold, prompt, architecture, comparison, retraining, or rerun is allowed. Current hardware is blocked.

**Implementation:**
1. Verify hashes and atomically mark token `started` before selector reveal.
2. Execute exactly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.evaluate \
  --config configs/recipes/medreason/frozen_system.yaml \
  --split lockbox --bootstrap-resamples 1000
```

   The freeze state machine may invoke `python -m medfm.challenges.medreason.freeze evaluate-lockbox` only if it records this exact approved evaluation command and hashes.
3. Persist raw/parsed predictions before exact MCQ and sequential proxy/RVF aggregation; compute 1,000 group intervals and controls.
4. Mark token `consumed` on success, failure, interruption, or partial disclosure; hash complete/partial report. Never rerun.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_lockbox_once.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_lockbox_once.py -m 'real_checkpoint and gpu'`. Failures: double use, crash after first output, hash mismatch, two systems, absent fallback prediction, or post-result mutation.

**Acceptance evidence:** Token hash/timestamps; research/split/config/judge hashes; prediction hash/count; exact MCQ and proxy GT/VA/RVF points/intervals; ambition comparisons only if observed; controls, latency, VRAM, failures, and sanitized logs. A crash yields consumed-token plus partial-report evidence, not retry permission.

**Non-goals and failure policy:** No rerun, selection, tuning, official-score claim, or attribution to a later all-data artifact.

**Handoff:** OPS-17 receives only frozen method/steps/calibration and lockbox report hash, never lockbox values as decisions.

## OPS-17 Train distinct all-label deployment system

**Depends on:** OPS-01, OPS-16, SEL-16, CLI-03, CLI-08, CLI-09, OPS-06.

**Parallel safety and exclusive file ownership:** Exactly one all-label run; research artifact stays read-only. Exclusive `artifacts/runs/medreason/all_label_deployment/` and `artifacts/models/medreason/all_label_deployment/`.

**Target paths/symbols:** All 17,722 released labelled cases, frozen architecture/steps/calibration, distinct deployment manifest, and eligibility evidence.

**Inputs:** Explicit organizer exception/contract, lockbox completion hash, selected method, full-label manifest, exact assets/profile.

**Outputs:** One all-label adapter/runtime bundle, offline-reload proof, deployment manifest with explicit non-attribution.

**Hard prerequisites and no-skip stop conditions:** This stage is only for late official hidden evaluation. Without OPS-01's written exception, write `STOP.json` and do not train. Only the approved data selector and artifact identity may differ from research config. No lockbox-driven change and no participant-validation output. Current 24 GB/364 GiB/no-host independently blocks execution.

**Implementation:**
1. Verify exception scope/deadline and selected hashes.
2. Structurally diff all-label config against research config; fail on any difference beyond selector/identity.
3. Execute one hash-bound `python -m medfm.challenges.medreason.freeze train-all-label` command to the frozen median steps.
4. Export/reload adapter-only artifacts and reuse OOF calibration/thresholds unchanged.
5. Freeze deployment manifest with `lockbox_score_attributable=false`; retain separate research manifest.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_all_label_deployment.py`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_all_label_deployment.py -m 'real_checkpoint and gpu'`. Failures: absent exception, participant IDs, changed steps/calibration, lockbox-driven config, two runs, or score-attribution leak.

**Acceptance evidence:** Exception hash; structural config diff; command/config/data/model hashes; training/VRAM logs; adapter/reload hashes; deployment-manifest SHA-256; distinct research link and `lockbox_score_attributable=false`. No deployment performance is invented.

**Non-goals and failure policy:** No ordinary post-challenge tuning, participant-validation data, method change, or research-score claim. Missing exception/resources produces durable `STOP`.

**Handoff:** Submission-mode OPS-18 consumes the deployment manifest. A research reproducibility Docker may consume OPS-15 only and must be labelled non-submission.

## OPS-18 Build validate and export final Docker

**Depends on:** DOC-01 through DOC-16, GOV-07, OPS-06, OPS-07, OPS-15, OPS-16; submission mode additionally depends on OPS-01 and OPS-17.

**Parallel safety and exclusive file ownership:** Serialize build, smoke, full run, duplicate run, network check, hardware measurement, and export against one immutable vendor manifest/base digest. Exclusive `docker/medreason/.build/`, `artifacts/submission/output/`, and `artifacts/submission/`.

**Target paths/symbols:** `docker/medreason/{Dockerfile,.dockerignore,requirements.lock,process.py,custom_system.py,test.sh,export.sh,UPSTREAM.md}`, official `medreason_docker/`, unchanged `tools/validate_output.py`, official fixture, `manifests/{official-template.json,wheelhouse.json,runtime-assets.json}`, vendor/package inventories, `medreason-gemma:final`, and final tarball.

**Inputs:** OPS-17 deployment manifest for submission mode or OPS-15 research manifest for clearly post-challenge mode; redistribution-approved assets only; 2,532-case runtime; local digest-pinned base; measured target profiles.

**Outputs:** No-pull/no-network image; fixture/full/duplicate/network/memory/package evidence; validated results; tarball and SHA-256 manifest.

**Hard prerequisites and no-skip stop conditions:** Before GPU allocation the manifest verifier must reject missing, extra, tampered, symlinked, path-escaping, or wrong-revision files. Judges and unapproved MedGemma must be absent. Require local digest base, ABI-matched wheelhouse, no network, complete inventory, and official one-per-ID version-1.0 schema. Submission mode stops without exception/deployment. Do not claim 48 GB without a matching physical run. Current 364 GiB/no-host/24 GB state blocks final acceptance; unpublished official time/archive limits must come from OPS-01.

**Implementation:**
1. Record `mode=research|submission`, bind one system manifest, and verify license/redistribution, package/wheel/runtime/official/base hashes.
2. Package inventory must include `medfm/challenges/medreason/` and intentional resources and exclude tests, source archives, judges, secrets, caches, optimizer checkpoints, lockbox references, and unapproved weights.
3. Build and smoke exactly:

```bash
ROOT="$PWD"
(cd docker/medreason && \
  docker build --pull=false --network=none -t medreason-gemma:final . && \
  ./test.sh medreason-gemma:final)
```

4. Run all participant-facing cases exactly:

```bash
docker run --rm --gpus all --network none \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e MEDFM_OFFLINE=1 \
  -v "$PWD/artifacts/data/medreason/validation_runtime:/input:ro" \
  -v "$PWD/artifacts/submission/output:/output" \
  medreason-gemma:final
```

5. Validate exactly:

```bash
uv run python docker/medreason/tools/validate_output.py \
  artifacts/submission/output/results.json \
  --input-json artifacts/data/medreason/validation_runtime/cases.json
```

6. Repeat the complete run into two fresh output directories and byte-compare only `results.json`; require runtime network attempts fail and logs pass the privacy scan.
7. Measure worst-case latency and allocated/reserved VRAM separately on each claimed 96 GB and 48 GB profile.
8. Export and hash exactly:

```bash
(cd docker/medreason && \
  ./export.sh medreason-gemma:final "$ROOT/artifacts/submission/medreason-gemma.tar.gz")
sha256sum artifacts/submission/medreason-gemma.tar.gz
```

9. Freeze image ID/digest, archive, vendor/package/system manifests, outputs, logs, and reports.

**Focused tests and exact commands:** `uv run pytest -q tests/challenges/medreason/test_docker_release.py tests/challenges/medreason/test_packaging.py tests/phase_01/test_packaging.py`. Packaging inventory coverage must update `tests/phase_01/test_packaging.py::SUBPACKAGES` to include `challenges`. Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_docker_release.py -m 'real_checkpoint and gpu'`; enabled missing Docker/CUDA/assets fails, never skips. Failures: unpinned base, online resolution, missing/extra file, judge inclusion, network success, wrong count/order/label, empty open fields, nondeterminism, text leak, 48 GB OOM, or export drift.

**Acceptance evidence:** Base/wheel/vendor/package/system hashes; official-template hash tied to commit `05748c0341b72dc08132bd108208b78dc14a2f0b`; image ID/digest; two-case fixture pass; exactly one valid output for 2,532 IDs (2,057 MCQ and 475 open); every MCQ is one supplied label; every open trace/answer is nonempty; duplicate result hashes equal; network attempts fail; zero privacy matches; measured latency/VRAM for each actually proven profile; validator exit 0; tarball SHA-256. Fixture proof is not full participant/hardware acceptance.

**Non-goals and failure policy:** No network/API/human runtime, judge image, mutable Hub resolution, tuning, or score claim. Never patch a validated image. Any mismatch requires a new attempt and the full sequence. Without an exception, retain only a clearly labelled research Docker and do not submit.

**Handoff:** Release custody receives mode, archive/image/vendor/package/system hashes, validator/duplicate/network/hardware evidence, exception hash if applicable, and only actually measured profile claims.