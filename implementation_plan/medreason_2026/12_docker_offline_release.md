# Docker offline release

This phase packages the already-selected all-label deployment system. It does not select models, retrain adapters, tune on participant validation, or claim official eligibility. All official files are pinned to `medreason26/MedReason-Challenge-Docker@05748c0341b72dc08132bd108208b78dc14a2f0b`. Git blob IDs and measured SHA-256 are distinct: release manifests must contain the full measured SHA-256, not a Git SHA-1 or an abbreviated value. Commands below are implementation/acceptance commands; none were executed while writing this plan.

## DOC-01 Create isolated official-compatible Docker context

- **Depends on:** GOV-07, GOV-08, SCH-01, SCH-04, EVA-01, RUN-15, SEL-16.
- **Parallel safety and exclusive file ownership:** May run with DOC-02, DOC-03, DOC-08, and DOC-09 after GOV-07 freezes source hashes. This task exclusively owns `docker/medreason/.dockerignore`, `docker/medreason/UPSTREAM.md`, `docker/medreason/tools/stage_context.py`, `tests/challenges/medreason/test_docker_context.py`, and the context layout. EVA-01 exclusively owns `docker/medreason/vendor/MedReason-Evaluation/`; consume it read-only. Other DOC tasks own only their named files.
- **Target paths/symbols:** new `docker/medreason/{app,manifests,test,tools,vendor}/`; `stage_context.py::{ContextFile,stage_context,validate_context_inventory}`; ignored generated `artifacts/submission/medreason_docker_context/`; existing `tests/phase_01/test_packaging.py::SUBPACKAGES` (add `"challenges"`). Vendor pinned runtime core under `docker/medreason/vendor/MedReason-Docker/medreason_docker/` and evaluator-only EVA files under `vendor/MedReason-Evaluation/`.
- **Inputs:** GOV-07's official-source SHA-256 manifest, EVA-01 evaluator bytes, source-controlled release files, and SEL-16's immutable deployment manifest. Official anchors include runtime `schema.py` (Git blob `3755e1c...`, verified SHA-256 prefix `c2beba09...`), `systems/base.py` (Git blob `8d60f09...`, verified SHA-256 prefix `3ef6d1ce...`), `io.py` (`b5dc514...`), `validation.py` (`b1d8b87...`), evaluator `scoring.py` (`cb8b249...`, verified SHA-256 prefix `b260a6fb...`), and validator `tools/validate_output.py` (`4e7e7bb...`). **External artifacts/hardware:** no GPU for fixture inventory; full staging needs approved model access and at least the planned 600 GiB storage reserve. Current free space is 364 GiB, so full staging is blocked.
- **Outputs:** a deny-by-default, copy-only context plus canonical `context-manifest.json` with schema version, sorted POSIX relative path, role, size, mode, SHA-256, and child-manifest hashes. It contains no `.git`, credentials, participant data, judges in the runtime image, caches, run outputs, or training state.
- **Ordered implementation:**
  1. Freeze the layout and source ownership above. Keep wheelhouse/runtime assets only in the ignored generated context.
  2. Implement copy containment checks; reject absolute/parent paths, symlinks, devices, sockets, executable data files, duplicates, and unlisted files.
  3. Preserve official files byte-for-byte; custom wrappers may call them but never silently alter official behavior.
  4. Make `.dockerignore` deny by default and explicitly allow only the manifest inventory.
  5. Reuse bundle-style path/symlink/executable checks and canonical sorted UTF-8 JSON. Do not use broad `AuditEvent` or `FailureReporter` in the container because they carry timestamps/random IDs/raw traces or paths.
  6. Add packaging inventory coverage: `medfm.challenges` imports without CUDA initialization, and the built wheel includes `medfm/challenges/medreason/**` while excluding tests/artifacts/secrets.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_docker_context.py tests/phase_01/test_packaging.py::test_subpackages_importable`. Inventory command: `uv build --wheel --out-dir artifacts/submission/wheel-dist && uv run --frozen python docker/medreason/tools/stage_context.py --check-wheel 'artifacts/submission/wheel-dist/medfm-*.whl' --inventory-only`. Test traversal, symlinks, an accidental judge/model cache, `.env`, optimizer state, unknown wheel, executable weight, and post-copy bit flip.
- **Acceptance evidence:** passing focused output; sorted context manifest and SHA-256; wheel inventory naming `medfm/challenges/medreason`; negative cases exiting nonzero. This is fixture/code acceptance, not real-artifact or GPU acceptance.
- **Non-goals/failure policy:** No participant data, proxy judges, training state, Hub caches, or secrets. **Fail closed:** no context is releasable with an unknown/missing path, symlink, hash/mode mismatch, unverified official byte, insufficient storage, or package inventory omission.
- **Handoff:** DOC-05 consumes the context/official manifest hashes; DOC-10 builds only this context; DOC-16 records both hashes.

## DOC-02 Pin CUDA Python base image by digest

- **Depends on:** GOV-03, GOV-09, GOV-10, DOC-01.
- **Parallel safety and exclusive file ownership:** May run with DOC-03/04/08/09. Exclusively owns `docker/medreason/base-image.lock.json`, `docker/medreason/Dockerfile`, `docker/medreason/tools/verify_base_image.py`, and `tests/challenges/medreason/test_docker_base.py`. DOC-03 supplies the fixed offline install stanza; DOC-06 supplies only the final entrypoint contract.
- **Target paths/symbols:** `BaseImageLock`, `parse_digest_reference`, `verify_local_base`. Existing `docker/Dockerfile` (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`) and `Dockerfile.ci` (`python:3.13-slim-bookworm`) are development patterns only: both use mutable tags and networked installation.
- **Inputs:** an approved CUDA/Python image compatible with selected PyTorch, CUDA/cuDNN, CPython, NVIDIA driver, and license policy. **External artifacts/hardware:** one-time registry access to resolve/pull the candidate; final build requires it already local. GPU fit is deferred to DOC-15.
- **Outputs:** `base-image.lock.json` with full `repository@sha256:<64>`, platform `linux/amd64`, local image/config digest, CUDA/cuDNN/Python versions, source registry, resolver evidence, and license-evidence hash. Dockerfile uses only a validated digest argument in `FROM`; no tag fallback, `apt`, remote `ADD`, or external `COPY --from`.
- **Ordered implementation:**
  1. Select an image already containing required CUDA/Python runtime libraries; do not assume the current CUDA development image or official CPU `python:3.11-slim` is suitable.
  2. Resolve and record the immutable OCI digest and local config identity; never invent a digest.
  3. Reject tag-only/malformed/non-SHA-256/multi-platform references and ABI/platform mismatch.
  4. Verify local identity via `docker image inspect` before an offline build; never silently substitute a same-name tag.
  5. Use a non-root runtime user and writable space only for `/output` and bounded temporary files.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_docker_base.py`. Preflight: `BASE_REF="$(uv run --frozen python docker/medreason/tools/verify_base_image.py --lock docker/medreason/base-image.lock.json --print-reference)" && docker image inspect "$BASE_REF" >/dev/null`. Test mutable tag, wrong arch/ABI, local config drift, networked Dockerfile instruction, and absent local image.
- **Acceptance evidence:** verified lock JSON, matching local image/config identity, ABI report, and static Dockerfile audit. Metadata-only success does not prove CUDA runtime or 48/96 GB support.
- **Non-goals/failure policy:** No hardware-fit or security claim from tag metadata. **Fail closed:** build is blocked until the approved exact digest is local and platform, ABI, license, and identity checks pass.
- **Handoff:** DOC-03 targets this ABI; DOC-10 receives the verified digest; DOC-15 and DOC-16 bind evidence/export to it.

## DOC-03 Build hash-locked offline Python wheelhouse

- **Depends on:** DOC-01, DOC-02, MOD-01, CLI-08.
- **Parallel safety and exclusive file ownership:** May run with DOC-04/08/09. Exclusively owns `docker/medreason/requirements.lock`, `tools/build_wheelhouse.py`, `tools/verify_wheelhouse.py`, `manifests/wheelhouse.json`, and `tests/challenges/medreason/test_wheelhouse.py`; do not modify `uv.lock` or `pyproject.toml` here.
- **Target paths/symbols:** `WheelRecord`, `build_inventory`, `verify_wheelhouse`; existing exact `uv.lock` (not itself a wheelhouse), `pyproject.toml` (Torch `2.9.0`; Transformers lock resolves `5.14.1`), and generated ignored `vendor/wheelhouse/`.
- **Inputs:** frozen `uv.lock`, production-only extras, first-party wheel, and DOC-02 CPython/platform/CUDA ABI. **External artifacts/hardware:** network and storage are allowed only during separate wheelhouse preparation; no GPU. Every redistributed dependency needs license evidence.
- **Outputs:** hash-complete `requirements.lock`; compatible wheels only; `wheelhouse.json` with filename, normalized project/version, tags, size, SHA-256, source identity, license hash, and lock provenance; first-party wheel inventory including the challenge package.
- **Ordered implementation:**
  1. Export production dependencies only; exclude tests/dev, proxy judges, TPU, trackers, download clients not used at runtime, and server extras.
  2. Populate wheels for exact CPython 3.11/Linux x86_64/CUDA ABI in a networked environment matching the base. Reject sdists/source builds, editable/VCS/mutable direct references, duplicates, and wrong tags.
  3. Build the first-party wheel from the frozen revision and hash it.
  4. Verify one-to-one lock/inventory coverage and scan wheel contents for secrets, data, caches, and omitted `medfm.challenges` modules.
  5. Docker install must be `python -m pip install --no-index --find-links=/opt/wheelhouse --require-hashes -r /opt/app/requirements.lock`; no resolution at build/runtime.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_wheelhouse.py tests/phase_01/test_packaging.py`. Preparation: `uv run --frozen python docker/medreason/tools/build_wheelhouse.py --uv-lock uv.lock --base-lock docker/medreason/base-image.lock.json --output artifacts/submission/wheelhouse`. Offline check: `env -u HTTPS_PROXY -u HTTP_PROXY uv run --frozen python docker/medreason/tools/verify_wheelhouse.py --requirements docker/medreason/requirements.lock --wheelhouse artifacts/submission/wheelhouse --manifest docker/medreason/manifests/wheelhouse.json`. Fail missing transitive wheel, changed byte, wrong ABI, sdist, duplicate, unhashed line, or absent challenge module.
- **Acceptance evidence:** zero unresolved requirements; sorted manifest/hash; wheel inventory; clean offline install in a base-matched empty environment. That does not prove real CUDA inference.
- **Non-goals/failure policy:** No build/runtime downloads or judge dependencies. **Fail closed:** any missing/incompatible/unhashed/unlicensed wheel, lock drift, sdist, or package-inventory omission blocks staging.
- **Handoff:** DOC-05 verifies the manifest; DOC-10 installs offline; DOC-16 records lock/wheelhouse hashes.

## DOC-04 Vendor model processor adapter and calibration artifacts

- **Depends on:** GOV-03, GOV-04, GOV-06, GOV-08, MOD-02, MOD-03, MOD-09, MOD-11, MCQ-15, OPEN-15, SEL-14, SEL-16, RUN-01.
- **Parallel safety and exclusive file ownership:** May run with DOC-02/03/08/09. Exclusively owns `docker/medreason/tools/stage_runtime_assets.py`, `manifests/runtime-assets.json`, and `tests/challenges/medreason/test_runtime_asset_staging.py`; immutable `artifacts/models/medreason/` inputs remain producer-owned.
- **Target paths/symbols:** `RuntimeAsset`, `stage_runtime_assets`, `reject_training_state`; existing `medfm.registry.weights` checksum verification and `medfm.inference.bundle.validate_bundle`/`load_bundle`; generated `vendor/runtime-assets/{models/primary,processor,adapters/mcq,adapters/open,calibration}`.
- **Inputs:** SEL-16's distinct all-label deployment manifest; exact Gemma 4 model revision, processor, tokenizer, chat template, configs/license notice, quantization hash, adapter-only safetensors, and OOF calibration/thresholds. **External artifacts/hardware:** approved gated access and 600 GiB storage. Current 364 GiB blocks full staging. MedGemma configs/templates are protected and redistribution is unresolved, so omit it unless explicit approval and exact file hashes exist.
- **Outputs:** content-addressed files and `runtime-assets.json` with role/path/size/SHA-256, model ID/exact revision, source manifest, license hash, processor/template/config hashes, and profile. Judges, participant data, optimizer/scheduler/scaler/RNG, and pickle-like state are excluded.
- **Ordered implementation:**
  1. Read SEL-16's manifest; never discover “latest”. Require exactly its primary and adapter/calibration files.
  2. Hash every Gemma 4 model/processor/template/config/license file. Immutable public files still require license notices.
  3. Validate adapter bundles, base compatibility, target modules, and safetensors-only policy; reuse weight/bundle containment, checksum, symlink, and executable checks.
  4. Prove `local_files_only=True` has a complete file inventory before copying; stage by content-addressed copy, never Hub-cache symlink.
  5. Exclude Llama/Qwen judges. Exclude MedGemma absent redistribution evidence. Keep deployment/research artifact identities separate.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_runtime_asset_staging.py tests/phase_17/test_bundle.py`. Protected staging: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen python docker/medreason/tools/stage_runtime_assets.py --deployment-manifest artifacts/models/medreason/deployment/manifest.json --output artifacts/submission/runtime-assets --manifest docker/medreason/manifests/runtime-assets.json`. Guard unset skips; guard enabled with missing access/assets/storage fails. Test revision/template mismatch, missing adapter, stale calibration, optimizer/pickle, symlink, judge inclusion, or unapproved MedGemma.
- **Acceptance evidence:** manifest verification, offline loader inventory, bundle checksum/base report, forbidden-file exclusion, and license hashes. Tiny fakes prove code only.
- **Non-goals/failure policy:** No selection, conversion, retuning, Hub fallback, judge shipping, or license inference. **Fail closed:** any missing/hash/license/profile mismatch, forbidden state, unapproved specialist, incomplete local load, or insufficient storage blocks context staging.
- **Handoff:** DOC-05 consumes `runtime-assets.json`; DOC-07 loads only named paths; DOC-15/16 bind telemetry/export to its hash.

## DOC-05 Verify vendored artifact manifest before startup

- **Depends on:** GOV-08, SCH-07, DOC-01, DOC-03, DOC-04.
- **Parallel safety and exclusive file ownership:** Starts after manifest schemas freeze; may run with DOC-08/09. Exclusively owns `docker/medreason/app/artifact_verifier.py`, `manifests/release-manifest.json`, and `tests/challenges/medreason/test_artifact_verifier.py`; DOC-06 only calls `verify_release_assets`.
- **Target paths/symbols:** `ReleaseManifest`, `verify_release_assets`, `verify_file_record`, `ManifestVerificationError`; reuse bundle/weight checks and canonical JSON, not broad audit/report objects.
- **Inputs:** context, official, wheelhouse, and runtime-asset manifests with expected root hashes baked into image labels. **External artifacts/hardware:** tiny fixtures need none; protected acceptance requires real staged bytes but no GPU allocation.
- **Outputs:** exhaustive canonical `release-manifest.json` and a startup status containing only release hash and sanitized code.
- **Ordered implementation:**
  1. Parse with duplicate-key rejection and strict schemas; reject unsafe paths, symlinks, special/executable asset files, duplicates, invalid hash, and size/hash mismatch.
  2. Stream-hash large files under `/opt/medreason`; reject extra runtime assets and never log sensitive paths.
  3. Verify official SHA-256, wheel/install provenance, runtime files, adapter bundle, and base compatibility.
  4. Run before model construction or `/input` reads; cache success only in-process and bind it to exact manifest bytes.
  5. Provide offline `--verify-only`; never repair/redownload.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_artifact_verifier.py`. Image preflight: `docker run --rm --network none --entrypoint python medreason-gemma:final /opt/app/process.py --verify-only`. Test bit flip, truncation, extra file, symlink, duplicate key, traversal, invalid hash, swapped adapter, and redacted failure.
- **Acceptance evidence:** successful exhaustive counts/root hash, stable sanitized negative codes, and proof that model/input loading was not reached on failure.
- **Non-goals/failure policy:** No repair, download, raw exception/path logging, `FailureReporter`, or `AuditEvent`. **Fail closed:** entrypoint exits nonzero before input on any manifest/inventory/hash/base discrepancy.
- **Handoff:** DOC-06 invokes verification first; DOC-10 records it; DOC-16 records root hash.

## DOC-06 Implement privacy-safe container process entrypoint

- **Depends on:** SCH-07, SCH-08, RUN-08, RUN-15, DOC-05.
- **Parallel safety and exclusive file ownership:** May run with DOC-07 after fixing its factory/predict interface. Exclusively owns `docker/medreason/process.py`, `docker/medreason/app/entrypoint.py`, entrypoint tests, and Dockerfile `ENTRYPOINT`; official vendored `MedReason-Docker/process.py` remains unchanged for conformance comparison.
- **Target paths/symbols:** `main`, `run_submission`, `safe_failure`, `atomic_write_results`. Official lifecycle is `python /opt/app/process.py`, setup once, `predict_case` in input order, teardown, validate, write. Upstream `process.py` logs mount paths/raw exception and runner lacks `finally`; the custom wrapper must preserve behavior while fixing privacy/teardown.
- **Inputs:** `/input/cases.json`, images under `/input`, DOC-05 verification, DOC-07 factory, version-1.0 serializer. **External artifacts/hardware:** fake system needs none; production requires assets/GPU with `MEDFM_RUN_REAL_CHECKPOINTS=1` and `MEDFM_RUN_GPU_TESTS=1`.
- **Outputs:** exactly atomic `/output/results.json`: top-level `name`, `type`, `answers`, `version={major:1,minor:0}`. Logs contain only bounded counts, case IDs, release hash, sanitized class—never question, answer/reference, metadata, trace, prompt/private thought, image/mount path, token, or raw traceback.
- **Ordered implementation:**
  1. Set offline/cache controls and deterministic seeds before model imports.
  2. Verify assets before input, then load official-compatible cases without logging paths/content.
  3. Setup once; process in input order. Per-case corrupt-image fallback lives inside `predict_case`; structural/global errors fail the run.
  4. Put teardown in `finally`; validate exact cardinality/task/labels/non-empty open fields before canonical atomic rename.
  5. Map boundary errors to typed safe codes; remove temp files and never leave partial/stale results.
  6. Run non-root, `/input:ro`, no shell/server/stdin/API/telemetry.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_docker_entrypoint.py tests/challenges/medreason/test_runtime_privacy.py`. Protected startup: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_docker_entrypoint.py`; enabled guard with absent GPU/assets fails. Test duplicate ID, invalid label, empty trace, corrupt case image, secret/path in setup error, disk-full/rename error, teardown exception, signal cleanup, Unicode, and output order.
- **Acceptance evidence:** canonical fixture output, atomic-write and teardown proof, log allowlist, sanitized codes, and non-root inspection. Fixture success is not full-model evidence.
- **Non-goals/failure policy:** No upstream raw logging, retries that change output, or position-prior fallback. **Fail closed:** startup, structural input, manifest, global inference, validation, or write failure exits nonzero without partial output; only declared per-case corrupt-image fallback continues.
- **Handoff:** DOC-07 supplies system factory; DOC-11–14 use this sole production entrypoint; DOC-13 hashes canonical bytes.

## DOC-07 Bridge custom system to MedReason runtime

- **Depends on:** SCH-04, RUN-01, RUN-02, RUN-03, RUN-04, RUN-05, RUN-06, RUN-07, RUN-08, RUN-15, SEL-16, DOC-04, DOC-06.
- **Parallel safety and exclusive file ownership:** May run beside DOC-08/09; only share the callable contract with DOC-06. Exclusively owns `docker/medreason/app/system.py`, `app/bridge.py`, and `tests/challenges/medreason/test_docker_bridge.py`; official schema/io/validation/base bytes remain unchanged.
- **Target paths/symbols:** `MedFMDeploymentSystem(MedReasonSystem)::{setup,predict_case,teardown}`, `to_runtime_case`, `to_official_prediction`, `create_system`; upstream official `MedReasonSystem` and dataclasses.
- **Inputs:** official `MedReasonCase`, DOC-04 assets, SEL-16 route/profile, and RUN interfaces for adapter switching, MCQ scoring, open parsing/one repair, fallbacks, and ordered results. **External artifacts/hardware:** fakes cover interface; exact base/adapters and selected GPU are protected.
- **Outputs:** official `MedReasonPrediction`; MCQ answer is exactly a supplied original label; open trace/answer are non-empty; no internal prompt/thought/path/judge data in metadata.
- **Ordered implementation:**
  1. Subclass unmodified official base and losslessly map case ID, task, image order, options, and Unicode metadata.
  2. Load only manifest-named local files with `local_files_only=True`, one base, selected adapters, and frozen OOF calibration; refuse revision/profile drift.
  3. Dispatch deterministically; preserve processor multimodal fields; strip private thought; keep corrupt refs inside per-case boundary for RUN-07 fallback.
  4. Release resources in teardown; no judge/service construction.
  5. Validate via official `validate_predictions_against_cases` and validator; do not silently change task aliases or output semantics.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_docker_bridge.py tests/challenges/medreason/test_runtime.py`. Protected: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_docker_bridge.py`. Test multi-image order, Unicode, option remap, thought stripping, one repair, corrupt fallback, teardown, and missing local file.
- **Acceptance evidence:** fake predictions accepted by official validation, call trace showing one base/correct adapter switches, no download calls, and protected report when available. Fake evidence is code acceptance only.
- **Non-goals/failure policy:** No training, agent loop, judges, tuning, or official vendor modification. **Fail closed:** unknown task, artifact/profile drift, invalid output, missing local file, or undeclared route stops release; only RUN-07 fallback is recoverable.
- **Handoff:** DOC-06 calls `create_system`; DOC-11/12 prove fixture/full execution.

## DOC-08 Vendor official two-case smoke fixture

- **Depends on:** GOV-07, DOC-01.
- **Parallel safety and exclusive file ownership:** May run with DOC-02–07/09. Exclusively owns `docker/medreason/test/cases.json`, `test/images/case_mcq_001.png`, `test/images/case_open_001.png`, `manifests/official-fixture.json`, and byte tests.
- **Target paths/symbols:** upstream `MedReason-Docker/test/cases.json` (Git blob `77cac657...`, verified SHA-256 prefix `bb1c6811...`) and two 224×224 RGB PNGs (`f527b063...`, `0c96c11...`) at commit `05748c...`; `tests/challenges/medreason/test_official_fixture_bytes.py`.
- **Inputs:** full GOV-07 SHA-256 values and pinned bytes. **External artifacts/hardware:** one-time upstream/mirror access; no GPU or protected data.
- **Outputs:** byte-identical three-file fixture and manifest with commit/path/Git blob/full SHA-256/size/role/image dimensions/mode.
- **Ordered implementation:**
  1. Copy pinned generic fixture only; never mutable branch/participant data.
  2. Verify bytes and distinguish Git SHA-1 from SHA-256.
  3. Assert exactly one MCQ, one open, and two contained relative image refs; do not rewrite/recompress.
  4. Stage read-only; commit no candidate-generated expected answer.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_official_fixture_bytes.py`. `uv run --frozen python docker/medreason/tools/verify_official_sources.py --manifest docker/medreason/manifests/official-fixture.json --root docker/medreason/test`. Fail changed newline/PNG, missing/extra file, path escape, wrong dimensions/mode, mutable-source blob, or hash-kind confusion.
- **Acceptance evidence:** exact SHA-256 report, commit/blob tie, and parsed `mcq=1/open=1`, `224x224 RGB` summary. Provenance only, not model/GPU quality.
- **Non-goals/failure policy:** No expected medical answer or protected image. **Fail closed:** missing/altered/unpinned/unmanifested fixture bytes block smoke and release.
- **Handoff:** DOC-11 mounts fixture; DOC-16 records manifest hash.

## DOC-09 Vendor and expose official output validator

- **Depends on:** GOV-07, DOC-01, DOC-08.
- **Parallel safety and exclusive file ownership:** May run with DOC-02–07. Exclusively owns `docker/medreason/tools/validate_output.py`, its manifest entry, tests, and executable mode. EVA-01 evaluator files remain read-only and are not runtime judges.
- **Target paths/symbols:** exact upstream `MedReason-Docker/tools/validate_output.py` Git blob `4e7e7bb...`; `get_answers`, `validate_basic`, `validate_against_input`, `main`; official `io.load_cases`/`normalize_task_type`.
- **Inputs:** byte-identical validator/core and GOV-07 full SHA-256. **External artifacts/hardware:** one-time upstream/mirror access; CPU only.
- **Outputs:** validator at exact command path, included in source/release manifests. Wrapper may select paths/capture bounded status but cannot alter decisions/exit code.
- **Ordered implementation:**
  1. Vendor validator and dependencies byte-for-byte and verify before use.
  2. Preserve its accepted payload forms; release output remains version-1.0 object.
  3. Always pass `--input-json` so missing/extra/duplicate IDs, task mismatch, bad MCQ label, and empty open fields fail.
  4. Keep host execution independent of models/network/GPU.
  5. Never substitute `MedReason-Evaluation/validate_submission.py` or scoring for this output-shape gate.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_official_validator.py`. `uv run --frozen python docker/medreason/tools/validate_output.py artifacts/submission/output/results.json --input-json artifacts/data/medreason/validation_runtime/cases.json`. Test missing/extra/duplicate, wrong task, out-of-set label, blank open trace/answer, malformed top-level, and Unicode.
- **Acceptance evidence:** exact source SHA-256; `[validate_output] OK: N predictions validated`; every negative returns nonzero. Schema acceptance is not metric evaluation.
- **Non-goals/failure policy:** No custom equivalent, proxy judge, or score claim. **Fail closed:** byte mismatch, missing `--input-json`, or validator nonzero blocks all smoke/full/export gates.
- **Handoff:** DOC-11/12 use exact validator; DOC-16 records source hash.

## DOC-10 Build no-pull no-network submission image

- **Depends on:** GOV-10, DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, DOC-09.
- **Parallel safety and exclusive file ownership:** Serial integration after inputs freeze. Exclusively owns `docker/medreason/build.sh`, `tests/challenges/medreason/test_offline_build.py`, image labels, and generated `artifacts/submission/build-evidence.json`; no concurrent staged-context mutation.
- **Target paths/symbols:** final image `medreason-gemma:final`, `record_build_evidence`, and generated context. Current Dockerfiles are not reused because they perform mutable/networked resolution.
- **Inputs:** locally present locked base, verified wheelhouse/assets/root manifest, source revision. **External artifacts/hardware:** Docker/BuildKit and at least 600 GiB planned disk; current 364 GiB blocks full build. GPU is not required to construct.
- **Outputs:** image/config digest and labels for source, official commit, base digest, root/deployment/profile hashes; sanitized `build-evidence.json`.
- **Ordered implementation:**
  1. Preflight local base, context/wheel/runtime/official hashes, free space, and forbidden files.
  2. Use BuildKit with pull false/network none; no apt/curl/git/Hub, remote ADD, external stage, or package index. Install via `--no-index --find-links --require-hashes`.
  3. Set offline defaults, non-root `ENTRYPOINT ["python","/opt/app/process.py"]`, and no credentials/writable Hub cache.
  4. Inspect layers/history for secrets, unexpected files, mutable refs, and label mismatch; run verify-only under no network.
  5. Bind all downstream evidence to immutable image/config digest, not tag alone.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_offline_build.py`. Build: `BASE_REF="$(uv run --frozen python docker/medreason/tools/verify_base_image.py --lock docker/medreason/base-image.lock.json --print-reference)"; (cd docker/medreason && DOCKER_BUILDKIT=1 docker build --pull=false --network=none --build-arg BASE_IMAGE="$BASE_REF" -t medreason-gemma:final .)`. Verify: `docker run --rm --network none --entrypoint python medreason-gemma:final /opt/app/process.py --verify-only`. Negative harness proves absent wheel/network-only RUN fails.
- **Acceptance evidence:** no-pull/no-download log, locked base/config, non-root entrypoint, layer scan, manifest verification, build evidence. Built image alone is not GPU acceptance.
- **Non-goals/failure policy:** No online fallback, secret args, judges/data, or mutable tags as identity. **Fail closed:** preflight/storage/network/secret/layer/manifest/label failure prevents final tag and validation.
- **Handoff:** DOC-11–15 consume image/config digest; DOC-16 refuses mismatched export.

## DOC-11 Exercise official two-case container round-trip

- **Depends on:** DOC-08, DOC-09, DOC-10.
- **Parallel safety and exclusive file ownership:** Use immutable image; may run with DOC-14 only in isolated outputs. Exclusively owns `docker/medreason/test.sh`, `tests/challenges/medreason/test_container_smoke_harness.py`, and `artifacts/submission/evidence/two-case/`.
- **Target paths/symbols:** official fixture, production `/opt/app/process.py`, `/output/results.json`, and vendored validator.
- **Inputs:** exact image digest and fixture. **External artifacts/hardware:** production acceptance requires real model/assets and compatible selected GPU. Guards require `MEDFM_RUN_GPU_TESTS=1` plus `MEDFM_RUN_REAL_CHECKPOINTS=1`; enabled with missing prerequisites fails. A fake/smoke route tests plumbing only.
- **Outputs:** two ordered answers, official-validator status, sanitized logs, elapsed time, input/result hashes.
- **Ordered implementation:**
  1. Mount fixture `/input:ro`, fresh `/output`, exact image, `--network none`, offline env, GPU.
  2. Run production route; do not set official template's `MEDREASON_SYSTEM=smoke` for release acceptance.
  3. Require zero exit, one results file, unchanged input, exact IDs/order, MCQ label membership, and non-empty open fields.
  4. Run exact validator with `--input-json`; scan logs against privacy allowlist.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_container_smoke_harness.py`. Protected: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 docker/medreason/test.sh medreason-gemma:final artifacts/submission/evidence/two-case/output`; internally `docker run --rm --gpus all --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e MEDFM_OFFLINE=1 -v "$PWD/docker/medreason/test:/input:ro" -v "$OUT:/output" medreason-gemma:final`, then validator. Test nonzero, missing/extra output, invalid label, blank trace, input mutation, or leak.
- **Acceptance evidence:** image digest, commands/env allowlist, zero statuses, result hash, two-ID summary, input pre/post equality, privacy scan. Fake fixture is insufficient.
- **Non-goals/failure policy:** No answer scoring or full-runtime inference. **Fail closed:** production-route smoke, official validation, read-only input, or privacy failure blocks full validation/export.
- **Handoff:** DOC-12 reuses harness; DOC-13 may retain fixture hash as secondary evidence.

## DOC-12 Exercise full participant validation package

- **Depends on:** OPS-02, OPS-03, DAT-05, DAT-06, RUN-15, DOC-09, DOC-10, DOC-11.
- **Parallel safety and exclusive file ownership:** Protected serial selected-GPU run; no shared GPU/tag/input/output mutation. Exclusively owns `tools/run_full_validation.py`, `tests/challenges/medreason/test_full_validation_harness.py`, and `artifacts/submission/evidence/full-validation/run-01/`.
- **Target paths/symbols:** `artifacts/data/medreason/validation_runtime/cases.json`, run-01 `results.json`, `FullValidationEvidence`, official validator.
- **Inputs:** immutable licensed participant package: 2,532 unlabeled cases (2,057 MCQ, 475 open), with no answer/reference fields; exact image. **External artifacts/hardware:** Synapse access, archive license/hash, sufficient storage, real assets, selected GPU. Local 24,576 MiB RTX 3090 and 364 GiB free disk cannot establish acceptance.
- **Outputs:** one valid result per ID, validator/log/input-integrity/privacy evidence, elapsed time, sanitized failure counts, and exact image/artifact/profile hashes. No metric/tuning artifact.
- **Ordered implementation:**
  1. Preflight package hash/no-answer invariant/count/task counts/path containment, empty output, and hardware profile.
  2. Run once with `--network none`, offline env, `/input:ro`, fresh `/output`, exact digest, no intervention/API.
  3. Require 2,532 unique input-ordered results; supplied MCQ labels and non-empty open trace/answer. Count only sanitized per-case fallback classes.
  4. Run official validator, verify input unchanged, and scan logs/result metadata for forbidden text/path/prompt/thought.
  5. Freeze run-01 for DOC-13; never tune/train/select from participant outputs.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_full_validation_harness.py`. Protected: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen python docker/medreason/tools/run_full_validation.py --image medreason-gemma:final --input artifacts/data/medreason/validation_runtime --output artifacts/submission/evidence/full-validation/run-01 --expected-cases 2532 --network none`; then exact validator. Enabled guard without package/GPU/assets fails. Test count 2,531, duplicate/extra, leaked answer field, invalid label, blank trace, mutation, stale output, or log leak.
- **Acceptance evidence:** archive/input/image/artifact hashes, task counts, validator `OK: 2532`, input equality, privacy report, zero exit, immutable result hash. Fixture tests accept harness only.
- **Non-goals/failure policy:** No scoring, tuning, hidden/private data, or official-limit claim. **Fail closed:** missing/changed/unlicensed input, count/answer leakage, incompatible hardware, incomplete run, validator/privacy/integrity failure, or image drift blocks release.
- **Handoff:** DOC-13 repeats same contract; DOC-15 measures same workload; DOC-16 records evidence hash, not protected contents.

## DOC-13 Verify byte-identical duplicate complete runs

- **Depends on:** SCH-08, RUN-02, RUN-15, DOC-12.
- **Parallel safety and exclusive file ownership:** Two serial clean runs on same selected GPU/image/input; no concurrent GPU workload/cache/output reuse. Exclusively owns `tools/compare_runs.py`, `tests/challenges/medreason/test_deterministic_container_runs.py`, `run-02/`, and comparison JSON.
- **Target paths/symbols:** `compare_result_bytes`, `compare_run_contract`, two `results.json`, `determinism-evidence.json`.
- **Inputs:** exact input/image/artifact/hardware/software identity and fresh output. **External artifacts/hardware:** same participant package and selected real GPU/profile; local 24 GB is not substitute evidence.
- **Outputs:** independently valid run-01/run-02 with byte-identical files/SHA-256 and matching sanitized contracts.
- **Ordered implementation:**
  1. Freeze seeds, deterministic algorithms, greedy/default decoding, case-hash sampling seeds if selected, ordering, and canonical serialization.
  2. Rerun all 2,532 cases into empty run-02 without copying output/cache state.
  3. Validate independently, then raw `cmp`; parsed JSON equality is insufficient.
  4. Keep timestamps/latencies outside `results.json`. On mismatch report byte offset and sanitized case ID, not answer/trace.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_deterministic_container_runs.py`. Repeat DOC-12 with run-02, then `cmp artifacts/submission/evidence/full-validation/run-01/results.json artifacts/submission/evidence/full-validation/run-02/results.json && sha256sum artifacts/submission/evidence/full-validation/run-{01,02}/results.json`. Test key order, newline, timestamp, stochastic seed, output order, and content mismatch.
- **Acceptance evidence:** both validator passes, `cmp` zero, equal SHA-256, matching contracts, separate telemetry. Tiny fixtures are code evidence only.
- **Non-goals/failure policy:** Do not normalize after generation, ignore trace differences, or rerun until lucky. **Fail closed:** any byte/contract drift, invalid/incomplete run, or uncontrolled concurrency blocks determinism acceptance.
- **Handoff:** DOC-15 may instrument without changing bytes; DOC-16 records determinism hash.

## DOC-14 Verify runtime network attempts always fail

- **Depends on:** RUN-08, DOC-10, DOC-11.
- **Parallel safety and exclusive file ownership:** May run beside DOC-11 only in isolated namespace/output. Exclusively owns `tools/network_probe.py`, `tests/challenges/medreason/test_container_network_isolation.py`, and network evidence.
- **Target paths/symbols:** `probe_network_denial`, `audit_runtime_processes`, test-only `--network-self-test`.
- **Inputs:** exact image, offline env, Docker `--network none`, clean proxy env. **External artifacts/hardware:** Docker namespace capability; no GPU/data for active probe, production fixture audit uses DOC-11 prerequisites.
- **Outputs:** matrix showing DNS/TCP/HTTP(S)/Hub/metadata/proxy attempts all fail; no unexpected listener/process; local fixture still runs.
- **Ordered implementation:**
  1. Require `--network none`; unset proxies/tokens and set Hub/Transformers/MEDFM offline flags.
  2. Attempt bounded DNS, IPv4/IPv6 TCP, HTTP(S), Hub resolution, and metadata endpoint access; classify every failure.
  3. Inspect image/process environment for URLs/tokens/API/tracker exporters/listeners. Static scan supplements active denial.
  4. Deliberately remove a local artifact and prove no download fallback: startup must fail closed.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_container_network_isolation.py`. `docker run --rm --network none --entrypoint python medreason-gemma:final /opt/app/process.py --network-self-test`; production reuses DOC-11. Test inherited proxy, host networking, DNS/connect success, metadata reachability, Hub download attempt, or listener.
- **Acceptance evidence:** mechanism/result matrix with zero successes, exact image digest, and production audit. Flag presence alone is insufficient.
- **Non-goals/failure policy:** No real clinical/API endpoint or credential. **Fail closed:** any successful resolution/connect/request, proxy/token inheritance, download attempt, listener, or missing active probe blocks release.
- **Handoff:** DOC-15 preserves isolation; DOC-16 records network evidence hash.

## DOC-15 Measure worst-case latency and peak VRAM

- **Depends on:** GOV-09, EVA-10, SEL-16, DOC-12, DOC-13, DOC-14.
- **Parallel safety and exclusive file ownership:** Protected exclusive-GPU task; no concurrent model/judge/training/telemetry workload. Exclusively owns `tools/measure_runtime.py`, `tests/challenges/medreason/test_runtime_measurement.py`, MedReason additions to `.github/workflows/protected-hardware.yml`, and `evidence/runtime/{48gb,96gb}/`.
- **Target paths/symbols:** `RuntimeMeasurement`, `measure_case`, `measure_full_run`, `verify_profile_hardware`; reuse existing `medfm.training.memory.MemorySnapshot` semantics plus synchronized CUDA allocated/reserved and independent device peak. Guards: existing `MEDFM_RUN_GPU_TESTS=1`, `MEDFM_RUN_REAL_CHECKPOINTS=1`, plus `MEDFM_RUN_MEDREASON_48GB=1`/`MEDFM_RUN_MEDREASON_96GB=1`.
- **Inputs:** exact image/artifact/input/profile, driver/CUDA metadata, and any written organizer limits. **External artifacts/hardware:** actual matching RTX 6000-class 48 GB and RTX Pro 6000 Blackwell-class 96 GB cards. Current RTX 3090 (24,576 MiB) proves neither; no SSH compute exists. Published resource ceilings are unavailable, and Qwen-example flags are not official limits.
- **Outputs:** per-profile JSON: GPU model/redacted identity/total memory, driver/CUDA, hashes, warmup, peak allocated/reserved/device memory, case latency, p50/p95/max/total, worst-case case ID only, sanitized failures, and ceiling verdict.
- **Ordered implementation:**
  1. Absent profile flag skips; enabled with wrong/missing GPU/assets/input/storage fails, never silently skips or substitutes 24/96 for 48.
  2. Synchronize CUDA, reset peaks after warmup, measure full processor→model→output, and independently sample device memory to capture non-PyTorch allocations.
  3. Exercise complete 2,532-case package and selected worst visual/multi-image routes without output changes; validate and compare deterministic bytes.
  4. 96 GB quality profile is selected 31B plus optional licensed/selected 4B; 48 GB compatibility order is NF4 31B then NF4 26B-A4B, no specialist. A 96 GB run with a configured 45 GiB allocation gate does not prove 48 GB kernels, latency, or end-to-end fit.
  5. Require a matching 48 GB full run before claiming 48 GB support. 96 evidence proves only exact 96 profile; 48 evidence only exact 48 model/quantization/image/driver. If organizer hardware is unknown, prefer the highest-scoring profile actually proven on 48 GB.
  6. Compare with written limits only. If limits stay unpublished, report measurements and block `official-ready` rather than invent a pass threshold.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_runtime_measurement.py`. Protected 48: `MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_MEDREASON_48GB=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_runtime_measurement.py --profile=48gb`; analogous 96 command with `MEDFM_RUN_MEDREASON_96GB=1 --profile=96gb`. Direct: `uv run --frozen python docker/medreason/tools/measure_runtime.py --image medreason-gemma:final --input artifacts/data/medreason/validation_runtime --profile 48gb --network none --output artifacts/submission/evidence/runtime/48gb`. Test unsynchronized timing, allocator undercount, wrong memory/profile, concurrent process, OOM, timeout, or byte drift.
- **Acceptance evidence:** immutable measurement hashes, GPU/driver identity, official-validator and output-equality pass, measured ceiling verdict only where written limit exists. Synthetic/24 GB evidence is explicitly unsupported.
- **Non-goals/failure policy:** No parameter-count estimates, extrapolation, route changes, or unpublished-limit claims. **Fail closed:** assigned-hardware official release is blocked without matching protected run, on peak/timeout/OOM/output drift/incomplete measurement, or while a required official limit is unknown.
- **Handoff:** DOC-16 accepts only evidence matching exported profile/image/artifact and states unsupported profiles.

## DOC-16 Export image archive and SHA-256 manifest

- **Depends on:** GOV-01, GOV-10, DOC-10, DOC-11, DOC-12, DOC-13, DOC-14, DOC-15.
- **Parallel safety and exclusive file ownership:** Final serial gate; image/tag/manifests/evidence immutable. Exclusively owns `docker/medreason/export.sh`, `tools/verify_release_evidence.py`, `tests/challenges/medreason/test_export_release.py`, and generated archive/checksum/export manifest.
- **Target paths/symbols:** `ReleaseEvidence`, `verify_release_evidence`, `write_export_manifest`; `artifacts/submission/medreason-gemma.tar.gz`, `.sha256`, `export-manifest.json`. Official export script blob `40466e8...` is an anchor; use `gzip -n` to eliminate gzip timestamp metadata.
- **Inputs:** exact image/config digest; base/context/wheel/runtime/official/release hashes; fixture/full/determinism/network/runtime evidence; written organizer exception when labeling official. **External artifacts/hardware:** export itself needs disk/Docker, not GPU, but matching protected 48/96 evidence must already exist.
- **Outputs:** archive; standard lowercase SHA-256 sidecar; canonical manifest with archive size/hash, image/config, profile, base/official/source/deployment/context/wheel/root hashes, evidence hashes/status, supported profiles, and `official_submission_ready`. Never duplicate protected inputs/results/model contents into evidence.
- **Ordered implementation:**
  1. Verify every evidence schema/hash/pass and binding to exact image, deployment artifact, participant input, and selected profile; reject stale tag/profile evidence.
  2. Eligibility boundary: absent written exception permits only explicitly labeled post-challenge research export, never `official_submission_ready=true`; require MedGemma redistribution evidence if present.
  3. Resolve tag to image ID; `docker save`, `gzip -n -9`, fsync/atomic rename, then compute SHA-256/size and canonical manifest/sidecar.
  4. In a clean offline Docker store when resources allow, import archive, verify config/image identity, and rerun verify-only plus official two-case round-trip under no network.
  5. Scan layers for secrets, participant/judge/training state, mutable refs, and unknown files. Mark 48 GB unsupported absent matching DOC-15 evidence; never substitute 96/45-GiB allocation evidence.
- **Focused tests and exact commands:** `uv run --frozen pytest -q tests/challenges/medreason/test_export_release.py`. `ROOT="$PWD"; (cd docker/medreason && ./export.sh medreason-gemma:final "$ROOT/artifacts/submission/medreason-gemma.tar.gz") && sha256sum -c artifacts/submission/medreason-gemma.tar.gz.sha256 && uv run --frozen python docker/medreason/tools/verify_release_evidence.py --manifest artifacts/submission/export-manifest.json --archive artifacts/submission/medreason-gemma.tar.gz`. Post-import smoke reuses DOC-11 in isolated offline Docker storage. Test stale image/profile, altered archive, gzip timestamp, missing hardware boundary, secret layer, unapproved specialist, or absent exception.
- **Acceptance evidence:** verified archive/sidecar/manifest hashes; image identity before/after import; layer scan; verify-only/fixture validator passes; full/determinism/network/runtime evidence hashes; explicit eligibility/supported profiles. Archive existence alone is insufficient.
- **Non-goals/failure policy:** No upload/submission/retag/official-win claim or protected-data inclusion. **Fail closed:** no artifact is labeled official-ready when any prior gate/hash/profile/eligibility/license evidence is missing, stale, mismatched, or failed; research-only export states every unresolved boundary.
- **Handoff:** OPS-18 receives immutable archive path/SHA-256, image/config digest, export-manifest SHA-256, supported profiles, and eligibility status.
