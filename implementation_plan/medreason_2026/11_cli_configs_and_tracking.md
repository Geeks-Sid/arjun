# CLI configs and tracking

Expose the approved stable module commands without routing production MedReason through `medfm.recipes.phase13`. All paths below are relative to `arjun/`. Every command must parse, strictly validate YAML and cross-field policy, resolve immutable manifests, and pass MedReason preflight before dataset decoding, processor/model/judge construction, CUDA initialization, or quantizer allocation. Default tests use CPU fakes. Real-checkpoint tests require `@pytest.mark.real_checkpoint` plus `MEDFM_RUN_REAL_CHECKPOINTS=1`; GPU tests additionally require `@pytest.mark.gpu` plus `MEDFM_RUN_GPU_TESTS=1`. A guarded skip is never artifact, license, judge, or hardware acceptance.

## CLI-01 Expose stable MedReason audit module command

**Depends on:** SCH-02, SCH-03, DAT-01, DAT-02, DAT-03, DAT-04, DAT-05, DAT-08, DAT-09, DAT-10, DAT-11, DAT-12, GOV-10.

**Parallel safety and exclusive file ownership:** May run with CLI-02 through CLI-09 after DAT-12 freezes its API. Own entry-point symbols in existing-after-DAT `medfm/challenges/medreason/data.py` and audit rows in new `tests/challenges/medreason/test_cli.py`; coordinate that shared test file. Do not alter DAT algorithms.

**Target paths/symbols:** Existing-after-DAT `medfm/challenges/medreason/data.py::{build_parser,_run_audit,main}`; existing-after-DAT `medfm/challenges/medreason/config.py` audit-request validator; existing-after-GOV `medfm/challenges/medreason/preflight.py`; new focused CLI tests and reused DAT fixtures under `tests/challenges/medreason/fixtures/`.

**Inputs:** `audit` with required `--train PATH`, `--validation PATH`, `--output DIR`; immutable ZIPs and governance evidence. Participant validation accepts no answer/prediction input.

**Outputs:** DAT-11 artifacts under `artifacts/data/medreason/derived/`, including source/extracted manifest SHA-256; deterministic stdout status and sanitized stderr. Never log questions, answers, metadata, prompts, image/source paths, or IDs beyond case ID.

**Config keys and validation order:** (1) `argparse` validates subcommand/flags; `--help` stops before heavy imports. (2) Inputs are distinct existing regular `.zip` files and output is a safe directory outside sources. (3) Validate governance evidence/package identities without inventing licenses/hashes. (4) Validate released schema and participant no-answer policy. (5) Only then securely extract, decode, hash, and atomically write. No model/CUDA allocation is valid for this command.

**Implementation:** Keep a thin adapter over DAT-12; return `2` usage, `1` sanitized validation/audit failure, `0` only after complete commit. Preserve exactly:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.data audit \
  --train artifacts/data/medreason/source/medreason2026_train.zip \
  --validation artifacts/data/medreason/source/medreason2026_validation_participant_facing.zip \
  --output artifacts/data/medreason/derived
```

**Tests:** `--help` at module and `audit` level returns `0`, shows all flags, writes nothing, and initializes no CUDA/model. Invalid missing/non-ZIP/same paths, unsafe members, answer leakage, duplicate IDs, or bad images stop before output/factories. Fixture audit proves deterministic manifests only. Commands, not run now:
```bash
uv run pytest -q tests/challenges/medreason/test_cli.py -k 'audit and (help or invalid or fixture)'
CUDA_VISIBLE_DEVICES='' uv run python -m medfm.challenges.medreason.data audit --help
```

**Acceptance evidence:** Help captures; matching hashes from two fixture runs; sanitized failures; zero-call model/CUDA spies. Protected Synapse audit remains OPS evidence.

**Non-goals and failure policy:** No downloads, hidden data, participant tuning, inferred hashes, or malformed-record repair. Missing protected prerequisites block protected execution, not fixture acceptance.

**Handoff:** Derived root, archive/extracted/dataset/audit hashes, schema version, and preflight receipt; downstream rejects mismatches.

## CLI-02 Expose stable MedReason evaluation module command

**Depends on:** SCH-04, SCH-05, EVA-02, EVA-03, EVA-04, EVA-08, EVA-09, EVA-10, EVA-11, EVA-14, SEL-02, SEL-03, SEL-04, SEL-05, CLI-08.

**Parallel safety and exclusive file ownership:** May run with CLI-01 and recipe cards. Own existing-after-EVA `medfm/challenges/medreason/evaluate.py` entry point and evaluation rows in `tests/challenges/medreason/test_cli.py`. Never edit generic `medfm/cli/evaluate.py` or vendored official files.

**Target paths/symbols:** Existing-after-EVA `evaluate.py::{build_parser,run_evaluation,main}`; CLI-08 `config.py::{load_evaluation_config,validate_evaluation_request}`; reuse `medfm/evaluation/advanced.py` only behind challenge metrics; focused CLI tests.

**Inputs:** Required `--config`; tournament requires `--split-manifest`; frozen accepts `--split {dev,oof,lockbox,participant_validation}` and `--bootstrap-resamples`. Overrides may narrow declared split/set approved count only, never replace frozen identity.

**Outputs:** Raw/parsed predictions, exact MCQ and labeled proxy GT/VA/RVF reports, telemetry, intervals, config hash under configured `artifacts/runs/medreason/`; unavailable exact judges emit `promotion_blocked`, never substitutes.

**Config keys and validation order:** (1) Flags; positive resamples and lockbox exactly `1000`. (2) Strict mapping: `schema_version`, `evaluation.id/mode/split/output_dir`, `seed`, no duplicates/unknowns. (3) Paths/hashes, split membership, candidate allowlist, lockbox state. (4) Official serializer/parser, judge IDs/revisions/licenses, rubric hashes, metrics/RVF cap, proxy labels, redaction, paired-bootstrap weights. (5) governance/judge/selection preflight. Allocate only after all; judges are sequential.

**Implementation:** Keep challenge evaluation separate from generic evaluation. Repository-relative resolution and canonical hash are deterministic. Return `2` usage, `1` typed failure, `0` complete report. Preserve exactly:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.evaluate \
  --config configs/recipes/medreason/zero_shot_tournament.yaml \
  --split-manifest artifacts/data/medreason/derived/splits.json

CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.challenges.medreason.evaluate \
  --config configs/recipes/medreason/frozen_system.yaml \
  --split lockbox --bootstrap-resamples 1000
```

**Tests:** `--help` lists stable flags with no outputs/allocation. Invalid mapping/unknown key/hash/split/resample/lockbox/proxy/judge config fails pre-factory. Saved synthetic predictions cover MCQ/RVF/bootstrap. Real judges/models are guarded. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_cli.py -k 'evaluate and (help or invalid or no_allocation)'
CUDA_VISIBLE_DEVICES='' uv run python -m medfm.challenges.medreason.evaluate --help
```

**Acceptance evidence:** Help/invalid captures, fixture report/config hashes, call order, `proxy: true` and `promotion_blocked`. No protected judge/lockbox claim.

**Non-goals and failure policy:** No generic alias, substitute judge, lockbox comparison, or invented official tie policy. Missing judge blocks proxy promotion.

**Handoff:** Config/split/candidate/judge/rubric hashes, prediction/report paths, proxy/block state, lockbox receipt.

## CLI-03 Integrate MedReason builders with training CLI

**Depends on:** MOD-01, MOD-02, MOD-03, MOD-09, MOD-10, MOD-11, MOD-12, MOD-13, MCQ-01, MCQ-02, MCQ-03, OPEN-01, OPEN-02, OPEN-03, OPEN-05, CLI-08.

**Parallel safety and exclusive file ownership:** May run with CLI-04 through CLI-07 after builder contract freeze. Own existing `medfm/cli/train.py::_builders_for_config` and new `tests/challenges/medreason/test_training_cli.py`; training owners retain existing-after-training `medfm/challenges/medreason/training.py::medreason_builders`. Do not alter Phase-13/14/15.

**Target paths/symbols:** Existing `medfm/cli/train.py`, `medfm/training/config.py::RunConfig`, `medfm/training/pipeline.py::TrainingPipeline`, new MedReason builders, focused tests.

**Inputs:** `recipe.family: medreason`, `recipe.mode: mcq_sft|open_sft`, exact model/revision, audited hashes, strict `extensions.medreason`.

**Outputs:** MedReason `ComponentBuilders`; dry-run JSON `allocated: false`; valid non-dry order remains `preflight -> backend -> registry -> dataset -> model -> peft -> task -> optimizer -> evaluator -> trainer -> checkpoint`, under `artifacts/runs/medreason/`.

**Config keys and validation order:** (1) `RunConfig` schema/accelerator/batch/memory/optimizer/PEFT/quantization/steps. (2) CLI-08 strict challenge extension validates family/mode/task/revisions/hashes/gates not expressible in generic `RunConfig`. (3) Exact normalized `medreason` dispatch; near spellings fail, never tiny. (4) `TrainingPipeline.preflight` registry/capabilities. (5) dry-run stops; non-dry allocates. Generic `eval_every_steps` is currently only parsed, so MedReason builders/trainers must implement fixed evaluation/early stopping or reject long-run mode; configs must not claim generic support.

**Implementation:** Lazily import `medreason_builders`; reject `offline_tiny` and Phase-13 production routing; preserve all current train flags. Validation failures call no registry weight resolver, decoder, backend, or model.

**Tests:** `--help` retains flags without heavy imports. Invalid family/route/task/offline-tiny/batch/model/revision/unsupported early-stop path fails pre-builder. Dispatch regression covers MedReason and existing families; dry-run no allocation. Real runs double guarded. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_training_cli.py -k 'help or dispatch or invalid or dry_run or early_stop'
CUDA_VISIBLE_DEVICES='' uv run python -m medfm.cli.train --config tests/challenges/medreason/fixtures/configs/invalid_route.yaml --dry-run --format json
```

**Acceptance evidence:** Selection/stage trace, unchanged dispatch, help, `allocated:false`, early-stop capability rejection/implementation evidence. Tiny fakes prove no hardware.

**Non-goals and failure policy:** No Phase-13 route, tiny fallback, default downloads, or falsely operational `eval_every_steps` claim.

**Handoff:** Dispatch/config hashes, revision/manifests, validation-before-allocation proof, and actual early-stop implementation status.

## CLI-04 Add zero-shot tournament recipe configuration

**Depends on:** GOV-03, GOV-04, GOV-06, MOD-04, MOD-05, MOD-06, MOD-07, EVA-11, EVA-12, EVA-13, CLI-02, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** May run with CLI-05 through CLI-07. Own new `configs/recipes/medreason/zero_shot_tournament.yaml` and rows in `tests/challenges/medreason/test_configs.py`; no evaluator/registry edits.

**Target paths/symbols:** New YAML, CLI-08 evaluation schema, config tests.

**Inputs/outputs:** Exact MOD registry records, split/official/judge hashes and audited data; output per-candidate clean/no-image/shuffled-image predictions, telemetry, metrics, advancement report under `artifacts/runs/medreason/zero_shot_tournament/`.

**Config keys and validation order:** (1) `schema_version:1`, ID/mode, seed `2026`, output. (2) Exact ordered candidates: `google/gemma-4-31B-it@419b2efe421994fdfd3394e621983d4cc511cd4f`, `google/gemma-4-26B-A4B-it@47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, diagnostic-only `google/medgemma-27b-it@2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f`. (3) controls `[clean,no_image,shuffled_image]`, deterministic generation, metrics/proxies/telemetry. (4) speed gates `mcq_pp:0.5`, `gt:0.10`, `va:0.10`, `min_speedup:1.5`, at most three trainable. (5) hashes/split/preflight then sequential allocation.

**Implementation:** Explicit YAML without hidden interpolation; fixed roles/controls/advancement/proxy blocking/local JSON; no fabricated protected paths. Use CLI-02 stable command.

**Tests:** Evaluation `--help`; invalid duplicate/extra/mutable candidate, trainable diagnostic, missing control, seed/threshold drift, substitute judge, >3 training candidates—all pre-allocation. Snapshot exact order. Protected tournament guarded. Command:
```bash
uv run pytest -q tests/challenges/medreason/test_configs.py -k 'zero_shot and (snapshot or invalid or help)'
```

**Acceptance evidence:** Canonical hash, exact snapshot, mutation rejection, zero-allocation spy; no score/license/hardware claim.

**Non-goals and failure policy:** No candidate additions, trainable diagnostic, substitute judge, inferred access. Missing evidence blocks governed route.

**Handoff:** Config hash, candidate roles/revisions, controls, output root, thresholds and judge/rubric hashes.

## CLI-05 Add Gemma Four MCQ QLoRA recipe

**Depends on:** MOD-04, MOD-09, MOD-10, MOD-12, MOD-13, MOD-14, MCQ-01, MCQ-02, MCQ-03, MCQ-08, MCQ-10, CLI-03, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** May run with CLI-04/CLI-06/CLI-07. Own new `configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml` and config-test rows; no builder/generic recipe edits.

**Target paths/symbols:** New YAML; existing `RunConfig`; challenge validator; MCQ builder; `tests/challenges/medreason/test_configs.py`.

**Inputs/outputs:** Audited train/dev hashes, exact 31B/processor/template, MCQ-09 selected LR evidence. Output config/run at `artifacts/runs/medreason/gemma4_31b_mcq_qlora/`; adapter publication later under `artifacts/models/medreason/`.

**Config keys and validation order:** (1) Schema/model/revision, family `medreason`, mode `mcq_sft`, task, seed `2026`, epochs `2`, paths/tracking. (2) target `<label>: <option text>`, assistant-only mask, case-ID/epoch permutation, manifests, no truncation. (3) NF4 4-bit/double quant BF16 compute/storage, rank `16`, alpha `32`, dropout `0.05`, discovered language targets, frozen vision, checkpointing, no KV cache. (4) microbatch `1`, accumulation/global `16`, fused AdamW, LR candidates `[2e-5,5e-5]`, warmup `0.03`, linear decay, norm `0.3`, pilot `250`; fixed evaluator/early-stop support must be builder-owned, not assumed from unused generic `eval_every_steps`. (5) buckets `[2048,4096,8192,16384]`, 100-real-batch `min(85 GiB,0.90*memory)` gate, immutable/preflight/selected-LR evidence before allocation.

**Implementation:** Use `recipe`/strict `extensions.medreason`; adapter-only safetensors plus resumable optimizer checkpoints/local JSON. Preserve exactly:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train \
  --config configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml
```

**Tests:** Train `--help`; invalid revision/NF4/BF16/LoRA/cache/vision/batch/truncation/memory/selected-LR/early-stop/full-backbone config pre-factory; snapshot/canonical round trip. Protected 31B/memory double guarded and cannot be accepted on local 24,576 MiB GPU. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_configs.py tests/challenges/medreason/test_training_cli.py -k 'mcq_qlora and (help or invalid or snapshot or dry_run)'
CUDA_VISIBLE_DEVICES='' uv run python -m medfm.cli.train --config configs/recipes/medreason/gemma4_31b_mcq_qlora.yaml --dry-run --format json
```

**Acceptance evidence:** Config hash/snapshot/mutation matrix/allocation-free dry-run; protected training remains OPS.

**Non-goals and failure policy:** No full fine-tune, silent reduction, base GRPO/consistency, placeholder hashes, or fake early stopping.

**Handoff:** Recipe/artifact hashes, selected LR/targets, bucket/memory/early-stop evidence, adapter contract.

## CLI-06 Add Gemma Four open QLoRA recipe

**Depends on:** MOD-04, MOD-09, MOD-10, MOD-12, MOD-13, MOD-14, OPEN-01, OPEN-02, OPEN-03, OPEN-05, OPEN-06, OPEN-07, OPEN-08, OPEN-09, CLI-03, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** May run with CLI-04/CLI-05/CLI-07. Own new `configs/recipes/medreason/gemma4_31b_open_qlora.yaml` and its test rows; no builder/MCQ edits.

**Target paths/symbols:** New YAML, `RunConfig`, strict challenge validator, open builder, config tests.

**Inputs/outputs:** Evidence-availability and group manifests, exact 31B/processor/template, selected LR. Output `artifacts/runs/medreason/gemma4_31b_open_qlora/`; frozen adapter later under `artifacts/models/medreason/`.

**Config keys and validation order:** (1) Base schema/model/revision/family/mode `open_sft`/task/seed `2026`/epochs `2`. (2) response `observations/reasoning/answer`, `supervision_mode:released_evidence|answer_only`, eligible evidence hash, `pseudo_traces:false`, `proxy_preferences:false`, assistant mask. (3) open/MCQ retention `0.5/0.5`, source-group oversampling, no duplicate case. (4) same QLoRA/optimizer/pilot as CLI-05, with builder-owned fixed evaluation/early stopping rather than generic `eval_every_steps`. (5) same buckets/memory/immutable/evidence/license checks; allocation last.

**Implementation:** Missing eligible evidence forces answer-only. Disable vision LoRA/thinking/consistency/self-consistency in base. Preserve exactly:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m medfm.cli.train \
  --config configs/recipes/medreason/gemma4_31b_open_qlora.yaml
```

**Tests:** Train `--help`; invalid evidence/pseudo supervision/mix/oversampling/optional component/QLoRA/memory/LR/early-stop/revision pre-factory; snapshot and evidence/answer-only fixtures. Protected 31B/GPU double guarded. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_configs.py tests/challenges/medreason/test_training_cli.py -k 'open_qlora and (help or invalid or snapshot or answer_only or dry_run)'
CUDA_VISIBLE_DEVICES='' uv run python -m medfm.cli.train --config configs/recipes/medreason/gemma4_31b_open_qlora.yaml --dry-run --format json
```

**Acceptance evidence:** Config hash, snapshot, both supervision branches, mutation/no-allocation proof; no grounding/hardware inference.

**Non-goals and failure policy:** No synthetic supervision, premature vision adaptation, answer-only grounding claim, silent changes, or fake early stop.

**Handoff:** Config/artifact/evidence hashes, mode/mix, LR/targets, bucket/memory/evaluator status.

## CLI-07 Add frozen-system evaluation recipe configuration

**Depends on:** RUN-01, RUN-02, RUN-03, RUN-04, RUN-05, RUN-06, RUN-07, RUN-08, RUN-15, SEL-01, SEL-08, SEL-09, SEL-10, SEL-11, SEL-14, SEL-15, CLI-02, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** May run with CLI-04 through CLI-06. Own new `configs/recipes/medreason/frozen_system.yaml` and tests; no runtime/frozen artifact mutation.

**Target paths/symbols:** New YAML, challenge evaluation schema, SEL manifest, RUN runtime, config tests.

**Inputs/outputs:** One frozen research manifest/hash, base/processor/template/adapters, OOF calibration/support/threshold, judge/rubric/parser hashes, split/lockbox ledger. Output `artifacts/runs/medreason/frozen_system/` predictions, proxies, intervals, telemetry, receipt.

**Config keys and validation order:** (1) Schema, frozen ID/mode, seed `2026`, deterministic output. (2) Research manifest/hash and exact runtime artifacts/profile/calibration/support/view values matching it. (3) MCQ and proxy GT/VA/RVF, rubric/parser hashes, group-paired `1000`, strata/Holm metadata. (4) split membership/unused ledger. (5) hashes/licenses/preflight/no overrides, then runtime; judges only after system release.

**Implementation:** Manifest is sole tunable truth; only approved split/resample flags. One-base/two-adapter bounded fallback/redaction; optional components only if selected/hashed. Use CLI-02 exact command and atomic single-use receipt.

**Tests:** Evaluation `--help`; invalid manifest/hash/adapter/calibration/component/sampling/resample/ledger/participant-scoring pre-allocation; tiny fixture only. Protected system/judges/lockbox double guarded. Command:
```bash
uv run pytest -q tests/challenges/medreason/test_configs.py tests/challenges/medreason/test_cli.py -k 'frozen_system and (help or invalid or manifest or lockbox)'
```

**Acceptance evidence:** Config/manifest hashes, mutation table, no-allocation spy, fixture receipt; no protected metric/profile.

**Non-goals and failure policy:** No selection/tuning/recalibration/lockbox comparison/retry/participant scoring/substitution.

**Handoff:** Frozen hashes/artifacts, split/receipt, reports, proxies, measured hardware only when observed.

## CLI-08 Validate challenge configs before allocating models

**Depends on:** GOV-08, GOV-10, SCH-05, MOD-04, MOD-05, MOD-06, MOD-08, MOD-09, MOD-12, MOD-13, EVA-08.

**Parallel safety and exclusive file ownership:** Shared prerequisite; callers proceed once fields freeze. Own new `medfm/challenges/medreason/config.py`, new `tests/challenges/medreason/test_config_validation.py`, fixtures. Avoid generic `RunConfig` changes unless coordinated.

**Target paths/symbols:** New `MedReasonConfigError`, `load_medreason_mapping`, `validate_audit_request`, `validate_training_config`, `load_evaluation_config`, `validate_evaluation_request`, canonical hashing; existing `RunConfig`, `TrainingPipeline.preflight`, MedReason preflight.

**Inputs/outputs:** YAML/CLI and immutable governance/data/model/judge/frozen references; produce typed requests, normalized references, canonical JSON/hash, ordered issues or sanitized error. No network, image decode, Torch/CUDA query, processor/model allocation.

**Config keys and validation order:** (1) UTF-8 strict mapping/duplicates/schema/allowed keys. (2) required types/ranges/enums/unknown keys. (3) generic accelerator/batch/memory/optimizer/freeze/PEFT/quantization/steps. (4) challenge family/mode/task/seed/masks/supervision/determinism/candidates/no forbidden data. (5) immutable revisions and processor/template/quantization/rubric/data/split/manifest hashes/path containment/no placeholders. (6) licenses/access/preflight/judges/pilot/measured memory/OOF/lockbox. (7) cross allocation boundary only on success.

**Implementation:** Strict typed validators and canonical serialization; separate pure/injected resolution but execute both pre-builder; deterministic sanitized field paths; one shared boundary. Preserve challenge values under `recipe`/`extensions.medreason`. Explicitly reject long-run configs whose requested fixed evaluation/early stopping is unsupported by the selected MedReason trainer; generic parsed-but-unused `eval_every_steps` is insufficient.

**Tests:** All module `--help` paths no resolver/factory/CUDA. Invalid syntax/type/route/revision/hash/quantization/batch/forbidden supervision/judge/pilot/memory/OOF/lockbox/early-stop at earliest layer. Spies cover decode/backend/AutoProcessor/AutoModel/judge/tracker. Hardware tests double guarded and matching 48/96 GB evidence required. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_config_validation.py
uv run pytest -q tests/challenges/medreason/test_cli.py tests/challenges/medreason/test_training_cli.py -k 'help or no_allocation'
```

**Acceptance evidence:** Canonical hashes, mutation table, ordered trace, zero spies; fixture policy is not access/hardware.

**Non-goals and failure policy:** No autocorrection/downloads/placeholders/accelerator initialization/weakened gate.

**Handoff:** Typed request, hash, trace/preflight receipt, allocation guarantee.

## CLI-09 Extend run metadata with challenge provenance

**Depends on:** GOV-08, DAT-11, MOD-02, MOD-09, MOD-10, EVA-05, EVA-09, EVA-10, CLI-08.

**Parallel safety and exclusive file ownership:** May run with recipes after keys freeze. Own additions to existing `medfm/training/run_metadata.py::{RunMetadata,capture_run_metadata}`, MedReason handoff in `medfm/training/trainer.py`, new `tests/challenges/medreason/test_tracking.py`; reuse `tracking.py`.

**Target paths/symbols:** Existing metadata/trainer/tracking and new tests. Prefer versioned `RunMetadata.extra['medreason']` if SCH-05 supplies record, preserving generic compatibility.

**Inputs/outputs:** Config/challenge/task/candidate; official commit `05748c0341b72dc08132bd108208b78dc14a2f0b`; archive/extracted/dataset/split/model/processor/template/adapter/quantization/target/rubric/artifact hashes; policy/state/measurements. Existing `<run>/tracking/params.json` gains versioned `medreason`; metrics remain JSONL.

**Config keys and validation order:** Required keys: `schema_version`, `config_sha256`, `official_commit`, `challenge_profile`, `task_route`, `candidate_id`, source/extracted/dataset/split hashes, `base_model_revision`, processor/template/adapter/quantization/targets/rubric hashes, seed/buckets/supervision/control/selection, artifact manifest. (1) CLI-08 canonicalizes. (2) required/null and lowercase 64-hex/exact commit. (3) cross-check manifest/config. (4) redact before writes: no content/metadata/paths/patient-study-article IDs/secrets. (5) append actual measurements after allocation without changing immutable identity. Metadata validation fails before model/tracker.

**Implementation:** Preserve canonical/config-hash semantics; thread validated hashes, never rederive mutable inputs; local JSON default; atomic params; raw predictions stay in EVA artifacts.

**Tests:** Training/evaluation `--help` creates no tracking. Invalid hash/commit/mismatch/participant answer/raw content fails pre-factory/write. Round-trip/redaction/determinism and Phase-01 compatibility. Real telemetry GPU guarded. Commands:
```bash
uv run pytest -q tests/challenges/medreason/test_tracking.py tests/phase_01/test_run_metadata.py tests/phase_01/test_tracking.py
uv run pytest -q tests/challenges/medreason/test_cli.py -k 'help and no_tracking_write'
```

**Acceptance evidence:** Metadata hash/key snapshot/mutation/redaction/no-write proof; fixture telemetry is not profile evidence.

**Non-goals and failure policy:** No clinical logging, placeholder evidence, default external upload, or inferred access/hardware.

**Handoff:** Privacy-safe identity hashes for training/EVA/SEL/Docker.

## CLI-10 Test module help and config failure paths

**Depends on:** CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, CLI-07, CLI-08, CLI-09.

**Parallel safety and exclusive file ownership:** Runs after interfaces stabilize. Own final consolidation of new `tests/challenges/medreason/{test_cli.py,test_training_cli.py,test_config_validation.py,test_configs.py,test_packaging.py}` and fixtures; coordinate prior owners. Do not change production just for tests.

**Target paths/symbols:** Those tests/fixtures; entry points; validators; builder/tracker boundaries; existing `tests/phase_01/test_packaging.py` conventions; existing `pyproject.toml` Hatch inventory (verify; implementation edits only if required).

**Inputs/outputs:** Four configs, synthetic archives/manifests/frozen records, mutation table and spies. Output CPU matrix for help/contracts/hashes/failure order/redaction/no allocation/hardware guards, plus wheel/source inventory for challenge modules and configs.

**Config keys and validation order:** (1) Inventory every audit/evaluation/RunConfig/recipe/provenance key with valid and missing/invalid conditional coverage. (2) usage `2`, syntax/type, cross-field, immutable, governance/artifact, allocation boundary `1`. (3) valid tiny fixture reaches fake factory after trace; real downloads/CUDA/network forbidden. (4) help/errors write nothing. (5) Packaging imports `medfm.challenges.medreason.data`, `.evaluate`, `.config`, `.training` from installed wheel with CUDA unavailable; release inventory contains `configs/recipes/medreason/{zero_shot_tournament,gemma4_31b_mcq_qlora,gemma4_31b_open_qlora,frozen_system}.yaml`, even if configs intentionally are not wheel package data.

**Implementation:** Follow direct `main([...])`/`capsys` and `sys.executable -m` subprocess conventions with repo `cwd`, capture, `check=False`. Parameterize help/mutations; assert earliest sanitized error, clean outputs, zero spies; regress exact MedReason dispatch and existing families. Test hardware/environment guard behavior; skip is not acceptance.

**Tests:** `--help` for data, data audit, evaluate, train. Invalid syntax/path/hash/revision/route/batch/QLoRA/bucket/pseudo/judge/split/lockbox/provenance/early-stop. Packaging failures include omitted module, eager CUDA/Transformers import, missing/renamed recipe. Commands, not run now:
```bash
uv run pytest -q tests/challenges/medreason/test_cli.py tests/challenges/medreason/test_training_cli.py tests/challenges/medreason/test_config_validation.py tests/challenges/medreason/test_configs.py tests/challenges/medreason/test_packaging.py
CUDA_VISIBLE_DEVICES='' uv run pytest -q tests/challenges/medreason -k 'help or invalid or no_allocation or packaging'
```

**Acceptance evidence:** Key-test inventory, help captures, hashes/traces, zero-call spies, clean outputs, guard assertions, wheel/source inventory. Implementation produces evidence later; this planning phase runs nothing.

**Non-goals and failure policy:** No protected execution, fake official/hardware results, source-text tests, full-suite expansion, or weakened policy. Protected and fixture acceptance remain distinct.

**Handoff:** Exact commands, mutation matrix, help snapshots, allocation proof, packaging inventory, hardware guards, and protected acceptance owed by OPS/Docker.
