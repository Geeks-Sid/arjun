# Models processors and PEFT

This phase builds MedReason's production model boundary. It extends the current fail-closed registry and reuses `medfm.peft`, `medfm.training`, `medfm.data.bucketing.BucketPlan`, and assistant-mask invariants from `medfm.data.textprep.tokenize`; it never routes production through the synthetic/offline builders in `medfm.recipes.phase13`. Base snapshots are immutable at `artifacts/models/medreason/base/<model_id>/<40-character-revision>/`, judges at `artifacts/judges/medreason/<model_id>/<revision>/`, and adapter-only outputs at `artifacts/models/medreason/adapters/<run_id>/<adapter_name>/`. Ordinary tests use fake processors/models and make no network call. Protected checks require `MEDFM_RUN_REAL_CHECKPOINTS=1`, explicit local snapshots, accepted gated access where applicable, and suitable measured hardware; a skip is not acceptance evidence.

## MOD-01 Implement native multimodal Hugging Face loader

**Depends on:** SCH-01, SCH-05, GOV-06.

**Parallel safety and exclusive file ownership:** May run beside MOD-04–MOD-08 after IDs are frozen. Exclusively owns `[new] medfm/challenges/medreason/models.py` and loader cases in `[new] tests/challenges/medreason/test_models.py`; do not edit registry YAML while registry-card owners are active.

**Target paths/symbols:** `[new] medfm/challenges/medreason/models.py`: `MedReasonModelRef`, `LoadedMultimodalModel`, `load_multimodal_model`; existing `medfm.registry.core.ModelRegistry`, `medfm.registry.catalog.ensure_v1_catalog`, and `medfm.peft.quantization.build_bitsandbytes_config`. Extend package-inventory coverage in `tests/phase_01/test_packaging.py`.

**Inputs:** Registered model ID, exact revision, immutable local snapshot/manifest hash, validated load mode, attention backend, dtype/device, optional `QuantizationConfig`.

**Outputs:** `LoadedMultimodalModel(processor, model, model_ref, snapshot_manifest_hash)` plus exact class/revision/load metadata.

**Implementation:**
1. Add exact-revision plumbing from registry through weight resolution to both factories; requested SHA must equal registry and snapshot-manifest SHAs before allocation.
2. Instantiate production VLMs through `AutoProcessor.from_pretrained` and `AutoModelForMultimodalLM.from_pretrained`, same local snapshot/SHA, `local_files_only=True`. MedGemma may resolve to its concrete Gemma-3 conditional-generation class, but callers use the multimodal auto boundary. Never use `AutoModelForCausalLM` for a VLM.
3. Extend the weights gate so cached weights for blocked/unaccepted records remain unusable. Verify every manifest-listed model/processor file, including chat-template `.jinja`; reject tag/branch/placeholder, path escape, mismatch, or fallback.
4. Pass only validated dtype, attention, device, and quantization arguments. Expose an injectable seam only for fake tests.
5. Import none of `medfm.models.language.gemma`, `medfm.models.language.medgemma`, `medfm.models.visual.medgemma_vision`, or `medfm.recipes.phase13`; a sentinel test proves Phase-13 is untouched.
6. Emit model-ID/error-class-only failures. Inventory the challenge module in wheels while excluding tests/artifacts.

**Tests:** `FakeAutoProcessor`/`FakeAutoModelForMultimodalLM` assert one exact SHA, local-only calls, class metadata, forwarding, and preallocation rejection. Failures: cached-but-blocked weights, missing `.jinja`, split revisions, causal substitution, Phase-13 call.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_models.py -k mod_01`
- Packaging: `uv run --frozen pytest -q tests/phase_01/test_packaging.py`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k native_loader`; explicit local path, never download.
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json` and `uv run --frozen medfm models validate medgemma-1.5-4b --format json`.

**Acceptance evidence:** Fake call logs show `AutoProcessor` plus `AutoModelForMultimodalLM`, matching SHA, and no Phase-13 invocation; wheel inventory contains the module. Protected JSON records manifest hash/classes. Fixtures prove no model access or GPU fit.

**Non-goals/failure policy:** No download, license acceptance, causal Gemma adapter, hardware claim, or Phase-13 promotion. Fail closed.

**Handoff:** MOD-02, MOD-03, MOD-09, MOD-12, MOD-13, MOD-14 consume the loaded object and exact manifest identity.

## MOD-02 Preserve checkpoint processor and chat template behavior

**Depends on:** MOD-01, GOV-06.

**Parallel safety and exclusive file ownership:** May run with registry tasks/MOD-09. Owns template symbols in `[new] medfm/challenges/medreason/processors.py` and its tests; coordinate same-file work with MOD-03/MOD-12.

**Target paths/symbols:** `[new] medfm/challenges/medreason/processors.py:NativeProcessorContract`, `load_native_processor_contract`, `apply_native_chat_template`; `medfm/data/textprep/tokenize.py:IGNORE_INDEX` supplies masking semantics, not production tokenization/truncation; `[new] tests/challenges/medreason/test_processors.py`.

**Inputs:** Loaded processor, manifest-verified processor/tokenizer/special-token/`.jinja` files, ordered messages/images.

**Outputs:** Native rendered encoding and `processor_contract.json` with revision, file/template hashes, class, special/image-token IDs, applied kwargs.

**Implementation:**
1. Use native `apply_chat_template`; never reconstruct template/image placeholders.
2. Preserve message/image order, generation-prompt mode, native normalization/resizing/special tokens. Record MedGemma 896x896/256-token behavior only when observed on the protected exact processor.
3. Reject absent/tampered/cross-revision templates; never log prompt/template text.
4. Mask system/user/padding/image-placeholder positions with `IGNORE_INDEX=-100` and supervise only native assistant spans. Do not re-tokenize or call `build_supervised_example` because it truncates.
5. Persist hashes, not contents.

**Tests:** Fake native template/image placeholder/assistant mask proves order, single tokenization, train/inference parity, assistant-only loss. Fail on missing `.jinja`, manual placeholder, mismatch, prompt-label leakage.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_processors.py -k mod_02`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k processor_template`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Contract hashes match manifest and fake masks pass. Real processor/template/896 facts require protected evidence.

**Non-goals/failure policy:** No project template, pseudo-trace, prompt logging, fallback tokenizer, or truncating textprep helper.

**Handoff:** MOD-03/MOD-12 and trainers consume processor hash/native encoding/mask contract.

## MOD-03 Preserve all processor-generated multimodal tensor fields

**Depends on:** MOD-01, MOD-02.

**Parallel safety and exclusive file ownership:** May run with registry tasks/MOD-09. Owns `prepare_native_batch`/`forward_native_batch` in `processors.py`; coordinate MOD-02/MOD-12.

**Target paths/symbols:** `[new] medfm/challenges/medreason/processors.py:NativeMultimodalBatch`, `prepare_native_batch`, `forward_native_batch`; `medfm/data/collators/vl.py` and `medfm/core/batch.py:MedicalBatch` are references only and must not narrow Gemma-4 native mappings; `[new] tests/challenges/medreason/test_processors.py`.

**Inputs:** Complete `AutoProcessor` `BatchEncoding`, separate labels.

**Outputs:** Open mapping preserving keys/values/dtypes/shapes/order plus contents-free field inventory/hash.

**Implementation:**
1. Preserve Gemma-4 `input_ids`, `attention_mask`, `mm_token_type_ids`, `pixel_values`, `image_position_ids`; Gemma-3/MedGemma `token_type_ids`, `pixel_values`; Qwen judge `mm_token_type_ids`, `pixel_values`, `image_grid_thw`; and any actual extras. Do not require omitted fields.
2. Forward `model(**native_fields, labels=labels)`; never rename `mm_token_type_ids`.
3. Move tensors without needless same-device copies or integer/boolean coercion; preserve image order.
4. Reject non-forwardable values with field-only errors. Explicitly record any signature-derived bookkeeping exclusion.
5. Record names/dtypes/shapes only.

**Tests:** Fake processors emit all family variants plus `native_extra_mask`; fake model captures kwargs. Assert key-for-key forwarding/no copy/no coercion/order. Fail on dropped/renamed fields, float masks, silent filtering.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_processors.py -k mod_03`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k processor_fields`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Fake kwargs equal processor mappings tensor-for-tensor. Each real family needs protected inventory/forward evidence.

**Non-goals/failure policy:** No universal fixed schema, `MedicalBatch` coercion, manual image flattening, or old-collator narrowing.

**Handoff:** Training/scoring/runtime receive native mapping and inventory hash.

## MOD-04 Add exact Gemma Four quality registry record

**Depends on:** GOV-03, GOV-06, SCH-05.

**Parallel safety and exclusive file ownership:** May run with MOD-05–MOD-08 if each owns its stanza; coordinate shared catalog-map integration.

**Target paths/symbols:** `model_registry/v1_scope.yaml`, `model_registry/licenses.yaml` key `gemma-4-31b-it`; `medfm/registry/catalog.py:_APPROX_PARAMS_B`, `_OUTPUTS`, `_GENERATIVE`, `load_v1_catalog`; `[new] tests/challenges/medreason/test_model_registry.py`.

**Inputs:** `google/gemma-4-31B-it@419b2efe421994fdfd3394e621983d4cc511cd4f`, GOV-03 evidence, GOV-06 manifest.

**Outputs:** Exact multimodal quality record; backends remain `UNTESTED` absent measured evidence.

**Implementation:** Add exact repo/SHA to both YAMLs and all catalog maps; add multimodal outputs/complete backend keys; plumb revision into weight resolution/MOD-01; retain honest legal/backend states and keep parameter count separate from memory evidence.

**Tests:** Registry and fake processor/model dispatch assert exact case/SHA/auto loader/maps/backends/legal gate. Fail on tag, causal loader, missing map, false `SUPPORTED_*`.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_model_registry.py -k mod_04`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k gemma_4_31b`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** CLI/manifest exact identity and honest states. Snapshot success proves access/integrity, not 48/96 GB fit.

**Non-goals/failure policy:** No support from vendor report/parameter count; incomplete legal/artifact evidence blocks load.

**Handoff:** Tournament/training/Docker consume stable ID/SHA/legal state/hash.

## MOD-05 Add exact Gemma Four speed registry record

**Depends on:** GOV-03, GOV-06, SCH-05.

**Parallel safety and exclusive file ownership:** Same registry coordination; owns `gemma-4-26b-a4b-it` only.

**Target paths/symbols:** `model_registry/v1_scope.yaml` and `model_registry/licenses.yaml` key `gemma-4-26b-a4b-it`; `medfm/registry/catalog.py:_APPROX_PARAMS_B`, `_OUTPUTS`, `_GENERATIVE`, `load_v1_catalog`; `[new] tests/challenges/medreason/test_model_registry.py`.

**Inputs:** `google/gemma-4-26B-A4B-it@47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, legal evidence, manifest.

**Outputs:** Exact conditional speed record, unmeasured backends `UNTESTED`.

**Implementation:** Add exact identity/capabilities to both YAMLs/maps; route MOD-01/native fields; label fallback without treating active parameters/1.5x as measurement; promote only from measured quality/latency/memory.

**Tests:** Fake dispatch/registry checks. Fail on dense/causal substitution, SHA typo, automatic support, reported speed as measurement.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_model_registry.py -k mod_05`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k gemma_4_26b_a4b`
- Registry: `uv run --frozen medfm models validate gemma-4-26b-a4b-it --format json`.

**Acceptance evidence:** CLI/manifests exact SHA; no protected measurements means no speed/fit claim.

**Non-goals/failure policy:** No estimated 1.5x, silent fallback, alternate revision.

**Handoff:** Selection consumes stable ID and measured artifacts.

## MOD-06 Update exact MedGemma specialist registry record

**Depends on:** GOV-04, GOV-06, SCH-05.

**Parallel safety and exclusive file ownership:** Owns existing `medgemma-1.5-4b` stanzas/tests; coordinate maps.

**Target paths/symbols:** `model_registry/v1_scope.yaml` and `model_registry/licenses.yaml` key `medgemma-1.5-4b`; `medfm/registry/catalog.py:_APPROX_PARAMS_B`, `_OUTPUTS`, `_GENERATIVE`, `load_v1_catalog`; `medfm/models/visual/specs.py:MEDGEMMA_MODEL_ID` must agree but is not the production loader; `[new] tests/challenges/medreason/test_model_registry.py`.

**Inputs:** `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, HAI-DEF/access/redistribution evidence, manifest.

**Outputs:** Exact optional-specialist record with unresolved terms blocking.

**Implementation:** Replace placeholders only after verification; update YAMLs/maps/revision plumbing; preserve native processor via MOD-01 auto boundary resolving concrete Gemma-3 model; keep existing adapters/Phase-13 outside production; optional open-evidence role only.

**Tests:** Fake processor/model checks exact SHA/gate/native fields/no Phase-13. Protected processor alone may record 896x896/256-token facts. Fail on invented acceptance/redistribution, placeholder.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_model_registry.py -k mod_06`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k medgemma_1_5_4b`
- Registry: `uv run --frozen medfm models validate medgemma-1.5-4b --format json`.

**Acceptance evidence:** Exact CLI legal/SHA and manifest. Shipping/access/processor claims require protected evidence.

**Non-goals/failure policy:** No redistribution, co-residence, hardware, benefit claim; omit if legal gate fails.

**Handoff:** Fusion receives only after all legal/artifact/metric/runtime/memory gates.

## MOD-07 Add frozen MedGemma scale diagnostic record

**Depends on:** GOV-04, GOV-06, SCH-05.

**Parallel safety and exclusive file ownership:** Owns `medgemma-27b-it` stanzas/tests; coordinate maps.

**Target paths/symbols:** `model_registry/v1_scope.yaml` and `model_registry/licenses.yaml` key `medgemma-27b-it`; `medfm/registry/catalog.py:_APPROX_PARAMS_B`, `_OUTPUTS`, `_GENERATIVE`, `load_v1_catalog`; `[new] tests/challenges/medreason/test_model_registry.py`.

**Inputs:** `google/medgemma-27b-it@2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f`, legal evidence, manifest.

**Outputs:** Frozen diagnostic record excluded from PEFT/fusion/deployment.

**Implementation:** Add exact repo/SHA/legal/backends to both YAMLs/maps; native MOD-01 route; encode diagnostic-only and reject train/adapter/deployment before allocation.

**Tests:** Fake processor/model forward and rejected fake training request. Fail on adapter/deployment, drift, invented acceptance.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_model_registry.py -k mod_07`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k medgemma_27b_diagnostic`
- Registry: `uv run --frozen medfm models validate medgemma-27b-it --format json`.

**Acceptance evidence:** CLI exact SHA/diagnostic state. Protected forward never proves training.

**Non-goals/failure policy:** Never train/fuse/export/deploy; missing access does not block core.

**Handoff:** Evaluation may consume tagged frozen predictions; trainers reject ID.

## MOD-08 Add exact proxy judge registry records

**Depends on:** GOV-05, GOV-06, SCH-05.

**Parallel safety and exclusive file ownership:** Owns two judge stanzas/maps/tests; evaluator prompts/scoring remain EVA scope.

**Target paths/symbols:** `model_registry/v1_scope.yaml` and `model_registry/licenses.yaml` keys `llama-3.1-70b-instruct-judge` and `qwen2.5-vl-72b-instruct-judge`; `medfm/registry/catalog.py:_APPROX_PARAMS_B`, `_OUTPUTS`, `_GENERATIVE`, `load_v1_catalog`; `[new] tests/challenges/medreason/test_model_registry.py`; judge-only artifact root `artifacts/judges/medreason/`.

**Inputs:** `meta-llama/Llama-3.1-70B-Instruct@1605565b47bb9346c5515c34102e054115b4f98b`; `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5`; evidence (Qwen exact custom license must not be guessed), manifests.

**Outputs:** Exact judge-only/sequential-load/nondeployment records with unresolved terms blocking.

**Implementation:** Add exact identities to both YAMLs/maps; no training/adapter/co-residence; Llama uses tokenizer + `AutoModelForCausalLM`; Qwen uses `AutoProcessor` + `AutoModelForMultimodalLM` and native fields; never call judges official/available without evidence.

**Tests:** Fake Llama text factories/Qwen multimodal factories and catalog exclusion. Fail on swapped SHA, Qwen causal load, co-residence, fake approval/official name.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_model_registry.py -k mod_08`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k proxy_judge_registry`
- Registry: `uv run --frozen medfm models validate llama-3.1-70b-instruct-judge --format json` and `uv run --frozen medfm models validate qwen2.5-vl-72b-instruct-judge --format json`.

**Acceptance evidence:** CLI exact SHA/legal/backend/judge role. Fixtures prove no proxy score/availability.

**Non-goals/failure policy:** No prompts/substitute/official claim/concurrent load/shipping; missing judge stops proxy promotion.

**Handoff:** EVA consumes identities/legal states/manifest hashes.

## MOD-09 Build hashed NF4 quantization configuration

**Depends on:** MOD-01, MOD-04, MOD-05, MOD-06.

**Parallel safety and exclusive file ownership:** May run with processor work. Owns `[new] medfm/challenges/medreason/peft.py:build_medreason_nf4_plan` and NF4 tests; reuse generic PEFT.

**Target paths/symbols:** `medfm.peft.config.QuantizationConfig`, `BackendPeftPlan`, `config_hash`, `validate_backend_combination`; `medfm.peft.quantization.build_bitsandbytes_config`, `prepare_model_for_kbit_training`, `disable_training_kv_cache`, `verify_compute_dtype`, `assert_qlora_trainability`.

**Inputs:** CUDA, exact base identity, NF4 four-bit/double-quant/BF16 compute and **BF16 quant storage**, LoRA config.

**Outputs:** Validated plan, upstream `BitsAndBytesConfig`, canonical SHA-256.

**Implementation:**
1. Extend canonical `QuantizationConfig` and `build_bitsandbytes_config` for explicit `bnb_4bit_quant_storage=torch.bfloat16` (currently missing); require NF4, four-bit, double quant, BF16 compute/storage.
2. Validate before allocation; reject FP4/float32/unsupported backend/unavailable capabilities.
3. Hash complete defaults including storage dtype/versions, never `repr()`.
4. Pass through MOD-01; disable cache, k-bit prepare, verify dtypes, exclude base optimizer parameters.
5. MedGemma BF16 LoRA remains distinct/unquantized/measured.

**Tests:** Fake processor/BitsAndBytes/model objects assert all upstream fields, stable hash, semantic-change sensitivity, preallocation checks, and frozen base weights. Fail on missing storage dtype, TPU/CPU NF4, enabled cache, or base optimizer parameters.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_peft.py -k mod_09`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k nf4_load`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Canonical hash round-trip includes BF16 storage; optimizer audit. Protected only proves exact device/checkpoint.

**Non-goals/failure policy:** No full tune, offload-as-fit, XLA equivalence, estimated acceptance.

**Handoff:** MOD-10/11/13 consume plan hash.

## MOD-10 Discover and record language LoRA targets

**Depends on:** MOD-01, MOD-09.

**Parallel safety and exclusive file ownership:** May run with MOD-12. Owns `peft.py:discover_language_lora_targets`; generic resolver changes require compatibility coverage.

**Target paths/symbols:** `medfm/peft/resolver.py:inspect_modules`, `resolve_targets`, `TargetResolution`, `_LLM_RULES`; `medfm/peft/lora.py:inject_lora`, `audit_trainable_parameters`; `[new] medfm/challenges/medreason/peft.py:discover_language_lora_targets`; `[new] tests/challenges/medreason/test_peft.py`.

**Inputs:** Loaded checkpoint, explicitly checked language submodule, architecture, rank-16/alpha-32/dropout-.05/no-bias config, hashes.

**Outputs:** `lora_targets.json` ordered names/types/shapes/counts/reasons/exclusions/audit/hash.

**Implementation:** Discover only in verified language boundary; match q/k/v/o and gate/up/down projections; exclude vision/projector/embedding/head/norm/base; fail unknown/zero/broad/outside; inject named adapters only after review; require same target hash on resume.

**Tests:** Fake multimodal model with tempting vision/projector linears and fake processor. Fail on root scan/outside/zero/broad/drift/trainable base.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_peft.py -k mod_10`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k lora_target_inventory`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Hashed selected/excluded inventory/audit; fake names are not real evidence.

**Non-goals/failure policy:** No all-linear, vision/projector LoRA, silent rediscovery.

**Handoff:** MOD-11/trainers consume hash/names.

## MOD-11 Save and reload adapter-only safetensors artifacts

**Depends on:** MOD-09, MOD-10.

**Parallel safety and exclusive file ownership:** May run with MOD-12/14. Owns `[new] medfm/challenges/medreason/checkpoints.py` and challenge checkpoint tests.

**Target paths/symbols:** `medfm.peft.checkpoint.save_adapter_checkpoint`, `load_adapter_checkpoint`, `load_checkpoint_manifest`, `compare_merged_unmerged`; competing `medfm.training.checkpoint.CheckpointManager`; `[new] save_medreason_adapter`, `load_medreason_adapter`.

**Inputs:** Exact base identity/architecture, named adapter, LoRA/quantization/processor/template/target hashes, run/split provenance.

**Outputs:** `adapter.safetensors` and `manifest.json` with `kind=adapter_only`, exact identity/hashes/tensor files.

**Implementation:**
1. Canonicalize exporters: `CheckpointManager` owns resumable state; final portable export delegates to `medfm.peft.checkpoint.save_adapter_checkpoint`. No second portable format.
2. Whitelist adapter tensors; reject base/vision/projector/embedding/optimizer/scheduler/data. Hash every file.
3. Reload only identical identity/config; fail corruption/mismatch/missing key; compare deterministic original/reloaded fake outputs.
4. Research/all-data artifacts use different run IDs/hashes. Final export must refuse overwrite, addressing current overwriteability. Never merge base.

**Tests:** Fake processor/model export/key inspection/reload/output parity. Fail frozen key, optimizer, wrong SHA, tamper, collision, missing hash, overwrite.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_checkpoints.py -k mod_11`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k adapter_round_trip`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Tensor inventory adapter-only, hashes, parity; real compatibility needs protected round trip.

**Non-goals/failure policy:** No full weights/optimizer publication/format fork/merge/alias/overwrite.

**Handoff:** Runtime/Docker consume adapter manifest/base/processor/quantization/target hashes.

## MOD-12 Enforce processor-length buckets without silent truncation

**Depends on:** MOD-02, MOD-03.

**Parallel safety and exclusive file ownership:** May run with MOD-10/11. Owns `processors.py:assign_processor_length_bucket`, `collate_native_bucket`; coordinate file.

**Target paths/symbols:** `[new] medfm/challenges/medreason/processors.py:ProcessorLength`, `assign_processor_length_bucket`, `collate_native_bucket`; reuse `medfm/data/collators/buckets.py:BucketPlan` constructed with `out_of_bucket_policy="error"` and its `assign` method; buckets 2048/4096/8192/16384; `[new] tests/challenges/medreason/test_processors.py`. Do not use truncating `medfm/data/textprep/tokenize.py:build_supervised_example` or force native mappings into `medfm/core/batch.py:MedicalBatch`.

**Inputs:** Complete native output after visual expansion and labels.

**Outputs:** Measured length, smallest bucket, padding-only native batch, sanitized overflow rejection.

**Implementation:** Derive length from processor/model dimensions, not estimate; use `BucketPlan` error policy and native pad semantics; preserve every field/mask; prohibit truncation/slicing/image dropping/placeholder deletion; overflow rejects before forward; larger bucket needs observed requirement/config review; ensure supervised tokens remain.

**Tests:** Fake lengths 2048/2049/4096/16384/16385 and position fields; model verifies prefixes/alignment. Fail 16385 truncation, text estimate, field drift, all labels masked.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_processors.py -k mod_12`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_real_checkpoints.py -k processor_length_buckets`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Boundary report measured lengths/zero truncation. Real distribution protected.

**Non-goals/failure policy:** No silent truncation/speculative bucket/runtime image selection/case exception.

**Handoff:** Trainers/MOD-13 consume bucket IDs/shapes/rejections.

## MOD-13 Enforce measured hundred-batch CUDA memory gate

**Depends on:** MOD-09, MOD-10, MOD-11, MOD-12, GOV-09.

**Parallel safety and exclusive file ownership:** May run with MOD-14. Owns `[new] medfm/challenges/medreason/memory.py`, challenge memory tests/artifact schema; use challenge wrapper rather than mutate shared trainer scheduling/checkpoint state.

**Target paths/symbols:** `medfm.training.backend.CudaBackend.memory_snapshot`, `reset_peak_memory_stats`, `MemorySnapshot`; `medfm.training.memory.MemoryPlanningError`; `[new] HundredBatchMemoryGate`, `run_hundred_batch_memory_gate`.

**Inputs:** Exact hashes, declared real bucket stream, device identity/total, microbatch 1/accumulation 16/checkpointing/no cache, exactly 100 real forward/backward batches.

**Outputs:** `artifacts/runs/medreason/<run_id>/memory_gate.json` environment/device/total/hashes/bucket/count/reset/peaks/threshold/pass/fail.

**Implementation:**
1. Preflight CUDA/BF16/artifacts/device; missing is BLOCKED. Local RTX 3090 24,576 MiB proves no 31B/26B or 48/96 GB profile.
2. Reset after defined setup/warmup; run 100 real processed batches through loss/backward/accumulation/optimizer. Challenge wrapper records its own batch sequence because generic trainer omits batch-sampler state.
3. Require peak allocated `< min(85 GiB, floor(0.90 * total_device_memory_bytes))`; reserved separate, no estimates.
4. Persist failure on OOM/nonfinite/rejection/fewer than 100/drift/no measurement. Changed strategy is a new artifact.

**Tests:** Fake CUDA/processor/model covers 99/100, equal/below, OOM/drift/reset. Fail estimate, synthetic-as-real, missing hashes, cross-device claim.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_memory_gate.py -k mod_13`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_memory_gate.py -k protected_hundred_batch`; also explicit snapshots/data/CUDA/suitable memory.
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Hardware acceptance requires non-skipped JSON with `real_batches=100`, exact hashes/device/total, measured peaks/threshold, pass.

**Non-goals/failure policy:** No hardware/co-residence/long-run claim from 24 GB; never estimate away runtime memory.

**Handoff:** Operations/selection consume gate hash for exact profile only.

## MOD-14 Verify attention backend numerical parity before promotion

**Depends on:** MOD-01, MOD-03, MOD-09, MOD-12.

**Parallel safety and exclusive file ownership:** May run with MOD-13. Owns `[new] medfm/challenges/medreason/attention.py`, challenge parity tests/artifacts.

**Target paths/symbols:** Existing `medfm/models/language/base.py:LanguageModelConfig` names `eager`, `sdpa`, `flash_attention_2`; `medfm/challenges/medreason/models.py:load_multimodal_model` `attn_implementation`; `[new] medfm/challenges/medreason/attention.py:AttentionParityResult`, `compare_attention_backends`; `[new] tests/challenges/medreason/test_attention_parity.py`.

**Inputs:** Exact artifact hashes, frozen representative batches across used buckets/tasks, eager reference, SDPA/Flash candidate, deterministic BF16 settings.

**Outputs:** `artifacts/runs/medreason/<run_id>/attention_parity.json` with environment/hashes/batches/shapes/finite checks/differences/tolerance/ranking/output/latency/memory/pass.

**Implementation:**
1. Compare clean identical loads/inputs in eval/no-sampling; eager reference.
2. Require finite relevant logits, preregistered BF16 `rtol=1e-2`, `atol=1e-2`, identical MCQ ranking and parsed greedy output.
3. Measure latency/peaks but never trade correctness. Missing Flash is BLOCKED, not SDPA failure.
4. Run separately per model/quantization/bucket/adapter/device/kernel; persist failure and leave unproven until protected pass.

**Tests:** Fake processor/backends cover exact/within/rank flip/nonfinite/greedy change/unavailable. Fail differing inputs/weights, sampling, post-hoc tolerance, averaging failed bucket, latency-only promotion.

- Fixture: `uv run --frozen pytest -q tests/challenges/medreason/test_attention_parity.py -k mod_14`
- Protected: `MEDFM_RUN_REAL_CHECKPOINTS=1 uv run --frozen pytest -q -m real_checkpoint tests/challenges/medreason/test_attention_parity.py -k protected_attention_parity`
- Registry: `uv run --frozen medfm models validate gemma-4-31b-it --format json`.

**Acceptance evidence:** Fixture artifacts prove comparator logic. Promotion requires non-skipped parity JSON for every exact candidate/bucket with all checks and measured latency/memory.

**Non-goals/failure policy:** Flash optional; no suppressed differences/post-hoc tolerance/cross-artifact hardware inference.

**Handoff:** Training/config/selection enable only backend named in exact parity artifact hash.
