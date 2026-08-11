# Schemas and package

This phase creates a CPU-only MedReason contract layer. Generic `MedicalSample`/manifest types cannot represent MedReason because they require patient/provenance fields, lack options/reasoning/grouping, and require an answer for VQA. Follow repository conventions in `medfm/core/sample.py` (frozen dataclasses, manual `to_dict`/`from_dict`), `medfm/core/errors.py` (typed failures), `medfm/core/serialization.py` (`canonical_json`, `config_hash`), and `medfm/data/manifests/` (integer versions, explicit migrations, fail-closed validation). Use the same frozen-dataclass and manual strict-validation convention at external JSON boundaries; do not introduce a second schema framework. Imports remain strict-mypy clean and CPU-safe.

Official compatibility is pinned to Docker commit `05748c0341b72dc08132bd108208b78dc14a2f0b`. Official result version 1.0 has top-level `name`, `type`, `answers`, `version`; answers contain `case_id`, normalized `task_type`, `answer`, `reasoning_trace`, and optional `confidence`/`metadata`. Fixture acceptance never proves protected data/model access, official judging, participant-validation completion, or 48/96 GB execution.

## SCH-01 Create isolated MedReason package and test skeleton

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with governance tasks that do not touch Python package files. Exclusively creates `medfm/challenges/__init__.py`, `medfm/challenges/medreason/__init__.py`, `tests/challenges/medreason/conftest.py`, and initial `tests/challenges/medreason/test_schemas.py`. SCH-09 owns final exports/packaging edits. Initializers must not import models, registries, CLI, Docker, Transformers, CUDA, or XLA.

**Target paths/symbols:** Empty explicit `__all__` in both initializers; synthetic fixture factories `mcq_case_payload`, `open_case_payload`, `released_example_payload`, `sha256_hex`; `TestPackageSkeleton` in the consolidated schema test.

**Inputs:** Agreed roots and published official two-case fields. Fixtures use generated temporary paths/hashes only—no protected records, labels, images, model files, credentials, or hardware observations.

**Outputs:** Importable namespaces and synthetic factories; no runtime artifact.

**Ordered implementation:**
1. Create mirrored source/test directories with future annotations.
2. Keep initializers free of eager imports and filesystem/environment side effects.
3. Build MCQ fixture with synthetic ID, `mcq`, relative image path, question, two unique `{label,text}` options, Unicode metadata.
4. Build open fixture with `open`, ordered image paths, no options, and nested unknown Unicode metadata.
5. Keep fixtures explicit/non-autouse and independent of inaccessible official assets.

**Focused tests and exact commands (do not run):** `TestPackageSkeleton` imports in a subprocess and asserts CUDA is uninitialized and `transformers`, `bitsandbytes`, `torch_xla` are absent; it rejects absolute paths, duplicate labels, and nonsynthetic IDs.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k PackageSkeleton`
- `uv run --frozen mypy medfm/challenges`

**Acceptance evidence:** Passing CPU import/fixture assertions and mypy output. This is fixture evidence. Protected tests require `MEDFM_RUN_REAL_CHECKPOINTS=1`, access, and matching hardware.

**Non-goals/failure policy:** Do not create implementation/config/registry/Docker files. CPU import failures block the phase and are never skipped for absent GPUs. No downloads.

**Handoff:** SCH-02–SCH-08 receive roots/fixtures; SCH-09 receives no accidental public API.

## SCH-02 Define normalized example option and task schemas

**Depends on:** SCH-01.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/schema.py` and `TestNormalizedSchema`. May run with SCH-03 if it stays in `manifests.py`; not with SCH-06/SCH-09.

**Target paths/symbols:** `MedReasonTaskType`, `MedReasonSplitRole`, `MedReasonOption`, `MedReasonExample`, manual codecs, typed errors. Do not extend generic `MedicalSample`.

**Inputs:** Exact normalized contract `MedReasonExample(case_id, task_type, image_paths, question, options, answer, reasoning_trace, metadata, group_id)`; canonical task values `mcq`/`open`. Raw official aliases belong to SCH-04.

**Outputs and concrete fields/invariants:**
- Strict enums: `MedReasonTaskType` exactly `MCQ="mcq"`, `OPEN="open"`; `MedReasonSplitRole` exactly `TRAIN="train"`, `DEV="dev"`, `LOCKBOX="lockbox"` for challenge-local split manifests (participant validation is a release/runtime role, never a selection split).
- Frozen `MedReasonOption(label: str, text: str)`.
- Frozen `MedReasonExample(case_id: str, task_type: MedReasonTaskType, image_paths: tuple[Path,...], question: str, options: tuple[MedReasonOption,...], answer: str|None, reasoning_trace: str|None, metadata: dict[str,JsonValue], group_id: str|None)`; SCH-06 finalizes `JsonValue` ownership.
- Manual deterministic dictionaries with exactly nine example keys and preserved order.

**Ordered implementation:**
1. Fully annotate immutable types; never coerce arbitrary values via `str()`.
2. Require non-whitespace ID/question/path/option fields and optional group, preserving original text without normalization.
3. Require image references but never stat/open/decode: released loaders fail closed later; runtime keeps broken references.
4. MCQ requires options with unique labels; open forbids options. Do not invent alphabet/count restrictions.
5. Present MCQ answer exactly matches a label; present open answer/trace is non-whitespace. Targets remain nullable for participant/runtime records.
6. Archive-role rules stay in DAT loaders; split roles do not become an extra normalized-example field.
7. Reject unknown normalized fields; SCH-06 captures unknown source fields into metadata.

**Focused tests and exact commands (do not run):** Round-trip MCQ/open, ordered images/options, null targets/group, and all three split roles. Reject unknown/case-changed enums, blank fields, duplicate labels, missing MCQ options, options on open, invalid answer, whitespace target. Prove a nonexistent `Path` is schema-valid while released policy rejects it.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k NormalizedSchema`
- `uv run --frozen mypy medfm/challenges/medreason/schema.py`

**Acceptance evidence:** Passing synthetic round trips/type checking and exact nine-key dictionaries. No decode/archive acceptance implied.

**Non-goals/failure policy:** Do not load, decode, group, or infer labels/tasks/splits. Malformed input raises typed schema errors; never generate missing targets.

**Handoff:** DAT constructs examples; SPL uses split enum/group/order; MCQ/OPEN consume exact labels/targets.

## SCH-03 Define released archive and provenance schemas

**Depends on:** SCH-01, GOV-08.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/manifests.py` and `TestReleaseManifests`. May run with SCH-02. Reuse but do not modify generic manifests. GOV-08 owns the generic immutable artifact contract.

**Target paths/symbols:** `MEDREASON_RELEASE_MANIFEST_VERSION`, `ReleaseRole`, `ExtractedFileRecord`, `CaseProvenanceRecord`, `ReleasedArchiveManifest`, validator and dictionary codec.

**Inputs:** Later governance source/license evidence, archive SHA-256/size, extracted inventory, DAT provenance. Released record keys are intentionally not modeled here because the protected training schema differs from runtime (`question type`, flat A–E) and must be inspected after access rather than guessed.

**Outputs and concrete fields:**
- `ExtractedFileRecord(relative_path: PurePosixPath, sha256: str, size_bytes: int)`.
- `CaseProvenanceRecord(case_id: str, source_relative_path: PurePosixPath, source_record_index: int, source_identifiers: dict[str,str])`, only for identifiers actually supplied.
- `ReleasedArchiveManifest(schema_version, dataset_name, dataset_version, role, source_page, license_id, archive_filename, archive_sha256, archive_size_bytes, extracted_files, case_provenance)` with exact typed fields.
- `ReleaseRole` only `TRAIN` and `PARTICIPANT_VALIDATION`; hidden/private split claims are impossible.

**Ordered implementation:**
1. Frozen dataclasses/manual codecs, independent integer version, explicit migrations, future-version rejection.
2. Require nonempty descriptive fields, lowercase 64-hex hashes, positive archive size, nonnegative file sizes/indexes, nonempty inventory.
3. Require relative POSIX paths without absolute/empty/`.`/`..`/backslash/NUL/control segments; DAT owns actual containment.
4. Reject duplicate extracted paths/case IDs and dangling provenance; canonicalize files by path and cases by ID while retaining indexes.
5. Validate supplied source identifiers without synthesizing identifiers from filenames or echoing rejected values.
6. Include no answers, traces, outputs, scores, inferred modality, or credentials; participant validation has nowhere to store leakage.
7. Released acceptance cannot pass until DAT-04 existence/decode/label checks pass; runtime does not use this policy.

**Focused tests and exact commands (do not run):** Round-trip synthetic train/participant manifests with Unicode. Reject future versions, bad hashes/sizes/indexes, traversal paths, duplicates/dangling records, and answer-like unknown fields. Missing/decode-failed released fixture blocks acceptance; equivalent runtime path remains representable.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k ReleaseManifests`
- `uv run --frozen mypy medfm/challenges/medreason/manifests.py`

**Acceptance evidence:** Canonical fixture/hash and sanitized failures. Real acceptance additionally needs GOV-02 and actual DAT-02 hashes; otherwise blocked/not-run.

**Non-goals/failure policy:** No download/extraction or guessed release keys/counts/licenses. Released duplicate IDs, malformed labels, missing/bad images, provenance defects, or target leakage fail closed; runtime tolerance never applies.

**Handoff:** DAT inspects/adapts actual released keys, writes inventories/provenance, and keys all audit/downstream artifacts by accepted hashes.

## SCH-04 Define runtime case and prediction schemas

**Depends on:** SCH-01, SCH-02, GOV-07.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/runtime_schema.py` and `TestRuntimeSchema` in `test_schemas.py`. May run with SCH-03/SCH-05. Do not vendor/modify official files (EVA-01/DOC-01) or implement prediction/fallback (RUN tasks).

**Target paths/symbols:** `TASK_TYPE_ALIASES`, `RuntimeOptionPayload`, `RuntimeCasePayload`, `RuntimeCasesPayload`, `SubmissionVersion`, `OfficialAnswerRecord`, `OfficialResultsPayload`, `normalize_runtime_task_type`, `validate_results_against_cases`. Strict frozen dataclass boundary codecs convert immediately to SCH-02 dataclasses.

**Inputs:** `/input/cases.json` as a raw list or `{"cases":[...]}`; official singular/plural image fields, aliases, dictionary/scalar options; ordered predictions. References may be absolute, uncontained, missing, duplicate, or corrupt as accepted by the pinned loader; DAT-07 later enforces containment, while missing/corrupt references must survive for fallback.

**Outputs and concrete fields/invariants:**
- `RuntimeOptionPayload(label: str, text: str)`.
- `RuntimeCasePayload(case_id: str, task_type: str, question: str, image_path: str|None, image_paths: list[str]|None, options: list[RuntimeOptionPayload]|list[str]|None, metadata: dict[str,JsonValue])`.
- Exact aliases: `mcq`, `closed`, `closed_ended`, `closed-ended`, `multiple_choice`, `multiple-choice` → `mcq`; `open`, `open_ended`, `open-ended`, `free_text`, `free-text` → `open`.
- `SubmissionVersion(major: Literal[1]=1, minor: Literal[0]=0)`.
- `OfficialAnswerRecord(case_id, task_type, answer, reasoning_trace="", confidence: float|None=None, metadata: dict[str,JsonValue]|None=None)`.
- `OfficialResultsPayload(name: str, type: Literal["Medical visual reasoning"], answers: list[OfficialAnswerRecord], version: SubmissionVersion)`.

**Ordered implementation:**
1. Reject null/numeric/boolean values where objects/lists/strings are required; disable implicit number-to-string coercion.
2. Accept both wrapper forms. Reject blank/duplicate case IDs, unknown task aliases, blank questions, malformed/no image fields, malformed options, and MCQ without options.
3. Preserve pinned ordering: singular `image_path` first, then `image_paths`; preserve duplicate path entries. Do not stat/read/decode. DAT-07 rejects escaping paths at the security boundary without discarding valid-but-broken in-root references.
4. Scalar option lists receive deterministic `A`, `B`, ... labels exactly as pinned `io.py`; dictionary options retain supplied labels/text. The pinned loader does not reject duplicate labels; the adapter must either preserve that fixture compatibility or, if DAT/RUN require uniqueness, reject explicitly before scoring and record the stricter policy—never silently deduplicate.
5. Preserve metadata/unknown fields through SCH-06 without shadowing known fields.
6. Cross-record results validation requires exact count, no duplicate/unknown/missing IDs, input order, task equality, nonempty answers, exact case-sensitive MCQ label membership, and nonempty open trace. Confidence, if accepted, is finite within `[0,1]`.
7. Emit fixed official version 1.0/type. Omit optional confidence/metadata only when absent.

**Focused tests and exact commands (do not run):** Cover raw/wrapper input, every alias, singular/plural/combined image order, duplicate/missing/absolute references, dictionary/scalar options, official MCQ/open output. Reject duplicate IDs, task mismatch, missing/extra/reordered predictions, wrong-case label, blank open fields, NaN/out-of-range confidence, wrong type/version. Cross-check preserved duplicate labels against the pinned fixture, and separately test any declared stricter scoring-boundary rejection.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k RuntimeSchema`
- `uv run --frozen mypy medfm/challenges/medreason/runtime_schema.py`
- After EVA-01 only: `uv run --frozen python docker/medreason/tools/validate_output.py tests/challenges/medreason/fixtures/results.json --input-json tests/challenges/medreason/fixtures/cases.json`

**Acceptance evidence:** Passing fixture/cross-record tests, exact version-1.0 payload, and proof broken references survive parsing. Only a later passing pinned official validator/two-case fixture supplies official-artifact compatibility; no 2,532-case or hardware claim follows.

**Non-goals/failure policy:** Do not predict, decode, score, repair, or perform Docker I/O. Structurally invalid input aborts. Structurally valid missing/corrupt images remain for RUN-07 case-local fallback. Released validation is never weakened.

**Handoff:** DAT-06/DAT-07 get lossless runtime cases; RUN-02/RUN-07 get normalized cases; RUN-15/DOC-07 get the exact result envelope and batch validator.

## SCH-05 Define experiment candidate and artifact records

**Depends on:** SCH-01, GOV-08.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/records.py` and `TestExperimentRecords`. May run with SCH-02–SCH-04. Do not modify `medfm/training/run_metadata.py`, `medfm/inference/bundle.py`, registries, or selection logic.

**Target paths/symbols:** `CandidateRoute`, `CandidateDecision`, `EvidenceScope`, `EvidenceStatus`, `ArtifactRecord`, `ExperimentCandidateRecord`, `AcceptanceEvidenceRecord`; mirror immutable `medfm/inference/bundle.py` records without replacing its bundle format.

**Inputs:** Governance-approved exact revisions/license decisions and later config, prompt, processor, quantization, data, split, adapter, and output hashes. Schemas never assert inaccessible artifacts exist.

**Outputs and concrete fields:**
- `ArtifactRecord(artifact_id: str, kind: str, relative_path: PurePosixPath, sha256: str, size_bytes: int, schema_version: int)`.
- `ExperimentCandidateRecord(candidate_id, parent_candidate_id, route, base_model_id, base_revision, adapter_artifact_ids, config_hash, prompt_hash, processor_hash, quantization_hash, data_manifest_hash, split_manifest_hash, seed, optional_component, official_eligible, ineligibility_reasons, decision, artifact_ids)`.
- Routes `mcq`, `open`, `system`; decisions `registered`, `evaluated`, `promoted`, `rejected`, `blocked`.
- `AcceptanceEvidenceRecord(evidence_id, candidate_id, scope, status, command, artifact_hashes, output_sha256, hardware_profile, blocker_codes)`; scopes `fixture`, `protected_artifact`, `hardware`; statuses `not_run`, `passed`, `failed`, `blocked`.

**Ordered implementation:**
1. Use frozen dataclasses/strict enums/manual codecs; require nonempty IDs/revisions, lowercase 64-hex hashes, nonnegative sizes/versions/seeds, relative paths, unique references.
2. `official_eligible=True` requires no ineligibility reasons; false requires stable reason codes. This records, not decides, eligibility.
3. Reject self-parenting, duplicate references, and promoted records without prior evaluated evidence in the set validator.
4. Keep config/input identity hashes distinct from output artifact hashes.
5. VRAM, latency, 48 GB, or 96 GB claims require `scope="hardware"` and a measured named profile. Fixture/protected evidence cannot satisfy them.
6. Passed evidence requires output digest/relevant artifacts; blocked requires blocker codes; not-run carries no success digest. Never prefill expected success.
7. Leave metrics/promotion math to EVA/SEL artifacts.

**Focused tests and exact commands (do not run):** Round-trip fixture candidate/evidence and stable hash under key reordering. Reject malformed hashes/paths, duplicate/self references, contradictory eligibility, passed-without-digest, blocked-without-blocker, and hardware claims labeled fixture. With real-checkpoint flag unset, validation performs no download/CUDA calls and protected/hardware evidence remains not-run/blocked.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k ExperimentRecords`
- `uv run --frozen mypy medfm/challenges/medreason/records.py`

**Acceptance evidence:** Canonical synthetic record digest and passing guards. The local RTX 3090/storage facts do not satisfy planned large-model hardware gates.

**Non-goals/failure policy:** Do not download, accept licenses, select systems, invent scores, or claim capacity. Invalid records fail closed; missing protected artifacts/hardware explicitly block dependent claims.

**Handoff:** MOD/CLI/EVA attach identities; SEL uses parentage/evidence; DOC/OPS publish only hashes backed by the correct evidence scope.

## SCH-06 Preserve Unicode context and unknown metadata

**Depends on:** SCH-02, SCH-03, SCH-04.

**Parallel safety and exclusive file ownership:** Exclusively owns new `medfm/challenges/medreason/metadata.py`, integration edits to `schema.py`, `manifests.py`, `runtime_schema.py`, and `TestMetadataPreservation`. Run after those file owners, not concurrently. May run with SCH-05.

**Target paths/symbols:** Recursive `JsonScalar`/`JsonValue`, `validate_json_value`, `clone_json_metadata`, `capture_unknown_metadata`. `medfm/data/textprep/unicode.py::normalize_unicode` is for derived comparison/tokenization text, never raw source metadata.

**Inputs:** Optional released `context`, unknown JSON keys, nested values, non-ASCII clinical text, combining marks, RTL text, supplementary code points.

**Outputs and concrete invariants:** Owned `dict[str,JsonValue]`, where values are only `None | bool | int | finite float | str | list[JsonValue] | dict[str,JsonValue]`. Optional top-level `context` and unknown source/runtime fields remain under original keys in example metadata because the approved normalized contract has no separate context field.

**Ordered implementation:**
1. Recursively validate/copy input so frozen records retain no caller-mutability alias. Reject bytes, sets, tuples pretending to be parsed JSON arrays, nonstring keys, NaN/infinity, unsupported objects.
2. Preserve raw strings exactly—no strip, case fold, NFC/NFKC, redaction, stringify, or in-memory ASCII escaping. Required domain fields keep separate nonempty rules.
3. Capture every unknown source/runtime key, including `context`, without overwriting known normalized keys; preserve nested spelling.
4. Use recursively immutable owned data or copy-on-read; every `to_dict` returns fresh JSON containers.
5. Normalized comparison/fingerprint values are separate derived fields; raw metadata remains unchanged.
6. Errors name only metadata path/type, never rejected values.

**Focused tests and exact commands (do not run):** Round-trip composed/decomposed accents, CJK, Arabic, emoji, and legal escaped controls; mutate source/serialized containers and prove stored data unchanged. Reject bytes/set/object/nonstring keys/NaN/infinity/collisions and assert leak canaries never enter errors. Prove `normalize_unicode` is not called implicitly.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k MetadataPreservation`
- `uv run --frozen mypy medfm/challenges/medreason/metadata.py medfm/challenges/medreason/schema.py medfm/challenges/medreason/runtime_schema.py`

**Acceptance evidence:** Unicode scalar equality/ownership fixtures and UTF-8 output. Actual released fields remain unknown until DAT ingestion.

**Non-goals/failure policy:** Do not infer modality/schema, extract PHI, or normalize raw context. Unsupported non-JSON values fail closed; metadata/rejected values are never logged.

**Handoff:** DAT preserves source context; SPL derives comparison keys without destroying raw values; SCH-08 receives JSON-safe owned metadata.

## SCH-07 Implement privacy-safe structured error taxonomy

**Depends on:** SCH-02, SCH-03, SCH-04, SCH-06.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/errors.py` and error-focused nodes in `tests/challenges/medreason/test_schemas.py`. It may run beside SCH-05 and DAT implementation only after those consumers agree to depend on this public error contract. No sibling may add a second challenge error enum or serialize raw third-party exceptions.

**Target paths/symbols:** `MedReasonErrorCode`, `MedReasonError`, `ReleasedDataError`, `RuntimeCaseError`, `ArtifactGateError`, `sanitize_error`, and `error_record`. Keep errors challenge-local; reuse the static-public-message pattern from `medfm/inference/errors.py` instead of widening generic data/inference APIs.

**Inputs:** Validation locations, stable error codes, case ID when policy permits it, component name, and an optional caught exception used only for type classification. Inputs may contain questions, options, answers, traces, metadata, prompts, image paths, credentials, or third-party decoder messages; none may cross the error boundary.

**Outputs:** A typed exception whose public string is static and whose deterministic record contains only `schema_version`, stable `code`, safe `component`, optional validated `case_id`, and retryability. Released-data errors remain fatal to the archive; recoverable runtime image/decode errors remain classifiable for RUN-07 per-case fallback; legal/artifact failures remain fail-closed.

**Ordered implementation:**
1. Define a closed error-code enum covering schema, duplicate ID/label, path containment, missing/decode-failed released image, preserved runtime image failure, target leakage, artifact hash/license/access, processor length, and structured-output failure classes.
2. Give every exception a static public message chosen by code; never call `str()`, `repr()`, or traceback formatting on an untrusted cause while constructing the public record.
3. Validate case IDs and component names against bounded printable rules before including them; omit an unsafe value rather than redact it heuristically.
4. Preserve the original exception only through Python exception chaining for local debugging; challenge logging receives `error_record()` only.
5. Make serialization total and deterministic for known errors and map unknown exceptions to one generic internal code without exposing their class module, message, arguments, or path.
6. Document which codes abort released audit, which permit runtime fallback, and which block artifact/model loading; callers may not reclassify them from free text.

**Focused tests and exact commands (do not run):** Plant unique canaries in every forbidden field and in nested exception messages/arguments, assert none occur in `str(error)`, dictionaries, JSON, captured logs, stdout, or stderr. Test every code's fatal/recoverable policy, unsafe IDs, unknown exceptions, deterministic key order, exception chaining, and equality of records across runs.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k PrivacySafeErrors`
- `uv run --frozen pytest tests/phase_17/test_inference.py -q -k error`
- `uv run --frozen mypy medfm/challenges/medreason/errors.py`

**Acceptance evidence:** Focused CPU results show the allowlisted record contains no planted content and every released/runtime/artifact code has exactly one declared policy. This proves error-boundary behavior only; the Docker stream scan and real decoder failures remain DOC-06/DOC-13 acceptance.

**Non-goals/failure policy:** Do not add telemetry transport, traceback persistence, regex-based PHI detection, retry orchestration, or diagnostic text fields. An unmapped released/artifact error fails closed. An unmapped per-case inference exception becomes the generic safe runtime class and may use only the preregistered schema-valid fallback.

**Handoff:** DAT receives fatal released-data codes, RUN receives recoverable case-local classes, EVA/SEL receive stable failure-class counters, and Docker receives the only error record allowed in runtime logs.

## SCH-08 Add deterministic JSON serialization round trips

**Depends on:** SCH-02, SCH-03, SCH-04, SCH-05, SCH-06, SCH-07.

**Parallel safety and exclusive file ownership:** Exclusively owns `medfm/challenges/medreason/serialization.py` and serialization-focused nodes in `tests/challenges/medreason/test_schemas.py`. Run after schema owners finish; do not concurrently edit their codecs. It wraps `medfm.core.serialization::{canonical_json,config_hash}` rather than introducing a second canonicalization algorithm.

**Target paths/symbols:** `MEDREASON_SERIALIZATION_VERSION`, `to_canonical_bytes`, `content_sha256`, `read_typed_json`, `write_typed_json_create_once`, and type-dispatch adapters for SCH records. JSON writing uses UTF-8, a final newline for files, finite numbers only, and atomic create-once semantics for immutable manifests.

**Inputs:** Any supported SCH dataclass or validated external payload, plus an expected schema version/type at read time. Paths remain challenge-local `Path`/`PurePosixPath` fields and serialize as normalized POSIX strings; unknown arbitrary objects are rejected.

**Outputs:** Canonical UTF-8 bytes, lowercase SHA-256, and a freshly owned typed object after read. Dictionary insertion order, process hash seed, locale, source container aliases, and escaped-versus-literal Unicode representations must not change the content digest.

**Ordered implementation:**
1. Convert only explicitly registered schema types through their manual `to_dict`; reject duck-typed mappings at typed entry points.
2. Recursively validate JSON values, convert declared paths to POSIX strings, reject NaN/infinity and unsupported values, and delegate ordering/encoding to the existing canonical serializer.
3. Compute hashes from canonical bytes, never from pretty-printed output, Python `repr`, file metadata, or absolute paths.
4. Read bytes with strict UTF-8 and duplicate-key detection, require the expected integer schema version and record kind before constructing a type, and reject trailing data.
5. Write to a same-directory temporary file, flush and atomically create the destination without overwrite. If the destination exists, accept only an explicit verify-existing operation whose bytes/hash already match; never silently replace a frozen artifact.
6. Keep submission pretty-printing separate: official `results.json` may use deterministic indentation, but artifact identity always derives from the documented canonical representation.

**Focused tests and exact commands (do not run):** Round-trip every SCH type with Unicode/nested metadata and compare object equality plus canonical hash. Reorder input dictionaries, vary locale/hash seed, mutate source/output containers, and assert identical bytes. Reject duplicate JSON keys, wrong/future versions, trailing objects, invalid UTF-8, NaN/infinity, absolute/traversal artifact paths, unsupported types, and overwrite/hash mismatch.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k DeterministicSerialization`
- `uv run --frozen pytest tests/phase_02/test_metadata_roundtrip.py -q`
- `uv run --frozen mypy medfm/challenges/medreason/serialization.py`

**Acceptance evidence:** Checked-in synthetic expected digests and focused tests prove deterministic round trips and no-clobber behavior on CPU. Real archive/model/adapter hashes are produced only by their owning tasks and must not be prefilled.

**Non-goals/failure policy:** Do not pickle, serialize tensors/model weights, pretty-print secrets, migrate unknown future versions, or make output ordering depend on filesystem traversal. A duplicate key, unknown kind/version, noncanonical path, or existing-byte mismatch fails closed.

**Handoff:** GOV/DAT/MOD/EVA/SEL/DOC use the canonical byte/hash contract for manifests and receipts; runtime output uses SCH-04's official envelope while recording its independent SHA-256.

## SCH-09 Export only intentional challenge package symbols

**Depends on:** SCH-01, SCH-02, SCH-03, SCH-04, SCH-05, SCH-06, SCH-07, SCH-08.

**Parallel safety and exclusive file ownership:** Exclusively owns final edits to `medfm/challenges/__init__.py`, `medfm/challenges/medreason/__init__.py`, the `challenges` entry in `tests/phase_01/test_packaging.py::SUBPACKAGES`, and export/import nodes in `tests/challenges/medreason/test_schemas.py`. Serialize with any later task touching these initializers or packaging inventory; downstream modules import owned submodules directly until this card lands.

**Target paths/symbols:** Public exports are the stable task/split enums, option/example/runtime prediction types, manifest/record types, public error base and code enum, and canonical serialization/hash functions. Keep validators/builders/loaders/trainers/judges, model classes, mutable registries, test fixtures, private helpers, and optional dependencies out of package-root exports.

**Inputs:** The finalized symbol tables from SCH-02–SCH-08 and repository packaging conventions. No model/data artifact is needed.

**Outputs:** Explicit sorted `__all__` tuples, an importable wheel-visible `medfm.challenges.medreason` namespace, and packaging inventory coverage proving CPU import has no accelerator, network, filesystem, environment, or registry side effects.

**Ordered implementation:**
1. Use LSP references on every candidate public symbol and resolve naming collisions before changing imports; do not add compatibility aliases or duplicate re-exports.
2. Import only CPU-safe schema/error/serialization modules at the package root and list each intentional name exactly once in `__all__`.
3. Keep `medfm.challenges.__init__` minimal; expose the `medreason` namespace without eagerly importing challenge implementation modules.
4. Add `challenges` to the phase-01 subpackage inventory and assert it is present in the built wheel/package discovery.
5. Exercise imports in a clean subprocess with offline variables and poisoned optional-module imports; assert no CUDA/XLA initialization, Hub/network call, artifact lookup, warning, stdout, or stderr.
6. Fail tests on any accidental extra root attribute that is first-party public API; consumers of nonpublic implementation details must import their owning submodule explicitly.

**Focused tests and exact commands (do not run):** Assert the exact stable root API by importing and using each exported contract in a small round trip, not by scraping source text. Poison `transformers`, `bitsandbytes`, `torch_xla`, sockets, and artifact paths; test a wheel/import inventory and clean captured streams.
- `uv run --frozen pytest tests/challenges/medreason/test_schemas.py -q -k PublicExports`
- `uv run --frozen pytest tests/phase_01/test_packaging.py -q`
- `uv run --frozen mypy medfm/challenges`

**Acceptance evidence:** Focused CPU/package results prove all intentional symbols are importable from the installed namespace and no forbidden optional/runtime work occurs. This is packaging-contract evidence, not a model/runtime/hardware claim.

**Non-goals/failure policy:** No convenience aliases, wildcard imports, eager plugin discovery, version fallback, model registration, or backwards-compatibility shim. An accidental eager dependency or unowned public symbol blocks the phase.

**Handoff:** Every implementation task consumes one stable schema/error/serialization surface; CLI and Docker can import the challenge package on CPU before validating configs and artifact prerequisites.
