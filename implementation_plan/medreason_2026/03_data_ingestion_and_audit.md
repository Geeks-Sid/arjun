# Data ingestion and audit

This phase converts an approved local package into challenge-local `MedReasonExample` records and immutable audit artifacts. MedReason must not be forced through generic `MedicalSample` or Phase-03 manifest schemas: those require fields the challenge may not release. Reuse their canonical JSON, SHA-256, reader, containment, privacy-safe error, and exact-checksum patterns, but not their schema or non-transitive split implementation.

Commands below are implementation acceptance commands; none were run while writing this backlog. Synthetic fixtures establish code behavior only. Protected Synapse data, their actual train schema/counts, and full-storage acceptance remain external gates.

**Shared fixtures.** `tests/challenges/medreason/conftest.py` supplies `valid_train_archive`, `valid_participant_archive`, `valid_runtime_root`, and parameterized `adversarial_archive`, generated in `tmp_path` from small non-clinical JSON and PNG/JPEG sources under `tests/challenges/medreason/fixtures/data/`. Valid fixtures cover MCQ/open, multiple ordered images, optional context, composed/decomposed Unicode, CJK/emoji, and unknown nested metadata. Adversarial cases cover archive traversal/types/bombs/corruption, malformed JSON, duplicate IDs, invalid labels, answer leakage, missing/corrupt/oversized images, identifier ambiguities, and immutable conflicts. Unsafe archives are never extracted into the repository.

**Immutable layout.** Source packages live at `artifacts/data/medreason/source/<archive_sha256>/archive.<suffix>`, extracted files at `artifacts/data/medreason/extracted/<archive_sha256>/payload/`, and final audits at `artifacts/data/medreason/audits/<archive_sha256>/<audit_contract_sha256>/`. Stage in a same-filesystem sibling, hash and fsync, then atomically rename. Existing destinations are accepted only after exact checksum verification; mismatches fail rather than overwrite. `.gitignore` is not immutability.

**Unicode/privacy.** Preserve original strings, key spelling, normalization form, and unknown metadata. Unicode normalization may produce explicitly named comparison keys only; it never replaces raw data. Logs contain case IDs and sanitized error classes only—never questions, options, answers, traces, metadata, image paths, or raw identifiers.

## DAT-01 Implement secure immutable archive extraction

**Depends on:** GOV-02, GOV-08, SCH-03, SCH-07.

**Parallel safety and exclusive file ownership:** May run with runtime-only DAT-06/DAT-07. Exclusively owns extraction symbols in `medfm/challenges/medreason/data.py` and `tests/challenges/medreason/test_data_extraction.py`; DAT-02 consumes the ledger without rewriting this policy.

**Target paths/symbols:** `ArchiveLimits`, `ArchiveMember`, `inspect_archive`, `extract_archive_immutable`, `_safe_archive_member_path`, `_stream_archive_member`. Reuse containment from `medfm.data.manifests.schema.validate_uri` and no-symlink/root checks from `medfm.inference.bundle.ModelBundle.path`.

**Inputs:** Read-only local `.zip`, `.tar`, `.tar.gz`, or `.tgz`; registered SCH-03 descriptor; trusted output root; explicit member-count, individual/aggregate bytes, path-length, nesting, and compression-ratio limits. No URL/credential input.

**Outputs:** Sealed `extracted/<archive_sha256>/payload/` plus ordered member ledger; never a partial publication.

**Ordered implementation:**

1. Validate actual archive format; reject encrypted/unsupported input before staging.
2. Reject empty/NUL/control names, absolute/drive/UNC paths, lexical/resolved `..`, and names colliding after separator, platform-case, or Unicode comparison normalization. Preserve original safe spelling.
3. Reject symlinks, hardlinks, sockets, devices, FIFOs, sparse and all non-regular/non-directory members. Nested archives are opaque unless a registered schema explicitly permits one bounded level.
4. Enforce declared and streamed limits. Never use `extractall`; create exclusive no-follow files with normalized non-executable permissions.
5. Stream bounded chunks, verify CRC/parser integrity and declared-size agreement, fsync, atomically publish by archive digest, and exact-verify an existing destination.
6. On failure remove task-owned staging only. Errors contain member ordinal/code, not names or parser text.

**Focused tests:** Use both valid archives and every path/type/bomb/corruption `adversarial_archive`. Assert raw Unicode names/content survive, no write escapes `tmp_path`, identical extraction is idempotent, and traversal, symlink/hardlink, normalized collision, CRC, forged size, limit, encryption, and immutable-conflict cases fail atomically. Plausible bugs: prefix containment, Windows paths accepted on Linux, partial extraction, or trust in declared size.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_data_extraction.py -k dat_01`.
- CLI acceptance: audit `<valid_train_archive>` succeeds; replacing it with any adversarial fixture exits nonzero with sanitized stderr and no audit directory.

**Acceptance evidence:** Passing format/security matrix, expected source/extracted descendants only, and a safe machine-readable ledger of counts/limits.

**External-data blocker:** Fixtures need no Synapse access. Actual package-format acceptance waits for OPS-02/OPS-03; synthetic success is not official-data evidence.

**Non-goals and failure policy:** No download, credentials, recursive discovery, quarantine, permissive fallback, or partial recovery. One unsafe/corrupt member rejects the package.

**Handoff:** DAT-02 receives archive digest, sealed root, original-plus-comparison member ledger, limits/version, and error taxonomy.

## DAT-02 Hash source archives and extracted file manifests

**Depends on:** DAT-01, GOV-08, SCH-03.

**Parallel safety and exclusive file ownership:** May overlap adapter design but must finish before real file reads. Exclusively owns hashing symbols and `tests/challenges/medreason/test_data_hashing.py`.

**Target paths/symbols:** `hash_file_stream`, `build_extracted_file_manifest`, `verify_extracted_file_manifest`, `canonical_manifest_sha256` in `data.py`. Reuse `canonical_json`, `manifest_content_hash` ordering, and bundle checksum exact coverage.

**Inputs:** Original archive, DAT-01 ledger/payload, and supplied source-page/package/license evidence with explicit verification status.

**Outputs:** Hash-addressed source copy, `source_manifest.json`, and `extracted_files.json`; each file row has preserved relative path, comparison key, byte count, SHA-256, ordinal, and only explicitly mapped media role.

**Ordered implementation:**

1. Hash/count original bytes, stream-copy while independently rehashing, and compare before publication.
2. Hash only ledger entries; reject missing, extra, non-regular, or symlink-replaced files.
3. Canonically sort comparison keys while retaining raw paths. Exclude mtimes, absolute paths, host/user/process data.
4. Record archive/extraction versions and evidence states (`verified`, `unverified`, `not_supplied`) without inventing facts.
5. Exact-verify staging and existing destinations; require verification before adapters open payloads.

**Focused tests:** Valid archives prove cross-copy/process determinism. Adversaries cover post-extraction mutation, added/missing file, symlink swap, same-size edit, comparison collision, malformed digest, and stale destination. Assert iteration/timestamps do not affect identity and unknown provenance stays unverified.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_data_hashing.py -k dat_02`.
- CLI acceptance: two audits of each valid archive emit identical archive/extracted hashes; tampered/adversarial inputs fail before record parsing.

**Acceptance evidence:** Exact coverage passes, deterministic digests match, and all tampering is detected pre-adaptation.

**External-data blocker:** Real hashes/evidence require OPS-02/OPS-03. Fixture digests must not enter experiment provenance.

**Non-goals and failure policy:** Do not trust ETags or CRC as SHA-256, infer provenance, or partially verify. One mismatch invalidates the artifact.

**Handoff:** DAT-03/DAT-05/DAT-11 consume verified roots, hashes, ordered file rows, and evidence states.

## DAT-03 Adapt released training records into normalized examples

**Depends on:** SCH-02, SCH-03, SCH-06, DAT-02.

**Parallel safety and exclusive file ownership:** May overlap runtime/image design. Exclusively owns released-train adapters and DAT-03 tests; DAT-04 adds separate validators.

**Target paths/symbols:** `ReleasedDataKind`, `ReleasedArchiveReader`, `adapt_training_record`, `load_training_examples`, `_preserve_unknown_metadata` in `data.py`; `tests/challenges/medreason/test_released_data.py`. Use SCH-02 `MedReasonExample`, not generic manifests/`MedicalSample`.

**Inputs:** Verified payload and a registered, versioned SCH-03 mapping for record files, aliases, task tags, image references, and context. Train answer/reasoning/metadata keys and question-type values are unpublished: unknown mappings must fail, never be guessed.

**Outputs:** Ordered `MedReasonExample(case_id, task_type, image_paths, question, options, answer, reasoning_trace, metadata, group_id=None)` plus safe counts.

**Ordered implementation:**

1. Verify exact files, then open only registered UTF-8 JSON/JSONL. Reject invalid encoding, duplicate keys, non-object rows, trailing corruption, and non-finite numbers.
2. Dispatch only registered schema version/task mappings; retain source ordinal and supplied image/option order.
3. Map required fields without ID/label coercion or task inference.
4. Preserve all unmapped raw keys/values recursively. Use separate normalized comparison keys only for ambiguity detection.
5. Preserve supplied identifier/modality metadata without guessing; leave `group_id=None` because generic splitting is unsuitable.

**Focused tests:** Valid train covers MCQ/open, ordered multi-image, context, optional trace, unknown nested metadata, and exact Unicode forms. Participant-under-train and adversarial duplicate key/ID, invalid UTF-8, unknown schema/task, comparison collision, undeclared record, and path-like metadata fail. Assert exact order/raw Unicode and no undeclared read.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_released_data.py -k dat_03`.
- CLI acceptance: registered valid train succeeds deterministically; participant under train or unknown/adversarial schema exits nonzero without sealed audit.

**Acceptance evidence:** Exact typed round trips with `group_id=None`; unknown train keys/task mappings fail closed.

**External-data blocker:** The unpublished train mapping cannot be finalized or claimed compatible before OPS-03 and a reviewed SCH-03 registration.

**Non-goals and failure policy:** No schema heuristics, pseudo-labels/traces, source lookup, splitting, decode, or generic-schema coercion. Unknown mapping is fatal.

**Handoff:** DAT-04 gets typed rows/ordinals; DAT-08 ordered images; DAT-10 raw metadata and explicit mappings.

## DAT-04 Reject malformed released labels answers and images

**Depends on:** DAT-03, SCH-02, SCH-07.

**Parallel safety and exclusive file ownership:** May overlap DAT-05 after interfaces freeze. Exclusively owns released validators and DAT-04 cases; it cannot weaken DAT-06 runtime behavior.

**Target paths/symbols:** `validate_released_example`, `validate_training_examples`, `preflight_released_images` in `data.py`; DAT-04 cases in `test_released_data.py`. Reuse aggregate fail-closed validation, `DataError` privacy, and strict `resolve_local_path` where compatible.

**Inputs:** Typed train examples, verified root, SCH-02 task/option invariants, registered image policy.

**Outputs:** Entire unchanged validated set or bounded deterministic case-ID/error aggregate; no partial subset.

**Ordered implementation:**

1. Reject empty/duplicate IDs/questions and invalid/unsafe image lists.
2. MCQ requires configured count, non-empty unique supplied labels/text, and answer exactly one supplied label with no trim/case repair.
3. Open requires non-empty answer; validate only an actually released optional trace.
4. Require every released image to be covered, in-root, non-symlink, regular, and decodable through DAT-08. Missing/corrupt/truncated/oversized/unsupported/mismatch rejects the whole set.
5. Report case IDs/error codes only. Keep released strict and runtime tolerant signatures distinct.

**Focused tests:** Both valid archives exercise strict image success. Adversaries cover duplicate IDs, empty fields, missing answers, answer outside labels, duplicate/empty labels, bad option count, traversal/symlink/missing/corrupt/truncated/unsupported/oversized images. Assert all-or-nothing failure and forbidden strings absent from logs/stderr.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_released_data.py -k dat_04`.
- CLI acceptance: valid train exits 0; every malformed released archive exits nonzero and publishes nothing.

**Acceptance evidence:** Exact label/answer/image matrix, every-image decode, atomic failure, sanitized diagnostics.

**External-data blocker:** Official label/image conventions remain unverified until OPS-03/OPS-07; observed additions require schema/tests, not leniency.

**Non-goals and failure policy:** No correction, remapping, row quarantine, image fallback, or text-only substitution. One malformed released case blocks use.

**Handoff:** DAT-08 receives complete strict image references; training/splits receive only all-or-nothing data and policy hash.

## DAT-05 Adapt participant validation records without answer leakage

**Depends on:** SCH-02, SCH-03, SCH-06, SCH-07, DAT-02, DAT-03, DAT-04.

**Parallel safety and exclusive file ownership:** May overlap runtime/audit. Exclusively owns participant adapters and `tests/challenges/medreason/test_participant_data.py`; it reuses strict released-image validation.

**Target paths/symbols:** `adapt_participant_record`, `load_participant_examples`, `assert_participant_unlabeled` in `data.py`.

**Inputs:** Verified participant package and registered mapping. The currently known participant projection retains exactly `case_id`, `question type`, `image_path`, `question`, and flat `A`–`E`; encode that only in the reviewed schema registration, not as a heuristic. Prediction/output files are never inputs.

**Outputs:** Strict image-valid examples with `answer=None`, `reasoning_trace=None`, and immutable kind `participant_validation`.

**Ordered implementation:**

1. Require explicit participant mode; never infer from absent answers or filenames.
2. Adapt known registered fields, preserving `image_path` order/shape as defined by the registration and raw Unicode. Unknown question-type values fail closed.
3. Recursively compare keys against registered/reserved answer/reference/rationale/reasoning aliases. Any such key—even null/empty/nested—rejects the archive rather than being dropped.
4. Force answer/trace null and forbid joins with results, caches, organizer outputs, or prior submissions.
5. Apply duplicate/flat A–E option/path/strict-image checks and provenance marking that training consumers reject.

**Focused tests:** Valid participant includes both tasks, flat A–E, and multilingual unknown-safe content; valid train under participant mode fails. Adversaries include unknown question type, missing/duplicate option fields, null/empty/nested answer aliases, answer sidecars, predictions files, duplicate IDs, and broken images. Assert all normalized answers/traces null and raw Unicode preserved.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_participant_data.py -k dat_05`.
- CLI acceptance: valid participant reports zero answer fields; train/leakage/unknown-type/broken-image adversaries exit nonzero with no artifact.

**Acceptance evidence:** No answer/trace survives, every leakage variant fails, and participant provenance is rejected by a fake trainer.

**External-data blocker:** Although current participant projection is known, its protected package must still be verified under OPS-03. Unpublished train fields must not be inferred from it.

**Non-goals and failure policy:** No tuning, pseudo-labels, output ingestion, answer recovery, hidden-data analysis, or participant-to-train conversion. Suspected leakage blocks use.

**Handoff:** Runtime/Docker may use participant shape for execution only; selection may not consume outputs.

## DAT-06 Preserve runtime cases with broken image references

**Depends on:** SCH-02, SCH-04, SCH-06, SCH-07.

**Parallel safety and exclusive file ownership:** May overlap released work. Exclusively owns runtime loading and `tests/challenges/medreason/test_runtime_data.py`; DAT-07 owns containment and RUN-07 fallback content.

**Target paths/symbols:** `RuntimeCaseLoad`, `RuntimeImageState`, `adapt_runtime_case`, `load_runtime_cases` in `data.py`. Wrap SCH-04's pinned official parser; do not route through released/generic readers.

**Inputs:** `/input/cases.json`-style root. Mapping comes only from official commit `05748c0341b72dc08132bd108208b78dc14a2f0b`. The official parser does not decode or check referenced images.

**Outputs:** One ordered normalized case per valid row with ordered references and state (`unopened`, `missing`, `not_regular`, `decode_failed` when probed). Safe missing/corrupt references remain for RUN-07 fallback.

**Ordered implementation:**

1. Parse through SCH-04, preserve order, validate IDs/task/question/options; malformed non-image fields/duplicate IDs fail globally before model allocation.
2. Use DAT-07 for security containment. Unsafe path escape/symlink is fatal; a safe in-root missing/corrupt path is preserved, not rejected.
3. Do not bulk-decode. Decode only inside each `predict_case`; a decode failure affects that case, never good siblings.
4. Never drop/reorder cases or attach reference answers. Preserve raw metadata in memory and exclude it from logs.
5. Keep released strict and runtime tolerant entry points type-distinct.

**Focused tests:** `valid_runtime_root` mixes good, missing, corrupt, and multi-image cases. Valid released archives regress strictness; adversaries cover traversal/symlink, duplicate IDs, bad options, malformed JSON, and mixed rows. Assert count/order preserved, good fake predictions run, safe broken cases reach fallback, unsafe/non-image failures stay global.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_runtime_data.py -k dat_06`.
- CLI acceptance: mixed runtime root exits 0 with safe broken counts/no paths; equivalent broken released archives exit nonzero.

**Acceptance evidence:** Mixed fixture preserves one case per input ID/order with case-local image failure and sanitized logs.

**External-data blocker:** Exact runtime compatibility depends on vendored pinned sources and full participant execution, not train-schema inference.

**Non-goals and failure policy:** No fallback answer generation here, skipping malformed rows, archive extraction, or released leniency. Only safe missing/corrupt references are tolerated.

**Handoff:** RUN-02/RUN-07 get ordered cases, safe references, states, and error classes.

## DAT-07 Enforce input-root image path containment

**Depends on:** SCH-04, SCH-07.

**Parallel safety and exclusive file ownership:** May overlap all released work. Exclusively owns containment symbols/DAT-07 tests; DAT-06 calls its interface and DAT-01 retains archive policy.

**Target paths/symbols:** `ContainedImageReference`, `resolve_input_image_path` in `data.py`; DAT-07 cases in `test_runtime_data.py`. Reuse `Path.resolve`/`is_relative_to`, bundle symlink rejection, and sanitized security errors.

**Inputs:** Real input root plus one official-parser reference. URLs are rejected unless the pinned contract explicitly requires them.

**Outputs:** Preserved relative reference, comparison key, resolved in-root path, and existence/type state; never an out-of-root path.

**Ordered implementation:**

1. Require non-symlink real root. Reject empty/control, URL, absolute/drive/UNC, and lexical `..`.
2. Resolve existing components; require root containment and reject parent/final/dangling symlinks.
3. Recheck with no-follow stat/open immediately before decode.
4. Classify safe missing/corrupt as recoverable for DAT-06. Traversal/symlink is security failure, but containment must not abort merely because an in-root file is missing/corrupt.
5. Return code only; caller adds case ID. Never echo path.

**Focused tests:** Valid nested/missing runtime references plus released regressions; adversaries cover traversal, absolute/drive/UNC/URL, lookalikes, symlink parent/final/root/dangling, prefix containment, and replacement between validation/open. Assert safe missing is retained and all escapes rejected.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_runtime_data.py -k dat_07`.
- CLI acceptance: valid runtime including safe missing exits 0; containment adversaries exit nonzero before decode/publication; released path adversaries fail.

**Acceptance evidence:** Path/TOCTOU matrix passes and diagnostics contain no supplied path.

**External-data blocker:** Organizer-specific reference semantics await SCH-04 observation; unsupported forms fail rather than being guessed.

**Non-goals and failure policy:** No remote fetch, URI cleanup, symlink following, basename search, or discovery fallback. Unsafe is fatal; safe broken is preserved.

**Handoff:** Runtime receives one reviewed path object and explicit safe-broken versus unsafe distinction.

## DAT-08 Decode and inventory every released image

**Depends on:** DAT-04, DAT-05, DAT-07.

**Parallel safety and exclusive file ownership:** May overlap DAT-09/DAT-10 after freezing `ImageAuditRow`. Exclusively owns decoder/inventory and `tests/challenges/medreason/test_image_audit.py`; DAT-09 owns hashes.

**Target paths/symbols:** `ImageAuditRow`, `ReleasedImageAuditor`, `decode_released_image`, `inventory_released_images` in new `audit.py`. Reuse `Reader`, `PayloadRead`, canonical tensor checks, and `PngJpegReader` (HWC uint8 grayscale/RGB, EXIF stripped); add only registered observed formats.

**Inputs:** Strict train/participant examples, verified root/files, format allowlist, and optional supplied modality. Runtime excluded.

**Outputs:** One row per `(case_id,image_index)` with ID/index, preserved reference, encoded SHA-256/bytes, decoder/version, declared format, width/height/channels, decoded shape/dtype/mode, supplied modality or null, decode status, and DAT-09 hash slots.

**Ordered implementation:**

1. Traverse deterministic case-ID/image-index order while retaining within-case order; require manifest coverage/containment.
2. Decode each distinct encoded SHA-256 once and fan out values without collapsing references.
3. Enforce decoded-pixel bounds; reject animation/multiframe, unsupported mode/format, metadata ambiguity, and bombs unless explicit canonical policy exists.
4. Derive geometry/mode from pixels; hard-fail supplied mismatch. Never infer modality.
5. One released decode failure blocks audit; runtime remains case-local.
6. Keep CPU-only: no CUDA calls or top-level `bitsandbytes`, `torch_xla`, `flash_attn`, `cucim` imports.

**Focused tests:** Valid archives cover grayscale/RGB, repeated bytes, ordered multi-image, supplied/absent modality. Adversaries cover missing/truncated/fake suffix/animation/oversize/unsupported/mismatch/EXIF/symlink. Assert exact coverage, one decode per digest, no modality inference, metadata stripping, CPU-only behavior, and atomic failure.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_image_audit.py -k dat_08`.
- CLI acceptance: valid audits produce deterministic rows/counts; image adversaries fail; runtime broken images remain accepted in runtime mode.

**Acceptance evidence:** Row count equals references, decoder counts prove reuse, structures match pixels, and CUDA remains uninitialized.

**External-data blocker:** Real formats/counts/cost await OPS-03/OPS-07. New codec support is explicit blocked work, not fallback.

**Non-goals and failure policy:** No resize, augmentation, OCR, modality inference, training tensors, or runtime eager decode. Unsupported released data is fatal.

**Handoff:** DAT-09 receives canonical pixels/structure; SPL/STR only supplied modality.

## DAT-09 Compute decoded-pixel and perceptual image hashes

**Depends on:** DAT-08, GOV-08.

**Parallel safety and exclusive file ownership:** May overlap DAT-10. Exclusively owns hash algorithms and DAT-09 image-audit tests; DAT-11 serializes only.

**Target paths/symbols:** `decoded_pixel_sha256`, `perceptual_hash64`, `PIXEL_HASH_VERSION`, `PERCEPTUAL_HASH_VERSION`, `attach_image_hashes` in `audit.py`.

**Inputs:** Canonical contiguous decoded pixels plus shape/dtype/mode.

**Outputs:** Lowercase 64-hex decoded-pixel SHA-256 and fixed-width lowercase pHash, with versions.

**Ordered implementation:**

1. Define `pixel-sha256-v1` over length-framed canonical header (version, shape, dtype, mode) and explicit-endian C-order bytes.
2. Pin a deterministic 64-bit DCT pHash: grayscale coefficients, alpha policy, resize, DCT cells, median/tie, leading zeros.
3. Compute during DAT-08's single decode; never reopen/differently transform duplicates.
4. Commit exact golden strings for uniform, impulse, gradient, RGB, and one-pixel perturbation.
5. Leave pHash distance/grouping to SPL-03.

**Focused tests:** Valid archives include identical pixels with different encoding and near duplicates; adversaries include corrupt/unsupported images and noncanonical layout/dtype. Assert alternate encodings share decoded hash, structural changes differ, exact golden pHashes are process-stable, and failures get no hashes.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_image_audit.py -k dat_09`.
- CLI acceptance: repeat valid audits emit identical image-audit digest/versions; adversaries fail instead of omitting hashes.

**Acceptance evidence:** Golden vectors, cross-process determinism, alternate-encoding equivalence, and perturbation behavior pass.

**External-data blocker:** Algorithms are fixture-acceptable; released duplicate prevalence/cost needs OPS-03/OPS-07 evidence.

**Non-goals and failure policy:** No threshold, grouping, deletion, embedding, or label action. Noncanonical input fails.

**Handoff:** SPL-03 gets exact hashes, pHashes, positions, versions; DAT-11 gets ready rows.

## DAT-10 Normalize source study patient article identifiers

**Depends on:** SCH-03, SCH-06, SCH-07, DAT-03, DAT-05.

**Parallel safety and exclusive file ownership:** May overlap image audit. Exclusively owns identifier normalization and `tests/challenges/medreason/test_identifier_audit.py`; grouping consumes outputs without reinterpreting raw metadata.

**Target paths/symbols:** `IdentifierKind`, `IdentifierLink`, `identifier_comparison_key`, `extract_identifier_links`, `IDENTIFIER_NORMALIZATION_VERSION` in `audit.py`. Reuse `hash_identifier` and Phase-03 no-echo privacy tests.

**Inputs:** Untouched metadata and explicit registered mappings to source/study/patient/article plus namespace. No mapping means no extraction.

**Outputs:** Sorted links containing case ID, kind, namespace, hashed value, comparison version, and field-locator hash—never raw identifier. Raw metadata remains unchanged; `group_id=None`.

**Ordered implementation:**

1. Extract only mapped fields and declared scalar/list forms. Reject ambiguous forms/comparison collisions; never infer from names/values/files/questions.
2. Preserve raw value; separately create documented Unicode/whitespace comparison key. Preserve case unless registration says otherwise; no ASCII folding/transliteration/heuristic numeric parsing.
3. Hash length-framed version/kind/namespace/comparison key. Validate/tag already hashed inputs instead of blind double-hash.
4. Scan artifacts/errors/logs/CLI to ensure raw values never leave protected example metadata.
5. Report mapped/absent/malformed/unclassified counts; do not claim patient-disjointness without evidence.

**Focused tests:** Valid archives cover all kinds, repeats/absence, composed/decomposed comparison equivalents, case distinctions, unrelated unknown metadata. Adversaries cover malformed clinical-looking values, collisions, wrong containers, duplicate namespaces, lookalike unmapped keys. Assert deterministic links, exact raw metadata, and no raw value elsewhere.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_identifier_audit.py -k dat_10`.
- CLI acceptance: valid audits print only hashed counts/versions; adversaries fail without echoing identifiers.

**Acceptance evidence:** Golden hashes, raw-metadata equality, privacy scans, and explicit no-evidence counts pass.

**External-data blocker:** Actual mappings/coverage await OPS-03. Missing authoritative fields must be reported, not fabricated.

**Non-goals and failure policy:** No grouping, disjointness claim, reverse lookup, enrichment, export, or filename grouping. Ambiguous mappings fail.

**Handoff:** SPL-02/SPL-06 receive hashed namespace links/coverage; DAT-11 receives versioned links.

## DAT-11 Write immutable audit and provenance artifacts

**Depends on:** GOV-08, SCH-03, SCH-05, SCH-06, DAT-02, DAT-04, DAT-05, DAT-08, DAT-09, DAT-10.

**Parallel safety and exclusive file ownership:** Runs after released producers; may overlap runtime-only work. Exclusively owns audit publication and `tests/challenges/medreason/test_data_artifacts.py`; DAT-12 only orchestrates.

**Target paths/symbols:** `MedReasonAuditManifest`, `build_audit_artifacts`, `write_audit_artifacts_immutable`, `verify_audit_artifacts` in `audit.py`. Reuse canonical serialization and bundle exact-checksum/safe-path patterns, not generic manifest schema.

**Inputs:** Source/extracted manifests, validated examples of one kind, image rows/hashes, hashed identifier links, versions, evidence states, safe counts.

**Outputs:** Audit directory containing `normalized_examples.jsonl`, `image_audit.jsonl`, `identifier_links.jsonl`, `source_manifest.json`, `extracted_files.json`, `audit_manifest.json`, `checksums.json`. Manifest records kind, input/contract hashes, counts, versions, raw-Unicode/comparison policies, validation and evidence states.

**Ordered implementation:**

1. Hash schema/mapping/validation/decoder/pixel/pHash/identifier/serialization versions into contract hash; exclude host/time/process state.
2. Serialize deterministic UTF-8 without ASCII escaping, preserving raw metadata/nulls; reject NaN/Infinity.
3. Recheck training answers, participant unlabeled state, exact image coverage, valid link case IDs, and source coverage.
4. Restrictive stage, hash children, exact checksum coverage, fsync, self-verify, atomic rename.
5. Verify identical existing content or fail conflict; never overwrite/merge.
6. Keep protected normalized content out of logs/summaries/links. Verifier rejects escape, symlink, missing/extra, noncanonical/digest/referential/provenance failure.

**Focused tests:** Both valid archives produce distinct repeat-byte-identical roots. Adversaries/tampering cover leakage, bad images, comparison collision, missing/extra/symlink/checksum files, partial root, referential mismatch, and conflict. Assert exact Unicode round-trip, participant nulls, no partial final root, and sensitive values only in intentionally protected examples.

- Exact behavior command: `uv run pytest -q tests/challenges/medreason/test_data_artifacts.py -k dat_11`.
- CLI acceptance: audit each valid package twice and verify produced root; all adversarial/tampered copies fail without changing original.

**Acceptance evidence:** Exact checksum coverage, deterministic bytes, verifier success, conflict preservation, and privacy-boundary scans.

**External-data blocker:** Real audit requires OPS-02/OPS-03/OPS-07 and at least 600 GiB reserved storage; measured local free space is 364 GiB, so full staging is blocked.

**Non-goals and failure policy:** No upload, redistribution, split/model work, or in-place migration. Any integrity/invariant failure prevents use.

**Handoff:** Downstream receives verified root, source/contract/checksum hashes, kind, versions, counts, and evidence states.

## DAT-12 Expose end-to-end MedReason data audit command

**Depends on:** DAT-01, DAT-02, DAT-03, DAT-04, DAT-05, DAT-06, DAT-07, DAT-08, DAT-09, DAT-10, DAT-11.

**Parallel safety and exclusive file ownership:** Final integration; may overlap unrelated work. Exclusively owns parser/orchestration and `tests/challenges/medreason/test_data_cli.py`. Coordinate `tests/phase_01/test_packaging.py` inventory with SCH-01; CLI-01 may route but not change this contract.

**Target paths/symbols:** `build_parser`, `audit_command`, `verify_command`, `main`, module entry point in `data.py`; CLI tests; package inventory/import coverage for `medfm.challenges` and `medfm.challenges.medreason`.

**Inputs:** Exactly one of `--archive` for `--kind train|participant` or `--input-root` for `--kind runtime`; required `--output-root`; explicit registered descriptor when needed; overrides may tighten limits only. `verify` accepts an audit root. No URL/credential.

**Outputs:** Train/participant deterministic JSON with status, kind, immutable path, source/audit/checksum hashes, counts, versions, evidence states. Runtime returns safe case/broken counts without claiming a released audit. Errors use stable nonzero classes and sanitized stderr; no partial stdout success.

**Ordered implementation:**

1. Provide `audit`/`verify`; validate source-kind combinations before extraction, torch import, CUDA query, or model allocation.
2. Orchestrate train DAT-01→02→03→04→08→09→10→11; participant substitutes DAT-05; runtime calls SCH-04/DAT-06/DAT-07 only.
3. Verify every interstage hash and clean task-owned staging on failure without touching sealed artifacts.
4. Emit canonical JSON and stable error classes for usage, access/provenance, archive security, schema, decode, integrity, conflict. Never continue after failure or emit sensitive content.
5. Reject remote input/download clients. `--help`, import, and fixtures must not initialize CUDA/XLA or import accelerator-only libraries.
6. Extend packaging inventory so built-package discovery—not editable-source luck—covers `medfm.challenges.medreason.data` and `audit`.
7. Help states registered-schema, protected-access, storage blockers, and strict released versus safe-broken runtime behavior.

**Focused tests:** Run both valid archives, valid mixed runtime, and complete adversarial catalog. Assert lightweight help, process determinism, verifiable roots, runtime safe-broken success, exact released failure codes, atomicity/privacy. Subprocess with `CUDA_VISIBLE_DEVICES=0` asserts CUDA uninitialized and `bitsandbytes`, `torch_xla`, `flash_attn`, `cucim` absent. Packaging inventory must fail if challenge modules are omitted.

- Exact behavior commands: `uv run pytest -q tests/challenges/medreason/test_data_cli.py -k dat_12` and `uv run pytest -q tests/phase_01/test_packaging.py -k 'subpackages or cpu_import or medreason'`.
- CLI acceptance: `uv run python -m medfm.challenges.medreason.data --help`; train/participant `audit --archive <fixture> --kind <kind> --output-root <tmp-root>`; runtime `audit --input-root <valid_runtime_root> --kind runtime --output-root <tmp-root>`; `verify --audit-root <produced-root>`.
- Plausible bugs: both source modes accepted, publication after validation failure, nondeterministic stdout, partial released success, global abort for safe broken runtime image, content leak, CUDA initialization, or wheel omission.

**Acceptance evidence:** Subprocess tests prove exit/status schema, two-process determinism, adversarial matrix, strict/tolerant distinction, verification, CPU-only behavior, and packaging inventory.

**External-data blocker:** Synthetic acceptance is local. Protected audit requires OPS-02/OPS-03, reviewed train mapping, evidence, and storage; official runtime also requires pinned vendored sources and participant package. The local 24,576 MiB RTX 3090 is irrelevant and proves no 48/96 GB claim.

**Non-goals and failure policy:** No network/download, model loading, split/evaluation/training, answer recovery, schema guessing, GPU fallback, or official result claim. Released failure is fatal/atomic; runtime tolerance is only safe in-root missing/corrupt images.

**Handoff:** CLI-01 receives stable arguments/exit schema; OPS-07 exact commands/blockers; downstream only verified DAT-11 roots/hashes.
