# MedReason 2026 — Governance and assets

This phase establishes legal, access, source-integrity, provenance, and machine-prerequisite gates before protected data, model, judge, or submission artifacts are touched. Current repository state is not acceptance evidence: `model_registry/licenses.yaml` has a blocked MedGemma 1.5 record, generic Gemma/Qwen records only, and no exact Gemma 4, MedGemma 27B, Llama judge, or Qwen judge record; `model_registry/v1_scope.yaml` reports MedGemma accelerator support as `UNTESTED`. Exact model rows must be added to both files atomically because `medfm.tools.governance.check_scope_consistency()` rejects orphan rows.

Verified public metadata establishes that all six planned commit SHAs exist; Gemma 4 31B/26B and Qwen are ungated, both MedGemma repositories are `gated:auto`, and Llama is `gated:manual`. Anonymous requests for protected config/template files returned 401. This proves revision existence/gating only—not access, terms acceptance, redistribution, or complete artifact integrity. Qwen uses the custom Qwen License, not the generic row's provisional Apache classification. Only a named human/license owner may review or accept terms; only a participant may attest Synapse access; only written organizer evidence may authorize a late official submission. Code must fail closed.

The measured local baseline on 2026-08-09 is an RTX 3090 with 24,576 MiB, 364 GiB free under the repository, and no configured SSH compute host, versus the plan's 600 GiB reserve and unproven 48/96 GB profiles. Commands below are future acceptance commands; this backlog writer ran no validation.

## GOV-01 Capture challenge eligibility and exception evidence

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-02 through GOV-05, GOV-07, and GOV-09. Exclusively owns `artifacts/runs/medreason/governance/challenge_eligibility.json` and `tests/challenges/medreason/fixtures/governance/challenge_eligibility_*.json`. Original organizer correspondence remains in a protected external store; this task must not edit registries, shared schemas, or another GOV artifact.

**Target paths/symbols:** Official challenge guide/FAQ from the approved plan; new ignored evidence at `artifacts/runs/medreason/governance/challenge_eligibility.json`; GOV-10 consumer `load_challenge_eligibility()` and blocker `challenge_exception_missing`.

**Inputs:** Canonical official URLs, retrieval timestamp, immutable page bytes/hash, documented normal-submission status/date, team registration and archival-track status, and—only if actually received—a written exception. Store a protected-document reference, SHA-256, scope, expiry, and human attestation; never message body, mailbox token, or personal correspondence.

**Outputs:** Strict canonical JSON with `schema_version`, `captured_at`, `official_page_urls`, `page_snapshot_sha256`, `normal_submission_status`, `normal_submission_closed_at`, `team_registration_attested`, `archival_track_eligible`, `written_exception`, `decision`, and ordered `blockers`. Default is `post_challenge_research_only`.

**Implementation:**
1. Capture and hash the exact official page bytes; do not treat an unpinned live page as permanent evidence.
2. Encode the approved-plan facts that normal submission closed July 22, 2026 and the Aug 8–17 archival track is not a route for a new team/new method, retaining source hashes.
3. Require a named human to attest team registration and a written organizer exception. Cookies, Synapse access, email-domain matches, and verbal statements prove neither.
4. Validate that an exception explicitly covers team, method/artifact, submission phase, and dates. Missing, expired, ambiguous, or unattested evidence blocks official submission.
5. Permit an explicitly labeled post-challenge research profile without an exception; never upgrade it to an official result.
6. Emit only normalized decisions, stable blocker codes, and hashes to GOV-08/GOV-10.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_eligibility.py` must accept a consistent synthetic record and reject absent signature, expiry, scope/team mismatch, missing page hash, and an archival-track claim used as submission permission. Logs must contain no source text.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_eligibility.py
MEDFM_RUN_MEDREASON_ACCESS_TESTS=1 uv run pytest -q tests/challenges/medreason/test_governance_eligibility.py -k protected
```

Protected tests skip without explicit opt-in and only validate a human-supplied file; they never contact organizers.

**Acceptance evidence:** Fixture acceptance proves parsing/fail-closed behavior only. Official acceptance additionally requires a human-attested real document hash that revalidates against the protected original and yields `official_submission_permitted`. Current external blocker: no organizer exception or team-registration attestation is available.

**Non-goals and failure policy:** No automatic contact, authenticated scraping, 2026-win claim, paper-track workaround, or correspondence storage. Ambiguity remains blocked.

**Handoff:** GOV-08 receives eligibility/source hashes; GOV-10 consumes decision/scope/expiry; OPS-01 is the external human action that can resolve the blocker.

## GOV-02 Capture Synapse access and data license evidence

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01 and GOV-03 through GOV-09. Exclusively owns `artifacts/runs/medreason/governance/synapse_access.json` and its fixtures. It must not download archives (OPS-03), store credentials, or write model-license registries.

**Target paths/symbols:** Synapse entity `syn74403682` and official wiki; ignored `synapse_access.json`; GOV-10 `load_data_access_evidence()` and blockers `synapse_access_unattested`, `data_terms_unaccepted`, `dataset_license_unresolved`.

**Inputs:** Entity/version, source-page bytes/hash, exact dataset terms/license bytes or protected reference plus hash, access-request state, named participant/license owner, acceptance timestamp, permitted/prohibited uses, and protected credential-store reference. Never record session/API tokens or account email in logs/manifests.

**Outputs:** Canonical JSON containing `schema_version`, `source_entity_id`, `source_page_url`, `source_page_sha256`, `terms_sha256`, `terms_version`, `license_identifier`, `access_status`, `terms_status`, `attested_by`, `attested_at`, `permitted_uses`, `prohibited_uses`, `archive_inventory`, `decision`, and blockers. Empty archive inventory means not downloaded, not verified.

**Implementation:**
1. Snapshot/hash the exact source page and presented terms; preserve entity/version.
2. Track independently: owner reviewed terms, human accepted terms, Synapse granted access, archives downloaded/hashed. None implies another.
3. Require human attestation; code must not log in or click acceptance.
4. Record enforceable restrictions while storing only a hash/reference when terms text is non-redistributable.
5. Leave archive names/sizes/SHA-256 absent until OPS-03 measures them; never insert planned values.
6. Fail closed on mutable/unhashed terms, missing license, pending/denied access, entity mismatch, or absent attestation.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_data_access.py` covers pending, denied, granted-but-unattested, terms mismatch, wrong entity, missing license, valid human attestation, and proves empty archive inventory cannot satisfy a data-ready gate.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_data_access.py
MEDFM_RUN_MEDREASON_ACCESS_TESTS=1 uv run pytest -q tests/challenges/medreason/test_governance_data_access.py -k protected
```

**Acceptance evidence:** Fixture tests prove validation only. Protected access requires a real human-attested record and reverified source/terms hashes. Data-ready acceptance additionally requires OPS-03 archive manifests. Current blockers: no human Synapse/access attestation, reviewed terms, or archive hashes.

**Non-goals and failure policy:** No automated acceptance/download, credential inference, or participant-validation answers/outputs. Uncertainty blocks protected data use; fixture work may continue.

**Handoff:** GOV-08 gets access/terms hashes; DAT-01/OPS-03 receive entity/version and later append measured archive manifests; GOV-10 consumes decisions; OPS-02 resolves human access.

## GOV-03 Review Gemma Four deployment license eligibility

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01, GOV-02, GOV-04, GOV-05, GOV-07, and GOV-09. Exclusively owns `artifacts/runs/medreason/governance/licenses/gemma4_review.json` and fixtures. Treat `licenses.yaml`/`v1_scope.yaml` as read-only; MOD-04/MOD-05 own their atomic paired update.

**Target paths/symbols:** `google/gemma-4-31B-it@419b2efe421994fdfd3394e621983d4cc511cd4f` and `google/gemma-4-26B-A4B-it@47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`. Proposed exact registry IDs are `gemma-4-31b-it` and `gemma-4-26b-a4b-it`. Existing `gemma-generic` describes Gemma 3, gated Gemma Terms, conditional uses, null acceptance, and `pending_review`; it cannot authorize Gemma 4.

**Inputs:** Exact model-card/license/notice bytes and page dates/hashes at each revision; code/weights license; gating; commercial/derivative/redistribution rights; prohibited uses; intended local training and Docker bundling; named owner/date. Public metadata says these snapshots are ungated and the approved plan reports Apache-2.0, but the owner must verify exact bytes.

**Outputs:** One decision per exact checkpoint with repository/revision, reviewed file hashes/page dates, proposed complete `LicenseRecord`, intended uses, bundling decision, owner/date, decision, and blockers.

**Implementation:**
1. Capture license/model-card bytes at each exact SHA without weight download; reject `main`/aliases.
2. Review code and weights separately for research, adapters, Docker redistribution, and commercial use; never inherit `gemma-generic`.
3. Require human review/approval. Ungated status removes click-through access but does not let software self-approve legal eligibility.
4. Apply `license_schema.json` cross-field policy: unresolved use fields require `blocked_unresolved`; no fabricated acceptance date.
5. Keep legal, artifact, and hardware states separate.
6. Hand both proposed rows to MOD-04/MOD-05; both `licenses.yaml` and `v1_scope.yaml` must change together with exact revisions and `UNTESTED` backends.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_gemma4_license.py` rejects generic-row reuse, wrong revision, license hash/date mismatch, unresolved redistribution marked approved, and unsigned review; accepts human-reviewed permissive fixtures without claiming remote artifact access.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_gemma4_license.py
uv run pytest -q tests/phase_00/test_license_schema.py tests/phase_00/test_scope_consistency.py
```

The phase-00 command runs only after MOD-04/MOD-05 update both registries.

**Acceptance evidence:** Fixture acceptance proves policy shape. Protected legal acceptance requires owner-signed exact license/model-card hashes. GOV-06/GOV-09 separately prove bytes/hardware. Until owner signature, exact checkpoints stay blocked despite public metadata.

**Non-goals and failure policy:** Do not accept terms, download weights, claim accelerator support, reuse Gemma 3 terms, or infer deployment rights from ungated access. Conflicting terms block only the affected snapshot.

**Handoff:** MOD-04/MOD-05 consume proposed rows/evidence hash; GOV-06 consumes access decision; GOV-10 requires exact IDs, not generic substitutions.

## GOV-04 Review MedGemma terms and redistribution eligibility

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01 through GOV-03, GOV-05, GOV-07, and GOV-09. Exclusively owns `artifacts/runs/medreason/governance/licenses/medgemma_review.json` and fixtures. MOD-06/MOD-07 own registry changes.

**Target paths/symbols:** `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` (`medgemma-1.5-4b`) and `google/medgemma-27b-it@2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f` (`medgemma-27b-it`). Both are `gated:auto`; anonymous protected config/template requests returned 401. Existing 4B row has unresolved weights/HAI-DEF/redistribution, null acceptance, `blocked_unresolved`; all accelerator states are `UNTESTED`.

**Inputs:** Exact HAI-DEF terms/model-card bytes, page dates and hashes for each SHA; gating and named human acceptance; research, derivative/adapter, redistribution clauses; prohibited uses; local and Docker intentions.

**Outputs:** Independent local research, adapter, and redistribution decisions per revision, with terms version/hash/date, `accepted_by`/date, proposed registry fields, diagnostic-only flag for 27B, and blockers.

**Implementation:**
1. Preserve current blocked state until exact terms are reviewed; never substitute another HAI-DEF version or marketing summary.
2. Only the named owner may accept terms. Code validates evidence; no browser/token automation.
3. Separate local access, adapter derivation, and redistribution. Access is not Docker permission.
4. Any unresolved use stays `blocked_unresolved` under existing policy.
5. If local access fails, disable both MedGemma routes. If redistribution fails, omit MedGemma from Docker while Gemma 4-only continues. Keep 27B frozen zero-shot diagnostic only.
6. Hand exact rows to MOD-06/MOD-07 with paired scope updates and `UNTESTED` hardware.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_medgemma_license.py` covers no acceptance, local-only with unresolved redistribution, prohibited redistribution, terms/revision mismatch, 27B promotion attempt, valid local-only route, and Gemma 4-only continuation.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_medgemma_license.py
MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -q tests/challenges/medreason/test_governance_medgemma_license.py -m real_checkpoint
uv run pytest -q tests/phase_00/test_license_schema.py tests/phase_00/test_scope_consistency.py
```

**Acceptance evidence:** Fixture acceptance proves routing/policy only. Protected legal acceptance needs human acceptance and exact terms hashes; Docker additionally needs affirmative redistribution. Current blockers are HAI-DEF/team acceptance and redistribution; no hardware claim is possible here.

**Non-goals and failure policy:** No click-through, access-as-redistribution inference, 27B training/fusion/deployment, or false availability. Unresolved redistribution removes MedGemma only, not Gemma 4.

**Handoff:** MOD-06/MOD-07 consume rows; GOV-06 consumes authorized access; DOC-04 gets include/omit decision; GOV-10 gets separate local and redistribution gates.

## GOV-05 Review proxy judge licenses and acceptance

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01 through GOV-04, GOV-07, and GOV-09. Exclusively owns `artifacts/runs/medreason/governance/licenses/judges_review.json` and fixtures. MOD-08 owns atomic registry/scope changes.

**Target paths/symbols:** `meta-llama/Llama-3.1-70B-Instruct@1605565b47bb9346c5515c34102e054115b4f98b` (`llama-3.1-70b-instruct-judge`) and `Qwen/Qwen2.5-VL-72B-Instruct@89c86200743eec961a297729e7990e8f2ddbc4c5` (`qwen2.5-vl-72b-instruct-judge`). Llama is `gated:manual`; Qwen is ungated but uses custom Qwen License—not `qwen-generic`'s provisional Apache classification. Judge assets remain under `artifacts/judges/medreason/` and never ship.

**Inputs:** Exact model-card/license bytes/page dates/hashes, access/gating, human acceptance, permissions for local quantized proxy use, restrictions, named owner/date, and separately measured artifact availability.

**Outputs:** One exact proposed `LicenseRecord`/decision per judge, with hashes/dates, human acceptance fields, allowed local use, prohibited redistribution, separate availability state, and blockers.

**Implementation:**
1. Review each exact snapshot independently; reject provider-family/generic substitutions.
2. Require human acceptance for Llama's gated/custom terms and human review for Qwen's custom license. Ungated Qwen does not mean Apache or preapproved.
3. Confirm local 4-bit sequential proxy use and no redistribution/API dependency.
4. Keep legal/artifact/runtime states separate.
5. Require both judges legally approved, content-verified, and runnable before GT/VA proxy promotion or winning-caliber claims. Diagnostics may not impersonate missing judges.
6. Hand exact paired records to MOD-08.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_judge_license.py` covers missing Meta row, generic-Qwen substitution, false Apache claim, one-of-two approval, unaccepted Llama terms, revision mismatch, prohibited quantization, and joint legal pass.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_judge_license.py
MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -q tests/challenges/medreason/test_governance_judge_license.py -m real_checkpoint
uv run pytest -q tests/phase_00/test_license_schema.py tests/phase_00/test_scope_consistency.py
```

**Acceptance evidence:** Fixture tests prove joint gating only. Protected legal acceptance needs both human-reviewed exact records. Promotion additionally needs GOV-06/EVA-15 manifests/hardware. Current blockers: exact team acceptance, snapshot access for Llama, storage, and 70B/72B hardware.

**Non-goals and failure policy:** No automatic terms acceptance, hosted API, substitute judge, judge redistribution, or inference of unpublished organizer prompts/quantization/ties. Either missing judge blocks promotion.

**Handoff:** MOD-08 consumes exact rows; GOV-06/OPS-05 consume access; EVA-06/EVA-07 receive IDs/evidence hashes; EVA-08/GOV-10 consume joint status.

## GOV-06 Verify every exact remote model revision

**Depends on:** GOV-03, GOV-04, GOV-05.

**Parallel safety and exclusive file ownership:** May run with GOV-01, GOV-02, GOV-07, and GOV-09 after legal reviews. Exclusively owns `artifacts/runs/medreason/governance/remote_revisions.json`, `snapshot_manifests/`, and `tests/challenges/medreason/test_governance_revisions.py`; no production downloads or registry edits.

**Target paths/symbols:** Verify all six exact IDs/SHAs from GOV-03 through GOV-05. New symbols later consumed by GOV-10: `ExpectedRevision`, `SnapshotFile`, `verify_snapshot_manifest()`. Public verification already proves all six SHA objects exist; it does not prove protected access or bytes.

**Inputs:** Exact repo IDs/40-character SHAs, legal decisions, remote commit metadata, file listing/hash identity, and local bytes only when authorized. Include processor/tokenizer/config/chat-template/model-card/license and all weight-index/shards. Existing `medfm.registry.weights.SAFE_EXTENSIONS` excludes `.jinja`; this task must inventory required `.jinja` templates explicitly rather than silently omit them.

**Outputs:** Canonical revision index and one manifest per snapshot with requested/resolved commit, gating, legal-evidence hash, page dates, file names/sizes/content hashes, required-file roles, `metadata_verified`, `content_verified`, and blockers.

**Implementation:**
1. Enforce fixed six-item allowlist and full SHA equality; reject branches/tags/aliases.
2. Metadata-query protected repos only as legally allowed; record 401/gating as blocked, never retry with guessed access.
3. Enumerate complete runtime files. Remote ETags/LFS pointers prove neither local SHA-256 nor complete bytes.
4. Require complete shard-index closure and required processor/chat-template assets; explicitly hand `.jinja` safe-file review to MOD-02.
5. Reject unclassified executable/custom-code files pending review; preserve April 2 Gemma 4 processor/template snapshot.
6. Re-hash all local bytes before use. Metadata-only status cannot satisfy load/release.
7. Normalize a canonical HF `repo_id` separately from human-facing URL; current `ModelSpec.repository` ambiguity must not feed a URL to Hub APIs.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_revisions.py` uses fake Hub metadata and covers all six, moving refs, resolved mismatch, missing/duplicate shard, changed bytes, absent processor/`.jinja`, URL-vs-repo-ID confusion, unknown executable, legal block, and metadata/content distinction.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_revisions.py
MEDFM_RUN_REAL_CHECKPOINTS=1 uv run pytest -q tests/challenges/medreason/test_governance_revisions.py -m real_checkpoint
```

**Acceptance evidence:** Fixture acceptance proves contracts only. Metadata acceptance records six exact resolved SHAs. Protected acceptance requires selected-route bytes to re-hash, including processors/templates. Current blockers: protected terms/access, 600 GiB staging, and absent local snapshots.

**Non-goals and failure policy:** No default download, `main`, invented hash, inference, or legal/hardware claim. Metadata cannot stand in for content.

**Handoff:** GOV-08 gets snapshot roots; MOD-01/MOD-02/MOD-04–08 receive exact repo-ID/revision/file contracts; DOC-04 gets content-verified selected assets; GOV-10 gets blockers.

## GOV-07 Pin official Docker sources and checksums

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01 through GOV-06 and GOV-09. Exclusively owns `model_registry/medreason_official_sources.yaml`, `artifacts/runs/medreason/governance/official_source_manifest.json`, and `tests/challenges/medreason/test_official_source_lock.py`. EVA-01 exclusively owns evaluator bytes under `docker/medreason/vendor/MedReason-Evaluation/` and their `docker/medreason/vendor/SHA256SUMS.json` entries; DOC-01 owns Docker context.

**Target paths/symbols:** Official upstream `medreason26/MedReason-Challenge-Docker` at exactly `05748c0341b72dc08132bd108208b78dc14a2f0b`. Lock observed upstream paths/bytes used for version-1.0 runtime schema, base/template, public evaluator, validator, and smoke contract; invent no path/digest before observing the commit tree.

**Inputs:** Repository URL, exact commit/tree, observed file list, raw bytes/SHA-256, upstream license/notice hashes.

**Outputs:** Checked-in deterministic lock with repository, commit/tree verification, allowlisted observed paths, roles, sizes, SHA-256, destination owner, and notices; ignored acquisition manifest with tool/version and verification result.

**Implementation:**
1. Resolve exact commit/tree, never `main`.
2. Identify actual relevant files at that tree and hash raw bytes without newline normalization.
3. Assign every role/destination owner and keep official bytes separate from custom wrappers.
4. Reject duplicate roles, traversal, missing file, checksum mismatch, unknown addition, or changed commit.
5. Leave byte-for-byte vending to EVA-01; wrappers cannot modify a vendored file while retaining its official checksum.
6. Require reviewed lock update for any upstream change.

**Focused tests and exact commands:** `tests/challenges/medreason/test_official_source_lock.py` uses a fake tree for commit/path/hash success and branch, commit mismatch, modified byte, omitted role, duplicate destination, traversal, and unknown file failures. Protected refresh only compares bytes.

```bash
uv run pytest -q tests/challenges/medreason/test_official_source_lock.py
MEDFM_RUN_MEDREASON_ACCESS_TESTS=1 uv run pytest -q tests/challenges/medreason/test_official_source_lock.py -k protected
```

**Acceptance evidence:** Fixture acceptance proves verifier behavior. Source acceptance requires exact commit plus measured hashes. Vendored acceptance is later EVA-01 byte comparison. No digest is claimed before measurement.

**Non-goals and failure policy:** Do not vendor/edit evaluator bytes, follow latest, infer judge behavior, or claim compatibility from fixtures. Any mismatch blocks official wrapper/release.

**Handoff:** GOV-08 receives lock root; EVA-01 consumes exact evaluator paths/digests; DOC-01/08/09 consume other official roles; GOV-10 requires verified source lock.

## GOV-08 Define immutable artifact provenance manifest contract

**Depends on:** GOV-01, GOV-02, GOV-06, GOV-07.

**Parallel safety and exclusive file ownership:** Starts after evidence field sets stabilize. Exclusively owns `model_registry/medreason_artifact_manifest.schema.json`, `ARTIFACT_MANIFEST_SCHEMA_PATH`, `validate_artifact_manifest()`, and `canonical_manifest_sha256()` additions in `medfm/tools/governance.py`, plus `tests/challenges/medreason/test_artifact_provenance.py`. Do not change model license or phase acceptance schemas.

**Target paths/symbols:** Reuse `medfm.tools.governance.load_json()`, `_schema_errors()`, date normalization, and fail-closed style. Cover source data, official code, models/processors, adapters, judges, rubrics, calibration, wheelhouse/base images, and research/deployment bundles under ignored artifact roots.

**Inputs:** Evidence/hash requirements from predecessors, existing license schema, exact revisions/official commit, restrictions, parent lineage.

**Outputs:** Draft-2020-12 strict schema (`additionalProperties: false`). Required fields: `schema_version`, `manifest_id`, `artifact_type`, `logical_name`, `created_at`, `created_by`, `source_uri`, `source_revision`, `license_record_id`, `license_evidence_sha256`, `access_evidence_sha256`, `files`, `parents`, `restrictions`, `acceptance_class`, `content_root_sha256`. File entries require relative `path`, `size_bytes`, lowercase SHA-256. Acceptance classes are `fixture`, `protected_artifact`, `hardware_measured`; hardware adds device/profile/smoke evidence.

**Implementation:**
1. Reference existing license/access records by hash; prohibit secrets, correspondence, raw case/reference text, credentials, and absolute paths.
2. Define canonical UTF-8 JSON and root over normalized metadata plus sorted files/parents, excluding only the root field.
3. Validate unique contained paths, sizes/hashes, exact revision, parents, and class-specific evidence.
4. Re-hash closed payloads; reject missing/extra/changed files and changed parents.
5. Prevent fixture manifests satisfying protected/hardware gates.
6. Corrections create new manifest/root linked to superseded root; never mutate evidence behind a root.
7. Keep validation CPU-safe and free of eager CUDA/HF imports.

**Focused tests and exact commands:** `tests/challenges/medreason/test_artifact_provenance.py` covers deterministic roots/mapping order, Unicode, duplicate/traversal/absolute paths, changed byte/size, missing/extra files, invalid legal hash, parent mutation, unknown field, and fixture-to-hardware promotion.

```bash
uv run pytest -q tests/challenges/medreason/test_artifact_provenance.py
uv run pytest -q tests/phase_00/test_license_schema.py tests/phase_00/test_acceptance_schema.py
```

**Acceptance evidence:** Fixture golden manifests prove schema/hash behavior only. Protected acceptance requires real bytes and accepted parent evidence. Hardware requires matching protected smoke. Schema success alone proves neither.

**Non-goals and failure policy:** No embedded payloads/PHI, duplicated license policy, mutable aliases, or absent-byte blessing. Invalid lineage/hash is fatal.

**Handoff:** DAT-11, MOD-11, EVA-09, SEL-14/16, DOC-16 and GOV-10 consume canonical roots, not informal filenames.

## GOV-09 Measure compute storage and runtime prerequisites

**Depends on:** none.

**Parallel safety and exclusive file ownership:** May run with GOV-01 through GOV-07. Exclusively owns `artifacts/runs/medreason/governance/prerequisites.json`, `tests/challenges/medreason/test_prerequisite_measurement.py`, and measurement behavior exposed by GOV-10. It must not change accelerator statuses.

**Target paths/symbols:** GOV-10 symbols `measure_prerequisites()`, `GpuMeasurement`, `StorageMeasurement`, `PrerequisiteDecision`. Ground in `v1_scope.yaml`/`check_accelerator_policy()`: `SUPPORTED_*` needs exact `smoke_config` plus `last_success_date`; blanket support is forbidden.

**Inputs:** Measured GPU model/hashed UUID/VRAM/driver/runtime, filesystem free bytes, sanitized compute inventory, selected 24/48/96 profile, required artifact roots, 600 GiB reserve, and exact smoke results if present.

**Outputs:** Canonical prerequisites JSON with observed time, probe/tool versions, local GPU/storage facts, remote summary, thresholds, per-profile status, hardware evidence roots, blockers. Initial facts are RTX 3090/24,576 MiB, 364 GiB, zero SSH hosts: fixture work actionable; full staging and 31B/26B/70B/72B plus 48/96 gates blocked.

**Implementation:**
1. Measure, never estimate; sanitize user/host/device IDs and never inspect credentials.
2. Compare integer bytes to 600 GiB, avoiding rounded passes.
3. Separate fixture capability from protected profiles; 24,576 MiB cannot prove 48/96 GB.
4. Require exact model/profile peak allocated/reserved VRAM, latency, driver/kernel, and smoke artifact before support status.
5. Keep 96 GB quality and 48 GB compatibility independent. A 45 GB allocation on 96 GB does not prove a 48 GB end-to-end run.
6. Keep storage shortfall hard-blocking; do not delete frozen evidence to pass.
7. Fail closed on missing tools, ambiguous GPU, stale evidence, or output mismatch.

**Focused tests and exact commands:** `tests/challenges/medreason/test_prerequisite_measurement.py` injects fake probes for threshold pass, one-byte shortfall, RTX-3090 fixture-only, no hosts, stale/ambiguous GPU, false 96-to-48 inference, and support without smoke. Real probe uses `@pytest.mark.gpu`/`MEDFM_RUN_GPU_TESTS=1`.

```bash
uv run pytest -q tests/challenges/medreason/test_prerequisite_measurement.py
MEDFM_RUN_GPU_TESTS=1 uv run pytest -q tests/challenges/medreason/test_prerequisite_measurement.py -m gpu
uv run pytest -q tests/phase_00/test_accelerator_policy.py
```

**Acceptance evidence:** Fixture tests prove parsing/thresholds. Current local report is accepted blocker evidence, not capability. Protected 48/96 acceptance requires a matching GOV-08 hardware manifest and smoke. Current blockers: fresh measurement of at least 600 GiB capacity, matching GPU access, and judge/training compute.

**Non-goals and failure policy:** No parameter-count estimates, cross-device support claims, provisioning, secret inspection, or registry promotion from probe presence. Missing/stale data blocks.

**Handoff:** GOV-10 consumes statuses/roots; MOD-13/EVA-15/DOC-15 provide protected measurements; OPS-06 resolves infrastructure.

## GOV-10 Implement fail-closed MedReason preflight gate

**Depends on:** GOV-01, GOV-02, GOV-03, GOV-04, GOV-05, GOV-06, GOV-07, GOV-08, GOV-09, SCH-01.

**Parallel safety and exclusive file ownership:** Runs after governance contracts and package skeleton. Exclusively owns `medfm/challenges/medreason/governance.py`, `tests/challenges/medreason/test_governance_preflight.py`, preflight fixtures, and the MedReason-specific nested import assertion in `tests/phase_01/test_packaging.py`. Coordinate packaging inventory with SCH-01, which owns initial `challenges` package creation. Do not edit evidence, registries, locks, or schema.

**Target paths/symbols:** `PreflightProfile`, `PreflightBlocker`, `PreflightReport`, `load_challenge_eligibility()`, `load_data_access_evidence()`, `load_license_reviews()`, `verify_snapshot_manifest()`, `measure_prerequisites()`, `evaluate_preflight()`, `main()`. Stable module command: `python -m medfm.challenges.medreason.governance`. Reuse current governance validators and remain CPU-safe.

Also gate existing `medfm.registry.weights.resolve_local_path()` and `download_weights()` call paths before cache resolution/network/model allocation: today cached resolution bypasses governance and download checks only acceptance, not a blocked license status. Require the caller to supply a passed preflight/evidence root; reject blocked/unaccepted exact IDs before calling either function. Normalize Hub `repo_id` separately from human-facing repository URL. Exact chat-template `.jinja` inclusion remains hash-allowlisted per GOV-06/MOD-02, not a blanket safe-extension relaxation.

**Inputs:** GOV-01–09 evidence, `licenses.yaml`, `v1_scope.yaml`, strict schemas, selected exact six-ID subset, operation (`fixture`, `post_challenge_research`, `proxy_promotion`, `official_submission`, `docker_release`), and expected frozen roots.

**Outputs:** Canonical `artifacts/runs/medreason/governance/preflight_report.json` with schema/profile, `passed|blocked`, ordered blocker codes, evidence roots, exact selected revisions, disabled optional routes, prerequisite measurements, timestamp, and sanitized remediations. Exit 0 only when requested profile passes. Fixture reports remain `acceptance_class: fixture`.

**Implementation:**
1. Strictly reject missing/unknown/duplicate/stale/malformed/mismatched evidence before cache lookup, model allocation, network, or data opening.
2. Validate paired registry/scope consistency. All six exact catalog IDs/revisions must have exact rows; `gemma-generic`/`qwen-generic` cannot substitute. Apply unresolved/gated acceptance rules.
3. Profiles: research requires legal/access/revision/storage/hardware for selected assets; official additionally requires written exception and Synapse attestation; proxy promotion requires both judges; Docker requires content manifests and redistribution for every bundled asset.
4. Optional MedGemma legal/redistribution failure visibly disables that route while approved Gemma 4-only may continue; never silently change selected architecture.
5. Enforce evidence class: fixture ≠ protected bytes ≠ matching 48/96 hardware.
6. Return stable sanitized codes; never log questions, answers, metadata, prompts, traces, credentials, correspondence, terms text, or protected paths.
7. Perform no terms acceptance, download, network, model allocation, or archive extraction. Write reports atomically and reject expected-root mismatch.
8. Add `measure-prerequisites` and `preflight` subcommands with argument validation before work.
9. Update packaging inventory so `medfm.challenges.medreason.governance` imports in CPU-only process without initializing CUDA/XLA or importing bitsandbytes/Transformers/Hub.
10. Migrate every MedReason model/cache/download caller to require the passed report/root; no bypass for already-cached weights.

**Focused tests and exact commands:** `tests/challenges/medreason/test_governance_preflight.py` builds a passing tiny fixture then mutates missing exception/access, unapproved exact license, generic substitution, wrong Qwen license, one missing judge, wrong revision, changed source/parent, insufficient storage, wrong GPU, fixture evidence in protected profile, optional MedGemma redistribution, sensitive text, cached blocked model, and download of blocked status. Assert nonzero, deterministic blockers, no pre-failure protected I/O, sanitized logs, and Gemma 4-only behavior. Add packaging inventory coverage.

```bash
uv run pytest -q tests/challenges/medreason/test_governance_preflight.py
uv run pytest -q tests/phase_01/test_packaging.py -k 'subpackages_importable or forbidden_top_level or cpu_import'
uv run python -m medfm.challenges.medreason.governance --help
uv run python -m medfm.challenges.medreason.governance preflight \
  --profile fixture \
  --evidence-dir tests/challenges/medreason/fixtures/governance \
  --output artifacts/runs/medreason/governance/preflight_report.json
```

Protected tests remain explicitly guarded:

```bash
MEDFM_RUN_REAL_CHECKPOINTS=1 MEDFM_RUN_GPU_TESTS=1 MEDFM_RUN_MEDREASON_ACCESS_TESTS=1 \
  uv run pytest -q tests/challenges/medreason/test_governance_preflight.py -m real_checkpoint
```

Environment variables opt into execution; they do not attest licenses/access.

**Acceptance evidence:** Fixture acceptance is focused output, help/fixture command, deterministic report hash, and packaging coverage. Protected research requires human legal/data evidence, content-verified artifacts, storage, and matching hardware. Proxy adds both judges/sequential evidence; official adds organizer exception; Docker adds redistribution and official-source verification. With current facts, every protected/official profile must block; a pass without new external evidence is a defect.

**Non-goals and failure policy:** No consent/access attestation, organizer contact, download, allocation, evidence repair, blocker suppression, model/judge substitution, or winning/official claim. Mandatory inconsistency fails before protected I/O; optional failure only disables the named optional route and stays visible.

**Handoff:** Every protected CLI/config calls preflight before data/model/judge/Docker work. Downstream consumes canonical report root, selected exact revision roots, route decisions, and blockers. OPS-01–OPS-06 supply the external evidence needed to unblock protected profiles.
